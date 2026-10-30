// Copyright 2026 Alpha Lam <arufa.hc@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "np_log.h"
#include "np_output.h"

#include <lcms2.h>
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "dcraw/gamma_curve.h"
#include "elle_icc_profiles/sRGB_elle_V2_g10.h"
#include "elle_icc_profiles/sRGB_elle_V2_srgbtrc.h"
#include "np_geometry.h"
#include "np_profile.h"
#include "nlohmann/json.hpp"
#include "tiff_head.h"

int write_tiff(LibRaw* proc, const std::string& attach_profile, const std::string& output,
               const std::vector<uint8_t>& embedded_icc, const uint16_t* rgb3) {
  unsigned* output_profile = NULL;
  unsigned profile_size = 0;
  bool dynamically_allocated = false;

  if (!attach_profile.empty()) {
    // If the profile to attach is not the psuedo "srgb" profile, the profile will be attached to the TIFF.
    // This is a hack to force LibRaw write the ICC profile in the TIFF without conversion.
    log_info("Attaching profile: %s\n", attach_profile.c_str());
    if (attach_profile == "srgb") {
      output_profile = reinterpret_cast<unsigned int*>(sRGB_elle_V2_srgbtrc_icc);
      profile_size = sRGB_elle_V2_srgbtrc_icc_len;
    } else if (attach_profile == "srgb-g10" || attach_profile == "srgb-linear") {
      output_profile = reinterpret_cast<unsigned int*>(sRGB_elle_V2_g10_icc);
      profile_size = sRGB_elle_V2_g10_icc_len;
    } else {
      if (read_profile(attach_profile, &output_profile, &profile_size)) {
        return -1;
      }
      dynamically_allocated = true;
    }
  } else if (!embedded_icc.empty()) {
    log_info("Attaching profile: [bundle embedded ICC] (%u bytes)\n", (unsigned)embedded_icc.size());
    output_profile = reinterpret_cast<unsigned int*>(const_cast<uint8_t*>(embedded_icc.data()));
    profile_size = (unsigned)embedded_icc.size();
  }

  const unsigned height = proc->imgdata.sizes.iheight;
  const unsigned width = proc->imgdata.sizes.iwidth;
  struct tiff_hdr header;
  tiff_head(proc, &header, profile_size);
  auto* fp = fopen(output.c_str(), "wb+");
  if (!fp) {
    log_error("Cannot open %s for writing\n", output.c_str());
    if (dynamically_allocated && output_profile) free(output_profile);
    return -1;
  }
  fwrite(&header, sizeof(header), 1, fp);
  if (profile_size) {
    fwrite(output_profile, profile_size, 1, fp);
  }
  if (dynamically_allocated && output_profile) {
    free(output_profile);
  }

  if (rgb3) {
    fwrite(rgb3, 3 * sizeof(uint16_t), (size_t)width * height, fp);
  } else {
    std::vector<ushort> row_buf(width * 3);
    for (unsigned row = 0; row < height; ++row) {
      for (unsigned col = 0; col < width; ++col) {
        row_buf[col * 3]     = proc->imgdata.image[row * width + col][0];
        row_buf[col * 3 + 1] = proc->imgdata.image[row * width + col][1];
        row_buf[col * 3 + 2] = proc->imgdata.image[row * width + col][2];
      }
      fwrite(row_buf.data(), 3 * sizeof(ushort), width, fp);
    }
  }
  fclose(fp);
  return 0;
}

#ifndef NEGICC_GIT_VERSION
#define NEGICC_GIT_VERSION "master"
#endif

using ordered_json = nlohmann::ordered_json;

// A number as the sidecar has always written it: printed with fmt, then read back (an integer when it prints as one),
// so the values do not change with the writer; null when it is not finite.
static ordered_json num(double val, const char* fmt = "%.6g") {
  if (!std::isfinite(val)) return nullptr;
  char buf[64];
  snprintf(buf, sizeof(buf), fmt, val);
  if (!strpbrk(buf, ".eE")) return strtoll(buf, nullptr, 10);
  return strtod(buf, nullptr);
}

// A number as text, printed with fmt (the cli_equivalent's arguments).
static std::string num_str(double val, const char* fmt) {
  char buf[64];
  snprintf(buf, sizeof(buf), fmt, val);
  return buf;
}

static ordered_json rect_json(const FilmRect& r) {
  return {{"cx", num(r.cx, "%.1f")}, {"cy", num(r.cy, "%.1f")}, {"w", num(r.w, "%.1f")}, {"h", num(r.h, "%.1f")},
          {"angle_deg", num(r.angle_deg, "%.3f")}};
}

static ordered_json frame_detection_json(const FrameDetection& fd) {
  // Sensor pixel coordinates throughout (the detector ran on the RAW before the decode). aspect and format describe
  // the frame as the previous detector did ("3:2", ...); film_format is the format the detection used. axis is "h" or
  // "v", or null when the detector left it undecided (a single square frame with nothing to orient it).
  const FilmRect& pr = fd.primary.rect;
  const float aspect = std::max(pr.w, pr.h) / std::max(1.0f, std::min(pr.w, pr.h));
  const char* ratio = "-";
  switch (fd.format) {
    case FILM_35MM: ratio = "3:2"; break;
    case FILM_645: ratio = "4:3"; break;
    case FILM_6X6: ratio = "1:1"; break;
    case FILM_6X7: ratio = "7:6"; break;
    case FILM_6X9: ratio = "3:2"; break;
    default: break;
  }
  ordered_json j;
  j["ok"] = fd.ok;
  j["axis"] = fd.axis == 'h' ? ordered_json("h") : fd.axis == 'v' ? ordered_json("v") : ordered_json(nullptr);
  j["confidence"] = num(fd.conf, "%.2f");
  j["low_confidence"] = fd.conf < kLowFrameConfidence;
  j["reason"] = fd.reason;
  ordered_json gaps = ordered_json::array();
  for (int g = 0; g < 2; ++g)
    if (fd.gaps[g][0] >= 0) gaps.push_back({fd.gaps[g][0], fd.gaps[g][1]});
  j["gaps"] = gaps;
  // segments: the frames on the strip, in strip order (secondaries are sorted by index, negative before the primary);
  // selected: the primary's position among them.
  int selected = 0;
  for (const FilmFrame& f : fd.secondaries) selected += f.index < 0 ? 1 : 0;
  j["segments"] = 1 + (int)fd.secondaries.size();
  j["n_gaps"] = fd.n_gaps;
  j["selected"] = selected;
  j["aspect"] = num(aspect, "%.4g");
  j["format"] = ratio;
  j["film_format"] = film_format_name(fd.format);
  j["orientation"] = fd.orientation;
  j["rotation_deg"] = num(fd.rotation_deg, "%.3f");
  j["px_per_mm"] = num(fd.px_per_mm, "%.2f");
  j["light_clipped"] = fd.light_clipped;
  j["crop"] = {fd.crop[0], fd.crop[1], fd.crop[2], fd.crop[3]};
  j["primary"] = rect_json(pr);
  ordered_json sec = ordered_json::array();
  for (const FilmFrame& f : fd.secondaries)
    sec.push_back({{"index", f.index}, {"rect", rect_json(f.rect)},
                   {"visible", {f.visible[0], f.visible[1], f.visible[2], f.visible[3]}},
                   {"visible_fraction", num(f.visible_fraction, "%.3f")}});
  j["secondaries"] = sec;
  j["film_base"] = fd.has_film_base ? rect_json(fd.film_base) : ordered_json(nullptr);
  j["base_patch"] = fd.has_base ? rect_json(fd.base_patch) : ordered_json(nullptr);
  j["sprockets"] = (int)fd.sprockets.size();
  return j;
}

static ordered_json auto_json(const ParamSearchResult& a, const int* frame_half, const Sidecar& sc) {
  ordered_json j;
  j["target"] = a.target_name;
  j["target_idx"] = a.target_idx;
  j["e"] = num(a.params.e, "%.9g");
  j["g"] = num(a.params.g, "%.9g");
  j["b"] = num(a.params.b, "%.9g");
  j["residual"] = num(a.residual, "%.6g");
  j["status"] = to_string(a.status);
  j["e_norm"] = num(a.e_norm, "%.9g");
  ordered_json m = ordered_json::array();
  for (int i = 0; i < 9; ++i) m.push_back(num(a.input_matrix[i], "%.9g"));
  j["input_matrix"] = m;
  j["frame_half"] = {frame_half[0], frame_half[1], frame_half[2], frame_half[3]};
  j["g_lo"] = num(a.g_lo, "%.6g");
  j["g_hi"] = num(a.g_hi, "%.6g");
  j["scene_ev"] = num(a.scene_ev, "%.4g");
  j["headroom"] = {{"applied", headroom_applied(a.headroom)}, {"ev", num(a.headroom_ev, "%.4g")},
                   {"reason", to_string(a.headroom)}};
  j["rescue"] = {{"branch", a.rescue == RescueOutcome::kNotNeeded ? "N" : "R"},
                 {"white_clip", num(a.rescue_white_clip, "%.6g")}, {"changed", a.rescue == RescueOutcome::kTaken}};
  ordered_json screening = ordered_json::array();
  for (const auto& r : a.screening)
    screening.push_back({{"target_idx", r.target_idx}, {"e", num(r.params.e)}, {"g", num(r.params.g)},
                         {"b", num(r.params.b)}, {"intent_loss", num(r.intent_loss)}});
  j["screening"] = screening;
  ordered_json tournament = ordered_json::array();
  for (const auto& r : a.tournament)
    tournament.push_back({{"target_idx", r.target_idx}, {"e", num(r.params.e)}, {"g", num(r.params.g)},
                          {"b", num(r.params.b)}, {"residual", num(r.residual)}, {"status", to_string(r.status)},
                          {"iters", r.steps}});
  j["tournament"] = tournament;
  j["model"] = sc.dino_model;
  j["backend"] = sc.dino_backend;
  j["inferences"] = a.inferences;
  j["time_ms"] = num(sc.t_auto_ms, "%.1f");
  return j;
}

// One step of the single-profile search: its parameters, losses and penalties.
template <typename It>
static ordered_json dino_step_json(const It& it, bool with_residual) {
  ordered_json j = {{"e", num(it.e)}, {"g", num(it.g)}, {"b", num(it.b)}};
  if (with_residual) j["residual"] = num(it.residual);
  j["loss_L"] = num(it.loss_L);
  j["loss_a"] = num(it.loss_a);
  j["loss_b"] = num(it.loss_b);
  j["clip_pct"] = num(it.clip_pct);
  j["penalty_clip"] = num(it.penalty_clip);
  j["penalty_anchor"] = num(it.penalty_anchor);
  j["objective"] = num(it.objective);
  return j;
}

static ordered_json dino_json(const DinoSolveResult& d, const Sidecar& sc) {
  ordered_json j;
  j["model"] = sc.dino_model;
  j["backend"] = sc.dino_backend;
  j["grid"] = {d.grid_w, d.grid_h};
  j["target"] = {d.target_w, d.target_h};
  j["e_center"] = num(d.e_center);
  j["e_bounds"] = {num(d.bounds_lo), num(d.bounds_hi)};
  j["status"] = d.status;
  j["best_iteration"] = d.best_iteration;
  j["inferences"] = d.total_inferences;
  j["solved_relative"] = {{"e", num(d.solved_relative.e)}, {"g", num(d.solved_relative.g)},
                          {"b", num(d.solved_relative.b)}};
  const bool has_bootstrap = !d.iterations.empty() && d.iterations[0].tag == 'b';
  j["bootstrap"] = has_bootstrap ? dino_step_json(d.iterations[0], false) : ordered_json(nullptr);
  ordered_json its = ordered_json::array();
  for (const auto& it : d.iterations)
    if (it.tag != 'b') its.push_back(dino_step_json(it, true));
  j["iterations"] = its;
  ordered_json tone = ordered_json::array();
  for (const auto& t : d.tone_mass) tone.push_back(num(t));
  j["tone_mass"] = tone;
  return j;
}

bool write_json_sidecar(const Sidecar& sc) {
  ordered_json j;
  j["schema"] = "negicc.neg_process.sidecar/1";
  j["tool"] = {{"name", "neg_process"}, {"version", NEGICC_GIT_VERSION}};
  ordered_json argv = ordered_json::array();
  for (int i = 0; i < sc.argc; ++i) argv.push_back(sc.argv[i]);
  j["argv"] = argv;

  std::string camera = "Unknown";
  if (sc.proc) {
    const std::string make = sc.proc->imgdata.idata.make;
    const std::string model = sc.proc->imgdata.idata.model;
    if (!make.empty() && !model.empty()) camera = model.find(make) == 0 ? model : make + " " + model;
    else if (!model.empty()) camera = model;
    else if (!make.empty()) camera = make;
  }
  j["inputs"] = {{"raw_files", sc.files},
                 {"camera", camera},
                 {"iso", sc.proc ? num(sc.proc->imgdata.other.iso_speed) : ordered_json(0)},
                 {"shutter", sc.proc ? num(sc.proc->imgdata.other.shutter) : ordered_json(0)}};

  ordered_json geom;
  geom["sensor_size"] = {sc.orig_w, sc.orig_h};
  geom["half_size"] = sc.half_size;
  geom["roi_source"] = sc.roi_source;
  geom["roi"] = sc.has_roi ? ordered_json({sc.roi[0], sc.roi[1], sc.roi[2], sc.roi[3]}) : ordered_json(nullptr);
  geom["rot_cw"] = sc.rot_cw;
  geom["hflip"] = sc.hflip;
  geom["vflip"] = sc.vflip;
  geom["output_size"] = {sc.out_w, sc.out_h};
  j["geometry"] = geom;

  j["frame_detection"] = sc.frame_det ? frame_detection_json(*sc.frame_det) : ordered_json(nullptr);

  ordered_json fb;
  fb["source"] = sc.film_base_source;
  fb["applied"] = sc.film_base_applied;
  if (sc.film_base_source == "none") {
    fb["film_base_rgb"] = nullptr;
    fb["std_density"] = nullptr;
  } else {
    fb["film_base_rgb"] = sc.film_base_rgb;
    fb["std_density"] = sc.film_base_source == "rebate" && sc.has_std_density ? num(sc.std_density, "%.6g")
                                                                               : ordered_json(nullptr);
  }
  fb["profile_film_base_rgb"] = sc.profile_film_base_rgb;
  j["film_base"] = fb;

  if (sc.profile_path.empty()) {
    j["profile"] = nullptr;
  } else if (sc.is_bundle) {
    j["profile"] = {{"path", sc.profile_path}, {"type", "bundle"}, {"bundle_target", sc.bundle_target_name},
                    {"input_space", to_string(sc.input_space)}};
  } else {
    j["profile"] = {{"path", sc.profile_path}, {"type", "icc"}, {"bundle_target", nullptr}};
  }

  ordered_json merged = ordered_json::array();
  for (int i = 0; i < 9; ++i) merged.push_back(num(sc.merged_matrix[i], "%.9g"));
  j["matrix"] = {{"merged", merged},
                 {"exposure_comp", num(sc.exposure_comp, "%.9g")},
                 {"gain_g", num(sc.gain_g, "%.9g")},
                 {"gain_b", num(sc.gain_b, "%.9g")},
                 {"pins", {{"e", (sc.pin_mask & PIN_E) != 0}, {"g", (sc.pin_mask & PIN_G) != 0},
                           {"b", (sc.pin_mask & PIN_B) != 0}}}};

  std::vector<std::string> cli = {"-E", num_str(sc.exposure_comp, "%.9g"), "--gain_g", num_str(sc.gain_g, "%.9g"),
                                  "--gain_b", num_str(sc.gain_b, "%.9g")};
  if (sc.shoulder.active()) cli.insert(cli.end(), {"--shoulder", num_str(sc.shoulder.x_hi, "%.9g")});
  if (sc.has_roi) {
    cli.push_back("--roi");
    for (int v : {sc.roi[0], sc.roi[1], sc.roi[2], sc.roi[3], sc.rot_cw, sc.hflip ? 1 : 0, sc.vflip ? 1 : 0})
      cli.push_back(std::to_string(v));
  }
  if (sc.is_bundle && sc.bundle_target_idx >= 0) cli.insert(cli.end(), {"--target", std::to_string(sc.bundle_target_idx)});
  j["cli_equivalent"] = cli;

  // auto: the conversion parameter search (multi_target_optimizer.h)
  j["auto"] = sc.auto_res ? auto_json(*sc.auto_res, sc.auto_frame_half, sc) : ordered_json(nullptr);
  j["dino"] = sc.dino ? dino_json(*sc.dino, sc) : ordered_json(nullptr);

  const ShoulderInfo& sh = sc.shoulder;
  j["tone"] = {{"knee", num(sc.knee, "%.9g")},
               {"shoulder", num(sh.x_hi, "%.9g")},
               {"shoulder_source", sh.source},
               {"shoulder_q995", sh.has_q995 ? num(sh.q995) : ordered_json(nullptr)},
               {"shoulder_knee", sh.has_knee ? num(sh.knee) : ordered_json(nullptr)},
               {"shoulder_guard", sh.has_guard ? ordered_json(sh.guard) : ordered_json(nullptr)}};

  j["output"] = {{"path", sc.output_path}, {"colorspace", sc.colorspace}, {"bits", 16}};

  ordered_json t;
  t["raw_decode"] = num(sc.t_raw_decode_ms, "%.1f");
  t["frame_detection"] = sc.frame_det ? num(sc.t_frame_det_ms, "%.1f") : ordered_json(nullptr);
  t["dino_views"] = sc.dino ? num(sc.dino->timing_views_ms, "%.1f") : ordered_json(nullptr);
  t["dino_render"] = sc.dino ? num(sc.dino->timing_render_ms, "%.1f") : ordered_json(nullptr);
  t["dino_inference"] = sc.dino ? num(sc.dino->timing_infer_ms, "%.1f") : ordered_json(nullptr);
  t["dino_search"] = sc.dino ? num(sc.dino->timing_search_ms, "%.1f") : ordered_json(nullptr);
  t["full_conversion"] = num(sc.t_full_conversion_ms, "%.1f");
  t["tiff_write"] = num(sc.t_tiff_write_ms, "%.1f");
  j["timings_ms"] = t;

  // Written to a temporary file and renamed, so a reader never sees a partial sidecar.
  const std::string json_path = sc.output_path + ".json";
  const std::string tmp_path = json_path + ".tmp";
  FILE* fp = fopen(tmp_path.c_str(), "w");
  if (!fp) {
    log_warn("Cannot open sidecar file %s for writing\n", tmp_path.c_str());
    return false;
  }
  // Paths and arguments need not be UTF-8; invalid bytes are written as U+FFFD rather than failing.
  const std::string text = j.dump(2, ' ', false, ordered_json::error_handler_t::replace) + "\n";
  const bool write_ok = fwrite(text.data(), 1, text.size(), fp) == text.size();
  const bool close_ok = fclose(fp) == 0;
  if (!write_ok || !close_ok) {
    log_warn("Failed to flush/close sidecar %s\n", tmp_path.c_str());
    unlink(tmp_path.c_str());
    return false;
  }
  if (rename(tmp_path.c_str(), json_path.c_str()) != 0) {
    log_warn("Failed to rename sidecar from %s to %s\n", tmp_path.c_str(), json_path.c_str());
    unlink(tmp_path.c_str());
    return false;
  }
  return true;
}
