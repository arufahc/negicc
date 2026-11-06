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
#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

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

// The search's own result: e, g, b and residual are its values before any curve fit, which a run with fitted curves
// converts at other values (input_curves.remeter's e, g and b, after the fit and the re-meter; the sidecar loader
// reads those as the knobs' base when they are there, and auto's otherwise).
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
  // The record keeps its historical keys and values: branch R is the clip fallback being tried or the highlight
  // pullback moving the result (the pullback used to report through the clip fallback's outcome), and changed is
  // either one moving the result.
  j["rescue"] = {{"branch", a.clip_fallback == ClipFallbackOutcome::kNotNeeded && !a.highlight_pullback ? "N" : "R"},
                 {"white_clip", num(a.clip_fallback_white_clip, "%.6g")},
                 {"changed", a.clip_fallback == ClipFallbackOutcome::kTaken || a.highlight_pullback}};
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

// The intent field the parameter search converted against (designs/plan_intent_field_sidecar.md in
// negicc-station): per DINOv3 grid cell, row-major over the oriented frame, the Lab colour the intent adapter predicts
// (mu) and its tolerance (sigma). The same block as the station's sidecar. Null when no search ran. source: "auto" (the
// bundle search, on the frame's inner crop_ratio crop) or "dino" (the single-bracket search, on the whole frame).
// The model's short name (the backbone's directory, e.g. dinov3-small), not the path the build embedded it from.
static std::string model_stem(std::string path) {
  const std::string kBuiltIn = "[built-in] ";
  if (path.rfind(kBuiltIn, 0) == 0) path = path.substr(kBuiltIn.size());
  const size_t slash = path.find_last_of("/\\");
  std::string file = slash == std::string::npos ? path : path.substr(slash + 1);
  if (slash != std::string::npos && file.rfind("backbone.", 0) == 0) {
    const std::string dir = path.substr(0, slash);
    const size_t s2 = dir.find_last_of("/\\");
    return s2 == std::string::npos ? dir : dir.substr(s2 + 1);
  }
  const size_t dot = file.find_last_of('.');
  return dot == std::string::npos ? file : file.substr(0, dot);
}

static ordered_json intent_field_json(const std::vector<float>& mu, const std::vector<float>& sigma, int grid_w,
                                      int grid_h, int target_idx, const char* source, float crop_ratio,
                                      const Sidecar& sc) {
  const size_t n = static_cast<size_t>(grid_w) * static_cast<size_t>(grid_h);
  if (n == 0 || mu.size() != n * 3 || sigma.size() != n * 3) return nullptr;
  ordered_json j;
  j["version"] = 1;
  j["grid"] = {{"rows", grid_h}, {"cols", grid_w}, {"patch_px", 16}, {"order", "row-major, oriented frame"}};
  j["source"] = source;
  j["crop_ratio"] = num(crop_ratio, "%.3g");
  j["target_idx"] = target_idx >= 0 ? ordered_json(target_idx) : ordered_json(nullptr);
  j["model"] = {{"name", model_stem(sc.dino_model)},
                {"path", sc.dino_model},
                {"backend", sc.dino_backend},
                {"negicc", NEGICC_GIT_VERSION}};
  ordered_json m = ordered_json::array(), s = ordered_json::array();
  for (size_t i = 0; i < n; ++i) {
    m.push_back({num(mu[i * 3], "%.2f"), num(mu[i * 3 + 1], "%.2f"), num(mu[i * 3 + 2], "%.2f")});
    s.push_back({num(sigma[i * 3], "%.4g"), num(sigma[i * 3 + 1], "%.4g"), num(sigma[i * 3 + 2], "%.4g")});
  }
  j["mu_lab"] = m;
  j["sigma_lab"] = s;
  return j;
}

// The fitted G/B input curves of the conversion's bracket (negicc-station designs/plan_fit_curves.md): what rebuilds
// the combined profile (the bracket's profile with its G and B input tables composed with the curves), the fit, and
// the parameters before it. The same block as the station's sidecar; numbers to 9 significant digits, so a replay
// (--fit_curves with the recorded numbers, as the cli_equivalent writes them) rebuilds the same tables.
static ordered_json input_curves_json(const Sidecar& sc) {
  const negicc::InputCurves& c = *sc.input_curves;
  auto n9 = [](double v) { return num(v, "%.9g"); };
  ordered_json j;
  j["version"] = 1;
  j["space"] = "profile input, after the input matrix and the knee";
  j["source"] = sc.input_curves_source;
  j["target_idx"] = sc.bundle_target_idx >= 0 ? ordered_json(sc.bundle_target_idx) : ordered_json(nullptr);
  ordered_json ch;
  const char* names[2] = {"G", "B"};
  // u_lo is known from a fit, and from recorded curves only when held (the ten-number form): an eight-number replay
  // has none, and before 0 was written for it as if it had been measured.
  const bool u_lo_known = sc.input_curves_source != "cli" || c.hold;
  for (int i = 0; i < 2; ++i) {
    ch[names[i]] = {{"a", n9(c.a[i])}, {"k", n9(c.k[i])}, {"ref", n9(c.ref[i])}, {"u_med", n9(c.u_med[i])}};
    if (u_lo_known) ch[names[i]]["u_lo"] = n9(c.u_lo[i]);
  }
  // Whether the export renders through curves. Not when the fit fell back (fallback: clip, shape or
  // intent_lightness, as remeter.fallback): the bracket's own tables, so no channels and no hold. Before, the fallback's
  // identity curves were written as channels (a = k = 0, with an anchor) and replayed as recorded curves.
  const bool applied = c.active();
  const CurveRefitResult* fr = sc.curve_fit;
  j["applied"] = applied;
  j["fallback"] = fr && fr->fallback ? ordered_json(fr->fallback) : ordered_json(nullptr);
  j["channels"] = applied ? ch : ordered_json(nullptr);
  // below u_lo the curve keeps its offset (InputCurves::hold); replay needs u_lo then
  j["hold"] = applied ? ordered_json(c.hold) : ordered_json(nullptr);
  if (const CurveRefitResult* f = sc.curve_fit) {
    // The options the fit ran with (before, the defaults were written whatever ran, and half of them were missing).
    const negicc::CurveFitOptions& op = f->options;
    j["lambda"] = n9(op.lambda);
    j["bound"] = n9(op.curve_bound);
    j["options"] = {{"curve_bound", n9(op.curve_bound)}, {"max_dev_stops", n9(op.max_dev_stops)},
                    {"g_bound", n9(op.g_bound)}, {"b_bound", n9(op.b_bound)}, {"lambda", n9(op.lambda)},
                    {"step0_gain", n9(op.step0_gain)}, {"step0_curve", n9(op.step0_curve)},
                    {"step_min", n9(op.step_min)}, {"max_evals", op.max_evals},
                    {"polish_step0", n9(op.polish_step0)}, {"polish_max_evals", op.polish_max_evals},
                    {"starts", op.starts}, {"start_a", n9(op.start_a)}, {"hold_slope", op.hold_slope},
                    {"l_weight", n9(op.l_weight)}, {"huber", n9(op.huber)}, {"solve_e", op.solve_e},
                    {"e_tol_L", n9(op.e_tol_L)}};
    j["fit"] = {{"loss_before", n9(f->fit.loss_before)}, {"loss_after", n9(f->fit.loss_after)},
                {"evaluations", f->fit.evaluations}, {"g", n9(f->fit.g)}, {"b", n9(f->fit.b)},
                {"solver", "joint"}, {"e", n9(f->fit_e)}, {"renders", f->fit_renders},
                {"iterations", f->fit_iterations}, {"converged", f->fit_converged},
                {"polish_evaluations", f->fit_polish}, {"start", f->fit_start}, {"passes", f->fit_passes},
                {"intent_repredicted", f->intent_repredicted}};
    j["params_before"] = {{"e", n9(f->params_before.e)}, {"g", n9(f->params_before.g)}, {"b", n9(f->params_before.b)},
                          {"x_hi", n9(f->params_before.shoulder)}};
    j["max_dev_stops"] = n9(op.max_dev_stops);
    // Measured on the export's render: its encoding and the roll-off it measures (x_hi_auto at Auto's parameters,
    // which the first fit ran with; x_hi_fit, which the last fit ran with; x_hi at the result, which the export uses).
    j["render"] = {{"encoding", f->encoding == ENCODE_SRGB ? "srgb" : f->encoding == ENCODE_SRGB_LINEAR ? "srgb-linear"
                                                                                                       : "custom"},
                   {"x_hi_auto", n9(f->x_hi_auto)}, {"x_hi_fit", n9(f->x_hi_fit)}};
    j["remeter"] = {{"e", n9(f->params.e)}, {"g", n9(f->params.g)}, {"b", n9(f->params.b)},
                    {"target_median_L", n9(f->target_median_L)}, {"median_L", n9(f->median_L)},
                    {"x_hi", n9(f->x_hi)}, {"passes", f->remeter_passes}, {"converged", f->remeter_converged},
                    {"e_lo", n9(f->e_lo)}, {"e_hi", n9(f->e_hi)},
                    {"shape_scale", n9(f->shape_scale)},
                    {"shape_scale_ch", {n9(f->shape_scale_ch[0]), n9(f->shape_scale_ch[1])}},
                    {"fallback", f->fallback ? ordered_json(f->fallback) : ordered_json(nullptr)},
                    {"intent_lightness_failed", f->intent_lightness_failed},
                    // On "intent_lightness": E re-metered to this grid median L* (clipped: a bound of E stopped it).
                    {"fallback_remeter", f->intent_lightness_failed
                                             ? ordered_json{{"target_L", n9(f->fallback_target_L)},
                                                            {"clipped", f->fallback_clipped},
                                                            {"e_moved", f->fallback_e_moved}}
                                             : ordered_json(nullptr)},
                    {"cells_excluded", f->cells_excluded}, {"cells_total", f->cells_total},
                    {"loss_L", n9(f->parts_export.L)}, {"loss_ab", n9(f->parts_export.ab)},
                    {"auto_loss_L", n9(f->parts_auto.L)}, {"auto_loss_ab", n9(f->parts_auto.ab)},
                    {"fit_loss_L", n9(f->parts_fit.L)}, {"fit_loss_ab", n9(f->parts_fit.ab)},
                    {"time_ms", num(f->time_ms, "%.1f")},
                    {"loss", n9(f->loss_export)}};
  } else {
    j["lambda"] = nullptr;
    j["bound"] = nullptr;
    j["options"] = nullptr;
    j["fit"] = nullptr;
    j["params_before"] = nullptr;
    j["max_dev_stops"] = nullptr;
    j["render"] = nullptr;
    j["remeter"] = nullptr;
  }
  return j;
}

const char* neg_process_version() { return NEGICC_GIT_VERSION; }

bool file_fingerprint(const std::string& path, uintmax_t& size, uint32_t& crc) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<char> buf(1 << 20);
  uLong c = crc32(0L, Z_NULL, 0);
  size = 0;
  while (f) {
    f.read(buf.data(), (std::streamsize)buf.size());
    const std::streamsize n = f.gcount();
    if (n <= 0) break;
    c = crc32_z(c, reinterpret_cast<const Bytef*>(buf.data()), (z_size_t)n);
    size += (uintmax_t)n;
  }
  crc = (uint32_t)c;
  return !f.bad();
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
  // Each input's size and CRC-32, so a later run reusing this sidecar (the knobs) can tell the same file from another
  // saved under the same name, also when the file was copied or downloaded again in between (a new modification time).
  ordered_json stats = ordered_json::array();
  for (size_t i = 0; i < sc.files.size(); ++i) {
    FileFingerprint fp;
    if (sc.file_fingerprints && i < sc.file_fingerprints->size() && (*sc.file_fingerprints)[i].ok) {
      fp = (*sc.file_fingerprints)[i];  // hashed once already, by the sidecar loader
    } else {
      fp.ok = file_fingerprint(sc.files[i], fp.size, fp.crc);
    }
    stats.push_back(fp.ok ? ordered_json({{"size", fp.size}, {"crc32", fp.crc}}) : ordered_json(nullptr));
  }
  j["inputs"] = {{"raw_files", sc.files},
                 {"raw_file_stats", stats},
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
  const negicc::FilmBaseDetection* d = sc.film_base_detection;
  auto num3 = [](const double* v) { return ordered_json{num(v[0], "%.10g"), num(v[1], "%.10g"), num(v[2], "%.10g")}; };
  if (sc.film_base_source == "none") {
    fb["film_base_rgb"] = nullptr;
  } else if (d && d->found) {
    fb["film_base_rgb"] = num3(d->rgb);
  } else {
    fb["film_base_rgb"] = sc.film_base_rgb;
  }
  // Read from the scan (film_base_detect.h): the base in the half-size decode's units, the rebate patch sampled and
  // the frame detector's record.
  const bool rebate = d && d->found && d->source == "rebate";
  fb["decode_rgb"] = d && d->found ? num3(d->decode_rgb) : ordered_json(nullptr);
  fb["box"] = rebate ? ordered_json{d->box[0], d->box[1], d->box[2], d->box[3]} : ordered_json(nullptr);
  fb["std_density"] = rebate ? num(d->std_d, "%.6g") : ordered_json(nullptr);
  if (d && d->has_detector) {
    ordered_json det;
    det["reason"] = d->reason;
    det["frame_conf"] = num(d->frame_conf, "%.3f");
    det["light_clipped"] = d->light_clipped;
    det["thinnest_over_patch_stops"] = d->has_outshone ? num(d->thinnest_over_patch_stops, "%.4f") : ordered_json(nullptr);
    fb["detector"] = det;
  } else {
    fb["detector"] = nullptr;
  }
  fb["profile_film_base_rgb"] = sc.profile_film_base_rgb;
  // The gains the white-light decode put on the camera channels to balance them on the film base (load_raw);
  // film_base_rgb and decode_rgb are read after them. Nine significant digits, which a single-precision gain round-trips
  // through: with six decimals a replay from the sidecar (negicc-station's) balanced the mosaic with other gains.
  const float* k = sc.balance_gains;
  fb["balance_gains"] =
      k ? ordered_json{num(k[0], "%.9g"), num(k[1], "%.9g"), num(k[2], "%.9g")} : ordered_json(nullptr);
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

  // The values -E, --gain_g and --gain_b take: the station's with a bundle, else the absolute ones.
  const float* egb = sc.station_params;
  const float abs_egb[3] = {sc.exposure_comp, sc.gain_g, sc.gain_b};
  if (!egb) egb = abs_egb;
  std::vector<std::string> cli = {"-E", num_str(egb[0], "%.9g"), "--gain_g", num_str(egb[1], "%.9g"), "--gain_b",
                                  num_str(egb[2], "%.9g")};
  if (sc.has_roi) {
    cli.push_back("--roi");
    for (int v : {sc.roi[0], sc.roi[1], sc.roi[2], sc.roi[3], sc.rot_cw, sc.hflip ? 1 : 0, sc.vflip ? 1 : 0})
      cli.push_back(std::to_string(v));
  }
  if (sc.is_bundle && sc.bundle_target_idx >= 0) cli.insert(cli.end(), {"--target", std::to_string(sc.bundle_target_idx)});
  if (sc.input_curves && (sc.cli_input_curves ? *sc.cli_input_curves : *sc.input_curves).active()) {
    // Replays the combined profile without fitting (the curves need -P, as when they were fitted), at the E, g and b
    // above: after the knobs, the curves re-anchored there.
    // The value leads with "ref:" and the refs (always positive): the older order led with aG, and a negative aG
    // made the value start with '-', which the parser rejects as an unknown option, so the line did not replay.
    // No curves (the fit fell back): nothing to replay, and the given values alone render the same pixels.
    const negicc::InputCurves& c = sc.cli_input_curves ? *sc.cli_input_curves : *sc.input_curves;
    cli.push_back("--fit_curves");
    std::string v8 = "ref:";
    for (float v : {c.ref[0], c.ref[1], c.u_med[0], c.u_med[1], c.a[0], c.k[0], c.a[1], c.k[1]})
      v8 += (v8.size() > 4 ? "," : "") + num_str(v, "%.9g");
    if (c.hold) v8 += "," + num_str(c.u_lo[0], "%.9g") + "," + num_str(c.u_lo[1], "%.9g");  // held below u_lo
    cli.push_back(v8);
  }
  j["cli_equivalent"] = cli;

  // auto: the conversion parameter search (multi_target_optimizer.h)
  if (!sc.cached_auto_json_str.empty()) {
    j["auto"] = ordered_json::parse(sc.cached_auto_json_str);
  } else {
    j["auto"] = sc.auto_res ? auto_json(*sc.auto_res, sc.auto_frame_half, sc) : ordered_json(nullptr);
  }
  if (!sc.cached_dino_json_str.empty()) {
    j["dino"] = ordered_json::parse(sc.cached_dino_json_str);
  } else {
    j["dino"] = sc.dino ? dino_json(*sc.dino, sc) : ordered_json(nullptr);
  }
  // The intent field Auto converted against; a run that reuses <output>.json (the knobs) keeps the previous run's.
  if (!sc.cached_intent_field_json_str.empty())
    j["intent_field"] = ordered_json::parse(sc.cached_intent_field_json_str);
  else if (sc.auto_res)
    j["intent_field"] = intent_field_json(sc.auto_res->intent_mu, sc.auto_res->intent_sigma,
                                          sc.auto_res->intent_grid_w, sc.auto_res->intent_grid_h,
                                          sc.auto_res->target_idx, "auto", ParamSearchOptions{}.crop_ratio, sc);
  else if (sc.dino)
    j["intent_field"] = intent_field_json(sc.dino->best_mu, sc.dino->best_sigma, sc.dino->grid_w, sc.dino->grid_h,
                                          sc.bundle_target_idx, "dino", 1.0f, sc);
  else
    j["intent_field"] = nullptr;

  if (!sc.cached_input_curves_json_str.empty())
    j["input_curves"] = ordered_json::parse(sc.cached_input_curves_json_str);
  else
    j["input_curves"] = sc.input_curves ? input_curves_json(sc) : ordered_json(nullptr);
  if (sc.has_knobs) {
    j["knobs"] = {
        {"brightness", num(sc.brightness, "%.6g")},
        {"tint", num(sc.tint, "%.6g")},
        {"warmth", num(sc.warmth, "%.6g")},
        {"limited", sc.knob_res.limited},
        {"base", {{"e", num(sc.knob_base[0], "%.9g")},
                  {"g", num(sc.knob_base[1], "%.9g")},
                  {"b", num(sc.knob_base[2], "%.9g")}}},
        {"egb", {{"e", num(sc.knob_res.e, "%.9g")},
                 {"g", num(sc.knob_res.g, "%.9g")},
                 {"b", num(sc.knob_res.b, "%.9g")}}},
        {"grey_point", {num(sc.knob_res.grey_lab[0], "%.4f"),
                        num(sc.knob_res.grey_lab[1], "%.4f"),
                        num(sc.knob_res.grey_lab[2], "%.4f")}},
    };
  } else {
    j["knobs"] = nullptr;
  }

  const ShoulderInfo& sh = sc.shoulder;
  j["tone"] = {{"knee", num(sc.knee, "%.9g")},
               {"shoulder", num(sh.x_hi, "%.9g")},
               {"shoulder_source", sh.source},
               {"shoulder_q995", sh.has_q995 ? num(sh.q995) : ordered_json(nullptr)},
               {"shoulder_knee", sh.has_knee ? num(sh.knee) : ordered_json(nullptr)}};

  j["output"] = {{"path", sc.output_path}, {"colorspace", sc.colorspace}, {"bits", 16}};

  ordered_json t;
  t["raw_decode"] = num(sc.t_raw_decode_ms, "%.1f");
  t["frame_detection"] = sc.frame_det ? num(sc.t_frame_det_ms, "%.1f") : ordered_json(nullptr);
  // The bracket analysis outside Auto's time (enclosures, knobs, roll-off); before, it counted as raw_decode.
  t["analysis"] = num(sc.t_analysis_ms, "%.1f");
  const bool has_cached_dino = !sc.cached_dino_json_str.empty();
  t["dino_views"] = sc.dino ? num(sc.dino->timing_views_ms, "%.1f")
                            : (has_cached_dino ? num(sc.cached_dino_views_ms, "%.1f") : ordered_json(nullptr));
  t["dino_render"] = sc.dino ? num(sc.dino->timing_render_ms, "%.1f")
                             : (has_cached_dino ? num(sc.cached_dino_render_ms, "%.1f") : ordered_json(nullptr));
  t["dino_inference"] = sc.dino ? num(sc.dino->timing_infer_ms, "%.1f")
                                : (has_cached_dino ? num(sc.cached_dino_infer_ms, "%.1f") : ordered_json(nullptr));
  t["dino_search"] = sc.dino ? num(sc.dino->timing_search_ms, "%.1f")
                             : (has_cached_dino ? num(sc.cached_dino_search_ms, "%.1f") : ordered_json(nullptr));
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
