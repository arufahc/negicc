// Copyright 2024-2026 Alpha Lam <arufa.hc@gmail.com>
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

// neg_process's output: the 16-bit TIFF with its ICC profile, and the JSON sidecar beside it. Their declarations
// are the output part of negicc.h.

#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "elle_icc_profiles/sRGB_elle_V2_g10.h"
#include "elle_icc_profiles/sRGB_elle_V2_srgbtrc.h"
#include "libraw/libraw.h"
#include "negicc.h"
#include "nlohmann/json.hpp"
#include "tiff_head.h"

// ====================================================================================================================
// Output: code
// The 16-bit TIFF with its ICC profile, and the JSON sidecar beside it.
// ====================================================================================================================

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
  // A classic TIFF addresses its data with 32-bit offsets: the header, the profile and the strip of three 16-bit
  // samples a pixel must end within them (as write_density_tiff checks, below).
  if (sizeof(tiff_hdr) + (uint64_t)profile_size + (uint64_t)width * height * 6 > 0xffffff00ull) {
    log_error("The image is too large for a TIFF file (%ux%u)\n", width, height);
    if (dynamically_allocated && output_profile) free(output_profile);
    return -1;
  }
  struct tiff_hdr header;
  tiff_head(proc, &header, profile_size);
  auto* fp = fopen(output.c_str(), "wb+");
  if (!fp) {
    log_error("Cannot open %s for writing\n", output.c_str());
    if (dynamically_allocated && output_profile) free(output_profile);
    return -1;
  }
  bool ok = fwrite(&header, sizeof(header), 1, fp) == 1;
  if (ok && profile_size) {
    ok = fwrite(output_profile, profile_size, 1, fp) == 1;
  }
  if (dynamically_allocated && output_profile) {
    free(output_profile);
  }

  if (ok && rgb3) {
    ok = fwrite(rgb3, 3 * sizeof(uint16_t), (size_t)width * height, fp) == (size_t)width * height;
  } else if (ok) {
    std::vector<ushort> row_buf(width * 3);
    for (unsigned row = 0; ok && row < height; ++row) {
      for (unsigned col = 0; col < width; ++col) {
        row_buf[col * 3]     = proc->imgdata.image[row * width + col][0];
        row_buf[col * 3 + 1] = proc->imgdata.image[row * width + col][1];
        row_buf[col * 3 + 2] = proc->imgdata.image[row * width + col][2];
      }
      ok = fwrite(row_buf.data(), 3 * sizeof(ushort), width, fp) == width;
    }
  }
  ok = (fclose(fp) == 0) && ok;
  if (!ok) {
    // A full disk or another write error: the truncated file is removed (not a device or pipe given as the output).
    log_error("Failed to write %s\n", output.c_str());
    std::error_code ec;
    if (std::filesystem::is_regular_file(output, ec)) unlink(output.c_str());
    return -1;
  }
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

// A tone curve in full (17 digits): the curve the image went through, exactly.
static ordered_json tone_curve_json(const ToneCurve& q) {
  return {{"black", num(q.black, "%.17g")}, {"white", num(q.white, "%.17g")}, {"gamma", num(q.gamma, "%.17g")},
          {"contrast", num(q.contrast, "%.17g")}};
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
  if (sc.search_renderer != "cpu") j["renderer"] = sc.search_renderer;
  if (!sc.shadow_json_str.empty()) j["shadow_alternative"] = ordered_json::parse(sc.shadow_json_str);
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

// The intent field the parameter search converted against: per DINOv3 grid cell, row-major over the oriented frame,
// the Lab colour (mu) and its tolerance (sigma) of the chosen step. The same block as the station's sidecar. Null when
// no search ran. source: "auto" (the bundle search or the print's search, on the frame's inner crop_ratio crop): a*
// and b* as the intent adapter predicted them from the chosen step's render, L* the frame's lightness target (the
// bootstrap field's; DESIGN.md 4.2); "dino" (the single-bracket search, on the whole frame): its best step's
// prediction in all three.
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

// The fitted G/B input curves of the conversion's bracket (the input curves part of negicc_core.h): what rebuilds
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
  // Whether the export renders through curves. Not when the fit fell back (fallback: clip, shape or intent_lightness,
  // as remeter.fallback): the bracket's own tables, so no channels and no hold.
  const bool applied = c.active();
  const WhiteLightFitResult* fr = sc.curve_fit;
  j["applied"] = applied;
  j["fallback"] = fr && fr->fallback ? ordered_json(fr->fallback) : ordered_json(nullptr);
  j["channels"] = applied ? ch : ordered_json(nullptr);
  // below u_lo the curve keeps its offset (InputCurves::hold); replay needs u_lo then
  j["hold"] = applied ? ordered_json(c.hold) : ordered_json(nullptr);
  if (const WhiteLightFitResult* f = sc.curve_fit) {
    // The options the fit ran with, all of them.
    const negicc::CurveFitOptions& op = f->options;
    j["bound"] = n9(op.curve_bound);
    j["options"] = {{"curve_bound", n9(op.curve_bound)}, {"max_dev_stops", n9(op.max_dev_stops)},
                    {"g_bound_stops", n9(op.g_bound_stops)}, {"b_bound_stops", n9(op.b_bound_stops)},
                    {"joint_lambda", n9(op.joint_lambda)},
                    {"hold_slope", op.hold_slope},
                    {"solve_e", op.solve_e}, {"e_tol_L", n9(op.e_tol_L)}};
    j["fit"] = {{"loss_before", n9(f->fit.loss_before)}, {"loss_after", n9(f->fit.loss_after)},
                // The fifth residual, in a*: the intent field's standard error, and its mean a* residual less
                // the lever share and the lean, at Auto and at the result (zero when there was no fit).
                {"intent_se", n9(f->fit.intent_se)}, {"intent_auto", n9(f->fit.intent_auto)},
                {"intent_after", n9(f->fit.intent_after)},
                {"evaluations", f->fit.evaluations}, {"g", n9(f->fit.g)}, {"b", n9(f->fit.b)},
                {"solver", "joint"}, {"e", n9(f->fit.e)}, {"renders", f->fit.renders},
                {"iterations", f->fit.iterations}, {"converged", f->fit.converged},
                // No longer varied (the coordinate polish and multiple seeds are gone); kept for readers.
                {"polish_evaluations", 0}, {"start", 0}, {"passes", f->fit_passes},
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
                    {"neutral", ordered_json{{"anchors", f->neutral_anchors ? "head" : "none"},
                                             {"m", f->neutral_export.m},
                                             {"auto_a", n9(f->neutral_auto.a_bar)},
                                             {"auto_b", n9(f->neutral_auto.b_bar)},
                                             {"export_a", n9(f->neutral_export.a_bar)},
                                             {"export_b", n9(f->neutral_export.b_bar)},
                                             {"auto_ka", n9(f->neutral_auto.k_a)},
                                             {"auto_kb", n9(f->neutral_auto.k_b)},
                                             {"export_ka", n9(f->neutral_export.k_a)},
                                             {"export_kb", n9(f->neutral_export.k_b)},
                                             // The evidence mass (sum of p^2 over the cells in range, as the
                                             // last fit pass saw them at its roll-off; earlier builds wrote it at
                                             // Auto's roll-off, up to 2 % apart), the shrink weight
                                             // m / (m + 12) inside the residuals, the anchors' L* spread and
                                             // slope share, and their mean a*'s standard deviation.
                                             {"evidence", n9(f->fit.evidence)},
                                             {"shrink", n9(f->fit.shrink)},
                                             {"span_L", n9(f->fit.span_L)},
                                             {"slope_share", n9(f->fit.slope_share)},
                                             {"anchor_sd", n9(f->fit.anchor_sd)}}},
                    // The never-greener guard of the last fit pass: Auto's and the fit's mean a* over all cells
                    // and weighted by p^2, whether each counted, and whether the guard pulled the fit.
                    {"guard", [&] {
                       const negicc::GreenGuard& gd = f->fit.guard;
                       return ordered_json{{"tol", n9(negicc::kGreenGuardTolAb)},
                                           {"counted", {gd.counted[0], gd.counted[1]}},
                                           {"auto_a", {n9(gd.auto_a[0]), n9(gd.auto_a[1])}},
                                           {"fit_a", {n9(gd.fit_a[0]), n9(gd.fit_a[1])}},
                                           {"binding", gd.binding},
                                           // The fit ran first without the guard; refit: that result broke
                                           // the bound and this is the guarded re-run.
                                           {"refit", gd.refit},
                                           {"free_fit_a", {n9(gd.free_fit_a[0]), n9(gd.free_fit_a[1])}},
                                           {"p_source", f->guard_p_source}};
                     }()},
                    {"time_ms", num(f->time_ms, "%.1f")},
                    {"loss", n9(f->loss_export)}};
  } else {
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

FileFingerprint memory_fingerprint(const void* data, size_t size) {
  FileFingerprint fp;
  fp.ok = data != nullptr;
  fp.size = size;
  fp.crc = fp.ok ? (uint32_t)crc32_z(crc32(0L, Z_NULL, 0), reinterpret_cast<const Bytef*>(data), (z_size_t)size) : 0;
  return fp;
}

std::string utf8_replaced(const std::string& s) {
  using nlohmann::ordered_json;
  return ordered_json::parse(ordered_json(s).dump(-1, ' ', false, ordered_json::error_handler_t::replace))
      .get<std::string>();
}

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


// The files the options name, as the sidecar records them (inputs.option_files): every argument (and every comma-separated
// piece of one, and either side of its first colon: --film_base's file:x,y,w,h, --dino's head:path) that names a
// regular file other than the inputs and the output, in the order given, with its size and CRC-32. A later run reuses
// the sidecar only when its own option files are these, unchanged: a bundle, flat field, output profile, print or head
// edited in place gives another size or CRC-32.
// bundle_files: the files a bundle given as a file reads besides itself (its brackets' and prints' ICC profiles,
// bundle_side_files), which no argument names.
ordered_json option_files_json(int argc, char** argv, const std::vector<std::string>& inputs,
                               const std::string& output, const std::vector<std::string>& bundle_files) {
  ordered_json out = ordered_json::array();
  std::vector<std::string> seen;
  auto consider = [&](const std::string& p) {
    if (p.empty() || !is_regular_file(p) || paths_match(output, p)) return;
    for (const std::string& f : inputs)
      if (paths_match(f, p)) return;
    for (const std::string& q : seen)
      if (q == p) return;
    seen.push_back(p);
    uintmax_t size = 0;
    uint32_t crc = 0;
    const bool ok = file_fingerprint(p, size, crc);
    out.push_back({{"arg", utf8_replaced(p)}, {"size", ok ? ordered_json(size) : ordered_json(nullptr)},
                   {"crc32", ok ? ordered_json(crc) : ordered_json(nullptr)}});
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    consider(a);
    size_t from = 0;
    while (from <= a.size()) {
      const size_t to = std::min(a.find(',', from), a.size());
      const std::string piece = a.substr(from, to - from);
      if (piece != a) consider(piece);
      const size_t colon = piece.find(':');
      if (colon != std::string::npos) {
        consider(piece.substr(0, colon));
        consider(piece.substr(colon + 1));
      }
      from = to + 1;
    }
  }
  for (const std::string& f : bundle_files) consider(f);
  return out;
}

// The ICC files beside a bundle given as a file (a split bundle's brackets and prints), as paths; none for a built-in
// bundle or one that holds its profiles inline.
std::vector<std::string> bundle_side_files(const ProfileBundle& b) {
  std::vector<std::string> out;
  if (!is_regular_file(b.path)) return out;
  const std::filesystem::path dir = std::filesystem::path(b.path).parent_path();
  for (const BundleTarget& t : b.targets)
    if (!t.icc_file.empty()) out.push_back((dir / t.icc_file).string());
  for (const BundlePrint& p : b.prints)
    if (p.icc_in_file && !p.icc_file.empty()) out.push_back((dir / p.icc_file).string());
  return out;
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
                 {"option_files", option_files_json(sc.argc, sc.argv, sc.files, sc.output_path, sc.bundle_files)},
                 {"camera", camera},
                 {"iso", sc.proc ? num(sc.proc->imgdata.other.iso_speed) : ordered_json(0)},
                 {"shutter", sc.proc ? num(sc.proc->imgdata.other.shutter) : ordered_json(0)}};
  // What the conversion used, as its log's block says it (conversion_used in pipeline.cc).
  j["used"] = sc.used_json.empty() ? ordered_json(nullptr) : ordered_json::parse(sc.used_json, nullptr, false);

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
  // A base given by a capture is taken from this record by a later run (--film_base auto), so it is written with the
  // digits that give its doubles back; a base read from the scan keeps the ten digits it has always had.
  const bool given_base = sc.film_base_source == "capture" || sc.film_base_source == "sidecar";
  const char* base_fmt = given_base ? "%.17g" : "%.10g";
  auto num3 = [base_fmt](const double* v) {
    return ordered_json{num(v[0], base_fmt), num(v[1], base_fmt), num(v[2], base_fmt)};
  };
  if (sc.film_base_source == "none") {
    fb["film_base_rgb"] = nullptr;
  } else if (d && d->found) {
    fb["film_base_rgb"] = num3(d->rgb);
  } else {
    fb["film_base_rgb"] = sc.film_base_rgb;
  }
  // Read from the scan (the film base detector part of negicc_core.h): the base in the half-size decode's units, the
  // rebate patch sampled and the frame detector's record.
  const bool rebate = d && d->found && d->source == "rebate";
  fb["decode_rgb"] = d && d->found ? num3(d->decode_rgb) : ordered_json(nullptr);
  fb["box"] = rebate ? ordered_json{d->box[0], d->box[1], d->box[2], d->box[3]} : ordered_json(nullptr);
  fb["std_density"] = rebate ? num(d->std_d, "%.6g") : ordered_json(nullptr);
  if (d && d->has_detector) {
    ordered_json det;
    det["reason"] = d->reason;
    det["frame_conf"] = num(d->frame_conf, "%.3f");
    det["light_clipped"] = d->light_clipped;
    det["thinnest_over_patch_stops"] =
        d->has_outshone ? num(d->thinnest_over_patch_stops, "%.4f") : ordered_json(nullptr);
    fb["detector"] = det;
  } else {
    fb["detector"] = nullptr;
  }
  if (!sc.film_base_capture_json.empty()) {
    fb["capture"] = ordered_json::parse(sc.film_base_capture_json, nullptr, false);
  }
  // A print on a base that is not a capture: film_base_rgb is the base raised to be no darker than the picture's
  // thinnest film (raise_base_to_print_floor, with print_floor_factor's factor); the base as found and the factor.
  if (sc.print_floor_applied) {
    fb["print_floor"] = {{"units", "the conversion's film base space at the scan's exposure (a bundle's base is "
                                    "scaled to it; film_base_rgb is the bundle's at its own exposure when nothing was "
                                    "raised)"},
                         {"found_source", sc.print_floor_found_source},
                         {"found_rgb", {num(sc.print_floor_found_rgb[0], "%.10g"),
                                        num(sc.print_floor_found_rgb[1], "%.10g"),
                                        num(sc.print_floor_found_rgb[2], "%.10g")}},
                         {"factor", num(sc.print_floor_k, "%.6g")},
                         {"frame", sc.print_floor_frame},
                         {"frame_conf", sc.print_floor_frame_conf < 0.0f ? ordered_json(nullptr)
                                                                         : num(sc.print_floor_frame_conf, "%.4g")}};
  }
  fb["profile_film_base_rgb"] = sc.profile_film_base_rgb;
  // The gains the white-light decode put on the camera channels to balance them on the film base (load_raw);
  // film_base_rgb and decode_rgb are read after them. Nine significant digits, which a single-precision gain
  // round-trips through: with six decimals a replay from the sidecar (negicc-station's) balanced the mosaic with other
  // gains.
  const float* k = sc.balance_gains;
  fb["balance_gains"] =
      k ? ordered_json{num(k[0], "%.9g"), num(k[1], "%.9g"), num(k[2], "%.9g")} : ordered_json(nullptr);
  j["film_base"] = fb;

  if (sc.profile_path.empty()) {
    j["profile"] = nullptr;
  } else if (sc.is_bundle) {
    j["profile"] = {{"path", sc.profile_path}, {"type", "bundle"}, {"bundle_target", sc.bundle_target_name},
                    {"input_space", to_string(sc.input_space)}};
    // --print: the brackets were rendered through a print material's print ICC (the print part of negicc.h), not the
    // chart's.
    if (!sc.print_id.empty()) {
      j["profile"]["print"] = {{"id", sc.print_id}, {"name", sc.print_name},
                               {"table", sc.print_from_bundle ? "the bundle's print ICC" : "computed by neg_process"},
                               {"curves", "computed at the print's mid grey A (a) from the bundle's "
                                          "grey_balance_curves, with the gains g and b after the knee"}};
      // A material given as a file: which file, and its content, so the replay can be checked against it.
      if (!sc.print_file.empty()) {
        j["profile"]["print"]["file"] = sc.print_file;
        j["profile"]["print"]["file_sha256"] = sc.print_file_sha256;
      }
      // The print's white, its mid grey A and the bracket's own, and its search (pipeline.cc:
      // search_print_continuous).
      j["profile"]["print"]["white"] = "paper";   // media-relative: the paper's base at the display's white
      j["profile"]["print"]["paper_white_xyz"] = {num(sc.print_paper_xyz[0], "%.6g"),
                                                  num(sc.print_paper_xyz[1], "%.6g"),
                                                  num(sc.print_paper_xyz[2], "%.6g")};
      j["profile"]["print"]["a"] = num(sc.print_a, "%.17g");
      if (std::isfinite(sc.print_bracket_a_mid))
        j["profile"]["print"]["bracket_a_mid"] = num(sc.print_bracket_a_mid, "%.17g");
      // The brackets' tilt (grey_relative.b) at that A, linear in A between brackets.
      if (std::isfinite(sc.print_tilt[0]))
        j["profile"]["print"]["tilt"] = {num(sc.print_tilt[0], "%.9g"), num(sc.print_tilt[1], "%.9g"),
                                         num(sc.print_tilt[2], "%.9g")};
      if (sc.print_searched) {
        auto three = [&](const float* v, double scale) {
          return ordered_json{num(scale * v[0], "%.4g"), num(scale * v[1], "%.4g"), num(scale * v[2], "%.4g")};
        };
        j["profile"]["print"]["search"] = {
            {"a0", num(sc.print_a0, "%.9g")},
            {"target_L", num(sc.print_target_l, "%.6g")},
            {"steps", sc.print_steps},
            {"converged", sc.print_converged},
            {"stopped_by", sc.print_stopped_by},
            {"above_knee_pct", num(100.0 * sc.print_above_knee, "%.4g")},
            {"base_factor", num(sc.print_floor_applied ? sc.print_floor_k : 1.0f, "%.6g")},
            {"past_knee_pct", three(sc.print_past_knee, 100.0)},
            {"past_cut_pct", three(sc.print_past_cut, 100.0)},
            {"above_white_pct", three(sc.print_above_white, 100.0)},
            {"at_white_pct", three(sc.print_at_white, 100.0)},
            {"a0_at_range_end", sc.print_a0_at_end},
            {"last_move", three(sc.print_last_move, 1.0)}};
      } else if (!sc.cached_print_search_json_str.empty()) {
        // A run that reused <output>.json (the knobs) starts from that run's A (knobs.base.a): its search's record
        // carries over.
        j["profile"]["print"]["search"] = ordered_json::parse(sc.cached_print_search_json_str);
      }
    }
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
  if (sc.is_bundle && sc.bundle_target_idx >= 0)
    cli.insert(cli.end(), {"--target", std::to_string(sc.bundle_target_idx)});
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
  if (sc.tone_curve && sc.tone_applied) {
    const ToneCurve& q = *sc.tone_curve;
    std::string v = "curve:";
    for (double x : {q.black, q.white, q.gamma, q.contrast})
      v += (v.size() > 6 ? "," : "") + num_str(x, "%.17g");
    cli.insert(cli.end(), {"--tone_stage", v});
  } else if (sc.tone_moved_nothing) {
    cli.insert(cli.end(), {"--tone_stage", "off"});  // the plan applied no curve: the same pixels without the stage
  }
  // A bundle conversion: the given values render these pixels on the replay, so each stage that runs after given values
  // and recorded nothing is turned off there, each on its own: --fit_curves off on white light (the fit's light) where
  // no curves are recorded, and --tone_stage off where no tone curve is recorded (nor already turned off above).
  if (sc.is_bundle && cli.size() >= 2) {
    if (sc.input_space == InputSpace::kRec2020Linear &&
        std::find(cli.begin(), cli.end(), "--fit_curves") == cli.end())
      cli.insert(cli.end(), {"--fit_curves", "off"});
    if (std::find(cli.begin(), cli.end(), "--tone_stage") == cli.end()) cli.insert(cli.end(), {"--tone_stage", "off"});
  }
  // A print is made by its own search (no given values with --print): its replay is the frame, the print and the knobs
  // if any, and re-running it re-runs the search (deterministic, with --dino, --dino_iters and --dino_backend when
  // given: the last keeps the passes where they ran), then the same knobs on its result.
  if (!sc.print_id.empty() && sc.density_json.empty()) {
    cli.clear();
    if (sc.has_roi) {
      cli.push_back("--roi");
      for (int v : {sc.roi[0], sc.roi[1], sc.roi[2], sc.roi[3], sc.rot_cw, sc.hflip ? 1 : 0, sc.vflip ? 1 : 0})
        cli.push_back(std::to_string(v));
    }
    if (!sc.dino_arg.empty()) cli.insert(cli.end(), {"--dino", sc.dino_arg});
    if (sc.dino_iters_given) cli.insert(cli.end(), {"--dino_iters", std::to_string(sc.dino_iters)});
    if (!sc.dino_backend_arg.empty()) cli.insert(cli.end(), {"--dino_backend", sc.dino_backend_arg});
    if (sc.has_knobs) {
      if (sc.brightness != 0.0f) cli.insert(cli.end(), {"--brightness", num_str(sc.brightness, "%.9g")});
      if (sc.tint != 0.0f) cli.insert(cli.end(), {"--tint", num_str(sc.tint, "%.9g")});
      if (sc.warmth != 0.0f) cli.insert(cli.end(), {"--warmth", num_str(sc.warmth, "%.9g")});
    }
  }
  // A print: the replay renders it too, from the same file when the material was given as one (its id may be a
  // built-in material's, or none).
  if (!sc.print_id.empty()) cli.insert(cli.end(), {"--print", sc.print_file.empty() ? sc.print_id : sc.print_file});
  // A density output takes no exposure, gains or bracket: its replay is the frame, --fit_curves off when a white-light
  // run without --print turned the fit off (for its suggested balance), and the --print materials and any knobs on
  // their CLFs as given: with --print the base not a capture is raised to the picture's thinnest film
  // (raise_base_to_print_floor) and the densities are written against it, so the replay needs --print to write the
  // same file and its CLFs.
  if (!sc.density_json.empty()) {
    cli.clear();
    if (sc.has_roi) {
      cli.push_back("--roi");
      for (int v : {sc.roi[0], sc.roi[1], sc.roi[2], sc.roi[3], sc.rot_cw, sc.hflip ? 1 : 0, sc.vflip ? 1 : 0})
        cli.push_back(std::to_string(v));
    }
    if (sc.density_fit_off) cli.insert(cli.end(), {"--fit_curves", "off"});
    // The search's model, iterations and backend, as on a print: the recorded balance and each print's search depend
    // on them.
    if (!sc.dino_arg.empty()) cli.insert(cli.end(), {"--dino", sc.dino_arg});
    if (sc.dino_iters_given) cli.insert(cli.end(), {"--dino_iters", std::to_string(sc.dino_iters)});
    if (!sc.dino_backend_arg.empty()) cli.insert(cli.end(), {"--dino_backend", sc.dino_backend_arg});
    for (const std::string& m : sc.density_prints) cli.insert(cli.end(), {"--print", m});   // once per material
    if (!sc.density_prints.empty()) {
      if (sc.brightness != 0.0f) cli.insert(cli.end(), {"--brightness", num_str(sc.brightness, "%.9g")});
      if (sc.tint != 0.0f) cli.insert(cli.end(), {"--tint", num_str(sc.tint, "%.9g")});
      if (sc.warmth != 0.0f) cli.insert(cli.end(), {"--warmth", num_str(sc.warmth, "%.9g")});
    }
  }
  // The format the film base's frame detection looked for, given or --roi auto's default (35mm): an explicit --roi
  // without it looks for every format, which may place the base's patch elsewhere. None when the base's detection
  // looked for every format (the CFA path's, which runs again for the base), as its replay does.
  if (!sc.film_format.empty()) cli.insert(cli.end(), {"--film_format", sc.film_format});
  j["cli_equivalent"] = cli;

  // auto: the conversion parameter search (the parameter search part of negicc.h)
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
  // The models: what --dino replaced, or the built-in ones.
  if (!sc.cached_heads_json_str.empty()) {
    j["heads"] = ordered_json::parse(sc.cached_heads_json_str);
  } else {
    ordered_json h;
    const char* names[3] = {"intent", "neutral", "tone"};
    for (int i = 0; i < 3; ++i) {
      if (sc.head_path[i].empty()) {
        h[names[i]] = nullptr;
        continue;
      }
      char crc[16];
      std::snprintf(crc, sizeof(crc), "%08x", sc.head_fp[i].crc);
      h[names[i]] = {{"path", sc.head_path[i]},
                     {"bytes", sc.head_fp[i].ok ? ordered_json(sc.head_fp[i].size) : ordered_json(nullptr)},
                     {"crc32", sc.head_fp[i].ok ? ordered_json(crc) : ordered_json(nullptr)}};
    }
    j["heads"] = h;
  }
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
        {"base", std::isfinite(sc.knob_base_a)
                     ? ordered_json{{"e", num(sc.knob_base[0], "%.9g")}, {"g", num(sc.knob_base[1], "%.9g")},
                                    {"b", num(sc.knob_base[2], "%.9g")}, {"a", num(sc.knob_base_a, "%.17g")}}
                     : ordered_json{{"e", num(sc.knob_base[0], "%.9g")}, {"g", num(sc.knob_base[1], "%.9g")},
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

  // The tone stage: the heads' predictions, the final render's descriptors before it, the targets, the
  // curve, and the clipping before (the clip check's sample) and after (the whole export).
  auto clip_json = [&]() {
    if (!(sc.tone_applied || sc.tone_moved_nothing) || !sc.tone_clip) return ordered_json(nullptr);
    const ToneClip& k = *sc.tone_clip;
    return ordered_json{{"white_before", num(k.white_before, "%.6f")}, {"white_after", num(k.white_after, "%.6f")},
                        {"black_before", num(k.black_before, "%.6f")}, {"black_after", num(k.black_after, "%.6f")}};
  };
  if (sc.tone_stage_plan) {
    auto n9 = [](double v) { return num(v, "%.9g"); };
    const ToneStagePlan& tp = *sc.tone_stage_plan;
    if (tp.planned && sc.tone_plan) {
      const TonePlanCurve& pl = *sc.tone_plan;
      ordered_json pred, pred_sd;
      for (int t = 0; t < ToneHead::kTargets; ++t) {
        pred[ToneHead::target_name(t)] = n9((float)tp.pred_mean[t]);
        pred_sd[ToneHead::target_name(t)] = n9((float)tp.pred_sd[t]);
      }
      auto desc = [&](const ToneDescriptors& d) {
        return ordered_json{{"L2", n9((float)d.L2)},   {"L10", n9((float)d.L10)}, {"L25", n9((float)d.L25)},
                            {"L50", n9((float)d.L50)}, {"L75", n9((float)d.L75)}, {"L90", n9((float)d.L90)},
                            {"L98", n9((float)d.L98)}, {"R9010", n9((float)d.range())}};
      };
      // As applied (capped on the final image), else as planned.
      const ToneTargets& tt = sc.tone_targets ? *sc.tone_targets : pl.targets;
      const ToneCurve& tc = sc.tone_applied && sc.tone_curve ? *sc.tone_curve : pl.curve;
      j["tone_stage"] = {
          {"applied", sc.tone_applied},
          {"source", "plan"},
          {"features", tp.features},
          {"predicted", pred},
          {"predicted_sd", pred_sd},
          {"render", desc(pl.render)},
          {"targets", {{"L2", n9((float)tt.L2)}, {"L50", n9((float)tt.L50)}, {"L98", n9((float)tt.L98)},
                       {"R9010", n9((float)tt.range)}}},
          // The share of the highlight and range moves kept, and of the shadow move: 1, or less where they would clip
          // the final image.
          {"moves", {{"shadow", n9((float)kToneShadowMove)}, {"highlight", n9((float)kToneHighlightMove)},
                     {"range", n9((float)kToneRangeMove)}, {"highlight_share", n9(sc.tone_share)},
                     {"shadow_share", n9(sc.tone_shadow_share)}}},
          // Where no share held both ends, the rule that chose the curve (--tone_stage); null where one held.
          {"fallback", sc.tone_fallback ? ordered_json(sc.tone_fallback) : ordered_json(nullptr)},
          // In full (17 digits): the curve the image went through, exactly; null where the shares moved nothing.
          {"curve", sc.tone_moved_nothing ? ordered_json(nullptr) : tone_curve_json(tc)},
          {"clip", clip_json()}};
    } else {
      j["tone_stage"] = {{"applied", false}, {"skipped", tp.skipped ? tp.skipped : ""}};
    }
  } else if (const ToneCurve* q = sc.tone_curve) {
    // A replayed curve (a previous run's, or given): the curve and the clipping on this image.
    j["tone_stage"] = {
        {"applied", sc.tone_applied},
        {"source", "replay"},
        {"curve", tone_curve_json(*q)},
        {"clip", clip_json()}};
  } else {
    j["tone_stage"] = nullptr;
  }

  // What a replay rendered from recorded values (null when nothing was replayed).
  if (!sc.replay_source.empty()) {
    j["replay"] = {{"source", sc.replay_source},
                   {"tone_curve", sc.tone_curve ? ordered_json(sc.tone_applied ? "applied" : "not applied")
                                                : ordered_json(nullptr)}};
  } else {
    j["replay"] = nullptr;
  }

  // A density output (the print part of negicc.h): its record, and the file's bit depth (a DPX holds 10 bits).
  ordered_json density = sc.density_json.empty() ? ordered_json(nullptr)
                                                 : ordered_json::parse(sc.density_json, nullptr, false);
  const int out_bits = density.is_object() && density.contains("encoding") ? density["encoding"].value("bits", 16) : 16;
  j["output"] = {{"path", sc.output_path}, {"colorspace", sc.colorspace}, {"bits", out_bits}};
  if (density.is_object()) j["density"] = density;

  ordered_json t;
  t["raw_decode"] = num(sc.t_raw_decode_ms, "%.1f");
  t["frame_detection"] = sc.frame_det ? num(sc.t_frame_det_ms, "%.1f") : ordered_json(nullptr);
  // The bracket analysis outside Auto's time (the knobs, the roll-off); before, it counted as raw_decode.
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

// ---- The density file (TIFF or DPX) and the print's CLFs (declared in the print part of negicc.h) -------------------

namespace {

// The capture's time as the TIFF and DPX date fields write it, "YYYY:MM:DD HH:MM:SS"; empty when the input records
// none, so that the file is the same on every run (its replay included) and claims no time it does not know, and when
// the time cannot be converted or its year takes more than the field's four digits.
std::string date_time(long long timestamp, char sep) {
  if (timestamp <= 0) return "";
  time_t t = (time_t)timestamp;
  struct tm tm_buf;
  if (!localtime_r(&t, &tm_buf)) return "";
  char buf[80];
  const int n = snprintf(buf, sizeof buf, "%04d:%02d:%02d%c%02d:%02d:%02d", tm_buf.tm_year + 1900, tm_buf.tm_mon + 1,
                         tm_buf.tm_mday, sep, tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
  if (n != (int)sizeof "YYYY:MM:DD HH:MM:SS" - 1) return "";
  return buf;
}

// Ends a write: on a failed one (a full disk, say) the file is removed, unless the output is a device or a pipe, so
// that nothing is left half written.
int finish_write(const std::string& path, bool ok) {
  if (ok) return 0;
  log_error("Failed to write %s\n", path.c_str());
  std::error_code ec;
  if (std::filesystem::is_regular_file(path, ec)) remove(path.c_str());
  return -1;
}

// ---- TIFF (little-endian, one strip) --------------------------------------------------------------------------------
struct TiffEntry {
  uint16_t tag, type;
  uint32_t count, value;
};

void put16(std::vector<uint8_t>& b, uint16_t v) {
  b.push_back((uint8_t)(v & 0xff));
  b.push_back((uint8_t)(v >> 8));
}
void put32(std::vector<uint8_t>& b, uint32_t v) {
  for (int k = 0; k < 4; ++k) b.push_back((uint8_t)((v >> (8 * k)) & 0xff));
}

// ---- DPX (big-endian) -----------------------------------------------------------------------------------------------
void be16(uint8_t* p, uint16_t v) {
  p[0] = (uint8_t)(v >> 8);
  p[1] = (uint8_t)(v & 0xff);
}
void be32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v >> 24);
  p[1] = (uint8_t)((v >> 16) & 0xff);
  p[2] = (uint8_t)((v >> 8) & 0xff);
  p[3] = (uint8_t)(v & 0xff);
}
void be_float(uint8_t* p, float f) {
  uint32_t v;
  memcpy(&v, &f, 4);
  be32(p, v);
}
void put_str(uint8_t* p, size_t field, const std::string& s) {
  memcpy(p, s.data(), std::min(field - 1, s.size()));
}

}  // namespace

int write_density_tiff(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                       const std::string& description, const DensityFileInfo& info) {
  const uint64_t strip_bytes = (uint64_t)width * (uint64_t)height * 6;
  // The values that do not fit an entry, laid out after the IFD.
  const std::string date = date_time(info.timestamp, ' ');
  std::vector<TiffEntry> entries;
  std::vector<uint8_t> extra;
  const int n_entries = 17 + (date.empty() ? 0 : 1) + (info.make.empty() ? 0 : 1) + (info.model.empty() ? 0 : 1);
  const uint32_t ifd_offset = 8;
  const uint32_t extra_offset = ifd_offset + 2 + (uint32_t)n_entries * 12 + 4;
  auto add_extra = [&](const void* data, size_t n) {
    const uint32_t at = extra_offset + (uint32_t)extra.size();
    const uint8_t* p = (const uint8_t*)data;
    extra.insert(extra.end(), p, p + n);
    if (extra.size() % 2) extra.push_back(0);
    return at;
  };
  // An ASCII value with its terminating NUL; four bytes or fewer are stored in the entry itself (TIFF 6.0).
  auto ascii = [&](uint16_t tag, const std::string& s) {
    const uint32_t n = (uint32_t)s.size() + 1;
    if (n <= 4) {
      uint32_t v = 0;
      for (uint32_t k = 0; k < n; ++k) v |= (uint32_t)(uint8_t)s[k] << (8 * k);   // the bytes in file order
      entries.push_back({tag, 2, n, v});
      return;
    }
    entries.push_back({tag, 2, n, add_extra(s.c_str(), n)});
  };
  const uint16_t bps[3] = {16, 16, 16};
  const uint32_t res[2] = {300, 1};
  entries.push_back({254, 4, 1, 0});                       // NewSubfileType
  entries.push_back({256, 4, 1, (uint32_t)width});         // ImageWidth
  entries.push_back({257, 4, 1, (uint32_t)height});        // ImageLength
  {
    std::vector<uint8_t> b;
    for (uint16_t v : bps) put16(b, v);
    entries.push_back({258, 3, 3, add_extra(b.data(), b.size())});  // BitsPerSample
  }
  entries.push_back({259, 3, 1, 1});                       // Compression: none
  entries.push_back({262, 3, 1, 2});                       // PhotometricInterpretation: RGB
  ascii(270, description);                                 // ImageDescription
  if (!info.make.empty()) ascii(271, info.make);
  if (!info.model.empty()) ascii(272, info.model);
  const size_t strip_entry = entries.size();
  entries.push_back({273, 4, 1, 0});                       // StripOffsets (filled below)
  entries.push_back({274, 3, 1, 1});                       // Orientation: top left
  entries.push_back({277, 3, 1, 3});                       // SamplesPerPixel
  entries.push_back({278, 4, 1, (uint32_t)height});        // RowsPerStrip
  entries.push_back({279, 4, 1, (uint32_t)strip_bytes});   // StripByteCounts
  {
    std::vector<uint8_t> b;
    put32(b, res[0]);
    put32(b, res[1]);
    const uint32_t at = add_extra(b.data(), b.size());
    entries.push_back({282, 5, 1, at});                    // XResolution
    entries.push_back({283, 5, 1, at});                    // YResolution
  }
  entries.push_back({284, 3, 1, 1});                       // PlanarConfiguration: chunky
  entries.push_back({296, 3, 1, 2});                       // ResolutionUnit: inch
  ascii(305, info.software);                               // Software
  if (!date.empty()) ascii(306, date);                     // DateTime: the capture's, when the input records one
  if ((int)entries.size() != n_entries) {
    log_error("Internal error: the TIFF directory has %d entries, expected %d\n", (int)entries.size(), n_entries);
    return -1;
  }
  // A classic TIFF addresses its data with 32-bit offsets: the strip, after the header and the values laid out after
  // the IFD (the description among them), must end within them.
  if ((uint64_t)extra_offset + extra.size() + strip_bytes > 0xffffff00ull) {
    log_error("The image is too large for a TIFF file (%dx%d, %zu bytes of metadata)\n", width, height,
              description.size());
    return -1;
  }
  const uint32_t data_offset = extra_offset + (uint32_t)extra.size();
  entries[strip_entry].value = data_offset;

  std::vector<uint8_t> head;
  head.push_back('I');
  head.push_back('I');
  put16(head, 42);
  put32(head, ifd_offset);
  put16(head, (uint16_t)entries.size());
  for (const TiffEntry& e : entries) {
    put16(head, e.tag);
    put16(head, e.type);
    put32(head, e.count);
    if (e.type == 3 && e.count == 1) {
      put16(head, (uint16_t)e.value);
      put16(head, 0);
    } else {
      put32(head, e.value);
    }
  }
  put32(head, 0);  // no further directory
  head.insert(head.end(), extra.begin(), extra.end());

  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) {
    log_error("Cannot open %s for writing\n", path.c_str());
    return -1;
  }
  bool ok = fwrite(head.data(), 1, head.size(), fp) == head.size();
  // The samples little-endian, as the header says ("II"), whatever the host's byte order; three per pixel.
  std::vector<uint8_t> row((size_t)width * 6);
  for (int j = 0; ok && j < height; ++j) {
    const uint16_t* src = pixels + (size_t)j * width * channels;
    for (int i = 0; i < width; ++i) {
      for (int c = 0; c < 3; ++c) {
        const uint16_t v = src[(size_t)i * channels + c];
        row[(size_t)i * 6 + 2 * c] = (uint8_t)(v & 0xff);
        row[(size_t)i * 6 + 2 * c + 1] = (uint8_t)(v >> 8);
      }
    }
    ok = fwrite(row.data(), 1, row.size(), fp) == row.size();
  }
  ok = (fclose(fp) == 0) && ok;
  return finish_write(path, ok);
}

int write_density_dpx(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                      const std::string& description, const DensityEncoding& enc, DensityMeasure measure,
                      const DensityFileInfo& info) {
  if (enc.bits != 10) {
    log_error("Internal error: a DPX holds 10-bit density codes\n");
    return -1;
  }
  // The layout is SMPTE ST 268:2014's (DPX, header version V2.0): the file (768 bytes, 5.1), image (640, 5.2) and
  // image source (256, 5.3, the orientation information below) headers make the generic 1664, the film (256, 6.1) and
  // television (128, 6.2) headers the industry 384, so the user-defined data starts at 2048 (clause 7; at most 1 MiB,
  // 4.1). An undefined integer or real field holds all ones (4.2). Descriptor 50 is R, G, B (Table 1); transfer and
  // colorimetric codes 0 and 1 are user defined and printing density (Tables 5A and 5B); packing 1 is filled to 32-bit
  // words, method A (Table 3B).
  // The user-defined data: a 32-byte identification, then the metadata JSON; the image starts on a multiple of 8.
  const uint32_t kHeader = 2048, kUserId = 32, kMaxUser = 1024 * 1024;
  uint32_t user_size = kUserId + (uint32_t)description.size() + 1;
  user_size = (user_size + 7u) & ~7u;
  if (user_size > kMaxUser) {
    log_error("The density metadata (%u bytes) does not fit a DPX file's user data (1 MiB)\n", user_size);
    return -1;
  }
  const uint64_t image_bytes = (uint64_t)width * (uint64_t)height * 4;
  const uint32_t image_offset = kHeader + user_size;
  if (image_bytes + image_offset > 0xffffff00ull) {
    log_error("The image is too large for a DPX file (%dx%d)\n", width, height);
    return -1;
  }
  std::vector<uint8_t> h(image_offset, 0);
  uint8_t* p = h.data();
  // File information (768 bytes).
  memcpy(p, "SDPX", 4);
  be32(p + 4, image_offset);
  put_str(p + 8, 8, "V2.0");
  be32(p + 16, (uint32_t)(image_offset + image_bytes));
  be32(p + 20, 1);                 // a new frame
  be32(p + 24, 1664);              // generic header: file, image and orientation information
  be32(p + 28, 384);               // industry header: film and television
  be32(p + 32, user_size);
  {
    const size_t slash = path.find_last_of("/\\");
    put_str(p + 36, 100, slash == std::string::npos ? path : path.substr(slash + 1));
  }
  put_str(p + 136, 24, date_time(info.timestamp, ':'));   // the capture's time, as the TIFF's DateTime (none: blank)
  put_str(p + 160, 100, info.software);
  be32(p + 660, 0xffffffffu);      // not encrypted
  // Image information (640 bytes).
  uint8_t* im = p + 768;
  be16(im, 0);                     // left to right, top to bottom
  be16(im + 2, 1);                 // one image element
  be32(im + 4, (uint32_t)width);
  be32(im + 8, (uint32_t)height);
  uint8_t* el = im + 12;
  be32(el, 0);                     // unsigned
  be32(el + 4, (uint32_t)enc.offset);                                      // reference low: the film base
  be_float(el + 8, 0.0f);
  be32(el + 12, (uint32_t)enc.max_code());                                 // reference high
  be_float(el + 16, (float)(enc.max_code() - enc.offset) / (float)enc.scale);
  el[20] = 50;                     // descriptor: R, G, B
  // The transfer characteristic and the colorimetric specification: 1, printing density, for RP 180 (the Cineon
  // convention) and for Academy Printing Density (the code tools know; the element's description says ADX); 0, user
  // defined, for Status M, which is not a printing density.
  const uint8_t kind = measure == kDensityStatusM ? 0 : 1;
  el[21] = kind;
  el[22] = kind;
  el[23] = 10;                     // bits
  be16(el + 24, 1);                // packing: filled to 32-bit words, method A
  be16(el + 26, 0);                // no run-length encoding
  be32(el + 28, image_offset);
  be32(el + 32, 0);
  be32(el + 36, 0);
  put_str(el + 40, 32, measure == kDensityStatusM ? "Status M density above base"
                           : measure == kDensityRp180 ? "RP 180 printing density"
                                                      : "ADX10, SMPTE ST 2065-3");
  for (int k = 1; k < 8; ++k) memset(im + 12 + k * 72, 0, 72);
  // Orientation information (256 bytes): undefined but for the original size and the input device.
  uint8_t* orient = p + 1408;
  be32(orient, 0);
  be32(orient + 4, 0);
  memset(orient + 8, 0xff, 8);     // centre: undefined
  be32(orient + 16, (uint32_t)width);
  be32(orient + 20, (uint32_t)height);
  put_str(orient + 148, 32, info.model.empty() ? info.make : info.model);
  memset(orient + 212, 0xff, 8);   // border validity
  be32(orient + 220, 1);           // pixel aspect ratio 1:1
  be32(orient + 224, 1);
  // Film information (256 bytes) and television information (128 bytes): undefined.
  uint8_t* film = p + 1664;
  memset(film + 48, 0xff, 20);     // frame position, sequence length, held count, frame rate, shutter angle
  uint8_t* tv = p + 1920;
  memset(tv, 0xff, 52);
  // User-defined data.
  put_str(p + kHeader, kUserId, "negicc.density/1");
  memcpy(p + kHeader + kUserId, description.data(), description.size());

  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) {
    log_error("Cannot open %s for writing\n", path.c_str());
    return -1;
  }
  bool ok = fwrite(h.data(), 1, h.size(), fp) == h.size();
  std::vector<uint8_t> row((size_t)width * 4);
  for (int j = 0; ok && j < height; ++j) {
    const uint16_t* src = pixels + (size_t)j * width * channels;
    for (int i = 0; i < width; ++i) {
      const uint32_t r = src[(size_t)i * channels] & 0x3ffu, g = src[(size_t)i * channels + 1] & 0x3ffu,
                     b = src[(size_t)i * channels + 2] & 0x3ffu;
      be32(&row[(size_t)i * 4], (r << 22) | (g << 12) | (b << 2));
    }
    ok = fwrite(row.data(), 4, (size_t)width, fp) == (size_t)width;
  }
  ok = (fclose(fp) == 0) && ok;
  return finish_write(path, ok);
}

void remove_print_clfs(const DensityPrints& prints) {
  for (const std::string& path : prints.paths) remove(path.c_str());
}

int write_print_clfs(const DensityRecord& rec, const DensityPrints& prints) {
  // Every material's nodes before any file is written, so a material with none writes nothing.
  std::vector<std::string> nodes;
  for (size_t k = 0; k < prints.frames.size(); ++k) {
    nodes.push_back(print_clf_nodes(prints.materials[k]));
    if (nodes.back().empty()) {
      log_error("No print CLF for %s: its log exposure range, its curves or its colours are not finite numbers\n",
                prints.materials[k].id.c_str());
      return 1;
    }
  }
  for (size_t k = 0; k < prints.frames.size(); ++k) {
    const FramePrint& print = prints.frames[k];
    const std::string clf = frame_print_clf(rec, print, nodes[k]);
    FILE* f = fopen(prints.paths[k].c_str(), "wb");
    const bool ok = f && fwrite(clf.data(), 1, clf.size(), f) == clf.size();
    if ((f && fclose(f) != 0) || !ok) {
      log_error("Cannot write '%s'\n", prints.paths[k].c_str());
      remove_print_clfs(prints);
      return 1;
    }
    log_info("Wrote '%s': this file as a print on %s, sRGB (bracket %s, exposure %.3f, slopes %.3f %.3f %.3f)\n",
             prints.paths[k].c_str(), print.name.c_str(), print.bracket.c_str(), print.exposure, print.slopes[0],
             print.slopes[1], print.slopes[2]);
  }
  return 0;
}
