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

#include "np_output.h"

#include <lcms2.h>
#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <unistd.h>

#include <cmath>
#include <string>

#include "dcraw/gamma_curve.h"
#include "elle_icc_profiles/sRGB_elle_V2_g10.h"
#include "elle_icc_profiles/sRGB_elle_V2_srgbtrc.h"
#include "np_geometry.h"
#include "np_profile.h"
#include "tiff_head.h"

int write_tiff(LibRaw* proc, const std::string& attach_profile, const std::string& output,
               const std::vector<uint8_t>& embedded_icc, const uint16_t* rgb3) {
  unsigned* output_profile = NULL;
  unsigned profile_size = 0;
  bool dynamically_allocated = false;

  if (!attach_profile.empty()) {
    // If the profile to attach is not the psuedo "srgb" profile, the profile will be attached to the TIFF.
    // This is a hack to force LibRaw write the ICC profile in the TIFF without conversion.
    printf("Attaching profile: %s\n", attach_profile.c_str());
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
    printf("Attaching profile: [bundle embedded ICC] (%u bytes)\n", (unsigned)embedded_icc.size());
    output_profile = reinterpret_cast<unsigned int*>(const_cast<uint8_t*>(embedded_icc.data()));
    profile_size = (unsigned)embedded_icc.size();
  }

  const unsigned height = proc->imgdata.sizes.iheight;
  const unsigned width = proc->imgdata.sizes.iwidth;
  struct tiff_hdr header;
  tiff_head(proc, &header, profile_size);
  auto* fp = fopen(output.c_str(), "wb+");
  if (!fp) {
    fprintf(stderr, "ERROR! Cannot open %s for writing\n", output.c_str());
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

static std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", (unsigned int)c);
          out += buf;
        } else {
          out += (char)c;
        }
        break;
    }
  }
  return out;
}

static std::string json_num(double val, const char* fmt = "%.6g") {
  if (!std::isfinite(val)) return "null";
  char buf[64];
  snprintf(buf, sizeof(buf), fmt, val);
  return buf;
}

bool write_json_sidecar(
    const std::string& output_path,
    int ac, char* av[],
    const LibRaw* proc,
    const std::vector<std::string>& files,
    int orig_w, int orig_h,
    bool half_size,
    bool has_roi, const int* roi_coords,
    const std::string& roi_source,
    const FrameDetection* frame_det,
    int rot_cw, bool hflip, bool vflip,
    int out_w, int out_h,
    bool film_base_applied,
    const std::vector<int>& film_base_rgb,
    const std::vector<int>& profile_film_base_rgb,
    const std::string& film_base_source,
    float std_density,
    bool has_std_density,
    const std::string& film_profile_path,
    const float* merged_matrix,
    float exposure_comp, float gain_g, float gain_b,
    unsigned pin_mask,
    float knee, const std::string& knee_source,
    const ShoulderInfo& shoulder,
    bool has_dino, const DinoSolveResult& dino_res,
    const std::string& dino_model_path,
    const std::string& dino_backend,
    const std::string& colorspace_name,
    double t_raw_decode_ms,
    double t_frame_det_ms,
    double t_full_conversion_ms,
    double t_tiff_write_ms,
    bool is_bundle,
    const std::string& bundle_target_name,
    int bundle_target_idx,
    const ParamSearchResult* auto_res,
    const int* auto_frame_half,
    double t_auto_ms,
    InputSpace input_space) {
  const std::string json_path = output_path + ".json";
  const std::string tmp_path = json_path + ".tmp";
  FILE* fp = fopen(tmp_path.c_str(), "w");
  if (!fp) {
    fprintf(stderr, "Warning: Cannot open sidecar file %s for writing\n", tmp_path.c_str());
    return false;
  }

  fprintf(fp, "{\n");
  fprintf(fp, "  \"schema\": \"negicc.neg_process.sidecar/1\",\n");
  fprintf(fp, "  \"tool\": {\"name\": \"neg_process\", \"version\": \"%s\"},\n", json_escape(NEGICC_GIT_VERSION).c_str());

  // argv
  fprintf(fp, "  \"argv\": [");
  for (int i = 0; i < ac; ++i) {
    fprintf(fp, "%s\"%s\"", (i > 0 ? ", " : ""), json_escape(av[i]).c_str());
  }
  fprintf(fp, "],\n");

  // inputs
  std::string camera = "Unknown";
  if (proc) {
    std::string make = proc->imgdata.idata.make;
    std::string model = proc->imgdata.idata.model;
    if (!make.empty() && !model.empty()) {
      if (model.find(make) == 0) camera = model;
      else camera = make + " " + model;
    } else if (!model.empty()) camera = model;
    else if (!make.empty()) camera = make;
  }
  fprintf(fp, "  \"inputs\": {\"raw_files\": [");
  for (size_t i = 0; i < files.size(); ++i) {
    fprintf(fp, "%s\"%s\"", (i > 0 ? ", " : ""), json_escape(files[i]).c_str());
  }
  fprintf(fp, "], \"camera\": \"%s\", \"iso\": %s, \"shutter\": %s},\n",
          json_escape(camera).c_str(),
          proc ? json_num(proc->imgdata.other.iso_speed).c_str() : "0",
          proc ? json_num(proc->imgdata.other.shutter).c_str() : "0");

  // geometry
  fprintf(fp, "  \"geometry\": {\n");
  fprintf(fp, "    \"sensor_size\": [%d, %d], \"half_size\": %s,\n",
          orig_w, orig_h, half_size ? "true" : "false");
  fprintf(fp, "    \"roi_source\": \"%s\", ", json_escape(roi_source).c_str());
  if (has_roi) {
    fprintf(fp, "\"roi\": [%d, %d, %d, %d],\n", roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3]);
  } else {
    fprintf(fp, "\"roi\": null,\n");
  }
  fprintf(fp, "    \"rot_cw\": %d, \"hflip\": %s, \"vflip\": %s,\n",
          rot_cw, hflip ? "true" : "false", vflip ? "true" : "false");
  fprintf(fp, "    \"output_size\": [%d, %d]\n", out_w, out_h);
  fprintf(fp, "  },\n");

  // frame_detection
  if (frame_det) {
    // Sensor pixel coordinates throughout (the detector ran on the RAW before the decode). aspect and format describe
    // the frame as the previous detector did ("3:2", ...); film_format is the format the detection used. axis is
    // "h" or "v", or null when the detector left it undecided (a single square frame with nothing to orient it).
    auto rect_json = [](const FilmRect& r) {
      char b[160];
      snprintf(b, sizeof(b), "{\"cx\": %.1f, \"cy\": %.1f, \"w\": %.1f, \"h\": %.1f, \"angle_deg\": %.3f}",
               r.cx, r.cy, r.w, r.h, r.angle_deg);
      return std::string(b);
    };
    const FilmRect& pr = frame_det->primary.rect;
    const float aspect = std::max(pr.w, pr.h) / std::max(1.0f, std::min(pr.w, pr.h));
    const char* ratio = "-";
    switch (frame_det->format) {
      case FILM_35MM: ratio = "3:2"; break;
      case FILM_645: ratio = "4:3"; break;
      case FILM_6X6: ratio = "1:1"; break;
      case FILM_6X7: ratio = "7:6"; break;
      case FILM_6X9: ratio = "3:2"; break;
      default: break;
    }
    fprintf(fp, "  \"frame_detection\": {\n");
    const bool has_axis = frame_det->axis == 'h' || frame_det->axis == 'v';
    fprintf(fp, "    \"ok\": %s, \"axis\": %s, \"confidence\": %s, \"low_confidence\": %s, \"reason\": \"%s\",\n",
            frame_det->ok ? "true" : "false", has_axis ? (frame_det->axis == 'h' ? "\"h\"" : "\"v\"") : "null", json_num(frame_det->conf, "%.2f").c_str(),
            frame_det->conf < kLowFrameConfidence ? "true" : "false", json_escape(frame_det->reason).c_str());
    fprintf(fp, "    \"gaps\": [");
    int g_cnt = 0;
    for (int g = 0; g < 2; ++g) {
      if (frame_det->gaps[g][0] >= 0) {
        if (g_cnt > 0) fprintf(fp, ", ");
        fprintf(fp, "[%d, %d]", frame_det->gaps[g][0], frame_det->gaps[g][1]);
        g_cnt++;
      }
    }
    // segments: the frames on the strip, in strip order (secondaries are sorted by index, negative before the
    // primary); selected: the primary's position among them.
    int selected = 0;
    for (const FilmFrame& f : frame_det->secondaries) selected += f.index < 0 ? 1 : 0;
    fprintf(fp, "], \"segments\": %d, \"n_gaps\": %d, \"selected\": %d,\n",
            1 + (int)frame_det->secondaries.size(), frame_det->n_gaps, selected);
    fprintf(fp, "    \"aspect\": %s, \"format\": \"%s\", \"film_format\": \"%s\", \"orientation\": \"%s\",\n",
            json_num(aspect, "%.4g").c_str(), ratio, film_format_name(frame_det->format),
            json_escape(frame_det->orientation).c_str());
    fprintf(fp, "    \"rotation_deg\": %s, \"px_per_mm\": %s, \"light_clipped\": %s,\n",
            json_num(frame_det->rotation_deg, "%.3f").c_str(), json_num(frame_det->px_per_mm, "%.2f").c_str(),
            frame_det->light_clipped ? "true" : "false");
    fprintf(fp, "    \"crop\": [%d, %d, %d, %d], \"primary\": %s,\n", frame_det->crop[0], frame_det->crop[1],
            frame_det->crop[2], frame_det->crop[3], rect_json(pr).c_str());
    fprintf(fp, "    \"secondaries\": [");
    for (size_t i = 0; i < frame_det->secondaries.size(); ++i) {
      const FilmFrame& f = frame_det->secondaries[i];
      fprintf(fp, "%s{\"index\": %d, \"rect\": %s, \"visible\": [%d, %d, %d, %d], \"visible_fraction\": %s}",
              i ? ", " : "", f.index, rect_json(f.rect).c_str(), f.visible[0], f.visible[1], f.visible[2],
              f.visible[3], json_num(f.visible_fraction, "%.3f").c_str());
    }
    fprintf(fp, "],\n");
    fprintf(fp, "    \"film_base\": %s, \"base_patch\": %s, \"sprockets\": %d\n",
            frame_det->has_film_base ? rect_json(frame_det->film_base).c_str() : "null",
            frame_det->has_base ? rect_json(frame_det->base_patch).c_str() : "null",
            (int)frame_det->sprockets.size());
    fprintf(fp, "  },\n");
  } else {
    fprintf(fp, "  \"frame_detection\": null,\n");
  }

  // film_base
  fprintf(fp, "  \"film_base\": {\n");
  fprintf(fp, "    \"source\": \"%s\", \"applied\": %s,\n",
          json_escape(film_base_source).c_str(),
          film_base_applied ? "true" : "false");
  if (film_base_source == "none") {
    fprintf(fp, "    \"film_base_rgb\": null, \"std_density\": null,\n");
  } else {
    fprintf(fp, "    \"film_base_rgb\": [%d, %d, %d], \"std_density\": %s,\n",
            film_base_rgb[0], film_base_rgb[1], film_base_rgb[2],
            (film_base_source == "rebate" && has_std_density) ? json_num(std_density, "%.6g").c_str() : "null");
  }
  fprintf(fp, "    \"profile_film_base_rgb\": [%d, %d, %d]\n",
          profile_film_base_rgb[0], profile_film_base_rgb[1], profile_film_base_rgb[2]);
  fprintf(fp, "  },\n");

  // profile
  if (!film_profile_path.empty()) {
    if (is_bundle) {
      fprintf(fp, "  \"profile\": {\"path\": \"%s\", \"type\": \"bundle\", \"bundle_target\": \"%s\", "
              "\"input_space\": \"%s\"},\n",
              json_escape(film_profile_path).c_str(), json_escape(bundle_target_name).c_str(),
              to_string(input_space));
    } else {
      fprintf(fp, "  \"profile\": {\"path\": \"%s\", \"type\": \"icc\", \"bundle_target\": null},\n",
              json_escape(film_profile_path).c_str());
    }
  } else {
    fprintf(fp, "  \"profile\": null,\n");
  }

  // matrix
  fprintf(fp, "  \"matrix\": {\"merged\": [");
  for (int i = 0; i < 9; ++i) {
    fprintf(fp, "%s%s", (i > 0 ? ", " : ""), json_num(merged_matrix[i], "%.9g").c_str());
  }
  fprintf(fp, "],\n");
  fprintf(fp, "             \"exposure_comp\": %s, \"gain_g\": %s, \"gain_b\": %s,\n",
          json_num(exposure_comp, "%.9g").c_str(),
          json_num(gain_g, "%.9g").c_str(),
          json_num(gain_b, "%.9g").c_str());
  fprintf(fp, "             \"pins\": {\"e\": %s, \"g\": %s, \"b\": %s}},\n",
          (pin_mask & PIN_E) ? "true" : "false",
          (pin_mask & PIN_G) ? "true" : "false",
          (pin_mask & PIN_B) ? "true" : "false");

  // cli_equivalent
  fprintf(fp, "  \"cli_equivalent\": [\"-E\", \"%s\", \"--gain_g\", \"%s\", \"--gain_b\", \"%s\", \"--knee\", \"%s\"",
          json_num(exposure_comp, "%.9g").c_str(),
          json_num(gain_g, "%.9g").c_str(),
          json_num(gain_b, "%.9g").c_str(),
          json_num(knee, "%.9g").c_str());
  if (shoulder.active()) {
    fprintf(fp, ", \"--shoulder\", \"%s\"", json_num(shoulder.x_hi, "%.9g").c_str());
  }
  if (has_roi) {
    fprintf(fp, ", \"--roi\", \"%d\", \"%d\", \"%d\", \"%d\", \"%d\", \"%d\", \"%d\"",
            roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3],
            rot_cw, hflip ? 1 : 0, vflip ? 1 : 0);
  }
  if (is_bundle && bundle_target_idx >= 0) {
    fprintf(fp, ", \"--target\", \"%d\"", bundle_target_idx);
  }
  fprintf(fp, "],\n");

  // auto: the conversion parameter search (multi_target_optimizer.h)
  if (auto_res) {
    const ParamSearchResult& a = *auto_res;
    fprintf(fp, "  \"auto\": {\n");
    fprintf(fp, "    \"target\": \"%s\", \"target_idx\": %d, \"e\": %s, \"g\": %s, \"b\": %s,\n",
            json_escape(a.target_name).c_str(), a.target_idx, json_num(a.params.e, "%.9g").c_str(),
            json_num(a.params.g, "%.9g").c_str(), json_num(a.params.b, "%.9g").c_str());
    fprintf(fp, "    \"residual\": %s, \"status\": \"%s\", \"e_norm\": %s, \"input_matrix\": [",
            json_num(a.residual, "%.6g").c_str(), to_string(a.status), json_num(a.e_norm, "%.9g").c_str());
    for (int i = 0; i < 9; ++i) fprintf(fp, "%s%s", (i > 0 ? ", " : ""), json_num(a.input_matrix[i], "%.9g").c_str());
    fprintf(fp, "],\n");
    fprintf(fp, "    \"frame_half\": [%d, %d, %d, %d], \"g_lo\": %s, \"g_hi\": %s, \"scene_ev\": %s,\n",
            auto_frame_half[0], auto_frame_half[1], auto_frame_half[2], auto_frame_half[3],
            json_num(a.g_lo, "%.6g").c_str(), json_num(a.g_hi, "%.6g").c_str(), json_num(a.scene_ev, "%.4g").c_str());
    fprintf(fp, "    \"headroom\": {\"applied\": %s, \"ev\": %s, \"reason\": \"%s\"},\n",
            headroom_applied(a.headroom) ? "true" : "false", json_num(a.headroom_ev, "%.4g").c_str(),
            to_string(a.headroom));
    fprintf(fp, "    \"rescue\": {\"branch\": \"%s\", \"white_clip\": %s, \"changed\": %s},\n",
            a.rescue == RescueOutcome::kNotNeeded ? "N" : "R", json_num(a.rescue_white_clip, "%.6g").c_str(),
            a.rescue == RescueOutcome::kTaken ? "true" : "false");
    fprintf(fp, "    \"screening\": [");
    for (size_t i = 0; i < a.screening.size(); ++i) {
      const auto& r = a.screening[i];
      fprintf(fp, "%s{\"target_idx\": %d, \"e\": %s, \"g\": %s, \"b\": %s, \"intent_loss\": %s}", (i > 0 ? ", " : ""),
              r.target_idx, json_num(r.params.e, "%.6g").c_str(), json_num(r.params.g, "%.6g").c_str(),
              json_num(r.params.b, "%.6g").c_str(),
              json_num(r.intent_loss, "%.6g").c_str());
    }
    fprintf(fp, "],\n    \"tournament\": [");
    for (size_t i = 0; i < a.tournament.size(); ++i) {
      const auto& r = a.tournament[i];
      fprintf(fp, "%s{\"target_idx\": %d, \"e\": %s, \"g\": %s, \"b\": %s, \"residual\": %s, \"status\": \"%s\", "
              "\"iters\": %d}", (i > 0 ? ", " : ""), r.target_idx, json_num(r.params.e, "%.6g").c_str(),
              json_num(r.params.g, "%.6g").c_str(), json_num(r.params.b, "%.6g").c_str(),
              json_num(r.residual, "%.6g").c_str(), to_string(r.status), r.steps);
    }
    fprintf(fp, "],\n");
    fprintf(fp, "    \"model\": \"%s\", \"backend\": \"%s\", \"inferences\": %d, \"time_ms\": %s\n",
            json_escape(dino_model_path).c_str(), json_escape(dino_backend).c_str(), a.inferences,
            json_num(t_auto_ms, "%.1f").c_str());
    fprintf(fp, "  },\n");
  } else {
    fprintf(fp, "  \"auto\": null,\n");
  }

  // dino
  if (has_dino) {
    fprintf(fp, "  \"dino\": {\n");
    fprintf(fp, "    \"model\": \"%s\", \"backend\": \"%s\", \"grid\": [%d, %d], \"target\": [%d, %d],\n",
            json_escape(dino_model_path).c_str(), json_escape(dino_backend).c_str(),
            dino_res.grid_w, dino_res.grid_h, dino_res.target_w, dino_res.target_h);
    fprintf(fp, "    \"e_center\": %s, \"e_bounds\": [%s, %s], \"status\": \"%s\", \"best_iteration\": %d, \"inferences\": %d,\n",
            json_num(dino_res.e_center).c_str(),
            json_num(dino_res.bounds_lo).c_str(),
            json_num(dino_res.bounds_hi).c_str(),
            json_escape(dino_res.status).c_str(), dino_res.best_iteration, dino_res.total_inferences);
    fprintf(fp, "    \"solved_relative\": {\"e\": %s, \"g\": %s, \"b\": %s},\n",
            json_num(dino_res.solved_relative.e).c_str(),
            json_num(dino_res.solved_relative.g).c_str(),
            json_num(dino_res.solved_relative.b).c_str());

    if (!dino_res.iterations.empty() && dino_res.iterations[0].tag == 'b') {
      const auto& b = dino_res.iterations[0];
      fprintf(fp, "    \"bootstrap\": {\"e\": %s, \"g\": %s, \"b\": %s, \"loss_L\": %s, \"loss_a\": %s, \"loss_b\": %s, "
                  "\"clip_pct\": %s, \"penalty_clip\": %s, \"penalty_anchor\": %s, \"objective\": %s},\n",
              json_num(b.e).c_str(), json_num(b.g).c_str(), json_num(b.b).c_str(),
              json_num(b.loss_L).c_str(), json_num(b.loss_a).c_str(), json_num(b.loss_b).c_str(),
              json_num(b.clip_pct).c_str(), json_num(b.penalty_clip).c_str(),
              json_num(b.penalty_anchor).c_str(), json_num(b.objective).c_str());
    } else {
      fprintf(fp, "    \"bootstrap\": null,\n");
    }

    fprintf(fp, "    \"iterations\": [");
    bool first_it = true;
    for (size_t i = 0; i < dino_res.iterations.size(); ++i) {
      if (dino_res.iterations[i].tag == 'b') continue;
      const auto& it = dino_res.iterations[i];
      if (!first_it) fprintf(fp, ",\n      ");
      else fprintf(fp, "\n      ");
      first_it = false;
      fprintf(fp, "{\"e\": %s, \"g\": %s, \"b\": %s, \"residual\": %s, \"loss_L\": %s, \"loss_a\": %s, \"loss_b\": %s, "
                  "\"clip_pct\": %s, \"penalty_clip\": %s, \"penalty_anchor\": %s, \"objective\": %s}",
              json_num(it.e).c_str(), json_num(it.g).c_str(), json_num(it.b).c_str(),
              json_num(it.residual).c_str(), json_num(it.loss_L).c_str(), json_num(it.loss_a).c_str(),
              json_num(it.loss_b).c_str(), json_num(it.clip_pct).c_str(),
              json_num(it.penalty_clip).c_str(), json_num(it.penalty_anchor).c_str(),
              json_num(it.objective).c_str());
    }
    if (!first_it) fprintf(fp, "\n    ");
    fprintf(fp, "],\n");

    fprintf(fp, "    \"tone_mass\": [");
    for (size_t i = 0; i < dino_res.tone_mass.size(); ++i) {
      fprintf(fp, "%s%s", (i > 0 ? ", " : ""), json_num(dino_res.tone_mass[i]).c_str());
    }
    fprintf(fp, "]\n");
    fprintf(fp, "  },\n");
  } else {
    fprintf(fp, "  \"dino\": null,\n");
  }

  // tone
  fprintf(fp, "  \"tone\": {\"knee\": %s, \"knee_source\": \"%s\", ",
          json_num(knee, "%.9g").c_str(), json_escape(knee_source).c_str());
  if (has_dino && dino_res.dynamic_knee.active) {
    fprintf(fp, "\"q995\": %s, \"knee_fit\": %s,\n",
            json_num(dino_res.dynamic_knee.q995).c_str(),
            json_num(dino_res.dynamic_knee.knee_fit).c_str());
  } else {
    fprintf(fp, "\"q995\": null, \"knee_fit\": null,\n");
  }
  fprintf(fp, "           \"shoulder\": %s, \"shoulder_source\": \"%s\", \"shoulder_q995\": %s, "
              "\"shoulder_knee\": %s, \"shoulder_guard\": %s},\n",
          json_num(shoulder.x_hi, "%.9g").c_str(), json_escape(shoulder.source).c_str(),
          shoulder.has_q995 ? json_num(shoulder.q995).c_str() : "null",
          shoulder.has_knee ? json_num(shoulder.knee).c_str() : "null",
          shoulder.has_guard ? (shoulder.guard ? "true" : "false") : "null");

  // output
  fprintf(fp, "  \"output\": {\"path\": \"%s\", \"colorspace\": \"%s\", \"bits\": 16},\n",
          json_escape(output_path).c_str(), json_escape(colorspace_name).c_str());

  // timings_ms
  fprintf(fp, "  \"timings_ms\": {\"raw_decode\": %s, \"frame_detection\": %s, ",
          json_num(t_raw_decode_ms, "%.1f").c_str(),
          frame_det ? json_num(t_frame_det_ms, "%.1f").c_str() : "null");
  if (has_dino) {
    fprintf(fp, "\"dino_views\": %s, \"dino_render\": %s, \"dino_inference\": %s, \"dino_search\": %s, ",
            json_num(dino_res.timing_views_ms, "%.1f").c_str(),
            json_num(dino_res.timing_render_ms, "%.1f").c_str(),
            json_num(dino_res.timing_infer_ms, "%.1f").c_str(),
            json_num(dino_res.timing_search_ms, "%.1f").c_str());
  } else {
    fprintf(fp, "\"dino_views\": null, \"dino_render\": null, \"dino_inference\": null, \"dino_search\": null, ");
  }
  fprintf(fp, "\"full_conversion\": %s, \"tiff_write\": %s}\n",
          json_num(t_full_conversion_ms, "%.1f").c_str(),
          json_num(t_tiff_write_ms, "%.1f").c_str());

  fprintf(fp, "}\n");
  const bool write_ok = (ferror(fp) == 0);
  const bool close_ok = (fclose(fp) == 0);
  if (!write_ok || !close_ok) {
    fprintf(stderr, "Warning: Failed to flush/close sidecar %s\n", tmp_path.c_str());
    unlink(tmp_path.c_str());
    return false;
  }

  if (rename(tmp_path.c_str(), json_path.c_str()) != 0) {
    fprintf(stderr, "Warning: Failed to rename sidecar from %s to %s\n", tmp_path.c_str(), json_path.c_str());
    unlink(tmp_path.c_str());
    return false;
  }
  return true;
}
