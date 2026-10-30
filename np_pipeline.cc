// Copyright 2024 Alpha Lam <arufa.hc@gmail.com>
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
#include "np_pipeline.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "cfa_pipeline.h"
#include "color_pipeline.h"
#include "embedded_data.h"
#include "np_conversion.h"
#include "np_geometry.h"
#include "np_input.h"
#include "np_output.h"
#include "np_profile.h"

void LibRawDeleter::operator()(LibRaw* p) const {
  if (!p) return;
  p->free_image();
  delete p;
}

void Scan::scan_exposure(const LibRaw& r, double& shutter_out, double& iso_out) {
  const bool have_shutter = (has_exif_exp && exif_shutter_d > 0.0) || r.imgdata.other.shutter > 0;
  const bool have_iso = (has_exif_exp && exif_iso_d > 0.0) || r.imgdata.other.iso_speed > 0;
  shutter_out = (has_exif_exp && exif_shutter_d > 0.0)
                    ? exif_shutter_d
                    : (r.imgdata.other.shutter > 0 ? (double)r.imgdata.other.shutter : 0.125);
  iso_out = (has_exif_exp && exif_iso_d > 0.0)
                ? std::round(exif_iso_d)
                : (r.imgdata.other.iso_speed > 0 ? std::round((double)r.imgdata.other.iso_speed) : 100.0);
  if ((!have_shutter || !have_iso) && !warned_default_exposure) {
    warned_default_exposure = true;
    log_warn("Missing EXIF exposure metadata in %s; falling back to %.4g s, ISO %.0f\n",
            exposure_file.c_str(), shutter_out, iso_out);
  }
}

RgbInput Scan::image() {
  RgbInput img;
  img.width = proc->imgdata.sizes.iwidth;
  img.height = proc->imgdata.sizes.iheight;
  if (!rgb3.empty()) {
    img.pixels = rgb3.data();
    img.channels = 3;
  } else {
    img.pixels = &proc->imgdata.image[0][0];
    img.channels = 4;
  }
  return img;
}

void CpuBackend::render(const MosaicInput& in, const Region& region, int dst_w, int dst_h, uint16_t* rgb_out) {
  if (in.n_frames == 1) {
    negicc::GrainAwareDebayerArgs ga{};
    ga.cfa_bayer = in.frames[0];
    ga.src_w = in.width;
    ga.src_h = in.height;
    ga.rx = region.x;
    ga.ry = region.y;
    ga.rw = region.w;
    ga.rh = region.h;
    ga.rot_cw = region.rot_cw;
    ga.hflip = region.hflip;
    ga.vflip = region.vflip;
    ga.dst_w = dst_w;
    ga.dst_h = dst_h;
    ga.filters = in.filters;
    ga.white_level = in.white_level;
    ga.eps = 4096.0f;
    ga.rgb_out = rgb_out;
    ga.flat = in.flat;
    ga.flat_x0 = region.x;
    ga.flat_y0 = region.y;
    ga.cfa_pitch = in.pitch;
    negicc::cpu_grain_aware_debayer(ga);
  } else {
    negicc::PixelShiftMergeArgs pm{};
    pm.bayer_f0 = in.frames[0];
    pm.bayer_f1 = in.frames[1];
    pm.bayer_f2 = in.frames[2];
    pm.bayer_f3 = in.frames[3];
    pm.src_w = in.width;
    pm.src_h = in.height;
    pm.rx = region.x;
    pm.ry = region.y;
    pm.rw = region.w;
    pm.rh = region.h;
    pm.rot_cw = region.rot_cw;
    pm.hflip = region.hflip;
    pm.vflip = region.vflip;
    pm.dst_w = dst_w;
    pm.dst_h = dst_h;
    pm.filters = in.filters;
    pm.white_level = in.white_level;
    pm.rgb_out = rgb_out;
    pm.flat = in.flat;
    pm.flat_x0 = region.x;
    pm.flat_y0 = region.y;
    pm.cfa_pitch = in.pitch;
    negicc::cpu_pixel_shift_merge(pm);
  }
}

void CpuBackend::convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
                         float knee, const ProfileData& prof, const RgbInput& img) {
  const int width = img.width, height = img.height, ch = img.channels;
  uint16_t* pixels = img.pixels;
  if (out.encoding == ENCODE_SRGB || out.encoding == ENCODE_SRGB_LINEAR) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        uint16_t* px = &pixels[((size_t)j * width + i) * ch];
        const negicc::Rgb v = negicc::convert_pixel(pc, merged_matrix, px);
        px[0] = quantize_u16(v.r);
        px[1] = quantize_u16(v.g);
        px[2] = quantize_u16(v.b);
      }
    }
  } else if (out.encoding == ENCODE_CUSTOM) {
    // Through Little-CMS from XYZ, a row at a time, into the image's own layout (3 samples, or LibRaw's 4).
    #pragma omp parallel
    {
      cmsHTRANSFORM thread_xform = cmsCreateTransform(out.xyz, TYPE_XYZ_FLT, out.custom,
                                                      ch == 3 ? TYPE_RGB_16 : TYPE_RGBA_16, INTENT_PERCEPTUAL, 0);
      std::vector<float> xyz_row(width * 3);

      #pragma omp for schedule(static)
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          const uint16_t* px = &pixels[((size_t)j * width + i) * ch];
          Rgb v = process_pixel(px, merged_matrix, knee, prof);
          xyz_row[i * 3 + 0] = v.r;
          xyz_row[i * 3 + 1] = v.g;
          xyz_row[i * 3 + 2] = v.b;
        }
        cmsDoTransform(thread_xform, xyz_row.data(), &pixels[(size_t)j * width * ch], width);
      }
      cmsDeleteTransform(thread_xform);
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        uint16_t* px = &pixels[((size_t)j * width + i) * ch];
        Rgb v = load_u16(px);
        v = apply_matrix(merged_matrix, v);
        v = apply_knee(v, knee);
        px[0] = quantize_u16(v.r);
        px[1] = quantize_u16(v.g);
        px[2] = quantize_u16(v.b);
      }
    }
  }
}

int prepare(const Options& o, Setup& s) {
  if (!o.flat.empty()) {
    try {
      s.flat_field = negicc::load_flat_field_file(o.flat);
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
  }

  // The profile: -p, else the built-in bundle of --film_stock and --light_source (or the light the input implies).
  ProfileBundle& bundle = s.bundle;
  if (!o.light_source.empty() && o.implied_light && o.light_source != o.implied_light && !o.tiff_input) {
    log_error("--light_source %s does not fit %s, which takes %s\n", o.light_source.c_str(),
            o.input_mode == InputMode::kLinearDng ? "a linear DNG"
            : o.input_mode == InputMode::kThreeShot ? "three captures, one per light"
                                                    : "a JPEG, converted from sRGB to linear Rec.2020",
            o.implied_light);
    return 1;
  }
  if (!o.light_source.empty() && o.film_stock.empty()) {
    log_error("--light_source requires --film_stock\n");
    return 1;
  }
  if (!o.film_profile.empty()) {
    s.film_prof_arg = o.film_profile;
    try {
      s.is_bundle = load_bundle_arg(s.film_prof_arg, bundle);
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
    if (!s.is_bundle && !is_regular_file(resolve_profile_path(s.film_prof_arg))) {
      log_error("No profile bundle JSON or ICC profile '%s'\n", s.film_prof_arg.c_str());
      return 1;
    }
  } else if (!o.film_stock.empty()) {
    if (o.light_source.empty() && !o.implied_light) {
      log_error("--film_stock requires --light_source (tri or d65)\n");
      return 1;
    }
    const std::string light = !o.light_source.empty() ? o.light_source : std::string(o.implied_light);
    if (o.light_source.empty()) log_info("Light source: %s, from the input\n", light.c_str());
    std::string error;
    const std::string key = embedded_bundle_key(o.film_stock, light, error);
    if (key.empty()) {
      log_error("%s (--list_profiles lists them)\n", error.c_str());
      return 1;
    }
    try {
      if (!load_embedded_bundle(key, bundle)) {
        log_error("Cannot load built-in profile bundle %s\n", key.c_str());
        return 1;
      }
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
    s.is_bundle = true;
  }
  if (s.is_bundle) {
    log_info("Profile bundle: %s (%s, %zu brackets)\n", bundle.key.c_str(), bundle.path.c_str(), bundle.targets.size());
  }
  s.has_profile = s.is_bundle || !s.film_prof_arg.empty();

  if (o.target_given) {
    if (!s.is_bundle) {
      log_error("--target requires a JSON profile bundle\n");
      return 1;
    }
    const std::string& tgt_str = o.target;
    char* end = nullptr;
    long val = strtol(tgt_str.c_str(), &end, 10);
    if (end && *end == '\0' && val >= 0 && val < (long)bundle.targets.size()) {
      s.manual_target_idx = (int)val;
    } else {
      for (size_t i = 0; i < bundle.targets.size(); ++i) {
        if (bundle.targets[i].name == tgt_str || bundle.targets[i].display_name == tgt_str) {
          s.manual_target_idx = (int)i;
          break;
        }
      }
    }
    if (s.manual_target_idx < 0) {
      log_error("Target '%s' not found in bundle\n", tgt_str.c_str());
      return 1;
    }
  }
  // The parameter search runs with a profile unless --gain_g or --gain_b is given. On a bundle it is
  // negicc-station's conversion parameter search (multi_target_optimizer.h), which picks the bracket, the exposure
  // and the gains; with --target or -E (and on a single ICC profile) it solves the gains, and the exposure unless -E
  // gives it, on one profile.
  s.exposure_given = o.exposure_given;
  s.use_dino = s.has_profile && !o.gain_g_given && !o.gain_b_given;
  s.bundle_search = s.use_dino && s.is_bundle && s.manual_target_idx < 0 && !s.exposure_given;
  if (!o.dino_adapter.empty() && !s.use_dino) {
    log_error("--dino requires the parameter search: a profile (--film_stock and --light_source, or "
                    "--film_profile) without --gain_g or --gain_b\n");
    return 1;
  }
  if (o.shoulder_auto && !s.use_dino) {
    log_error("--shoulder auto requires the parameter search (it measures the shoulder from the render "
                    "at the searched parameters); give an x_hi instead\n");
    return 1;
  }

  // The crosstalk correction: the bundle's, with the rows -r, -g and -b give in its place (for the search too).
  s.r_coeff = o.r_coeff;
  s.g_coeff = o.g_coeff;
  s.b_coeff = o.b_coeff;
  const bool* row_given = o.row_given;
  std::vector<float>* rows[3] = {&s.r_coeff, &s.g_coeff, &s.b_coeff};
  // A rec2020_linear (white-light) bundle takes the white-light decode, which has no crosstalk to correct and, as in
  // negicc-station, decodes single-shot captures only.
  // py-ref: negicc_station/src/pipeline_api.py @ 813166c (prepare_image)
  s.white_light = s.is_bundle && bundle.input_space == InputSpace::kRec2020Linear;
  s.camera_rgb_input = o.tiff_input && s.is_bundle && !s.white_light;
  if (s.white_light) {
    if (row_given[0] || row_given[1] || row_given[2]) {
      log_error("-r, -g and -b cannot be used with the rec2020_linear profile bundle %s: its white-light "
                      "decode has no crosstalk to correct\n", bundle.key.c_str());
      return 1;
    }
    if (o.files.size() == 4) {
      log_error("The rec2020_linear profile bundle %s converts single-shot captures only, not pixel "
                      "shift\n", bundle.key.c_str());
      return 1;
    }
    if (o.input_mode == InputMode::kThreeShot) {
      log_error("The rec2020_linear profile bundle %s is for white light; three captures, one per "
                      "light, take a camera_raw_rgb (tri) bundle\n", bundle.key.c_str());
      return 1;
    }
  }
  // Three captures, one per light, have no crosstalk between the lights to correct: each channel comes from the
  // capture under its own light, so the correction is the identity unless -r, -g or -b give one.
  if (o.input_mode == InputMode::kThreeShot && s.is_bundle) {
    for (int r = 0; r < 3; ++r) {
      if (row_given[r]) continue;
      for (int c = 0; c < 3; ++c) {
        bundle.crosstalk_matrix[r * 3 + c] = r == c ? 1.0f : 0.0f;
        bundle.crosstalk_matrix_d[r * 3 + c] = r == c ? 1.0 : 0.0;
      }
    }
    bundle.has_crosstalk = true;
  }
  if (s.is_bundle) {
    for (int r = 0; r < 3; ++r) {
      if (row_given[r]) {
        for (int c = 0; c < 3; ++c) {
          bundle.crosstalk_matrix[r * 3 + c] = (*rows[r])[c];
          bundle.crosstalk_matrix_d[r * 3 + c] = (double)(*rows[r])[c];
        }
      } else if (bundle.has_crosstalk) {
        rows[r]->assign(bundle.crosstalk_matrix + r * 3, bundle.crosstalk_matrix + r * 3 + 3);
      }
    }
  }
  s.has_crosstalk = row_given[0] || row_given[1] || row_given[2] || (s.is_bundle && bundle.has_crosstalk);
  // An RGB TIFF or JPEG is colour-managed RGB, converted to linear Rec.2020: not the camera RGB a crosstalk
  // correction (a camera_raw_rgb bundle) needs.
  if (o.input_mode == InputMode::kRgbFile && !s.camera_rgb_input && s.has_crosstalk) {
    log_error("%s is an RGB TIFF or JPEG, converted to linear Rec.2020: give a rec2020_linear (d65) "
                    "profile bundle, not one with a crosstalk correction (a TIFF of linear camera RGB takes "
                    "--light_source tri)\n", o.files[0].c_str());
    return 1;
  }
  if (o.input_mode == InputMode::kLinearDng && s.is_bundle && !s.white_light) {
    log_error("A linear DNG is converted to linear Rec.2020 with its own colour data: give a "
                    "rec2020_linear (d65) profile bundle\n");
    return 1;
  }
  s.raw_decode = s.white_light ? RawDecode::kWhiteLight
                 : s.has_crosstalk ? RawDecode::kCameraRgb
                                   : RawDecode::kRec2020;

  if (s.use_dino) {
    const EmbeddedFile* trt = find_embedded_backbone(true);
    const EmbeddedFile* onnx = find_embedded_backbone(false);
    const EmbeddedFile* backbone = trt ? trt : onnx;
    if (!backbone) {
      log_error("This neg_process was built without the DINOv3 backbone of the parameter search (make "
                      "dinov3_model, or make dinov3_engine for USE_CUDA=1, then rebuild); give --gain_g or --gain_b "
                      "to convert without it\n");
      return 1;
    }
    s.dino_path = std::string("[built-in] ") + backbone->name;
    log_info("Loading DINOv3 backbone %s (backend: %s)...\n", s.dino_path.c_str(), o.dino_backend.c_str());
    try {
      s.engine.load_memory(trt ? trt->data : nullptr, trt ? trt->size : 0, onnx ? onnx->data : nullptr,
                           onnx ? onnx->size : 0, o.dino_backend);
      // The file that runs: the .onnx when the engine was not loaded.
      if (trt && std::strcmp(s.engine.backend_used(), "gpu") != 0) {
        s.dino_path = std::string("[built-in] ") + onnx->name;
      }
      if (!o.dino_adapter.empty()) {
        std::string adapter = o.dino_adapter;
        if (is_directory(adapter)) adapter = (std::filesystem::path(adapter) / "adapter.onnx").string();
        s.engine.load_adapter_file(adapter);
      } else {
        std::string adapter_dir;
        const std::string profile_name = !o.film_stock.empty() ? o.film_stock
                                         : s.is_bundle ? bundle.key : s.film_prof_arg;
        const EmbeddedFile* a = find_embedded_adapter(profile_name, adapter_dir);
        if (!a) {
          log_error("No intent adapter for profile '%s'\n", profile_name.c_str());
          return 1;
        }
        s.engine.set_adapter(a->data, a->size, adapter_dir);
      }
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
    log_info("DINOv3 engine loaded [%s] adapter %s (memory: %.1f MB)\n", s.engine.backend_name(),
           s.engine.adapter_name().c_str(), (double)s.engine.memory_bytes() / (1024.0 * 1024.0));
  }
  return 0;
}

int decode(const Options& o, Setup& s, Scan& scan) {
  const std::vector<std::string>& files = o.files;
  ProfileBundle& bundle = s.bundle;
  DinoV3Engine& engine = s.engine;
  const bool use_dino = s.use_dino, bundle_search = s.bundle_search;
  const bool is_roi_auto = o.roi_auto;
  const InputMode input_mode = o.input_mode;
  const bool camera_rgb_input = s.camera_rgb_input;
  const RawDecode raw_decode = s.raw_decode;
  const negicc::FlatFieldPtr& flat_field = s.flat_field;
  const ShoulderInfo& shoulder_info = o.shoulder;
  const bool is_shoulder_auto = o.shoulder_auto;
  int* roi_coords = scan.roi;
  bool& has_roi = scan.has_roi;
  int& rot_cw = scan.rot_cw;
  bool &hflip = scan.hflip, &vflip = scan.vflip;
  FrameDetection& frame_det = scan.frame_det;
  bool& has_auto_res = scan.has_auto_res;
  ParamSearchResult& auto_res = scan.auto_res;
  int* auto_frame_half = scan.auto_frame_half;
  double &t_auto_ms = scan.t_auto_ms, &t_frame_det_ms = scan.t_frame_det_ms, &t_raw_decode_ms = scan.t_raw_decode_ms;
  double& t_raw_0 = scan.t_raw_0;
  LibRawPtr* ps_extra_procs = scan.ps_extra;
  std::string& input_source = scan.input_source;
  std::shared_ptr<const negicc::FlatTables>& flat_full_tables = scan.flat_full_tables;
  std::copy(o.roi, o.roi + 4, roi_coords);
  has_roi = o.has_roi;
  rot_cw = o.rot_cw;
  hflip = o.hflip;
  vflip = o.vflip;

  scan.exposure_file = files[0];
  scan.has_exif_exp = read_exif_exposure_d(files[0], scan.exif_shutter_d, scan.exif_iso_d);
  auto get_scan_exposure = [&](const LibRaw& r, double& shutter_out, double& iso_out) {
    scan.scan_exposure(r, shutter_out, iso_out);
  };

  HalfSizeAnalysis analyse;
  if (bundle_search) {
    analyse = [&](const LibRaw& raw) -> bool {
      const double t_auto_0 = omp_get_wtime();
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      get_scan_exposure(raw, scan_shutter_s, scan_iso);
      // --roi halved, or the whole image.
      const int half_w = raw.imgdata.sizes.iwidth, half_h = raw.imgdata.sizes.iheight;
      int* bbox = auto_frame_half;
      bbox[0] = 0;
      bbox[1] = 0;
      bbox[2] = half_w;
      bbox[3] = half_h;
      if (has_roi) {
        bbox[0] = roi_coords[0] / 2;
        bbox[1] = roi_coords[1] / 2;
        bbox[2] = std::max(1, (roi_coords[0] + roi_coords[2]) / 2 - bbox[0]);
        bbox[3] = std::max(1, (roi_coords[1] + roi_coords[3]) / 2 - bbox[1]);
        clamp_rect(bbox, half_w, half_h);
      }
      // Step 3 & Step 4: zero-copy RgbView over the half-size image, cropped and oriented in place.
      negicc::RgbView half_view = negicc::RgbView::from_interleaved_rgb(
          &raw.imgdata.image[0][0], half_w, half_h, 4, (ptrdiff_t)half_w * 4);
      negicc::RgbView frame = half_view.crop_and_orient(bbox[0], bbox[1], bbox[2], bbox[3], rot_cw, hflip, vflip);
      ParamSearchOptions opts;
      if (o.dino_iters_given) opts.max_steps = o.dino_iters;
      opts.shoulder = shoulder_info.x_hi;
      opts.shoulder_auto = is_shoulder_auto;
      opts.enclosure_rescue = o.enclosure_rescue;
      if (!search_conversion_params(bundle, engine, frame, scan_shutter_s, scan_iso, opts, auto_res)) {
        log_error("The conversion parameter search failed\n");
        return false;
      }
      has_auto_res = true;
      t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
      return true;
    };
  }

  FrameDetectionParams det_params;
  det_params.format = o.film_format;
  det_params.axis_hint = o.roi_axis_hint;
  RawRegionFinder find_frame;
  if (is_roi_auto && files.size() != 4) {
    find_frame = [&](LibRaw& raw, int roi[4]) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(raw, mosaic)) {
        log_error("--roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        return false;
      }
      det_params.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, det_params, frame_det);
      t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
      if (frame_det.conf < kLowFrameConfidence) {
        roi_coords[0] = 0;
        roi_coords[1] = 0;
        roi_coords[2] = mosaic.width;
        roi_coords[3] = mosaic.height;
      } else {
        roi_coords[0] = frame_det.crop[0];
        roi_coords[1] = frame_det.crop[1];
        roi_coords[2] = frame_det.crop[2];
        roi_coords[3] = frame_det.crop[3];
      }
      has_roi = true;
      for (int i = 0; i < 4; ++i) roi[i] = roi_coords[i];
      return true;
    };
  }

  auto print_frame_detection = [&]() {
    const char* axis_name = frame_det.axis == 'h' ? "h" : (frame_det.axis == 'v' ? "v" : "undecided");
    log_info("Film frame: axis=%s format=%s orientation=%s rotation=%.2f deg gaps=%d frames=%d conf=%.2f%s%s\n",
           axis_name, film_format_name(frame_det.format), frame_det.orientation.c_str(), frame_det.rotation_deg,
           frame_det.n_gaps, 1 + (int)frame_det.secondaries.size(), frame_det.conf,
           frame_det.reason.empty() ? "" : " reason: ", frame_det.reason.c_str());
    if (frame_det.n_gaps > 0) {
      log_info("  Gaps (sensor px):");
      for (int g = 0; g < 2; ++g)
        if (frame_det.gaps[g][0] >= 0) log_info(" [%d, %d]", frame_det.gaps[g][0], frame_det.gaps[g][1]);
      log_info("\n");
    }
    log_info("  Frame (sensor px): x=%d y=%d w=%d h=%d (frame centre %.1f, %.1f, size %.1f x %.1f, %.1f px/mm)\n",
           roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3], frame_det.primary.rect.cx,
           frame_det.primary.rect.cy, frame_det.primary.rect.w, frame_det.primary.rect.h, frame_det.px_per_mm);
    for (const FilmFrame& f : frame_det.secondaries)
      log_info("  Frame %+d (sensor px): visible x=%d y=%d w=%d h=%d (%.0f%% of a frame)\n", f.index, f.visible[0],
             f.visible[1], f.visible[2], f.visible[3], 100.0f * f.visible_fraction);
  };

  const int qual_flag = o.quality;
  const bool no_crop_flag = o.no_crop;

  // Determine whether the 5-step zero-copy CFA pipeline applies (Step 1..5).
  bool use_cfa_pipeline = false;
  negicc::InputFormat input_format = (files.size() == 4) ? negicc::InputFormat::SONY_PIXEL_SHIFT_4
                                                         : negicc::InputFormat::CFA_BAYER_OR_XTRANS;
  t_raw_0 = omp_get_wtime();
  LibRaw *proc = nullptr;
  LibRaw *unpacked_probe = nullptr;
  float cfa_white_level = 65535.0f;
  int cfa_rx = 0, cfa_ry = 0, cfa_rw = 0, cfa_rh = 0;
  negicc::RgbView full_half_view;
  negicc::RgbView oriented_half_view;
  negicc::RgbView search_half_view;
  negicc::FlatSampleArgs flat_full{};

  if ((input_mode == InputMode::kRaw || input_mode == InputMode::kLinearDng) &&
      (files.size() == 4 || (raw_decode == RawDecode::kCameraRgb && qual_flag < 0))) {
    LibRaw* probe = new LibRaw();
    log_info("Loading RAW file %s\n", files[0].c_str());
    int ret = probe->open_file(files[0].c_str());
    if (ret != LIBRAW_SUCCESS) {
      log_error("Cannot open %s: %s\n", files[0].c_str(), libraw_strerror(ret));
      delete probe;
      return 1;
    }
    log_info("Image size: %dx%d\n", probe->imgdata.sizes.iwidth, probe->imgdata.sizes.iheight);
    if ((ret = probe->unpack()) != LIBRAW_SUCCESS) {
      log_error("Cannot unpack %s: %s\n", files[0].c_str(), libraw_strerror(ret));
      delete probe;
      return 1;
    }
    if (files.size() == 4) {
      if (!is_sony_a7rm4(probe)) {
        log_error("4-shot pixel shift merge is only available with Sony A7RM4 camera (got %s %s)\n",
                probe->imgdata.idata.make, probe->imgdata.idata.model);
        delete probe;
        return 1;
      }
      use_cfa_pipeline = true;
    } else if (is_sony_cfa_geometry(probe) && probe->imgdata.rawdata.raw_image) {
      use_cfa_pipeline = true;
    }
    if (use_cfa_pipeline) {
      proc = probe;
    } else {
      unpacked_probe = probe;
    }
  }


  if (use_cfa_pipeline) {
    (void)input_format;
    if (is_roi_auto) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(*proc, mosaic)) {
        log_error("--roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        delete proc;
        return 1;
      }
      det_params.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, det_params, frame_det);
      t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
    }

    // Step 1: Per-channel black subtraction in-place on the unpacked CFA mosaic.
    if (!subtract_cfa_black_in_place(proc, cfa_white_level)) {
      log_error("Invalid CFA mosaic geometry in %s\n", files[0].c_str());
      delete proc;
      return 1;
    }
    const int src_w = proc->imgdata.sizes.width;
    const int src_h = proc->imgdata.sizes.height;
    const size_t cfa_pitch = proc->imgdata.sizes.raw_pitch / 2;
    const unsigned int filters = proc->imgdata.idata.filters;
    const uint16_t* cfa_origin = proc->imgdata.rawdata.raw_image +
                                 (size_t)proc->imgdata.sizes.top_margin * cfa_pitch +
                                 proc->imgdata.sizes.left_margin;

    // For 4x pixel shift, unpack and black-subtract frames 1..3 in Step 1 and keep all 4 CFA mosaics in memory.
    if (files.size() == 4) {
      for (int mi = 1; mi < 4; ++mi) {
        ps_extra_procs[mi - 1].reset(new LibRaw());
        LibRaw* fp = ps_extra_procs[mi - 1].get();
        log_info("Loading RAW file %s\n", files[mi].c_str());
        int fret = fp->open_file(files[mi].c_str());
        if (fret != LIBRAW_SUCCESS || (fret = fp->unpack()) != LIBRAW_SUCCESS) {
          log_error("Cannot open/unpack %s: %s\n", files[mi].c_str(), libraw_strerror(fret));
          delete proc;
          return 1;
        }
        const size_t fpitch = fp->imgdata.sizes.raw_pitch / 2;
        if (!is_sony_a7rm4(fp) || fp->imgdata.sizes.width != src_w || fp->imgdata.sizes.height != src_h ||
            fpitch != cfa_pitch || fp->imgdata.idata.filters != filters ||
            fp->imgdata.sizes.top_margin != proc->imgdata.sizes.top_margin ||
            fp->imgdata.sizes.left_margin != proc->imgdata.sizes.left_margin ||
            std::fabs(fp->imgdata.other.iso_speed - proc->imgdata.other.iso_speed) > 1e-3f ||
            std::fabs(fp->imgdata.other.shutter - proc->imgdata.other.shutter) > 1e-6f) {
          log_error("4-shot pixel shift merge is only available with matching Sony A7RM4 frames\n");
          delete proc;
          return 1;
        }
        float fw_white = 0.0f;
        if (!subtract_cfa_black_in_place(fp, fw_white)) {
          log_error("Invalid CFA mosaic geometry in %s\n", files[mi].c_str());
          delete proc;
          return 1;
        }
      }
    }

    // Step 2: Downsizing via zero-copy RgbView over the CFA mosaic.
    const int half_w = src_w / 2;
    const int half_h = src_h / 2;
    std::shared_ptr<const negicc::FlatTables> flat_half_tables;
    negicc::FlatSampleArgs flat_half{};
    if (flat_field) {
      try {
        flat_half_tables = flat_field->tables(half_w, half_h, true);
        flat_half = flat_half_tables->host();
        flat_full_tables = flat_field->tables(src_w, src_h, false);
        flat_full = flat_full_tables->host();
      } catch (const std::exception& e) {
        log_error("%s\n", e.what());
        delete proc;
        return 1;
      }
    }
    full_half_view = negicc::RgbView::from_cfa_half(
        cfa_origin, cfa_pitch, src_w, src_h, filters, nullptr, flat_half, flat_half_tables);

    // Step 3: Crop, rotate, flip & inline ROI search on the zero-copy view.
    if (is_roi_auto) {
      if (frame_det.conf < kLowFrameConfidence) {
        roi_coords[0] = 0;
        roi_coords[1] = 0;
        roi_coords[2] = src_w;
        roi_coords[3] = src_h;
      } else {
        roi_coords[0] = frame_det.crop[0];
        roi_coords[1] = frame_det.crop[1];
        roi_coords[2] = frame_det.crop[2];
        roi_coords[3] = frame_det.crop[3];
      }
      has_roi = true;
      print_frame_detection();

      if (frame_det.conf < kLowFrameConfidence) {
        log_warn("low detection confidence (conf=%.2f%s%s); falling back to the full sensor frame\n",
               frame_det.conf, frame_det.reason.empty() ? "" : ", reason: ", frame_det.reason.c_str());
      }
      if (use_dino && frame_det.axis == 'v' && rot_cw == 0) {
        log_warn("axis=v, rot=0 (vertical frame without rotation may degrade DINOv3 intent accuracy)\n");
      }
    }

    cfa_rx = 0;
    cfa_ry = 0;
    cfa_rw = src_w;
    cfa_rh = src_h;
    if (has_roi) {
      if (roi_coords[0] < 0 || roi_coords[1] < 0 || roi_coords[0] >= src_w || roi_coords[1] >= src_h ||
          roi_coords[2] <= 0 || roi_coords[3] <= 0) {
        log_error("ROI [%d, %d, %d, %d] is outside sensor boundaries (%dx%d)\n",
                roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3], src_w, src_h);
        delete proc;
        return 1;
      }
      cfa_rx = roi_coords[0];
      cfa_ry = roi_coords[1];
      cfa_rw = std::min(roi_coords[2], src_w - cfa_rx);
      cfa_rh = std::min(roi_coords[3], src_h - cfa_ry);
    } else if (!no_crop_flag && files.size() == 1) {
      const auto& crop_ref = proc->imgdata.sizes.raw_inset_crops[0];
      const bool has_inset_crop = crop_ref.cwidth > 0 && crop_ref.cheight > 0 &&
                                  crop_ref.cleft != 0xffff && crop_ref.ctop != 0xffff;
      if (has_inset_crop && (crop_ref.cleft || crop_ref.ctop)) {
        cfa_rx = std::max(0, std::min((int)crop_ref.cleft, src_w - 1));
        cfa_ry = std::max(0, std::min((int)crop_ref.ctop, src_h - 1));
        cfa_rw = std::max(1, std::min((int)crop_ref.cwidth, src_w - cfa_rx));
        cfa_rh = std::max(1, std::min((int)crop_ref.cheight, src_h - cfa_ry));
      }
    }

    int out_half[4] = {
        cfa_rx / 2,
        cfa_ry / 2,
        std::max(1, (cfa_rx + cfa_rw) / 2 - cfa_rx / 2),
        std::max(1, (cfa_ry + cfa_rh) / 2 - cfa_ry / 2),
    };
    clamp_rect(out_half, half_w, half_h);
    for (int i = 0; i < 4; ++i) auto_frame_half[i] = out_half[i];
    search_half_view = full_half_view.crop_and_orient(
        auto_frame_half[0], auto_frame_half[1], auto_frame_half[2], auto_frame_half[3],
        rot_cw, hflip, vflip);
    oriented_half_view = search_half_view;

    // Step 4: Conversion parameter search inline on the zero-copy oriented RgbView.
    if (bundle_search) {
      const double t_auto_0 = omp_get_wtime();
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      get_scan_exposure(*proc, scan_shutter_s, scan_iso);
      ParamSearchOptions opts;
      if (o.dino_iters_given) opts.max_steps = o.dino_iters;
      opts.shoulder = shoulder_info.x_hi;
      opts.shoulder_auto = is_shoulder_auto;
      opts.enclosure_rescue = o.enclosure_rescue;
      if (!search_conversion_params(bundle, engine, search_half_view, scan_shutter_s, scan_iso, opts, auto_res)) {
        log_error("The conversion parameter search failed\n");
        delete proc;
        return 1;
      }
      has_auto_res = true;
      t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
    }
    t_raw_decode_ms = (omp_get_wtime() - t_raw_0) * 1000.0 - t_auto_ms - t_frame_det_ms;

    // The mosaic and the frame, for render.
    scan.mosaic.frames[0] = cfa_origin;
    scan.mosaic.n_frames = files.size() == 4 ? 4 : 1;
    for (int i = 0; i + 1 < scan.mosaic.n_frames; ++i) {
      LibRaw* fp = ps_extra_procs[i].get();
      const size_t fpitch = fp->imgdata.sizes.raw_pitch / 2;
      scan.mosaic.frames[i + 1] = fp->imgdata.rawdata.raw_image + (size_t)fp->imgdata.sizes.top_margin * fpitch +
                                  fp->imgdata.sizes.left_margin;
    }
    scan.mosaic.pitch = cfa_pitch;
    scan.mosaic.width = src_w;
    scan.mosaic.height = src_h;
    scan.mosaic.filters = filters;
    scan.mosaic.white_level = cfa_white_level;
    scan.mosaic.flat = flat_full;
    scan.cfa_region = {cfa_rx, cfa_ry, cfa_rw, cfa_rh, rot_cw, hflip, vflip};
  } else {
    if (input_mode == InputMode::kRgbFile || input_mode == InputMode::kThreeShot) {
      proc = load_rgb_input(files, input_mode, camera_rgb_input, qual_flag, !no_crop_flag && !has_roi && !is_roi_auto,
                            (!is_roi_auto && has_roi) ? roi_coords : nullptr, analyse,
                            is_roi_auto ? &frame_det : nullptr, det_params, input_source);
    } else {
      proc = load_raw(files[0], true,
                      qual_flag,
                      !no_crop_flag && !has_roi && !is_roi_auto,
                      raw_decode,
                      (!is_roi_auto && has_roi) ? roi_coords : nullptr,
                      analyse, find_frame, flat_field.get(),
                      unpacked_probe);
    }
    unpacked_probe = nullptr;
    if (!proc) {
      log_error("Cannot open %s\n", files[0].c_str());
      return 1;
    }
    if (is_roi_auto) {
      print_frame_detection();
      if (frame_det.conf < kLowFrameConfidence) {
        log_warn("low detection confidence (conf=%.2f%s%s); falling back to the full sensor frame\n",
               frame_det.conf, frame_det.reason.empty() ? "" : ", reason: ", frame_det.reason.c_str());
      }
      if (use_dino && frame_det.axis == 'v' && rot_cw == 0) {
        log_warn("axis=v, rot=0 (vertical frame without rotation may degrade DINOv3 intent accuracy)\n");
      }
    }
    if (rot_cw != 0 || hflip || vflip) {
      if (!extract_roi_and_transform(proc, 0, 0, proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                                     rot_cw, hflip, vflip)) {
        proc->free_image();
        delete proc;
        return 1;
      }
      log_info("Transformed subrect: %dx%d (rot=%d, hflip=%d, vflip=%d)\n",
             proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight, rot_cw, hflip ? 1 : 0, vflip ? 1 : 0);
    }
    t_raw_decode_ms = (omp_get_wtime() - t_raw_0) * 1000.0 - t_auto_ms - t_frame_det_ms;
  }

  const int orig_w = proc->imgdata.rawdata.sizes.width ? proc->imgdata.rawdata.sizes.width : proc->imgdata.sizes.width;
  const int orig_h = proc->imgdata.rawdata.sizes.height ? proc->imgdata.rawdata.sizes.height : proc->imgdata.sizes.height;

  scan.proc.reset(proc);
  scan.cfa = use_cfa_pipeline;
  scan.orig_w = orig_w;
  scan.orig_h = orig_h;
  log_info("ISO Speed: %f\n", proc->imgdata.other.iso_speed);
  log_info("Shutter Speed: %f\n", proc->imgdata.other.shutter);
  return 0;
}

int begin_conversion(const Options& o, const Setup& s, Conversion& c) {
  const bool is_bundle = s.is_bundle, bundle_search = s.bundle_search;
  const ProfileBundle& bundle = s.bundle;
  std::vector<int>& effective_profile_film_base_rgb = c.profile_film_base_rgb;
  const std::vector<int> effective_film_base_rgb = {1, 1, 1};
  unsigned& pin_mask = c.pin_mask;
  float* merged_matrix = c.merged_matrix;
  float &global_exposure_comp = c.exposure_comp, &gain_g = c.gain_g, &gain_b = c.gain_b;
  global_exposure_comp = o.exposure_comp;
  gain_g = o.gain_g;
  gain_b = o.gain_b;
  c.knee = kKnee;
  c.shoulder = o.shoulder;
  const std::vector<float> &r_coeff = s.r_coeff, &g_coeff = s.g_coeff, &b_coeff = s.b_coeff;

  // The sidecar records the bundle's film base. No film base of the capture is given (TODO at --film_stock: read it
  // from an image), so none is applied: the film base scale of the input matrix is 1.
  c.sidecar_profile_film_base_rgb = {1, 1, 1};
  if (is_bundle && bundle.has_film_base) {
    c.sidecar_profile_film_base_rgb = {bundle.profile_film_base_rgb[0], bundle.profile_film_base_rgb[1],
                                       bundle.profile_film_base_rgb[2]};
  }
  effective_profile_film_base_rgb = {1, 1, 1};

  if (bundle_search) {
    // The bundle's film base is folded into the solved exposure and gains (not applied as a film base scale).
    c.sidecar_film_base_rgb = c.sidecar_profile_film_base_rgb;
    c.film_base_source = "bundle";
  }

  if (o.exposure_given) pin_mask |= PIN_E;
  if (o.gain_g_given) pin_mask |= PIN_G;
  if (o.gain_b_given) pin_mask |= PIN_B;

  adjust_correction_matrix(r_coeff, g_coeff, b_coeff,
                           global_exposure_comp,
                           gain_g, gain_b,
                           effective_profile_film_base_rgb,
                           effective_film_base_rgb,
                           merged_matrix);
  return 0;
}

int render(const Options& o, const Setup& s, Scan& scan, Backend& backend) {
  (void)o;
  (void)s;
  if (!scan.cfa) return 0;  // LibRaw's decode, a TIFF, a JPEG or three captures: the image is in scan.proc already
  const double t_step5_debayer_0 = omp_get_wtime();
  LibRaw* proc = scan.proc.get();
  const Region& r = scan.cfa_region;
  int norm_rot = r.rot_cw;
  bool norm_hf = r.hflip, norm_vf = r.vflip;
  normalize_rotation(norm_rot, norm_hf, norm_vf);
  const bool swap_dims = (norm_rot == 90 || norm_rot == 270);
  const int dst_w = swap_dims ? r.h : r.w;
  const int dst_h = swap_dims ? r.w : r.h;
  scan.rgb3.resize((size_t)dst_w * dst_h * 3);
  log_info(scan.mosaic.n_frames == 1 ? "Debayer quality: grain-aware\n" : "Merging 4 images...\n");
  backend.render(scan.mosaic, r, dst_w, dst_h, scan.rgb3.data());

  // The mosaics are not needed any more: LibRaw's raw buffer is freed (proc keeps its metadata), and a pixel shift's
  // other frames go.
  if (proc->imgdata.rawdata.raw_alloc && !proc->imgdata.image) {
    proc->imgdata.image = (ushort (*)[4])proc->imgdata.rawdata.raw_alloc;
    proc->imgdata.rawdata.raw_alloc = nullptr;
    proc->imgdata.rawdata.raw_image = nullptr;
    proc->free_image();
  }
  proc->recycle_datastream();
  for (auto& p : scan.ps_extra) p.reset();
  scan.mosaic = MosaicInput();

  proc->imgdata.sizes.iwidth = dst_w;
  proc->imgdata.sizes.iheight = dst_h;
  proc->imgdata.idata.colors = 3;
  proc->imgdata.params.output_bps = 16;
  log_info("Processed image size: %dx%d\n", dst_w, dst_h);
  scan.t_raw_decode_ms += (omp_get_wtime() - t_step5_debayer_0) * 1000.0;
  return 0;
}

int solve(const Options& o, Setup& s, Scan& scan, Conversion& c) {
  const bool is_bundle = s.is_bundle, bundle_search = s.bundle_search, use_dino = s.use_dino;
  ProfileBundle& bundle = s.bundle;
  DinoV3Engine& engine = s.engine;
  const std::string& film_prof_arg = s.film_prof_arg;
  const int manual_target_idx = s.manual_target_idx;
  const bool use_cfa_pipeline = scan.cfa;
  std::vector<uint16_t>& cfa_export_rgb = scan.rgb3;
  LibRaw* proc = scan.proc.get();
  const bool has_auto_res = scan.has_auto_res;
  const ParamSearchResult& auto_res = scan.auto_res;
  unsigned& pin_mask = c.pin_mask;
  float* merged_matrix = c.merged_matrix;
  float &global_exposure_comp = c.exposure_comp, &gain_g = c.gain_g, &gain_b = c.gain_b;
  float& knee = c.knee;
  ShoulderInfo& shoulder_info = c.shoulder;
  const bool is_shoulder_auto = o.shoulder_auto;
  ParsedFilmProfile& film_prof = c.film_prof;
  std::vector<uint8_t>& embedded_icc = c.embedded_icc;
  std::string& bundle_target_name = c.bundle_target_name;
  int& bundle_target_idx = c.bundle_target_idx;
  DinoSolveResult& dino_res = c.dino_res;
  const std::vector<float> &r_coeff = s.r_coeff, &g_coeff = s.g_coeff, &b_coeff = s.b_coeff;
  const std::vector<int>& effective_profile_film_base_rgb = c.profile_film_base_rgb;
  const std::vector<int> effective_film_base_rgb = {1, 1, 1};
  // With a bundle, E, g and b are the station's (the search's, and what -E, --gain_g and --gain_b take): the
  // conversion is the station's, the bundle's merged matrix (crosstalk, film base and the scan's exposure ratio)
  // scaled per row by E, E*g and E*b, so neg_process renders what negicc-station renders. row_scale turns them into
  // the absolute exposure and gains on the crosstalk (the sidecar's matrix.*) and back.
  float station_base[9];
  double row_scale[3] = {1.0, 1.0, 1.0};
  if (is_bundle) {
    double scan_shutter_s = 0.125, scan_iso = 100.0;
    scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
    compute_station_merged_matrix(bundle, scan_shutter_s, scan_iso, nullptr, false, station_base, nullptr, row_scale);
  }
  auto apply_station = [&](float e, float g, float b) {
    c.station_params = std::array<float, 3>{e, g, b};
    negicc::candidate_matrix(station_base, e, g, b, merged_matrix);
    global_exposure_comp = (float)(row_scale[0] * (double)e);
    gain_g = (float)(row_scale[1] / row_scale[0] * (double)g);
    gain_b = (float)(row_scale[2] / row_scale[0] * (double)b);
  };
  const std::vector<int> unit_fb = {1, 1, 1};

  if (is_bundle && !bundle_search) {
    if (use_dino || manual_target_idx < 0 || !(pin_mask & PIN_E)) {
      float g_lo = 0.0f, g_hi = 0.0f, scene_ev = 0.0f;
      int num_feasible = 0;
      if (use_cfa_pipeline) {
        // TODO: Run `compute_bundle_enclosures` on `oriented_half_view` (zero-copy half-size CFA view) instead of the
        // full-resolution `cfa_export_rgb` buffer once single-target enclosure baselines are updated.
        compute_bundle_enclosures(bundle,
                                  negicc::RgbView::from_interleaved_rgb(
                                      cfa_export_rgb.data(), proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight),
                                  g_lo, g_hi, scene_ev, num_feasible);
      } else {
        compute_bundle_enclosures(bundle,
                                  (const uint16_t (*)[4])proc->imgdata.image,
                                  proc->imgdata.sizes.iwidth,
                                  proc->imgdata.sizes.iheight,
                                  g_lo, g_hi, scene_ev, num_feasible);
      }
    }
  } else if (!is_bundle) {
    if (!film_prof_arg.empty() && !load_film_profile(film_prof_arg, film_prof)) {
      log_error("Cannot load film profile %s\n", film_prof_arg.c_str());
    }
  }

  if (has_auto_res) {
    bundle_target_idx = auto_res.target_idx;
    bundle_target_name = auto_res.target_name;
    film_prof = bundle.targets[bundle_target_idx].film_prof;
    film_prof.rebind_pointers();
    // The search's E, g and b, converted as the station converts them (the sidecar's cli_equivalent reproduces it).
    apply_station(auto_res.params.e, auto_res.params.g, auto_res.params.b);
    shoulder_info.x_hi = auto_res.params.shoulder;
    if (is_shoulder_auto) {
      shoulder_info.has_q995 = true;
      shoulder_info.q995 = auto_res.shoulder_q995;
    }
    if (shoulder_info.active()) {
      shoulder_info.has_knee = true;
      shoulder_info.knee = auto_res.shoulder_knee;
      shoulder_info.has_guard = true;
      shoulder_info.guard = auto_res.shoulder_guard;
    }
    log_info("Auto: %s E %.4f g %.4f b %.4f (resid %.4f, %s), %d inferences\n", auto_res.target_name.c_str(),
           auto_res.params.e, auto_res.params.g, auto_res.params.b, auto_res.residual, to_string(auto_res.status),
           auto_res.inferences);
    auto ms = [](std::chrono::steady_clock::duration d) {
      return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
    };
    log_info("Timings (ms): render %lld, infer %lld, residual %lld, search %lld, total %lld\n",
           ms(auto_res.render_time), ms(auto_res.infer_time), ms(auto_res.residual_time), ms(auto_res.search_time),
           ms(auto_res.total_time));
    log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g --target %d\n\n", auto_res.params.e, auto_res.params.g,
           auto_res.params.b, bundle_target_idx);
  } else if (use_dino) {
    if (!is_bundle && !film_prof.data.has_profile) {
      log_error("The parameter search requires a film profile to evaluate intent losses.\n");
      return 1;
    }

    SolverConfig cfg;
    cfg.knee = knee;
    cfg.shoulder = shoulder_info.x_hi;
    cfg.shoulder_auto = is_shoulder_auto;

    const int max_iters = o.dino_iters;
    IntentGains user_gains{global_exposure_comp, gain_g, gain_b};
    if (is_bundle && (pin_mask & PIN_E)) {
      // A pinned -E is the station's: the search starts from it (and the station's g = b = 1) in absolute units.
      apply_station(global_exposure_comp, 1.0f, 1.0f);
      user_gains = {global_exposure_comp, gain_g, gain_b};
      adjust_correction_matrix(r_coeff, g_coeff, b_coeff, global_exposure_comp, gain_g, gain_b, unit_fb, unit_fb,
                               merged_matrix);
    }

    if (is_bundle) {
      // -E without --target: the bracket the bundle picks without the search.
      bundle_target_idx = manual_target_idx >= 0 ? manual_target_idx
                                                 : select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
      log_info("Selected target: %s\n", bundle.targets[bundle_target_idx].display_name.c_str());
      const auto& tgt = bundle.targets[bundle_target_idx];
      bundle_target_name = tgt.name;
      film_prof = tgt.film_prof;
      film_prof.rebind_pointers();
      set_target_bounds(cfg, tgt);
    } else {
      cfg.prof = &film_prof.data;
    }

    // TODO: Run `solve_dinov3_intent` on `oriented_half_view` (zero-copy half-size CFA view) instead of the
    // full-resolution `cfa_export_rgb` buffer once single-target solver baselines are updated.
    const bool solve_ok = use_cfa_pipeline
        ? solve_dinov3_intent(engine, cfg, merged_matrix,
                              negicc::RgbView::from_interleaved_rgb(
                                  cfa_export_rgb.data(), proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight),
                              user_gains, pin_mask, max_iters, dino_res)
        : solve_dinov3_intent(engine, cfg, merged_matrix, (const uint16_t (*)[4])proc->imgdata.image,
                              proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                              user_gains, pin_mask, max_iters, dino_res);
    if (!solve_ok) {
      log_error("DINOv3 intent solver failed.\n");
      return 1;
    }

    shoulder_info.x_hi = dino_res.shoulder.x_hi;
    if (is_shoulder_auto) {
      shoulder_info.has_q995 = true;
      shoulder_info.q995 = dino_res.shoulder.q995;
    }
    if (dino_res.shoulder.active) {
      shoulder_info.has_knee = true;
      shoulder_info.knee = dino_res.shoulder.knee;
      shoulder_info.has_guard = true;
      shoulder_info.guard = dino_res.shoulder.guard;
    }

    std::string pins_str = "none";
    if (pin_mask != 0) {
      pins_str = "";
      if (pin_mask & PIN_E) pins_str += "E ";
      if (pin_mask & PIN_G) pins_str += "g ";
      if (pin_mask & PIN_B) pins_str += "b";
      if (!pins_str.empty() && pins_str.back() == ' ') pins_str.pop_back();
    }
    log_info("\nDINOv3 solver: grid %dx%d (%dx%d), backend %s, E_c %.3f, bounds [%.3f, %.3f], max iters %d, pins %s\n",
           dino_res.grid_w, dino_res.grid_h, dino_res.target_w, dino_res.target_h,
           engine.backend_name(), dino_res.e_center, dino_res.bounds_lo, dino_res.bounds_hi,
           max_iters, pins_str.c_str());
    if (dino_res.anchor_clamped) {
      log_warn("ANCHOR_CLAMPED (median L* 50 not reached within E_c 2^[-4, 4])\n");
    }
    log_info(" it      E      g      b   resid    L_L    L_a    L_b  clip%%  P_clip  P_anch       J\n");
    for (const auto& it : dino_res.iterations) {
      if (it.residual < 0.0f) {
        log_info("  %c  %5.3f  %5.3f  %5.3f       -  %5.3f  %5.3f  %5.3f  %5.2f  %6.3f  %6.3f  %6.3f\n",
               it.tag, it.e, it.g, it.b, it.loss_L, it.loss_a, it.loss_b, it.clip_pct,
               it.penalty_clip, it.penalty_anchor, it.objective);
      } else {
        log_info("  %c  %5.3f  %5.3f  %5.3f  %6.3f  %5.3f  %5.3f  %5.3f  %5.2f  %6.3f  %6.3f  %6.3f\n",
               it.tag, it.e, it.g, it.b, it.residual, it.loss_L, it.loss_a, it.loss_b, it.clip_pct,
               it.penalty_clip, it.penalty_anchor, it.objective);
      }
    }
    log_info("Status: %s after %d inferences (best it %d, resid %.3f)\n",
           dino_res.status.c_str(), dino_res.total_inferences, dino_res.best_iteration, dino_res.best_residual);
    log_info("Solved (relative): E %.3f g %.3f b %.3f\n",
           dino_res.solved_relative.e, dino_res.solved_relative.g, dino_res.solved_relative.b);
    if (is_bundle) {
      // The station's values, which -E, --gain_g and --gain_b take with a bundle.
      // The station's values of the solved absolute ones.
      c.station_params = std::array<float, 3>{(float)(dino_res.solved_absolute.e / row_scale[0]),
                                              (float)(dino_res.solved_absolute.g * row_scale[0] / row_scale[1]),
                                              (float)(dino_res.solved_absolute.b * row_scale[0] / row_scale[2])};
      log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g --target %d\n", (*c.station_params)[0],
               (*c.station_params)[1], (*c.station_params)[2], bundle_target_idx);
    } else {
      log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g\n", dino_res.solved_absolute.e,
               dino_res.solved_absolute.g, dino_res.solved_absolute.b);
    }
    if (dino_res.solved_relative.g < 0.5f || dino_res.solved_relative.g > 2.0f ||
        dino_res.solved_relative.b < 0.5f || dino_res.solved_relative.b > 2.0f) {
      log_warn("GAIN_RANGE (relative g or b outside [0.5, 2.0])\n");
    }
    if (dino_res.best_iteration >= 0 && (size_t)dino_res.best_iteration + 1 < dino_res.iterations.size()) {
      const auto& bit = dino_res.iterations[dino_res.best_iteration + 1];
      float log2_ratio = (bit.e > 0.0f) ? std::log2(bit.e) : 0.0f;
      log_info("Penalties: clip %.3f (%.2f%% %s 2.50%%), anchor %.3f (log2(E/E_c) %+.3f)\n",
             bit.penalty_clip, bit.clip_pct, bit.clip_pct <= 2.5f ? "<=" : ">", bit.penalty_anchor, log2_ratio);
    }
    log_info("Timings (ms): views %.1f, render %.1f, infer %.1f, residual %.1f, search %.1f\n\n",
           dino_res.timing_views_ms, dino_res.timing_render_ms, dino_res.timing_infer_ms,
           dino_res.timing_residual_ms, dino_res.timing_search_ms);

    if (is_bundle) {
      // Rendered as the station renders those values, so the sidecar's cli_equivalent reproduces it.
      apply_station((*c.station_params)[0], (*c.station_params)[1], (*c.station_params)[2]);
    } else {
      global_exposure_comp = dino_res.solved_absolute.e;
      gain_g = dino_res.solved_absolute.g;
      gain_b = dino_res.solved_absolute.b;
      adjust_correction_matrix(r_coeff, g_coeff, b_coeff, global_exposure_comp, gain_g, gain_b,
                               effective_profile_film_base_rgb, effective_film_base_rgb, merged_matrix);
    }
  } else if (is_bundle) {
    if (manual_target_idx >= 0) {
      bundle_target_idx = manual_target_idx;
    } else {
      bundle_target_idx = select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
    }
    log_info("Selected target: %s\n", bundle.targets[bundle_target_idx].display_name.c_str());
    bundle_target_name = bundle.targets[bundle_target_idx].name;
    film_prof = bundle.targets[bundle_target_idx].film_prof;
    film_prof.rebind_pointers();
    // The given values are the station's (E defaults to 1, g and b to 1), converted as the search's result is.
    apply_station(global_exposure_comp, gain_g, gain_b);
  }

  if (is_bundle && bundle_target_idx >= 0) {
    embedded_icc = bundle.targets[bundle_target_idx].icc_bytes;
  }
  return 0;
}

int convert(const Options& o, const Setup& s, Scan& scan, Conversion& c, Backend& backend) {
  OutputSpec& out = c.out;
  if (!o.colorspace.empty()) {
    const std::string& colorspace_str = o.colorspace;
    if (colorspace_str == "srgb") {
      out.encoding = ENCODE_SRGB;
      out.attach_profile = "srgb";
      log_info("Using elle sRGB TRC as output profile\n");
    } else if (colorspace_str == "srgb-g10" || colorspace_str == "srgb-linear") {
      out.encoding = ENCODE_SRGB_LINEAR;
      out.attach_profile = "srgb-g10";
      log_info("Using elle sRGB Linear (Gamma = 1.0) as output profile\n");
    } else {
      std::string resolved_custom = resolve_profile_path(colorspace_str);
      out.custom = cmsOpenProfileFromFile(resolved_custom.c_str(), "r");
      if (!out.custom) {
        log_error("Cannot read output ICC profile: %s\n", colorspace_str.c_str());
        return 1;
      }
      out.xyz = cmsCreateXYZProfile();
      out.encoding = ENCODE_CUSTOM;
      out.attach_profile = resolved_custom;
      log_info("Reading output ICC profile: %s\n", resolved_custom.c_str());
    }
  } else if (s.has_profile) {
    out.encoding = ENCODE_NONE;
    if (!s.is_bundle) out.attach_profile = s.film_prof_arg;
  }

  const double t_start = omp_get_wtime();
  // The shared pipeline (color_pipeline.h) with the frozen shoulder: the stage every consumer runs. Only the sRGB
  // encoding applies it.
  const negicc::PipelineConfig export_pc = make_pipeline_config(c.film_prof.data, c.knee, c.shoulder.x_hi, out.encoding);
  backend.convert(out, export_pc, c.merged_matrix, c.knee, c.film_prof.data, scan.image());
  if (out.encoding == ENCODE_CUSTOM) {
    cmsCloseProfile(out.xyz);
    cmsCloseProfile(out.custom);
    out.xyz = out.custom = nullptr;
  }
  const double t_end = omp_get_wtime();
  scan.t_full_conversion_ms = (t_end - t_start) * 1000.0;
  log_info("Processing time: %f\n", t_end - t_start);
  return 0;
}

int write(const Options& o, const Setup& s, Scan& scan, Conversion& c) {
  LibRaw* proc = scan.proc.get();
  const bool is_bundle = s.is_bundle, use_dino = s.use_dino, is_roi_auto = o.roi_auto;
  const ProfileBundle& bundle = s.bundle;
  const DinoV3Engine& engine = s.engine;
  const std::string& film_prof_arg = s.film_prof_arg;
  const std::string& dino_path = s.dino_path;
  const std::string& dino_backend = o.dino_backend;
  const std::string& attach_profile = c.out.attach_profile;
  const OutputEncoding encoding = c.out.encoding;
  const std::vector<uint16_t>& cfa_export_rgb = scan.rgb3;
  const std::vector<uint8_t>& embedded_icc = c.embedded_icc;
  const int orig_w = scan.orig_w, orig_h = scan.orig_h;
  const bool has_roi = scan.has_roi;
  const int* roi_coords = scan.roi;
  const FrameDetection& frame_det = scan.frame_det;
  const int rot_cw = scan.rot_cw;
  const bool hflip = scan.hflip, vflip = scan.vflip;
  const int width = proc->imgdata.sizes.iwidth, height = proc->imgdata.sizes.iheight;
  const bool film_base_applied = false;  // no film base of the capture (see the TODO at --film_stock)
  const std::vector<int>& sidecar_film_base_rgb = c.sidecar_film_base_rgb;
  const std::vector<int>& sidecar_profile_film_base_rgb = c.sidecar_profile_film_base_rgb;
  const std::string& film_base_source = c.film_base_source;
  const float* merged_matrix = c.merged_matrix;
  const float global_exposure_comp = c.exposure_comp, gain_g = c.gain_g, gain_b = c.gain_b;
  const unsigned pin_mask = c.pin_mask;
  const float knee = c.knee;
  const ShoulderInfo& shoulder_info = c.shoulder;
  const bool has_auto_res = scan.has_auto_res;
  const DinoSolveResult& dino_res = c.dino_res;
  const double t_raw_decode_ms = scan.t_raw_decode_ms, t_frame_det_ms = scan.t_frame_det_ms;
  const double t_full_conversion_ms = scan.t_full_conversion_ms, t_auto_ms = scan.t_auto_ms;
  double& t_tiff_write_ms = scan.t_tiff_write_ms;
  const std::string& bundle_target_name = c.bundle_target_name;
  const int bundle_target_idx = c.bundle_target_idx;
  const ParamSearchResult& auto_res = scan.auto_res;
  const int* auto_frame_half = scan.auto_frame_half;

  const std::string& output = o.output;
  log_info("Writing TIFF '%s'\n", output.c_str());
  const double t_tiff_0 = omp_get_wtime();
  int ret = write_tiff(proc, attach_profile, output,
                       is_bundle ? embedded_icc : std::vector<uint8_t>{},
                       cfa_export_rgb.empty() ? nullptr : cfa_export_rgb.data());
  t_tiff_write_ms = (omp_get_wtime() - t_tiff_0) * 1000.0;

  if (ret != 0 || o.no_metadata) {
    unlink((output + ".json").c_str());
  } else {
    std::string cs_name = "camera-linear";
    if (encoding == ENCODE_SRGB) {
      cs_name = "srgb";
    } else if (encoding == ENCODE_SRGB_LINEAR) {
      cs_name = "srgb-linear";
    } else if (encoding == ENCODE_CUSTOM) {
      cs_name = attach_profile;
    } else if (encoding == ENCODE_NONE) {
      cs_name = attach_profile.empty() ? "camera-linear" : attach_profile;
    }

    std::string roi_source = "none";
    if (is_roi_auto) {
      roi_source = (frame_det.conf < kLowFrameConfidence) ? "auto_fallback" : "auto";
    } else if (has_roi) {
      roi_source = "cli";
    }

    Sidecar sc;
    sc.output_path = output;
    sc.argc = o.argc;
    sc.argv = o.argv;
    sc.proc = proc;
    sc.files = o.files;
    sc.orig_w = orig_w;
    sc.orig_h = orig_h;
    sc.has_roi = has_roi;
    sc.roi = roi_coords;
    sc.roi_source = roi_source;
    sc.frame_det = is_roi_auto ? &frame_det : nullptr;
    sc.rot_cw = rot_cw;
    sc.hflip = hflip;
    sc.vflip = vflip;
    sc.out_w = width;
    sc.out_h = height;
    sc.film_base_applied = film_base_applied;
    sc.film_base_rgb = sidecar_film_base_rgb;
    sc.profile_film_base_rgb = sidecar_profile_film_base_rgb;
    sc.film_base_source = film_base_source;
    sc.std_density = frame_det.base_std_d;
    sc.has_std_density = frame_det.has_base;
    sc.profile_path = is_bundle ? bundle.path : film_prof_arg;
    sc.is_bundle = is_bundle;
    sc.bundle_target_name = bundle_target_name;
    sc.bundle_target_idx = bundle_target_idx;
    sc.input_space = bundle.input_space;
    sc.merged_matrix = merged_matrix;
    sc.exposure_comp = global_exposure_comp;
    sc.gain_g = gain_g;
    sc.gain_b = gain_b;
    sc.station_params = c.station_params ? c.station_params->data() : nullptr;
    sc.pin_mask = pin_mask;
    sc.knee = knee;
    sc.shoulder = shoulder_info;
    sc.dino = use_dino && !has_auto_res ? &dino_res : nullptr;
    sc.dino_model = dino_path;
    sc.dino_backend = use_dino ? std::string(engine.backend_used()) : dino_backend;
    sc.auto_res = has_auto_res ? &auto_res : nullptr;
    sc.auto_frame_half = auto_frame_half;
    sc.colorspace = cs_name;
    sc.t_raw_decode_ms = t_raw_decode_ms;
    sc.t_frame_det_ms = t_frame_det_ms;
    sc.t_full_conversion_ms = t_full_conversion_ms;
    sc.t_tiff_write_ms = t_tiff_write_ms;
    sc.t_auto_ms = t_auto_ms;
    write_json_sidecar(sc);
  }

  return ret;
}
