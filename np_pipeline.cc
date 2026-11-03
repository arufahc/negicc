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
                         const RgbInput& img) {
  const int width = img.width, height = img.height, ch = img.channels;
  uint16_t* pixels = img.pixels;
  // The sRGB encodings and ENCODE_NONE (the corrected RGB, already in [0, 1]) write 16-bit codes in place.
  auto convert_in_place = [&](auto stage) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        uint16_t* px = &pixels[((size_t)j * width + i) * ch];
        const Rgb v = stage(px);
        px[0] = negicc::quantize_u16(v.r);
        px[1] = negicc::quantize_u16(v.g);
        px[2] = negicc::quantize_u16(v.b);
      }
    }
  };
  if (out.encoding == ENCODE_SRGB || out.encoding == ENCODE_SRGB_LINEAR) {
    convert_in_place([&](const uint16_t* px) { return negicc::convert_pixel(pc, merged_matrix, px); });
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
          const Rgb v = negicc::convert_pixel_pcs(pc, merged_matrix, px);
          xyz_row[i * 3 + 0] = v.r;
          xyz_row[i * 3 + 1] = v.g;
          xyz_row[i * 3 + 2] = v.b;
        }
        cmsDoTransform(thread_xform, xyz_row.data(), &pixels[(size_t)j * width * ch], width);
      }
      cmsDeleteTransform(thread_xform);
    }
  } else {
    convert_in_place([&](const uint16_t* px) { return negicc::convert_pixel_corrected(pc, merged_matrix, px); });
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

  // The crosstalk correction: the bundle's, with any rows -r, -g and -b give in place of its own (for the search too).
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

// The station's E of given values: -E, else the exposure of bracket target_idx's e_center (absolute, as when no
// exposure is given) on the station's scale.
static float given_station_e(const Options& o, const ProfileBundle& bundle, int target_idx, double scan_shutter_s,
                             double scan_iso, const double* film_base_rgb) {
  if (o.exposure_given || bundle.targets[target_idx].e_center <= 0.0f) return o.exposure_comp;
  float base[9];
  double row_scale[3];
  compute_station_merged_matrix(bundle, scan_shutter_s, scan_iso, film_base_rgb, film_base_rgb != nullptr, base, nullptr,
                                row_scale);
  return (float)((double)bundle.targets[target_idx].e_center / row_scale[0]);
}

int decode(const Options& o, Setup& s, Scan& scan) {
  ProfileBundle& bundle = s.bundle;
  std::copy(o.roi, o.roi + 4, scan.roi);
  scan.has_roi = o.has_roi;
  scan.rot_cw = o.rot_cw;
  scan.hflip = o.hflip;
  scan.vflip = o.vflip;

  scan.exposure_file = o.files[0];
  scan.has_exif_exp = read_exif_exposure_d(o.files[0], scan.exif_shutter_d, scan.exif_iso_d);

  // The bracket of given values (the search's is its own), for the roll-off measured on the half-size frame.
  const int given_target_idx = !s.is_bundle ? -1
                               : s.manual_target_idx >= 0 ? s.manual_target_idx
                                                          : select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
  auto measure_rolloff = [&](const negicc::RgbView& frame, double scan_shutter_s, double scan_iso) {
    const int target_idx = scan.has_auto_res ? scan.auto_res.target_idx : given_target_idx;
    const float e = scan.has_auto_res ? scan.auto_res.params.e
                                 : given_station_e(o, bundle, target_idx, scan_shutter_s, scan_iso,
                                                   scan.film_base_rgb());
    const float g = scan.has_auto_res ? scan.auto_res.params.g : o.gain_g;
    const float b = scan.has_auto_res ? scan.auto_res.params.b : o.gain_b;
    scan.rolloff_x_hi = measure_rolloff_x_hi(bundle, target_idx, frame, scan_shutter_s, scan_iso, e, g, b,
                                             ParamSearchOptions{}.crop_ratio, &scan.rolloff_q995, scan.film_base_rgb());
    scan.has_rolloff = true;
  };

  // The film base read from the scan for a bundle conversion (film_base_detect.h): from the whole half-size image
  // (flat field applied) and the frame detector's base patch on the mosaic, as negicc-station's Auto reads it. On the
  // zero-copy CFA path (Sony RAWs, pixel shift) and on a RAW through LibRaw (white light, -q, other cameras) the frame
  // detector runs on the mosaic for every bundle conversion, with or without --roi auto, so the rebate is found when
  // the scan shows one, as negicc-station reads it. RGB files take the base from the statistics (with a white-light
  // bundle also from --roi auto's detection).
  // load_raw runs the half-size analysis before it crops to the frame, so with --roi auto too the statistics see
  // the whole scan.
  // TODO: a film base from an image (a capture of unexposed film, as negicc-station's Film Base tab takes) in its place.
  auto detect_base = [&](const negicc::RgbView& half, const FrameDetection* det, int sensor_w, int sensor_h) {
    if (!s.is_bundle) return;
    const double* xt = (s.bundle.has_crosstalk && !s.white_light) ? s.bundle.crosstalk_matrix_d : nullptr;
    if (negicc::detect_film_base(half, det, sensor_w, sensor_h, xt, scan.film_base)) {
      log_info("Film base (%s): R=%.1f G=%.1f B=%.1f\n", scan.film_base.source.c_str(), scan.film_base.rgb[0],
               scan.film_base.rgb[1], scan.film_base.rgb[2]);
    } else {
      log_info("Film base: none read from the scan; the bundle's\n");
    }
  };

  // A RAW through LibRaw: the frame detector with its default parameters on the unpacked mosaic (load_raw's
  // base_det), for the film base's rebate, as on the CFA path below and in negicc-station.
  std::optional<RawFrameDetection> raw_base_det;
  HalfSizeAnalysis analyse;
  if (s.bundle_search || (s.is_bundle && !s.use_dino)) {
    analyse = [&](const LibRaw& raw) -> bool {
      const double t_auto_0 = omp_get_wtime();
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      scan.scan_exposure(raw, scan_shutter_s, scan_iso);
      // --roi halved, or the whole image.
      const int half_w = raw.imgdata.sizes.iwidth, half_h = raw.imgdata.sizes.iheight;
      int* bbox = scan.auto_frame_half;
      bbox[0] = 0;
      bbox[1] = 0;
      bbox[2] = half_w;
      bbox[3] = half_h;
      if (scan.has_roi) {
        bbox[0] = scan.roi[0] / 2;
        bbox[1] = scan.roi[1] / 2;
        bbox[2] = std::max(1, (scan.roi[0] + scan.roi[2]) / 2 - bbox[0]);
        bbox[3] = std::max(1, (scan.roi[1] + scan.roi[3]) / 2 - bbox[1]);
        clamp_rect(bbox, half_w, half_h);
      }
      // Step 3 & Step 4: zero-copy RgbView over the half-size image, cropped and oriented in place.
      negicc::RgbView half_view = negicc::RgbView::from_interleaved_rgb(
          &raw.imgdata.image[0][0], half_w, half_h, 4, (ptrdiff_t)half_w * 4);
      // The film base from the whole half-size image, with the rebate of the frame detection on a RAW's mosaic
      // (raw_base_det); for an RGB file with a white-light bundle, --roi auto's detection.
      if (raw_base_det) {
        detect_base(half_view, &raw_base_det->det, raw_base_det->sensor_w, raw_base_det->sensor_h);
      } else {
        detect_base(half_view, (s.white_light && o.roi_auto) ? &scan.frame_det : nullptr,
                    s.white_light ? raw.imgdata.sizes.width : 0, s.white_light ? raw.imgdata.sizes.height : 0);
      }
      negicc::RgbView frame =
          half_view.crop_and_orient(bbox[0], bbox[1], bbox[2], bbox[3], scan.rot_cw, scan.hflip, scan.vflip);
      if (s.bundle_search) {
        ParamSearchOptions opts;
        if (o.dino_iters_given) opts.max_steps = o.dino_iters;
        opts.film_base_rgb_d = scan.film_base_rgb();
        opts.film_base_from_scan = opts.film_base_rgb_d != nullptr;
        if (!search_conversion_params(bundle, s.engine, frame, scan_shutter_s, scan_iso, opts, scan.auto_res)) {
          log_error("The conversion parameter search failed\n");
          return false;
        }
        scan.has_auto_res = true;
      }
      measure_rolloff(frame, scan_shutter_s, scan_iso);
      scan.t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
      return true;
    };
  }

  // --roi auto's frame: the detected crop, or the whole full_w x full_h image when the detection is not confident.
  auto take_detected_frame = [&](int full_w, int full_h) {
    if (scan.frame_det.conf < kLowFrameConfidence) {
      scan.roi[0] = 0;
      scan.roi[1] = 0;
      scan.roi[2] = full_w;
      scan.roi[3] = full_h;
    } else {
      for (int i = 0; i < 4; ++i) scan.roi[i] = scan.frame_det.crop[i];
    }
    scan.has_roi = true;
  };

  FrameDetectionParams det_params;
  det_params.format = o.film_format;
  det_params.axis_hint = o.roi_axis_hint;
  RawRegionFinder find_frame;
  if (o.roi_auto && o.files.size() != 4) {
    find_frame = [&](LibRaw& raw, int roi[4]) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(raw, mosaic)) {
        log_error("--roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        return false;
      }
      det_params.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, det_params, scan.frame_det);
      scan.t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
      take_detected_frame(mosaic.width, mosaic.height);
      for (int i = 0; i < 4; ++i) roi[i] = scan.roi[i];
      return true;
    };
  }

  // Logs --roi auto's detection, and warns when it fell back or the frame is vertical and not turned.
  auto print_frame_detection = [&]() {
    const FrameDetection& det = scan.frame_det;
    const char* axis_name = det.axis == 'h' ? "h" : (det.axis == 'v' ? "v" : "undecided");
    log_info("Film frame: axis=%s format=%s orientation=%s rotation=%.2f deg gaps=%d frames=%d conf=%.2f%s%s\n",
           axis_name, film_format_name(det.format), det.orientation.c_str(), det.rotation_deg,
           det.n_gaps, 1 + (int)det.secondaries.size(), det.conf,
           det.reason.empty() ? "" : " reason: ", det.reason.c_str());
    if (det.n_gaps > 0) {
      log_info("  Gaps (sensor px):");
      for (int g = 0; g < 2; ++g)
        if (det.gaps[g][0] >= 0) log_info(" [%d, %d]", det.gaps[g][0], det.gaps[g][1]);
      log_info("\n");
    }
    log_info("  Frame (sensor px): x=%d y=%d w=%d h=%d (frame centre %.1f, %.1f, size %.1f x %.1f, %.1f px/mm)\n",
           scan.roi[0], scan.roi[1], scan.roi[2], scan.roi[3], det.primary.rect.cx,
           det.primary.rect.cy, det.primary.rect.w, det.primary.rect.h, det.px_per_mm);
    for (const FilmFrame& f : det.secondaries)
      log_info("  Frame %+d (sensor px): visible x=%d y=%d w=%d h=%d (%.0f%% of a frame)\n", f.index, f.visible[0],
             f.visible[1], f.visible[2], f.visible[3], 100.0f * f.visible_fraction);
    if (det.conf < kLowFrameConfidence) {
      log_warn("low detection confidence (conf=%.2f%s%s); falling back to the full sensor frame\n",
               det.conf, det.reason.empty() ? "" : ", reason: ", det.reason.c_str());
    }
    if (s.use_dino && det.axis == 'v' && scan.rot_cw == 0) {
      log_warn("axis=v, rot=0 (vertical frame without rotation may degrade DINOv3 intent accuracy)\n");
    }
  };


  // Determine whether the 5-step zero-copy CFA pipeline applies (Step 1..5).
  bool use_cfa_pipeline = false;
  scan.t_raw_0 = omp_get_wtime();
  LibRaw *proc = nullptr;
  LibRaw *unpacked_probe = nullptr;
  float cfa_white_level = 65535.0f;
  int cfa_rx = 0, cfa_ry = 0, cfa_rw = 0, cfa_rh = 0;
  negicc::RgbView full_half_view;
  negicc::RgbView oriented_half_view;
  negicc::RgbView search_half_view;
  negicc::FlatSampleArgs flat_full{};

  if ((o.input_mode == InputMode::kRaw || o.input_mode == InputMode::kLinearDng) &&
      (o.files.size() == 4 || (s.raw_decode == RawDecode::kCameraRgb && o.quality < 0))) {
    LibRaw* probe = new LibRaw();
    log_info("Loading RAW file %s\n", o.files[0].c_str());
    int ret = open_and_unpack_raw_file(probe, o.files[0]);
    if (ret != LIBRAW_SUCCESS) {
      log_error("Cannot open/unpack %s: %s\n", o.files[0].c_str(), libraw_strerror(ret));
      delete probe;
      return 1;
    }
    if (o.files.size() == 4) {
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
    if (o.roi_auto) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(*proc, mosaic)) {
        log_error("--roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        delete proc;
        return 1;
      }
      det_params.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, det_params, scan.frame_det);
      scan.t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
    }

    // Step 1: Per-channel black subtraction in-place on the unpacked CFA mosaic.
    if (!subtract_cfa_black_in_place(proc, cfa_white_level)) {
      log_error("Invalid CFA mosaic geometry in %s\n", o.files[0].c_str());
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
    if (o.files.size() == 4) {
      for (int mi = 1; mi < 4; ++mi) {
        scan.ps_extra[mi - 1].reset(new LibRaw());
        LibRaw* fp = scan.ps_extra[mi - 1].get();
        log_info("Loading RAW file %s\n", o.files[mi].c_str());
        int fret = fp->open_file(o.files[mi].c_str());
        if (fret != LIBRAW_SUCCESS || (fret = fp->unpack()) != LIBRAW_SUCCESS) {
          log_error("Cannot open/unpack %s: %s\n", o.files[mi].c_str(), libraw_strerror(fret));
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
          log_error("Invalid CFA mosaic geometry in %s\n", o.files[mi].c_str());
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
    if (s.flat_field) {
      try {
        flat_half_tables = s.flat_field->tables(half_w, half_h, true);
        flat_half = flat_half_tables->host();
        scan.flat_full_tables = s.flat_field->tables(src_w, src_h, false);
        flat_full = scan.flat_full_tables->host();
      } catch (const std::exception& e) {
        log_error("%s\n", e.what());
        delete proc;
        return 1;
      }
    }
    full_half_view = negicc::RgbView::from_cfa_half(
        cfa_origin, cfa_pitch, src_w, src_h, filters, nullptr, flat_half, flat_half_tables);
    if (s.is_bundle) {
      // The frame detector on the black-subtracted mosaic of the first frame, with its default parameters, as
      // negicc-station runs it for the film base (CapturedImage::film_frame_detection).
      FrameDetection base_det;
      FilmImage mosaic;
      const float no_black[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const bool has_det =
          filters > 1000 &&
          film_image_from_cfa(cfa_origin, src_w, src_h, (ptrdiff_t)cfa_pitch, 2,
                              [filters](int row, int col) { return negicc::bayer_color_at(row, col, filters); },
                              no_black, cfa_white_level, mosaic);
      if (has_det) detect_film_frame(mosaic, FrameDetectionParams(), base_det);
      detect_base(full_half_view, has_det ? &base_det : nullptr, src_w, src_h);
    }

    // Step 3: Crop, rotate, flip & inline ROI search on the zero-copy view.
    if (o.roi_auto) {
      take_detected_frame(src_w, src_h);
      print_frame_detection();
    }

    cfa_rx = 0;
    cfa_ry = 0;
    cfa_rw = src_w;
    cfa_rh = src_h;
    if (scan.has_roi) {
      if (scan.roi[0] < 0 || scan.roi[1] < 0 || scan.roi[0] >= src_w || scan.roi[1] >= src_h ||
          scan.roi[2] <= 0 || scan.roi[3] <= 0) {
        log_error("ROI [%d, %d, %d, %d] is outside sensor boundaries (%dx%d)\n",
                scan.roi[0], scan.roi[1], scan.roi[2], scan.roi[3], src_w, src_h);
        delete proc;
        return 1;
      }
      cfa_rx = scan.roi[0];
      cfa_ry = scan.roi[1];
      cfa_rw = std::min(scan.roi[2], src_w - cfa_rx);
      cfa_rh = std::min(scan.roi[3], src_h - cfa_ry);
    } else if (!o.no_crop && o.files.size() == 1) {
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
    for (int i = 0; i < 4; ++i) scan.auto_frame_half[i] = out_half[i];
    search_half_view = full_half_view.crop_and_orient(
        scan.auto_frame_half[0], scan.auto_frame_half[1], scan.auto_frame_half[2], scan.auto_frame_half[3],
        scan.rot_cw, scan.hflip, scan.vflip);
    oriented_half_view = search_half_view;

    // Step 4: Conversion parameter search inline on the zero-copy oriented RgbView.
    if (s.bundle_search) {
      const double t_auto_0 = omp_get_wtime();
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
      ParamSearchOptions opts;
      if (o.dino_iters_given) opts.max_steps = o.dino_iters;
      opts.film_base_rgb_d = scan.film_base_rgb();
      opts.film_base_from_scan = opts.film_base_rgb_d != nullptr;
      if (!search_conversion_params(bundle, s.engine, search_half_view, scan_shutter_s, scan_iso, opts,
                                    scan.auto_res)) {
        log_error("The conversion parameter search failed\n");
        delete proc;
        return 1;
      }
      scan.has_auto_res = true;
      scan.t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
    }
    if (s.bundle_search || (s.is_bundle && !s.use_dino)) {
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
      measure_rolloff(search_half_view, scan_shutter_s, scan_iso);
    }
    scan.t_raw_decode_ms = (omp_get_wtime() - scan.t_raw_0) * 1000.0 - scan.t_auto_ms - scan.t_frame_det_ms;

    // The mosaic and the frame, for render.
    scan.mosaic.frames[0] = cfa_origin;
    scan.mosaic.n_frames = o.files.size() == 4 ? 4 : 1;
    for (int i = 0; i + 1 < scan.mosaic.n_frames; ++i) {
      LibRaw* fp = scan.ps_extra[i].get();
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
    scan.cfa_region = {cfa_rx, cfa_ry, cfa_rw, cfa_rh, scan.rot_cw, scan.hflip, scan.vflip};
  } else {
    if (o.input_mode == InputMode::kRgbFile || o.input_mode == InputMode::kThreeShot) {
      proc = load_rgb_input(o.files, o.input_mode, s.camera_rgb_input, o.quality,
                            !o.no_crop && !scan.has_roi && !o.roi_auto,
                            (!o.roi_auto && scan.has_roi) ? scan.roi : nullptr, analyse,
                            o.roi_auto ? &scan.frame_det : nullptr, det_params, scan.input_source);
    } else {
      proc = load_raw(o.files[0], true,
                      o.quality,
                      !o.no_crop && !scan.has_roi && !o.roi_auto,
                      s.raw_decode,
                      (!o.roi_auto && scan.has_roi) ? scan.roi : nullptr,
                      analyse, find_frame, s.flat_field.get(),
                      unpacked_probe, s.is_bundle ? &raw_base_det : nullptr);
    }
    unpacked_probe = nullptr;
    if (!proc) {
      log_error("Cannot open %s\n", o.files[0].c_str());
      return 1;
    }
    if (o.roi_auto) {
      if (o.input_mode == InputMode::kRgbFile || o.input_mode == InputMode::kThreeShot) {
        const auto& rs = proc->imgdata.rawdata.sizes;
        const auto& is = proc->imgdata.sizes;
        take_detected_frame(rs.width ? rs.width : is.width, rs.height ? rs.height : is.height);
      }
      print_frame_detection();
    }
    if (scan.rot_cw != 0 || scan.hflip || scan.vflip) {
      if (!extract_roi_and_transform(proc, 0, 0, proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                                     scan.rot_cw, scan.hflip, scan.vflip)) {
        proc->free_image();
        delete proc;
        return 1;
      }
      log_info("Transformed subrect: %dx%d (rot=%d, hflip=%d, vflip=%d)\n",
             proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight, scan.rot_cw, scan.hflip ? 1 : 0,
             scan.vflip ? 1 : 0);
    }
    scan.t_raw_decode_ms = (omp_get_wtime() - scan.t_raw_0) * 1000.0 - scan.t_auto_ms - scan.t_frame_det_ms;
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
  const std::vector<int> unit_fb = {1, 1, 1};
  c.exposure_comp = o.exposure_comp;
  c.gain_g = o.gain_g;
  c.gain_b = o.gain_b;
  c.knee = kKnee;

  // The sidecar records the bundle's film base. The film base scale of the input matrix is 1: a bundle conversion's
  // film base (the scan's, else the bundle's) is in the station's merged matrix (solve).
  c.sidecar_profile_film_base_rgb = {1, 1, 1};
  if (s.is_bundle && s.bundle.has_film_base) {
    c.sidecar_profile_film_base_rgb = {s.bundle.profile_film_base_rgb[0], s.bundle.profile_film_base_rgb[1],
                                       s.bundle.profile_film_base_rgb[2]};
  }
  c.profile_film_base_rgb = {1, 1, 1};

  if (s.is_bundle) {
    // The bundle's film base (when none is read from the scan) is in the station's merged matrix, so in E, g and b,
    // not applied as a film base scale.
    c.sidecar_film_base_rgb = c.sidecar_profile_film_base_rgb;
    c.film_base_source = "bundle";
  }

  if (o.exposure_given) c.pin_mask |= PIN_E;
  if (o.gain_g_given) c.pin_mask |= PIN_G;
  if (o.gain_b_given) c.pin_mask |= PIN_B;

  adjust_correction_matrix(s.r_coeff, s.g_coeff, s.b_coeff,
                           c.exposure_comp,
                           c.gain_g, c.gain_b,
                           c.profile_film_base_rgb,
                           unit_fb,
                           c.merged_matrix);
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
  ProfileBundle& bundle = s.bundle;
  LibRaw* proc = scan.proc.get();
  // With a bundle, E, g and b are the station's (the search's, and what -E, --gain_g and --gain_b take): the
  // conversion is the station's, the bundle's merged matrix (crosstalk, film base and the scan's exposure ratio)
  // scaled per row by E, E*g and E*b, so neg_process renders what negicc-station renders. row_scale turns them into
  // the absolute exposure and gains on the crosstalk (the sidecar's matrix.*) and back.
  float station_base[9];
  double row_scale[3] = {1.0, 1.0, 1.0};
  if (s.is_bundle) {
    double scan_shutter_s = 0.125, scan_iso = 100.0;
    scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
    compute_station_merged_matrix(bundle, scan_shutter_s, scan_iso, scan.film_base_rgb(), scan.film_base_rgb() != nullptr,
                                  station_base, nullptr, row_scale);
  }
  auto apply_station = [&](float e, float g, float b) {
    c.station_params = std::array<float, 3>{e, g, b};
    negicc::candidate_matrix(station_base, e, g, b, c.merged_matrix);
    c.exposure_comp = (float)(row_scale[0] * (double)e);
    c.gain_g = (float)(row_scale[1] / row_scale[0] * (double)g);
    c.gain_b = (float)(row_scale[2] / row_scale[0] * (double)b);
  };
  const std::vector<int> unit_fb = {1, 1, 1};

  if (s.is_bundle && !s.bundle_search) {
    if (s.use_dino || s.manual_target_idx < 0 || !(c.pin_mask & PIN_E)) {
      // Sets each bracket's exposure range on the bundle (e_min, e_max, e_center, is_feasible), which the target
      // selection, set_target_bounds and given_station_e read below. The scene statistics it returns are unused.
      float g_lo = 0.0f, g_hi = 0.0f, scene_ev = 0.0f;
      int num_feasible = 0;
      if (scan.cfa) {
        // TODO: Run `compute_bundle_enclosures` on `oriented_half_view` (zero-copy half-size CFA view) instead of the
        // full-resolution `cfa_export_rgb` buffer once single-target enclosure baselines are updated.
        compute_bundle_enclosures(bundle,
                                  negicc::RgbView::from_interleaved_rgb(
                                      scan.rgb3.data(), proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight),
                                  g_lo, g_hi, scene_ev, num_feasible);
      } else {
        compute_bundle_enclosures(bundle,
                                  (const uint16_t (*)[4])proc->imgdata.image,
                                  proc->imgdata.sizes.iwidth,
                                  proc->imgdata.sizes.iheight,
                                  g_lo, g_hi, scene_ev, num_feasible);
      }
    }
  } else if (!s.is_bundle) {
    if (!s.film_prof_arg.empty() && !load_film_profile(s.film_prof_arg, c.film_prof)) {
      log_error("Cannot load film profile %s\n", s.film_prof_arg.c_str());
    }
  }

  if (scan.has_auto_res) {
    const ParamSearchResult& res = scan.auto_res;
    c.bundle_target_idx = res.target_idx;
    c.bundle_target_name = res.target_name;
    c.film_prof = bundle.targets[c.bundle_target_idx].film_prof;
    c.film_prof.rebind_pointers();
    // The search's E, g and b, converted as the station converts them (the sidecar's cli_equivalent reproduces it).
    apply_station(res.params.e, res.params.g, res.params.b);
    log_info("Auto: %s E %.4f g %.4f b %.4f (resid %.4f, %s), %d inferences\n", res.target_name.c_str(),
           res.params.e, res.params.g, res.params.b, res.residual, to_string(res.status),
           res.inferences);
    auto ms = [](std::chrono::steady_clock::duration d) {
      return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
    };
    log_info("Timings (ms): render %lld, infer %lld, residual %lld, search %lld, total %lld\n",
           ms(res.render_time), ms(res.infer_time), ms(res.residual_time), ms(res.search_time),
           ms(res.total_time));
    log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g --target %d\n\n", res.params.e, res.params.g,
           res.params.b, c.bundle_target_idx);
  } else if (s.use_dino) {
    if (!s.is_bundle && !c.film_prof.data.has_profile) {
      log_error("The parameter search requires a film profile to evaluate intent losses.\n");
      return 1;
    }

    SolverConfig cfg;
    cfg.knee = c.knee;
    cfg.shoulder_auto = s.is_bundle;  // the roll-off, as the bracket search renders it

    const int max_iters = o.dino_iters;
    IntentGains user_gains{c.exposure_comp, c.gain_g, c.gain_b};
    if (s.is_bundle && (c.pin_mask & PIN_E)) {
      // A pinned -E is the station's: the search starts from it (and the station's g = b = 1) in absolute units.
      apply_station(c.exposure_comp, 1.0f, 1.0f);
      user_gains = {c.exposure_comp, c.gain_g, c.gain_b};
      adjust_correction_matrix(s.r_coeff, s.g_coeff, s.b_coeff, c.exposure_comp, c.gain_g, c.gain_b, unit_fb, unit_fb,
                               c.merged_matrix);
    }

    if (s.is_bundle) {
      // -E without --target: the bracket the bundle picks without the search.
      c.bundle_target_idx = s.manual_target_idx >= 0 ? s.manual_target_idx
                                                 : select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
      log_info("Selected target: %s\n", bundle.targets[c.bundle_target_idx].display_name.c_str());
      const auto& tgt = bundle.targets[c.bundle_target_idx];
      c.bundle_target_name = tgt.name;
      c.film_prof = tgt.film_prof;
      c.film_prof.rebind_pointers();
      set_target_bounds(cfg, tgt);
    } else {
      cfg.prof = &c.film_prof.data;
    }

    // TODO: Run `solve_dinov3_intent` on `oriented_half_view` (zero-copy half-size CFA view) instead of the
    // full-resolution `cfa_export_rgb` buffer once single-target solver baselines are updated.
    const bool solve_ok = scan.cfa
        ? solve_dinov3_intent(s.engine, cfg, c.merged_matrix,
                              negicc::RgbView::from_interleaved_rgb(
                                  scan.rgb3.data(), proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight),
                              user_gains, c.pin_mask, max_iters, c.dino_res)
        : solve_dinov3_intent(s.engine, cfg, c.merged_matrix, (const uint16_t (*)[4])proc->imgdata.image,
                              proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                              user_gains, c.pin_mask, max_iters, c.dino_res);
    if (!solve_ok) {
      log_error("DINOv3 intent solver failed.\n");
      return 1;
    }


    std::string pins_str = "none";
    if (c.pin_mask != 0) {
      pins_str = "";
      if (c.pin_mask & PIN_E) pins_str += "E ";
      if (c.pin_mask & PIN_G) pins_str += "g ";
      if (c.pin_mask & PIN_B) pins_str += "b";
      if (!pins_str.empty() && pins_str.back() == ' ') pins_str.pop_back();
    }
    log_info("\nDINOv3 solver: grid %dx%d (%dx%d), backend %s, E_c %.3f, bounds [%.3f, %.3f], max iters %d, pins %s\n",
           c.dino_res.grid_w, c.dino_res.grid_h, c.dino_res.target_w, c.dino_res.target_h,
           s.engine.backend_name(), c.dino_res.e_center, c.dino_res.bounds_lo, c.dino_res.bounds_hi,
           max_iters, pins_str.c_str());
    if (c.dino_res.anchor_clamped) {
      log_warn("ANCHOR_CLAMPED (median L* 50 not reached within E_c 2^[-4, 4])\n");
    }
    log_info(" it      E      g      b   resid    L_L    L_a    L_b  clip%%  P_clip  P_anch       J\n");
    for (const auto& it : c.dino_res.iterations) {
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
           c.dino_res.status.c_str(), c.dino_res.total_inferences, c.dino_res.best_iteration, c.dino_res.best_residual);
    log_info("Solved (relative): E %.3f g %.3f b %.3f\n",
           c.dino_res.solved_relative.e, c.dino_res.solved_relative.g, c.dino_res.solved_relative.b);
    if (s.is_bundle) {
      // The station's values, which -E, --gain_g and --gain_b take with a bundle.
      // The station's values of the solved absolute ones.
      c.station_params = std::array<float, 3>{(float)(c.dino_res.solved_absolute.e / row_scale[0]),
                                              (float)(c.dino_res.solved_absolute.g * row_scale[0] / row_scale[1]),
                                              (float)(c.dino_res.solved_absolute.b * row_scale[0] / row_scale[2])};
      log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g --target %d\n", (*c.station_params)[0],
               (*c.station_params)[1], (*c.station_params)[2], c.bundle_target_idx);
    } else {
      log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g\n", c.dino_res.solved_absolute.e,
               c.dino_res.solved_absolute.g, c.dino_res.solved_absolute.b);
    }
    if (c.dino_res.solved_relative.g < 0.5f || c.dino_res.solved_relative.g > 2.0f ||
        c.dino_res.solved_relative.b < 0.5f || c.dino_res.solved_relative.b > 2.0f) {
      log_warn("GAIN_RANGE (relative g or b outside [0.5, 2.0])\n");
    }
    if (c.dino_res.best_iteration >= 0 && (size_t)c.dino_res.best_iteration + 1 < c.dino_res.iterations.size()) {
      const auto& bit = c.dino_res.iterations[c.dino_res.best_iteration + 1];
      float log2_ratio = (bit.e > 0.0f) ? std::log2(bit.e) : 0.0f;
      log_info("Penalties: clip %.3f (%.2f%% %s 2.50%%), anchor %.3f (log2(E/E_c) %+.3f)\n",
             bit.penalty_clip, bit.clip_pct, bit.clip_pct <= 2.5f ? "<=" : ">", bit.penalty_anchor, log2_ratio);
    }
    log_info("Timings (ms): views %.1f, render %.1f, infer %.1f, residual %.1f, search %.1f\n\n",
           c.dino_res.timing_views_ms, c.dino_res.timing_render_ms, c.dino_res.timing_infer_ms,
           c.dino_res.timing_residual_ms, c.dino_res.timing_search_ms);

    if (s.is_bundle) {
      // Rendered as the station renders those values, so the sidecar's cli_equivalent reproduces it.
      apply_station((*c.station_params)[0], (*c.station_params)[1], (*c.station_params)[2]);
    } else {
      c.exposure_comp = c.dino_res.solved_absolute.e;
      c.gain_g = c.dino_res.solved_absolute.g;
      c.gain_b = c.dino_res.solved_absolute.b;
      adjust_correction_matrix(s.r_coeff, s.g_coeff, s.b_coeff, c.exposure_comp, c.gain_g, c.gain_b,
                               c.profile_film_base_rgb, unit_fb, c.merged_matrix);
    }
  } else if (s.is_bundle) {
    if (s.manual_target_idx >= 0) {
      c.bundle_target_idx = s.manual_target_idx;
    } else {
      c.bundle_target_idx = select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
    }
    log_info("Selected target: %s\n", bundle.targets[c.bundle_target_idx].display_name.c_str());
    c.bundle_target_name = bundle.targets[c.bundle_target_idx].name;
    c.film_prof = bundle.targets[c.bundle_target_idx].film_prof;
    c.film_prof.rebind_pointers();
    // The given values are the station's (E defaults to the bracket's e_center, g and b to 1).
    double scan_shutter_s = 0.125, scan_iso = 100.0;
    scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
    apply_station(given_station_e(o, bundle, c.bundle_target_idx, scan_shutter_s, scan_iso, scan.film_base_rgb()),
                  c.gain_g, c.gain_b);
  }

  if (s.is_bundle && c.bundle_target_idx >= 0) {
    c.embedded_icc = bundle.targets[c.bundle_target_idx].icc_bytes;
  }
  if (s.is_bundle && c.bundle_target_idx >= 0 && c.station_params) {
    // The highlight roll-off of the conversion, as the bracket search measures it at its final parameters (on the
    // half-size frame), so given values (cli_equivalent) reproduce it: Auto's result and given values were measured
    // in decode; a pinned -E or --target search keeps its solver's own x_hi.
    float q995 = 0.0f;
    if (scan.has_rolloff) {
      c.shoulder.x_hi = scan.rolloff_x_hi;
      q995 = scan.rolloff_q995;
    } else {
      c.shoulder.x_hi = c.dino_res.shoulder.x_hi;
      q995 = c.dino_res.shoulder.q995;
    }
    c.shoulder.has_q995 = true;
    c.shoulder.q995 = q995;
    if (c.shoulder.active()) {
      c.shoulder.source = "rolloff";
      negicc::PipelineConfig pc{};
      negicc::set_shoulder(pc, c.shoulder.x_hi);
      c.shoulder.has_knee = true;
      c.shoulder.knee = pc.shoulder_x_k;
    }
    log_info("Roll-off: q995 %.3f, x_hi %.3f (sRGB output only)\n", q995, c.shoulder.x_hi);
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
  backend.convert(out, export_pc, c.merged_matrix, scan.image());
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
  const std::string& attach_profile = c.out.attach_profile;
  const OutputEncoding encoding = c.out.encoding;
  const int out_w = proc->imgdata.sizes.iwidth, out_h = proc->imgdata.sizes.iheight;

  log_info("Writing TIFF '%s'\n", o.output.c_str());
  const double t_tiff_0 = omp_get_wtime();
  int ret = write_tiff(proc, attach_profile, o.output, s.is_bundle ? c.embedded_icc : std::vector<uint8_t>{},
                       scan.rgb3.empty() ? nullptr : scan.rgb3.data());
  scan.t_tiff_write_ms = (omp_get_wtime() - t_tiff_0) * 1000.0;

  if (ret != 0 || o.no_metadata) {
    unlink((o.output + ".json").c_str());
    return ret;
  }

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
  if (o.roi_auto) {
    roi_source = (scan.frame_det.conf < kLowFrameConfidence) ? "auto_fallback" : "auto";
  } else if (scan.has_roi) {
    roi_source = "cli";
  }

  // A bundle conversion takes the film base read from the scan (Scan::film_base), else the bundle's.
  const bool film_base_applied = s.is_bundle && scan.film_base.found;

  Sidecar sc;
  sc.output_path = o.output;
  sc.argc = o.argc;
  sc.argv = o.argv;
  sc.proc = proc;
  sc.files = o.files;
  sc.orig_w = scan.orig_w;
  sc.orig_h = scan.orig_h;
  sc.has_roi = scan.has_roi;
  sc.roi = scan.roi;
  sc.roi_source = roi_source;
  sc.frame_det = o.roi_auto ? &scan.frame_det : nullptr;
  sc.rot_cw = scan.rot_cw;
  sc.hflip = scan.hflip;
  sc.vflip = scan.vflip;
  sc.out_w = out_w;
  sc.out_h = out_h;
  sc.film_base_applied = film_base_applied;
  sc.film_base_rgb.assign(c.sidecar_film_base_rgb.begin(), c.sidecar_film_base_rgb.end());
  sc.profile_film_base_rgb = c.sidecar_profile_film_base_rgb;
  sc.film_base_source = film_base_applied ? scan.film_base.source : c.film_base_source;
  sc.film_base_detection = s.is_bundle ? &scan.film_base : nullptr;
  sc.profile_path = s.is_bundle ? s.bundle.path : s.film_prof_arg;
  sc.is_bundle = s.is_bundle;
  sc.bundle_target_name = c.bundle_target_name;
  sc.bundle_target_idx = c.bundle_target_idx;
  sc.input_space = s.bundle.input_space;
  sc.merged_matrix = c.merged_matrix;
  sc.exposure_comp = c.exposure_comp;
  sc.gain_g = c.gain_g;
  sc.gain_b = c.gain_b;
  sc.station_params = c.station_params ? c.station_params->data() : nullptr;
  sc.pin_mask = c.pin_mask;
  sc.knee = c.knee;
  sc.shoulder = c.shoulder;
  if (encoding != ENCODE_SRGB && sc.shoulder.active()) {
    // Only the sRGB output applies the roll-off: record it as off (keeping the measured 99.5% point).
    sc.shoulder.x_hi = 1.0f;
    sc.shoulder.source = "off";
    sc.shoulder.has_knee = false;
  }
  sc.dino = s.use_dino && !scan.has_auto_res ? &c.dino_res : nullptr;
  sc.dino_model = s.dino_path;
  sc.dino_backend = s.use_dino ? std::string(s.engine.backend_used()) : o.dino_backend;
  sc.auto_res = scan.has_auto_res ? &scan.auto_res : nullptr;
  sc.auto_frame_half = scan.auto_frame_half;
  sc.colorspace = cs_name;
  sc.t_raw_decode_ms = scan.t_raw_decode_ms;
  sc.t_frame_det_ms = scan.t_frame_det_ms;
  sc.t_full_conversion_ms = scan.t_full_conversion_ms;
  sc.t_tiff_write_ms = scan.t_tiff_write_ms;
  sc.t_auto_ms = scan.t_auto_ms;
  write_json_sidecar(sc);
  return 0;
}
