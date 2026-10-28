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

#include "profile_bundle.h"

#include <lcms2_plugin.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

static double srgb_enc_d(double l) {
  return (l <= 0.0031308) ? (l * 12.92) : (1.055 * std::pow(l, 1.0 / 2.4) - 0.055);
}

static double srgb_dec_d(double s) {
  return (s <= 0.04045) ? (s / 12.92) : std::pow((s + 0.055) / 1.055, 2.4);
}

static float apply_fit3_hl_shoulder_u(float u_in) {
  constexpr double kWhiteU = 0.75;
  constexpr double kRolloff = 0.5;
  constexpr double kKneeU = 0.42;
  constexpr double kExp = 8.0;
  constexpr double kCeilS = 0.988;  // ~251.9 in 8-bit sRGB (matches FIT3 highlight asymptote <= 252)
  const double u = (double)u_in;
  if (u <= kKneeU) return u_in;
  double H;
  if (u <= kWhiteU) {
    H = u / kWhiteU;
  } else if (u <= 0.90) {
    const double k = (1.0 - kWhiteU) / (kWhiteU * kRolloff);
    const double t = (u - kWhiteU) / (1.0 - kWhiteU);
    H = 1.0 + k * t / (1.0 - t);
  } else {
    // C1-continuous tangent continuation above u = 0.90 (t = 0.60, H = 2.0, dH/du = 50/3)
    const double du = u - 0.90;
    H = 2.0 + (50.0 / 3.0) * du + 25.0 * du * du;
  }
  const double H0 = kKneeU / kWhiteU;
  const double ev = std::log2(H / H0);
  if (ev <= 0.0) return u_in;
  const double s0 = srgb_enc_d(kKneeU);
  const double ds_du = (1.055 / 2.4) * std::pow(kKneeU, 1.0 / 2.4 - 1.0);
  const double slope_ev = ds_du * std::log(2.0) * kKneeU;
  const double rem = kCeilS - s0;
  const double span_ev = (rem * kExp) / slope_ev;
  if (ev >= span_ev) return (float)srgb_dec_d(kCeilS);
  const double s_sh = kCeilS - rem * std::pow((span_ev - ev) / span_ev, kExp);
  const double u_sh = srgb_dec_d(s_sh);
  return (float)std::min(u, u_sh);
}

Rgb apply_fit3_highlight_shoulder_rgb(Rgb lin) {
  constexpr float kMaxU = 1.50f;
  constexpr float kInvMaxU = 1.0f / kMaxU;
  static const std::vector<float> kLut = []() {
    constexpr int kN = 6144;
    std::vector<float> lut(kN);
    for (int i = 0; i < kN; ++i) {
      lut[i] = apply_fit3_hl_shoulder_u(((float)i / (float)(kN - 1)) * kMaxU);
    }
    return lut;
  }();
  return {
      negicc::apply_trc(negicc::clamp01(lin.r * kInvMaxU), kLut.data(), (int)kLut.size()),
      negicc::apply_trc(negicc::clamp01(lin.g * kInvMaxU), kLut.data(), (int)kLut.size()),
      negicc::apply_trc(negicc::clamp01(lin.b * kInvMaxU), kLut.data(), (int)kLut.size()),
  };
}

static void read_curve_set_local(_cmsStageToneCurvesData* data,
                                 std::vector<float> (&out)[3],
                                 int (&out_sizes)[3],
                                 const float* (&out_ptrs)[3]) {
  for (int c = 0; c < 3; ++c) {
    if (!data || c >= (int)data->nCurves || !data->TheCurves[c]) {
      out[c].clear();
      out_sizes[c] = 0;
      out_ptrs[c] = nullptr;
      continue;
    }
    cmsToneCurve* tc = data->TheCurves[c];
    const int entries = cmsGetToneCurveEstimatedTableEntries(tc);
    const cmsUInt16Number* table = cmsGetToneCurveEstimatedTable(tc);
    if (entries > 0 && table) {
      out[c].resize(entries);
      for (int i = 0; i < entries; ++i) {
        out[c][i] = table[i] / 65535.0f;
      }
    } else {
      out[c].resize(4096);
      for (int i = 0; i < 4096; ++i) {
        float in_v = (float)i / 4095.0f;
        out[c][i] = cmsEvalToneCurveFloat(tc, in_v);
      }
    }
    out_sizes[c] = (int)out[c].size();
    out_ptrs[c] = out[c].data();
  }
}

bool parse_film_profile_pipeline(cmsHPROFILE hprof, ParsedFilmProfile& prof) {
  if (!hprof) return false;
  cmsPipeline* pipeline = (cmsPipeline*)cmsReadTag(hprof, cmsSigAToB0Tag);
  if (!pipeline) return false;

  prof.data.has_profile = 1;
  const float ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  memcpy(prof.data.matrix, ident, sizeof(ident));
  prof.data.offset[0] = prof.data.offset[1] = prof.data.offset[2] = 0.0f;

  int curve_sets = 0;
  for (cmsStage* st = cmsPipelineGetPtrToFirstStage(pipeline); st; st = cmsStageNext(st)) {
    const cmsStageSignature type = cmsStageType(st);
    if (type == cmsSigCurveSetElemType) {
      auto* cd = (_cmsStageToneCurvesData*)cmsStageData(st);
      if (curve_sets == 0) {
        read_curve_set_local(cd, prof.in_trc, prof.data.in_trc_size, prof.data.in_trc);
      } else {
        read_curve_set_local(cd, prof.out_trc, prof.data.out_trc_size, prof.data.out_trc);
      }
      curve_sets++;
    } else if (type == cmsSigMatrixElemType) {
      auto* md = (_cmsStageMatrixData*)cmsStageData(st);
      for (int i = 0; i < 9; ++i) prof.data.matrix[i] = (float)md->Double[i];
      for (int i = 0; i < 3; ++i) prof.data.offset[i] = md->Offset ? (float)md->Offset[i] : 0.0f;
    } else if (type == cmsSigCLutElemType) {
      auto* cl = (_cmsStageCLutData*)cmsStageData(st);
      for (int i = 0; i < 3; ++i) prof.data.clut_dim[i] = cl->Params->nSamples[i];
      const size_t n = (size_t)prof.data.clut_dim[0] * prof.data.clut_dim[1] * prof.data.clut_dim[2] * 3;
      prof.clut.resize(n);
      if (cl->HasFloatValues) {
        for (size_t i = 0; i < n; ++i) prof.clut[i] = cl->Tab.TFloat[i];
      } else {
        for (size_t i = 0; i < n; ++i) prof.clut[i] = cl->Tab.T[i] / 65535.0f;
      }
      prof.data.clut = prof.clut.data();
    }
  }
  return true;
}

bool is_json_bundle(const std::string& raw_path) {
  auto open_regular = [](const std::string& p) -> FILE* {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec)) return nullptr;
    return fopen(p.c_str(), "rb");
  };
  FILE* fp = open_regular(raw_path);
  if (!fp) {
    std::string alt = "profiles/" + raw_path;
    fp = open_regular(alt);
    if (!fp) {
      size_t last_slash = raw_path.find_last_of("/\\");
      if (last_slash != std::string::npos) {
        std::string base = "profiles/" + raw_path.substr(last_slash + 1);
        fp = open_regular(base);
      }
    }
  }
  if (!fp) return false;

  char buf[4096];
  size_t n = fread(buf, 1, sizeof(buf), fp);
  fclose(fp);
  if (n == 0) return false;

  size_t i = 0;
  if (n >= 3 && (uint8_t)buf[0] == 0xEF && (uint8_t)buf[1] == 0xBB && (uint8_t)buf[2] == 0xBF) {
    i = 3;
  }
  while (i < n && (buf[i] == ' ' || buf[i] == '\t' || buf[i] == '\r' || buf[i] == '\n')) {
    i++;
  }
  return (i < n && buf[i] == '{');
}

static inline bool base64_decode(const std::string& in, std::vector<uint8_t>& out) {
  out.clear();
  out.reserve(in.size() * 3 / 4);
  static const int8_t T[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1, -2,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
  };
  int val = 0, valb = -8;
  for (unsigned char c : in) {
    if (c == '=') break;
    int d = T[c];
    if (d < 0) continue;
    val = (val << 6) | d;
    valb += 6;
    if (valb >= 0) {
      out.push_back((uint8_t)((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return !out.empty();
}

const char* to_string(InputSpace space) {
  return space == InputSpace::kRec2020Linear ? "rec2020_linear" : "camera_raw_rgb";
}

static bool parse_3x3_matrix(const nlohmann::json& node, float out_m[9], double out_d[9] = nullptr) {
  if (!node.is_array() || node.size() != 3) return false;
  for (int r = 0; r < 3; ++r) {
    if (!node[r].is_array() || node[r].size() != 3) return false;
    for (int c = 0; c < 3; ++c) {
      if (!node[r][c].is_number()) return false;
      const double vd = node[r][c].get<double>();
      out_m[r * 3 + c] = (float)vd;
      if (out_d) out_d[r * 3 + c] = vd;
    }
  }
  return true;
}

double shutter_to_seconds_d(const std::string& shutter) {
  std::string s = shutter;
  const size_t b = s.find_first_not_of(" \t\r\n");
  const size_t e = s.find_last_not_of(" \t\r\n");
  s = (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
  while (!s.empty() && s.back() == 's') s.pop_back();
  if (s.empty()) throw std::invalid_argument("bad shutter speed '" + shutter + "'");
  auto to_double = [&](const std::string& t) {
    if (t.empty()) throw std::invalid_argument("bad shutter speed '" + shutter + "'");
    size_t used = 0;
    const double v = std::stod(t, &used);
    if (used != t.size() || !std::isfinite(v)) throw std::invalid_argument("bad shutter speed '" + shutter + "'");
    return v;
  };
  const size_t slash = s.find('/');
  double val = 0.0;
  if (slash != std::string::npos) {
    const double den = to_double(s.substr(slash + 1));
    if (den <= 0.0) throw std::invalid_argument("bad shutter speed '" + shutter + "'");
    val = to_double(s.substr(0, slash)) / den;
  } else {
    val = to_double(s);
  }
  if (!std::isfinite(val) || val <= 0.0) throw std::invalid_argument("bad shutter speed '" + shutter + "'");
  return val;
}

// The profiles record a bracket's shutter speed as the camera displays it, a string: a fraction ("1/8s"), or
// decimal seconds ("0.4s", "2"), with or without the trailing "s". Trims the string and the "s", then divides a
// fraction or reads the decimal.
// TODO: Normalize the shutter speed into float seconds in the profile bundle files and drop this parsing.
float shutter_to_seconds(const std::string& shutter) {
  return (float)shutter_to_seconds_d(shutter);
}

bool read_exif_exposure_d(const std::string& filepath, double& shutter_s, double& iso) {
  FILE* fp = fopen(filepath.c_str(), "rb");
  if (!fp) return false;
  auto rd16 = [&](bool le, uint16_t& out) {
    uint8_t b[2];
    if (fread(b, 1, 2, fp) != 2) return false;
    out = le ? (uint16_t(b[0]) | (uint16_t(b[1]) << 8)) : ((uint16_t(b[0]) << 8) | uint16_t(b[1]));
    return true;
  };
  auto rd32 = [&](bool le, uint32_t& out) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) return false;
    out = le ? (uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24))
             : ((uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]));
    return true;
  };
  long tiff_base = 0;
  uint8_t hdr[16];
  if (fread(hdr, 1, 16, fp) == 16 && std::memcmp(hdr, "FUJIFILMCCD-RAW ", 16) == 0) {
    uint32_t jpeg_off = 0, jpeg_len = 0;
    if (fseek(fp, 84, SEEK_SET) != 0 || !rd32(false, jpeg_off) || !rd32(false, jpeg_len) ||
        jpeg_off == 0 || jpeg_len < 12 || fseek(fp, (long)jpeg_off, SEEK_SET) != 0) {
      fclose(fp);
      return false;
    }
    uint8_t soi[2];
    if (fread(soi, 1, 2, fp) != 2 || soi[0] != 0xFF || soi[1] != 0xD8) {
      fclose(fp);
      return false;
    }
    long pos = (long)jpeg_off + 2;
    const long end = (long)jpeg_off + (long)jpeg_len;
    bool found_exif = false;
    while (pos + 4 <= end) {
      if (fseek(fp, pos, SEEK_SET) != 0) break;
      uint8_t m[2];
      uint16_t seg_len = 0;
      if (fread(m, 1, 2, fp) != 2 || m[0] != 0xFF || !rd16(false, seg_len) || seg_len < 2) break;
      if (m[1] == 0xE1 && seg_len >= 8) {
        char exif_hdr[6];
        if (fread(exif_hdr, 1, 6, fp) == 6 && std::memcmp(exif_hdr, "Exif\0\0", 6) == 0) {
          tiff_base = pos + 10;
          found_exif = true;
          break;
        }
      }
      if (m[1] == 0xDA || m[1] == 0xD9) break;
      pos += 2 + (long)seg_len;
    }
    if (!found_exif) {
      fclose(fp);
      return false;
    }
  }
  if (fseek(fp, tiff_base, SEEK_SET) != 0) {
    fclose(fp);
    return false;
  }
  uint8_t bo[2];
  if (fread(bo, 1, 2, fp) != 2 || !((bo[0] == 'I' && bo[1] == 'I') || (bo[0] == 'M' && bo[1] == 'M'))) {
    fclose(fp);
    return false;
  }
  const bool le = (bo[0] == 'I');
  uint16_t magic = 0;
  uint32_t ifd0_off = 0;
  if (!rd16(le, magic) || magic != 42 || !rd32(le, ifd0_off) || fseek(fp, tiff_base + (long)ifd0_off, SEEK_SET) != 0) {
    fclose(fp);
    return false;
  }
  uint16_t n0 = 0;
  if (!rd16(le, n0)) {
    fclose(fp);
    return false;
  }
  uint32_t exif_off = 0;
  for (uint16_t i = 0; i < n0; ++i) {
    uint16_t tag = 0, fmt = 0;
    uint32_t cnt = 0, val = 0;
    if (!rd16(le, tag) || !rd16(le, fmt) || !rd32(le, cnt) || !rd32(le, val)) break;
    if (tag == 0x8769) {
      exif_off = val;
      break;
    }
  }
  if (!exif_off || fseek(fp, tiff_base + (long)exif_off, SEEK_SET) != 0) {
    fclose(fp);
    return false;
  }
  uint16_t ne = 0;
  if (!rd16(le, ne)) {
    fclose(fp);
    return false;
  }
  struct TagEntry { uint16_t tag, fmt; uint32_t cnt, val; };
  std::vector<TagEntry> entries;
  entries.reserve(ne);
  for (uint16_t i = 0; i < ne; ++i) {
    TagEntry e{};
    if (!rd16(le, e.tag) || !rd16(le, e.fmt) || !rd32(le, e.cnt) || !rd32(le, e.val)) break;
    entries.push_back(e);
  }
  bool got_shutter = false;
  for (const auto& e : entries) {
    if (e.tag == 0x829A) {
      uint32_t num = 0, den = 0;
      if (fseek(fp, tiff_base + (long)e.val, SEEK_SET) == 0 && rd32(le, num) && rd32(le, den) && den > 0) {
        shutter_s = (double)num / (double)den;
        got_shutter = true;
      }
    } else if (e.tag == 0x8827 && e.fmt == 3) {
      if (e.cnt == 1) {
        iso = (double)(le ? (e.val & 0xFFFFu) : ((e.val >> 16) & 0xFFFFu));
      } else if (e.cnt > 1 && fseek(fp, tiff_base + (long)e.val, SEEK_SET) == 0) {
        uint16_t v16 = 0;
        if (rd16(le, v16)) iso = (double)v16;
      }
    } else if (e.tag == 0x8827 && e.fmt == 4) {
      if (e.cnt == 1) {
        iso = (double)e.val;
      } else if (e.cnt > 1 && fseek(fp, tiff_base + (long)e.val, SEEK_SET) == 0) {
        uint32_t v32 = 0;
        if (rd32(le, v32)) iso = (double)v32;
      }
    }
  }
  fclose(fp);
  return got_shutter;
}

void compute_station_merged_matrix(const ProfileBundle& bundle, double scan_shutter_s, double scan_iso,
                                   const double* custom_film_base_rgb, bool film_base_from_scan,
                                   float* out_f32, double* out_f64, double* out_row_scale_f64) {
  const double* fb = custom_film_base_rgb ? custom_film_base_rgb : bundle.film_base_avg_d;
  const double base_t = bundle.film_base_shutter.empty() ? 0.125 : shutter_to_seconds_d(bundle.film_base_shutter);
  const double base_iso = (double)bundle.film_base_iso;
  const double exposure_base = base_t * (base_iso / 100.0);
  const double exposure_scan = scan_shutter_s * (scan_iso / 100.0);
  const double exposure_ratio =
      film_base_from_scan ? 1.0 : (exposure_scan > 0.0 ? (exposure_base / exposure_scan) : 1.0);
  for (int r = 0; r < 3; ++r) {
    const double scale = (fb[r] > 0.0) ? (bundle.normalization_target / fb[r]) * exposure_ratio : 1.0;
    if (out_row_scale_f64) out_row_scale_f64[r] = scale;
    for (int c = 0; c < 3; ++c) {
      const double v = bundle.crosstalk_matrix_d[r * 3 + c] * scale;
      if (out_f64) out_f64[r * 3 + c] = v;
      if (out_f32) out_f32[r * 3 + c] = (float)v;
    }
  }
}

void parse_profile_stem(const std::string& raw_name, std::string& bundle_key, std::string& stock_key) {
  bundle_key.clear();
  stock_key.clear();
  std::string s = raw_name;
  const size_t b = s.find_first_not_of(" \t\r\n");
  const size_t e = s.find_last_not_of(" \t\r\n");
  if (b == std::string::npos) return;
  s = s.substr(b, e - b + 1);
  const size_t slash = s.find_last_of("/\\");
  if (slash != std::string::npos) s = s.substr(slash + 1);
  auto lower = [](std::string v) {
    for (char& c : v) c = (char)std::tolower((unsigned char)c);
    return v;
  };
  if (s.size() >= 5 && lower(s.substr(s.size() - 5)) == ".json") s = s.substr(0, s.size() - 5);
  std::vector<std::string> parts;
  size_t start = 0;
  for (;;) {
    const size_t us = s.find('_', start);
    parts.push_back(s.substr(start, us == std::string::npos ? std::string::npos : us - start));
    if (us == std::string::npos) break;
    start = us + 1;
  }
  std::string bundle = s;
  if (lower(parts[0]) == "profile" && parts.size() >= 2) {
    std::string kept;
    bool any = false;
    for (size_t i = 1; i < parts.size(); ++i) {
      const std::string& p = parts[i];
      const bool is_digits = !p.empty() && std::all_of(p.begin(), p.end(), [](char c) { return c >= '0' && c <= '9'; });
      if (is_digits && p.size() >= 8) break;
      kept += (any ? "_" : "") + p;
      any = true;
    }
    bundle = any ? kept : s;
  }
  bundle.erase(std::remove(bundle.begin(), bundle.end(), ' '), bundle.end());
  if (bundle.empty()) return;
  bundle_key = bundle;
  stock_key = lower(bundle.substr(0, bundle.find('-')));
  stock_key = stock_key.substr(0, stock_key.find('_'));
}

static void parse_profile_bundle(const nlohmann::json& j, const std::string& raw_path,
                                 const BundleFileReader& read_file, ProfileBundle& bundle);

// nlohmann's errors do not name the file, so they are rethrown with it.
static nlohmann::json parse_bundle_json(const char* begin, const char* end, const std::string& name) {
  try {
    return nlohmann::json::parse(begin, end);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("profile bundle " + name + ": " + e.what());
  }
}

void load_profile_bundle_from_memory(const char* data, size_t size, const std::string& name,
                                     const BundleFileReader& read_file, ProfileBundle& bundle) {
  parse_profile_bundle(parse_bundle_json(data, data + size, name), name, read_file, bundle);
}

static bool read_file_bytes(const std::filesystem::path& path, std::vector<uint8_t>& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) return false;
  out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

void load_profile_bundle(const std::string& raw_path, ProfileBundle& bundle) {
  namespace fs = std::filesystem;
  const fs::path raw(raw_path);
  std::vector<fs::path> candidates = {raw, fs::path("profiles") / raw};
  if (raw.has_parent_path()) candidates.push_back(fs::path("profiles") / raw.filename());
  fs::path opened;
  std::vector<uint8_t> json_bytes;
  for (fs::path c : candidates) {
    std::error_code ec;
    if (fs::is_directory(c, ec)) c /= "bundle.json";
    if (read_file_bytes(c, json_bytes)) {
      opened = c;
      break;
    }
  }
  if (opened.empty()) throw std::runtime_error("cannot open profile bundle " + raw_path);

  // ICC profiles of a split bundle (profiles/<Key>/bundle.json) sit next to it.
  const fs::path dir = opened.parent_path();
  const BundleFileReader read_file = [&dir](const std::string& name, std::vector<uint8_t>& out) {
    return read_file_bytes(dir / name, out);
  };
  const char* begin = reinterpret_cast<const char*>(json_bytes.data());
  parse_profile_bundle(parse_bundle_json(begin, begin + json_bytes.size(), raw_path), raw_path, read_file, bundle);
}

static void parse_profile_bundle(const nlohmann::json& j, const std::string& raw_path,
                                 const BundleFileReader& read_file, ProfileBundle& bundle) {
  bundle.path = raw_path;
  if (j.contains("export") && j["export"].is_object()) {
    bundle.key = j["export"].value("bundle_key", "");
    bundle.source_file = j["export"].value("source_file", "");
  }
  if (bundle.key.empty()) {
    std::string stock;
    parse_profile_stem(raw_path, bundle.key, stock);
  }
  bundle.camera_name = j.value("camera_name", j.value("camera_model", "Unknown"));
  bundle.normalization_target = j.value("normalization_target", 55000.0);

  // The colour path the bundle was measured in. Bundles from before the field are camera_raw_rgb.
  // py-ref: negicc_station/src/film_profiling.py @ 813166c (FilmProfile.__init__)
  const nlohmann::json& space = j.contains("input_space") ? j["input_space"] : nlohmann::json("camera_raw_rgb");
  if (space == "camera_raw_rgb") {
    bundle.input_space = InputSpace::kCameraRawRgb;
  } else if (space == "rec2020_linear") {
    bundle.input_space = InputSpace::kRec2020Linear;
  } else {
    throw std::runtime_error("profile bundle " + raw_path + ": unknown input_space " + space.dump() +
                             " (camera_raw_rgb or rec2020_linear)");
  }
  if (bundle.input_space == InputSpace::kRec2020Linear) {
    // A common colour space needs no correction; a matrix here would be applied to every camera's decode.
    const bool nested = j.contains("crosstalk_profile") && j["crosstalk_profile"].is_object() &&
                        j["crosstalk_profile"].contains("crosstalk_correction_matrix") &&
                        !j["crosstalk_profile"]["crosstalk_correction_matrix"].is_null();
    const bool top = j.contains("crosstalk_correction_matrix") && !j["crosstalk_correction_matrix"].is_null();
    if (nested || top) {
      throw std::runtime_error("profile bundle " + raw_path + ": a rec2020_linear bundle must not carry a crosstalk "
                               "correction matrix (re-export it from negicc-station, whose "
                               "tools/fix_d65_profiles.py removed it)");
    }
    // crosstalk_matrix stays the identity and has_crosstalk false.
  } else if (j.contains("crosstalk_profile") && j["crosstalk_profile"].is_object()) {
    const auto& ct = j["crosstalk_profile"];
    if (ct.contains("crosstalk_correction_matrix")) {
      bundle.has_crosstalk = parse_3x3_matrix(ct["crosstalk_correction_matrix"], bundle.crosstalk_matrix,
                                              bundle.crosstalk_matrix_d);
    }
  } else if (j.contains("crosstalk_correction_matrix")) {
    bundle.has_crosstalk = parse_3x3_matrix(j["crosstalk_correction_matrix"], bundle.crosstalk_matrix,
                                            bundle.crosstalk_matrix_d);
  }

  if (j.contains("film_base") && j["film_base"].is_object()) {
    const auto& fb = j["film_base"];
    double r_avg = 0.0, g_avg = 0.0, b_avg = 0.0;
    if (fb.contains("r") && fb["r"].contains("avg") && fb["r"]["avg"].is_number()) {
      r_avg = fb["r"]["avg"].get<double>();
    }
    if (fb.contains("g") && fb["g"].contains("avg") && fb["g"]["avg"].is_number()) {
      g_avg = fb["g"]["avg"].get<double>();
    }
    if (fb.contains("b") && fb["b"].contains("avg") && fb["b"]["avg"].is_number()) {
      b_avg = fb["b"]["avg"].get<double>();
    }
    bundle.film_base_avg[0] = (float)r_avg;
    bundle.film_base_avg[1] = (float)g_avg;
    bundle.film_base_avg[2] = (float)b_avg;
    bundle.film_base_avg_d[0] = r_avg;
    bundle.film_base_avg_d[1] = g_avg;
    bundle.film_base_avg_d[2] = b_avg;
    if (fb.contains("shutter")) {
      if (fb["shutter"].is_string()) bundle.film_base_shutter = fb["shutter"].get<std::string>();
      else if (fb["shutter"].is_number()) bundle.film_base_shutter = std::to_string(fb["shutter"].get<double>());
      else throw std::invalid_argument("bad film_base shutter in " + raw_path);
      (void)shutter_to_seconds_d(bundle.film_base_shutter);
    }
    if (fb.contains("iso") && fb["iso"].is_number()) bundle.film_base_iso = fb["iso"].get<float>();
    if (r_avg > 0.0 && g_avg > 0.0 && b_avg > 0.0) {
      bundle.profile_film_base_rgb[0] = (int)std::round(r_avg);
      bundle.profile_film_base_rgb[1] = (int)std::round(g_avg);
      bundle.profile_film_base_rgb[2] = (int)std::round(b_avg);
      bundle.has_film_base = true;
    }
  }

  if (!j.contains("targets") || !j["targets"].is_array()) {
    throw std::runtime_error("profile bundle " + raw_path + " has no targets array");
  }

  bundle.targets.clear();
  for (const auto& tj : j["targets"]) {
    if (!tj.is_object()) continue;
    BundleTarget tgt;
    tgt.index = (int)bundle.targets.size();
    tgt.name = tj.value("name", "Target " + std::to_string(tgt.index + 1));
    tgt.iso = tj.value("iso", 100.0);
    if (tj.contains("shutter")) {
      if (tj["shutter"].is_string()) tgt.shutter = tj["shutter"].get<std::string>();
      else if (tj["shutter"].is_number()) tgt.shutter = std::to_string(tj["shutter"].get<double>());
    }

    if (tj.contains("target_ev") && tj["target_ev"].is_number()) {
      tgt.has_target_ev = true;
      tgt.target_ev = tj["target_ev"].get<double>();
    } else if (tj.contains("ev") && tj["ev"].is_number()) {
      tgt.has_target_ev = true;
      tgt.target_ev = tj["ev"].get<double>();
    }

    // py-ref: negicc_station/src/target_selection.py @ 93be1ec (bracket_exposure)
    tgt.exposure_s = std::max(1e-6f, shutter_to_seconds(tgt.shutter) * float(tgt.iso) / 100.0f);

    if (tj.contains("patches") && tj["patches"].is_object()) {
      std::vector<float> gs_g;
      for (auto it = tj["patches"].begin(); it != tj["patches"].end(); ++it) {
        if (it.value().is_object() && it.value().contains("g") && it.value()["g"].is_number()) {
          tgt.patches_g.push_back(it.value()["g"].get<float>());
        }
      }
      // Tonal capacity of the grey wedge gs0..gs23, or of all patches without one.
      // py-ref: negicc_station/src/target_selection.py @ 93be1ec (bracket_tonal_capacity)
      for (int k = 0; k < 24; ++k) {
        const std::string key = "gs" + std::to_string(k);
        if (tj["patches"].contains(key) && tj["patches"][key].is_object() && tj["patches"][key].contains("g") &&
            tj["patches"][key]["g"].is_number()) {
          gs_g.push_back(tj["patches"][key]["g"].get<float>());
        }
      }
      const std::vector<float>& cap_g = gs_g.empty() ? tgt.patches_g : gs_g;
      if (!cap_g.empty()) {
        const float mx = *std::max_element(cap_g.begin(), cap_g.end());
        const float mn = *std::min_element(cap_g.begin(), cap_g.end());
        tgt.tonal_capacity_ev = std::log2(mx / std::max(1e-6f, mn));
      }
      if (!tgt.patches_g.empty()) {
        tgt.tgt_min_g = *std::min_element(tgt.patches_g.begin(), tgt.patches_g.end());
        tgt.tgt_max_g = *std::max_element(tgt.patches_g.begin(), tgt.patches_g.end());
      }
    }
    if (tj.contains("profcheck_output") && tj["profcheck_output"].is_string()) {
      tgt.profcheck_output = tj["profcheck_output"].get<std::string>();
    }

    bool have_icc = false;
    if (tj.contains("icc_profile_base64") && tj["icc_profile_base64"].is_string()) {
      have_icc = base64_decode(tj["icc_profile_base64"].get<std::string>(), tgt.icc_bytes);
    } else if (tj.contains("icc_profile") && tj["icc_profile"].is_string()) {
      const std::string icc_name = tj["icc_profile"].get<std::string>();
      if (!read_file || !read_file(icc_name, tgt.icc_bytes)) {
        throw std::runtime_error("profile bundle " + raw_path + ": cannot read ICC profile " + icc_name +
                                 " of target " + tgt.name);
      }
      have_icc = true;
    }
    {
      if (have_icc && !tgt.icc_bytes.empty()) {
        cmsHPROFILE hprof = cmsOpenProfileFromMem(tgt.icc_bytes.data(), tgt.icc_bytes.size());
        if (hprof) {
          if (parse_film_profile_pipeline(hprof, tgt.film_prof)) {
            tgt.film_prof.rebind_pointers();
          }
          cmsCloseProfile(hprof);
        }
      }
    }

    if (tgt.film_prof.data.has_profile == 0) {
      std::cerr << "Warning: Target '" << tgt.name << "' has no valid ICC profile; skipping" << std::endl;
      continue;
    }

    tgt.index = (int)bundle.targets.size();
    bundle.targets.push_back(std::move(tgt));
  }

  if (bundle.targets.empty()) {
    throw std::runtime_error("profile bundle " + raw_path + " has no valid targets");
  }

  int nominal_idx = -1;
  for (size_t i = 0; i < bundle.targets.size(); ++i) {
    if (bundle.targets[i].has_target_ev && std::abs(bundle.targets[i].target_ev) < 0.01) {
      nominal_idx = (int)i;
      break;
    }
  }
  if (nominal_idx < 0) {
    for (size_t i = 0; i < bundle.targets.size(); ++i) {
      std::string lower_name = bundle.targets[i].name;
      for (char& c : lower_name) c = (char)std::tolower((unsigned char)c);
      if (lower_name.find("box_speed") != std::string::npos ||
          lower_name.find("nominal") != std::string::npos ||
          lower_name.find("0.0ev") != std::string::npos ||
          lower_name.find("+0ev") != std::string::npos ||
          lower_name.find("_0ev") != std::string::npos ||
          lower_name.find(" 0ev") != std::string::npos ||
          lower_name == "0ev") {
        nominal_idx = (int)i;
        break;
      }
    }
  }
  if (nominal_idx < 0) {
    for (size_t i = 0; i < bundle.targets.size(); ++i) {
      std::string lower_name = bundle.targets[i].name;
      for (char& c : lower_name) c = (char)std::tolower((unsigned char)c);
      if (lower_name == "target 5" || lower_name == "target5" || lower_name == "target_5") {
        nominal_idx = (int)i;
        break;
      }
    }
  }
  if (nominal_idx < 0) {
    nominal_idx = (int)bundle.targets.size() / 2;
  }

  double nom_center = 0.0;
  if (nominal_idx >= 0 && nominal_idx < (int)bundle.targets.size()) {
    const auto& nt = bundle.targets[nominal_idx];
    if (nt.tgt_min_g > 0.0f && nt.tgt_max_g > 0.0f) {
      nom_center = std::sqrt((double)nt.tgt_min_g * (double)nt.tgt_max_g);
    }
  }

  for (auto& tgt : bundle.targets) {
    if (tgt.has_target_ev) {
      char buf[32];
      snprintf(buf, sizeof(buf), " (%+.2f EV)", tgt.target_ev);
      tgt.display_name = tgt.name + buf;
    } else if (tgt.name.find("EV") != std::string::npos || tgt.name.find("ev") != std::string::npos) {
      tgt.display_name = tgt.name;
    } else if (nom_center > 0.0 && tgt.tgt_min_g > 0.0f && tgt.tgt_max_g > 0.0f) {
      double tgt_center = std::sqrt((double)tgt.tgt_min_g * (double)tgt.tgt_max_g);
      double delta_ev = std::log2(tgt_center / nom_center);
      char buf[32];
      snprintf(buf, sizeof(buf), " (%+.2f EV)", delta_ev);
      tgt.display_name = tgt.name + buf;
    } else {
      tgt.display_name = tgt.name;
    }
  }
}

void compute_bundle_enclosures(ProfileBundle& bundle, const uint16_t (*image)[4],
                              int width, int height, float& g_lo, float& g_hi,
                              float& scene_ev, int& num_feasible) {
  num_feasible = 0;
  const size_t total_px = (size_t)width * height;
  if (total_px == 0) return;

  std::vector<uint32_t> hist(65536, 0);
  for (size_t i = 0; i < total_px; ++i) {
    hist[image[i][1]]++;
  }

  const double p005 = 0.005 * (double)total_px;
  const double p995 = 0.995 * (double)total_px;

  uint64_t cum = 0;
  int g_005 = 1, g_995 = 65535;
  bool found_005 = false;
  for (int v = 0; v < 65536; ++v) {
    cum += hist[v];
    if (!found_005 && cum >= p005) {
      g_005 = v;
      found_005 = true;
    }
    if (cum >= p995) {
      g_995 = v;
      break;
    }
  }

  g_lo = std::max(1.0f, (float)g_005);
  g_hi = std::max(g_lo + 1.0f, (float)g_995);
  scene_ev = (float)std::log2(g_hi / g_lo);

  for (auto& tgt : bundle.targets) {
    if (tgt.patches_g.empty() || tgt.tgt_min_g <= 0.0f || tgt.tgt_max_g <= 0.0f) {
      tgt.is_feasible = false;
      continue;
    }
    tgt.e_min = tgt.tgt_min_g / g_lo;
    tgt.e_max = tgt.tgt_max_g / g_hi;
    tgt.is_feasible = (tgt.e_max >= tgt.e_min);
    tgt.tgt_capacity_ev = (float)std::log2(tgt.tgt_max_g / tgt.tgt_min_g);
    tgt.e_center = (float)std::sqrt(tgt.e_min * tgt.e_max);
    tgt.slack_ev = (float)std::log2(tgt.e_max / tgt.e_min);
    tgt.fill_ratio = (g_hi / g_lo) / (tgt.tgt_max_g / tgt.tgt_min_g);
    if (tgt.is_feasible) num_feasible++;
  }
}

void compute_bundle_enclosures(ProfileBundle& bundle, const negicc::RgbView& view,
                              float& g_lo, float& g_hi, float& scene_ev, int& num_feasible) {
  num_feasible = 0;
  const size_t total_px = (size_t)view.width * view.height;
  if (total_px == 0) return;

  std::vector<uint32_t> hist(65536, 0);
  if (view.is_contiguous_rgb()) {
    for (int y = 0; y < view.height; ++y) {
      const uint16_t* g_row = view.g_ptr + (ptrdiff_t)y * view.stride;
      for (int x = 0; x < view.width; ++x) {
        hist[g_row[(ptrdiff_t)x * 3]]++;
      }
    }
  } else {
    for (int y = 0; y < view.height; ++y) {
      for (int x = 0; x < view.width; ++x) {
        uint16_t px[3];
        view.sample(x, y, px);
        hist[px[1]]++;
      }
    }
  }

  const double p005 = 0.005 * (double)total_px;
  const double p995 = 0.995 * (double)total_px;

  uint64_t cum = 0;
  int g_005 = 1, g_995 = 65535;
  bool found_005 = false;
  for (int v = 0; v < 65536; ++v) {
    cum += hist[v];
    if (!found_005 && cum >= p005) {
      g_005 = v;
      found_005 = true;
    }
    if (cum >= p995) {
      g_995 = v;
      break;
    }
  }

  g_lo = std::max(1.0f, (float)g_005);
  g_hi = std::max(g_lo + 1.0f, (float)g_995);
  scene_ev = (float)std::log2(g_hi / g_lo);

  for (auto& tgt : bundle.targets) {
    if (tgt.patches_g.empty() || tgt.tgt_min_g <= 0.0f || tgt.tgt_max_g <= 0.0f) {
      tgt.is_feasible = false;
      continue;
    }
    tgt.e_min = tgt.tgt_min_g / g_lo;
    tgt.e_max = tgt.tgt_max_g / g_hi;
    tgt.is_feasible = (tgt.e_max >= tgt.e_min);
    tgt.tgt_capacity_ev = (float)std::log2(tgt.tgt_max_g / tgt.tgt_min_g);
    tgt.e_center = (float)std::sqrt(tgt.e_min * tgt.e_max);
    tgt.slack_ev = (float)std::log2(tgt.e_max / tgt.e_min);
    tgt.fill_ratio = (g_hi / g_lo) / (tgt.tgt_max_g / tgt.tgt_min_g);
    if (tgt.is_feasible) num_feasible++;
  }
}

void set_target_bounds(SolverConfig& cfg, const BundleTarget& tgt) {
  cfg.prof = &tgt.film_prof.data;
  if (tgt.e_center <= 0.0f || tgt.e_min <= 0.0f || tgt.e_max <= 0.0f) {
    cfg.has_custom_bounds = false;
    cfg.custom_e_center = 1.0f;
    cfg.custom_e_bounds[0] = 0.354f;
    cfg.custom_e_bounds[1] = 2.828f;
    cfg.custom_e_bounds[2] = 1.0f;
    return;
  }
  cfg.has_custom_bounds = true;
  cfg.custom_e_center = tgt.e_center;
  const float lo = std::min(tgt.e_min, tgt.e_max);
  const float hi = std::max(tgt.e_min, tgt.e_max);
  cfg.custom_e_bounds[0] = lo / cfg.custom_e_center;
  cfg.custom_e_bounds[1] = hi / cfg.custom_e_center;
  cfg.custom_e_bounds[2] = 1.0f;
}

int select_bundle_target_without_dino(const ProfileBundle& bundle, int min_target_idx) {
  if (bundle.targets.empty()) return -1;
  const int n = (int)bundle.targets.size();
  const int start_idx = (n >= 5) ? std::min(min_target_idx, n - 1) : 0;

  auto rank_fn = [&](int idx) -> float {
    const auto& r = bundle.targets[idx];
    if (!r.is_feasible || r.slack_ev < 0.0f) {
      return 1000.0f + std::abs(r.slack_ev) * 100.0f;
    }
    return std::abs(r.fill_ratio - 0.85f);
  };

  int best_feasible_idx = -1;
  float best_feasible_rank = 1e9f;

  int best_any_idx = -1;
  float best_any_rank = 1e9f;

  for (int i = start_idx; i < n; ++i) {
    const auto& t = bundle.targets[i];
    float r = rank_fn(i);
    if (t.is_feasible && t.slack_ev >= 0.0f) {
      if (r < best_feasible_rank) {
        best_feasible_rank = r;
        best_feasible_idx = i;
      }
    }
    if (r < best_any_rank) {
      best_any_rank = r;
      best_any_idx = i;
    }
  }

  if (best_feasible_idx >= 0) return best_feasible_idx;
  if (best_any_idx >= 0) return best_any_idx;
  return start_idx;
}
