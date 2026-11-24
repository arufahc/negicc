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

#include "profile_bundle.h"

#include "dinov3.h"
#include "np_log.h"

#include <lcms2_plugin.h>
#include <netinet/in.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

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
  // Every return leaves prof as it was made here: empty unless every stage was taken (has_profile).
  prof = ParsedFilmProfile{};
  if (!hprof) return false;
  if (cmsGetColorSpace(hprof) != cmsSigRgbData || cmsGetPCS(hprof) != cmsSigXYZData) {
    log_error("The profile is not RGB to XYZ, as a film profile is\n");
    return false;
  }
  cmsPipeline* pipeline = (cmsPipeline*)cmsReadTag(hprof, cmsSigAToB0Tag);
  if (!pipeline) {
    log_error("The profile has no AToB0 tag\n");
    return false;
  }
  // The one layout apply_profile applies: input curves, a cLUT, output curves, three channels throughout (a lut8Type or
  // lut16Type with the identity matrix, or a lutAtoBType with A curves, a cLUT and B curves). A lutAtoBType's matrix
  // after the cLUT is taken when it is the identity with no offset and no M curves come before it. Anything else is
  // refused with the stages it has: a lut8Type's or lut16Type's matrix (Little CMS keeps it, before the input curves,
  // only when it is not the identity), M curves, a missing cLUT or set of curves, or another stage.
  auto stage_name = [](cmsStageSignature t) -> const char* {
    switch (t) {
      case cmsSigCurveSetElemType: return "curves";
      case cmsSigCLutElemType: return "cLUT";
      case cmsSigMatrixElemType: return "matrix";
      default: return "another stage";
    }
  };
  std::string layout;
  for (cmsStage* st = cmsPipelineGetPtrToFirstStage(pipeline); st; st = cmsStageNext(st))
    layout += std::string(layout.empty() ? "" : ", ") + stage_name(cmsStageType(st));
  auto refuse = [&](const char* why) {
    prof = ParsedFilmProfile{};
    log_error("The profile's AToB0 (%s) is not input curves, a cLUT and output curves: %s\n", layout.c_str(), why);
    return false;
  };
  if (cmsPipelineInputChannels(pipeline) != 3 || cmsPipelineOutputChannels(pipeline) != 3)
    return refuse("it does not take three channels to three");

  // The position in the layout: 0 before the input curves, 1 before the cLUT, 2 after it (an identity matrix may
  // follow), 3 after the output curves.
  int at = 0;
  bool matrix_seen = false;
  for (cmsStage* st = cmsPipelineGetPtrToFirstStage(pipeline); st; st = cmsStageNext(st)) {
    if (cmsStageInputChannels(st) != 3 || cmsStageOutputChannels(st) != 3)
      return refuse("a stage does not take three channels to three");
    const cmsStageSignature type = cmsStageType(st);
    if (type == cmsSigCurveSetElemType) {
      auto* cd = (_cmsStageToneCurvesData*)cmsStageData(st);
      if (at == 0) {
        read_curve_set_local(cd, prof.in_trc, prof.data.in_trc_size, prof.data.in_trc);
        at = 1;
      } else if (at == 2) {
        read_curve_set_local(cd, prof.out_trc, prof.data.out_trc_size, prof.data.out_trc);
        at = 3;
      } else if (at == 1) {
        return refuse("a second set of curves before any cLUT (input curves without a cLUT, or a lutAtoBType's M or B "
                      "curves taken for input curves)");
      } else {
        return refuse("a set of curves after the output curves (a lutAtoBType's M curves)");
      }
    } else if (type == cmsSigMatrixElemType) {
      if (at != 2 || matrix_seen) return refuse("a matrix where apply_profile does not apply one");
      auto* md = (_cmsStageMatrixData*)cmsStageData(st);
      bool identity = true;
      for (int i = 0; i < 9; ++i) identity = identity && md->Double[i] == ((i % 4 == 0) ? 1.0 : 0.0);
      for (int i = 0; i < 3 && md->Offset; ++i) identity = identity && md->Offset[i] == 0.0;
      if (!identity) return refuse("a matrix other than the identity after the cLUT");
      matrix_seen = true;
    } else if (type == cmsSigCLutElemType) {
      if (at != 1) return refuse(at == 0 ? "a cLUT without input curves before it" : "a second cLUT");
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
      at = 2;
    } else {
      return refuse("a stage this parser does not know");
    }
  }
  if (at != 3) return refuse(at == 2 ? "no output curves after the cLUT" : "no cLUT");
  const float ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  memcpy(prof.data.matrix, ident, sizeof(ident));
  prof.data.offset[0] = prof.data.offset[1] = prof.data.offset[2] = 0.0f;
  prof.data.has_profile = 1;
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
  // A UTF-8 byte order mark (EF BB BF, RFC 3629 section 6) is skipped.
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
  // The standard base64 alphabet (RFC 4648 section 4, Table 1): A-Z 0-25, a-z 26-51, 0-9 52-61, '+' 62, '/' 63;
  // '=' (padding) is -2, everything else -1 and skipped.
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
  // Unsigned, so the bits shifted out at the top wrap away: only the low 12 bits are ever read.
  uint32_t val = 0;
  int valb = -8;
  for (unsigned char c : in) {
    if (c == '=') break;
    int d = T[c];
    if (d < 0) continue;
    val = (val << 6) | (uint32_t)d;
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

bool crosstalk_matrix_invertible(const float m[9]) {
  double x[9];
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(m[k])) return false;
    x[k] = m[k];
  }
  const double det = x[0] * (x[4] * x[8] - x[5] * x[7]) - x[1] * (x[3] * x[8] - x[5] * x[6]) +
                     x[2] * (x[3] * x[7] - x[4] * x[6]);
  // The determinant of dependent rows is not computed as 0 but as rounding residue (two equal rows of general values
  // give about 1e-17), so it is compared with the rounding error this computation can carry: a product of two floats
  // is exact in double, each term then takes two rounded operations and the sum two more, at most four units of
  // roundoff (2 x DBL_EPSILON) of the terms' magnitudes. A determinant within that of 0 is 0 as far as the arithmetic
  // can tell.
  const double magnitude = std::fabs(x[0]) * (std::fabs(x[4] * x[8]) + std::fabs(x[5] * x[7])) +
                           std::fabs(x[1]) * (std::fabs(x[3] * x[8]) + std::fabs(x[5] * x[6])) +
                           std::fabs(x[2]) * (std::fabs(x[3] * x[7]) + std::fabs(x[4] * x[6]));
  if (std::fabs(det) <= 2.0 * std::numeric_limits<double>::epsilon() * magnitude) return false;
  return std::fabs(det) >= (double)std::numeric_limits<float>::min();
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

const char* density_measure_name(DensityMeasure m) {
  return m == kDensityStatusM ? "statusm" : m == kDensityRp180 ? "rp180" : "apd";
}

const char* density_measure_key(DensityMeasure m) {
  return m == kDensityStatusM ? "status_m" : m == kDensityRp180 ? "rp180_printing_density" : "academy_printing_density";
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
    // Decimals only (an optional exponent): std::stod also reads hex floats, "inf" and "nan".
    if (t.find_first_not_of("0123456789+-.eE") != std::string::npos)
      throw std::invalid_argument("bad shutter speed '" + shutter + "'");
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
  // A Fujifilm RAF starts with "FUJIFILMCCD-RAW " and has the big-endian offset and length of its embedded JPEG at byte
  // 84, as LibRaw reads it (identify.cpp); the JPEG's markers (SOI FFD8, APP1 FFE1, SOS FFDA, EOI FFD9) are ITU-T T.81
  // Table B.1's, and its APP1 holds the Exif TIFF. Any other file is itself the TIFF. The TIFF starts with "II" or "MM"
  // and 42 (TIFF 6.0, section 2); the Exif tags (CIPA DC-008) are 0x8769 (Exif IFD pointer), 0x829A (ExposureTime, a
  // RATIONAL) and 0x8827 (PhotographicSensitivity, formerly ISOSpeedRatings, as a SHORT, type 3, or LONG, type 4).
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
                                 const BundleFileReader& read_file, ProfileBundle& bundle,
                                 const std::string& json_file = "");

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
  // Copied into the bundle's print ICC readers (BundlePrint::read_icc), which outlive this call.
  const BundleFileReader read_file = [dir](const std::string& name, std::vector<uint8_t>& out) {
    return read_file_bytes(dir / name, out);
  };
  const char* begin = reinterpret_cast<const char*>(json_bytes.data());
  parse_profile_bundle(parse_bundle_json(begin, begin + json_bytes.size(), raw_path), raw_path, read_file, bundle,
                       opened.string());
}

// json_file: the file the bundle was read from (empty for one read from memory), which a station bundle's print ICCs
// are read from again when a print asks for one.
static void parse_profile_bundle(const nlohmann::json& j, const std::string& raw_path,
                                 const BundleFileReader& read_file, ProfileBundle& bundle,
                                 const std::string& json_file) {
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
  if (j.contains("camera_models")) {
    const nlohmann::json& models = j["camera_models"];
    bool ok = models.is_array();
    for (size_t i = 0; ok && i < models.size(); ++i) ok = models[i].is_string();
    if (!ok) throw std::runtime_error("profile bundle " + raw_path + ": camera_models is not a list of model names");
    for (const auto& m : models) bundle.camera_models.push_back(m.get<std::string>());
    bundle.has_camera_models = true;
  }
  // A field given with a type the loader cannot use refuses the bundle, naming it; a field left out takes its default.
  // (The JSON parser refuses a number outside double's range, so every number read is finite.)
  auto refuse = [&raw_path](const std::string& field, const nlohmann::json& v, const char* why) {
    throw std::runtime_error("profile bundle " + raw_path + ": " + field + " " + v.dump() + " " + why);
  };

  // The 16-bit level the film base is scaled to (DESIGN.md 2.3), a code of the 16-bit output: from 1 to the full
  // scale. Every channel is multiplied by it over the film base (at most 2147483647, below), which then stays a normal
  // float, and n0 = normalization_target / 65535 a positive double.
  if (j.contains("normalization_target")) {
    const nlohmann::json& v = j["normalization_target"];
    if (!v.is_number() || !(v.get<double>() >= 1.0 && v.get<double>() <= 65535.0)) {
      refuse("normalization_target", v, "is not a number in [1, 65535]");
    }
    bundle.normalization_target = v.get<double>();
  }

  // The density export's matrices and the light's record (profile_bundle.h); absent from older bundles.
  if (j.contains("light_source") && j["light_source"].is_object()) bundle.light_source_json = j["light_source"].dump();
  if (j.contains("density_matrices") && !j["density_matrices"].is_object()) {
    refuse("density_matrices", j["density_matrices"], "is not an object");
  }
  if (j.contains("density_matrices")) {
    const nlohmann::json& dm = j["density_matrices"];
    // Everything but the matrices and the brackets is the fit's record, kept for the output's metadata.
    nlohmann::json fit = nlohmann::json::object();
    for (auto it = dm.begin(); it != dm.end(); ++it) {
      bool is_measure = it.key() == "brackets";
      for (int m = 0; m < kDensityMeasures; ++m) is_measure |= it.key() == density_measure_key((DensityMeasure)m);
      if (!is_measure) fit[it.key()] = it.value();
    }
    bundle.density_fit_json = fit.dump();
    auto finite = [](const double* m) {
      for (int k = 0; k < 9; ++k) {
        if (!std::isfinite(m[k])) return false;
      }
      return true;
    };
    // The matrix of a measure is kept in two parts: light_to_film, the light's density to the film's dye amounts
    // (it changes with the light), and the measure's film_to_measure, the dye amounts to its density (it does not).
    // The matrix applied is their product, with the measure's light_density_factors on the light's densities.
    float unused[9];
    double light_to_film[9];
    const bool has_light = dm.contains("light_to_film") && dm["light_to_film"].is_object() &&
                           dm["light_to_film"].contains("matrix") &&
                           parse_3x3_matrix(dm["light_to_film"]["matrix"], unused, light_to_film) &&
                           finite(light_to_film);
    for (int m = 0; m < kDensityMeasures; ++m) {
      const char* key = density_measure_key((DensityMeasure)m);
      if (!dm.contains(key)) continue;
      const nlohmann::json& node = dm[key];
      DensityMatrix& out = bundle.density_matrix[m];
      double film_to_measure[9];
      if (!has_light || !node.is_object() || !node.contains("film_to_measure") ||
          !parse_3x3_matrix(node["film_to_measure"], unused, film_to_measure) || !finite(film_to_measure)) {
        throw std::runtime_error("profile bundle " + raw_path + ": density_matrices needs a finite 3x3 "
                                 "light_to_film.matrix and " + key + ".film_to_measure");
      }
      double factor[3] = {1.0, 1.0, 1.0};
      if (node.contains("light_density_factors")) {
        const nlohmann::json& f = node["light_density_factors"];
        if (!f.is_array() || f.size() != 3 || !f[0].is_number() || !f[1].is_number() || !f[2].is_number()) {
          throw std::runtime_error("profile bundle " + raw_path + ": density_matrices." + key +
                                   ".light_density_factors needs three positive numbers");
        }
        for (int c = 0; c < 3; ++c) {
          factor[c] = f[c].get<double>();
          if (!std::isfinite(factor[c]) || !(factor[c] > 0.0)) {
            throw std::runtime_error("profile bundle " + raw_path + ": density_matrices." + key +
                                     ".light_density_factors needs three positive numbers");
          }
        }
      }
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          double v = 0.0;
          for (int k = 0; k < 3; ++k) v += film_to_measure[r * 3 + k] * light_to_film[k * 3 + c];
          out.m[r * 3 + c] = v * factor[c];
          if (!std::isfinite(out.m[r * 3 + c])) {
            throw std::runtime_error("profile bundle " + raw_path + ": density_matrices." + key +
                                     ": the product of film_to_measure and light_to_film is not finite");
          }
        }
      }
      for (int k = 0; k < 9; ++k) {
        out.film_to_measure[k] = film_to_measure[k];
        out.light_to_film[k] = light_to_film[k];
      }
      for (int c = 0; c < 3; ++c) out.light_density_factors[c] = factor[c];
      out.present = true;
      out.target_curves = node.value("target_curves", std::string());
    }
    // What a print of a density file takes from the chart, per exposure bracket (np_print.h): the mid grey as the
    // light's density, and the slopes of the film's grey scale.
    if (dm.contains("brackets") && (!dm["brackets"].is_object() || !dm["brackets"].contains("targets") ||
                                    !dm["brackets"]["targets"].is_object())) {
      refuse("density_matrices.brackets", dm["brackets"], "is not an object with an object of targets");
    }
    if (dm.contains("brackets")) {
      auto three = [](const nlohmann::json& a, double out[3]) {
        if (!a.is_array() || a.size() != 3) return false;
        for (int c = 0; c < 3; ++c) {
          if (!a[c].is_number()) return false;
          out[c] = a[c].get<double>();
          if (!std::isfinite(out[c])) return false;
        }
        return true;
      };
      const nlohmann::json& targets = dm["brackets"]["targets"];
      for (auto it = targets.begin(); it != targets.end(); ++it) {
        const std::string at = "density_matrices.brackets.targets." + it.key();
        const nlohmann::json& e = it.value();
        if (!e.is_object()) refuse(at, e, "is not an object");
        DensityBracket bracket;
        bracket.name = it.key();
        // The slopes are optional: only a bundle without the grey balance curve reads them (frame_print).
        if (e.contains("slopes")) {
          bracket.has_slopes = three(e["slopes"], bracket.slopes) && bracket.slopes[0] > 0.0 &&
                               bracket.slopes[1] > 0.0 && bracket.slopes[2] > 0.0;
          if (!bracket.has_slopes) refuse(at + ".slopes", e["slopes"], "is not three positive numbers");
        }
        if (e.contains("grey_relative")) {
          if (!e["grey_relative"].is_object()) refuse(at + ".grey_relative", e["grey_relative"], "is not an object");
          if (e["grey_relative"].contains("b")) {
            bracket.has_tilt = three(e["grey_relative"]["b"], bracket.tilt);
            if (!bracket.has_tilt) refuse(at + ".grey_relative.b", e["grey_relative"]["b"], "is not three numbers");
          }
        }
        // A bracket without its mid grey is left out.
        if (e.contains("grey_light_density")) {
          if (!three(e["grey_light_density"], bracket.grey_light)) {
            refuse(at + ".grey_light_density", e["grey_light_density"], "is not three numbers");
          }
          bundle.density_brackets.push_back(bracket);
        }
      }
    }
  }

  // The film's grey balance curve: three rising columns on one light-density grid. A block that is not whole is left
  // out, with a warning (a print then says the bundle has no grey balance curve).
  if (j.contains("density_matrices") && j["density_matrices"].is_object() &&
      j["density_matrices"].contains("grey_balance_curves") &&
      j["density_matrices"]["grey_balance_curves"].is_object()) {
    const nlohmann::json& g = j["density_matrices"]["grey_balance_curves"];
    GreyBalanceCurves& gb = bundle.grey_balance;
    bool ok = g.contains("light_density_start") && g["light_density_start"].is_number() &&
              g.contains("light_density_step") && g["light_density_step"].is_number() && g.contains("apd") &&
              g["apd"].is_array() && g["apd"].size() == 3;
    if (ok) {
      gb.start = g["light_density_start"].get<double>();
      gb.step = g["light_density_step"].get<double>();
      ok = gb.step > 0.0;
    }
    for (int c = 0; ok && c < 3; ++c) {
      for (const auto& v : g["apd"][c]) {
        ok = ok && v.is_number() && std::isfinite(v.get<double>()) &&
             (gb.apd[c].empty() || v.get<double>() > gb.apd[c].back());
        if (ok) gb.apd[c].push_back(v.get<double>());
      }
      ok = ok && gb.apd[c].size() >= 2 && gb.apd[c].size() == gb.apd[0].size();
    }
    gb.present = ok;
    if (!ok) {
      gb = GreyBalanceCurves{};
      log_warn("Profile bundle %s: density_matrices.grey_balance_curves is not whole (a start, a positive step and "
               "three rising columns of one length): left out\n", raw_path.c_str());
    }
  }

  // The bundle's print ICCs, by print material: where each is, read when a print asks for it (BundlePrint).
  if (j.contains("print_profiles") && j["print_profiles"].is_object()) {
    auto text = [](const nlohmann::json& e, const char* key, const std::string& fallback) {
      return e.contains(key) && e[key].is_string() ? e[key].get<std::string>() : fallback;
    };
    for (auto it = j["print_profiles"].begin(); it != j["print_profiles"].end(); ++it) {
      const nlohmann::json& e = it.value();
      if (!e.is_object()) continue;
      BundlePrint p;
      p.id = it.key();
      p.name = text(e, "name", p.id);
      p.reference_bracket = text(e, "reference_bracket", "");
      p.spektrafilm_sha256 = text(e, "spektrafilm_sha256", "");
      p.icc_sha256 = text(e, "icc_profile_sha256", "");
      const std::string where = "profile bundle " + raw_path;
      // An ICC profile is at least its 128-byte header and 4-byte tag count (ICC.1:2022 7.2 and 7.3.2): 132 bytes.
      if (e.contains("icc_profile") && e["icc_profile"].is_string()) {
        // An exported bundle: a file beside bundle.json, read through the bundle's own reader.
        const std::string file = e["icc_profile"].get<std::string>();
        p.icc_in_file = true;
        p.icc_file = file;
        p.read_icc = [read_file, file, where](std::vector<uint8_t>& icc, std::string& err) {
          if (read_file && read_file(file, icc) && icc.size() >= 132) return true;
          return err = where + ": cannot read print ICC " + file, false;
        };
      } else if (e.contains("icc_profile_base64") && e["icc_profile_base64"].is_string()) {
        // A station bundle: the base64 in its JSON. Read from the file again when asked (the JSON is not kept); a
        // bundle read from memory has no file, so its ICC is decoded now and held.
        const std::string id = p.id;
        auto decode = [id, where](const nlohmann::json& entry, std::vector<uint8_t>& icc, std::string& err) {
          if (entry.is_object() && entry.contains("icc_profile_base64") && entry["icc_profile_base64"].is_string() &&
              base64_decode(entry["icc_profile_base64"].get<std::string>(), icc) && icc.size() >= 132) {
            return true;
          }
          return err = where + ": the print ICC of " + id + " cannot be decoded", false;
        };
        if (!json_file.empty()) {
          p.read_icc = [json_file, id, where, decode](std::vector<uint8_t>& icc, std::string& err) {
            std::vector<uint8_t> bytes;
            if (!read_file_bytes(json_file, bytes)) return err = where + ": cannot be read again", false;
            const nlohmann::json again = nlohmann::json::parse(bytes.begin(), bytes.end(), nullptr, false);
            if (again.is_discarded() || !again.contains("print_profiles") || !again["print_profiles"].is_object() ||
                !again["print_profiles"].contains(id)) {
              return err = where + ": the print ICC of " + id + " is no longer in it", false;
            }
            return decode(again["print_profiles"][id], icc, err);
          };
        } else {
          // One that does not decode refuses its print when asked, as a missing file does, not the whole bundle.
          std::vector<uint8_t> bytes;
          std::string err;
          const bool ok = decode(e, bytes, err);
          auto held = std::make_shared<const std::vector<uint8_t>>(std::move(bytes));
          p.read_icc = [held, ok, err](std::vector<uint8_t>& icc, std::string& why) {
            if (!ok) return why = err, false;
            return icc = *held, true;
          };
        }
      } else {
        continue;
      }
      bundle.prints.push_back(std::move(p));
    }
  }

  // The film stock's spectra, for a print material given as a file (np_print.h). A block that is not whole is left
  // out: such a print then has no matrix and says so.
  if (j.contains("film_model") && j["film_model"].is_object()) {
    const nlohmann::json& fm = j["film_model"];
    auto numbers = [](const nlohmann::json& a, std::vector<double>& out) {
      if (!a.is_array()) return false;
      out.clear();
      for (const auto& v : a) {
        if (!v.is_number() || !std::isfinite(v.get<double>())) return false;
        out.push_back(v.get<double>());
      }
      return true;
    };
    FilmModel film;
    film.name = fm.value("name", std::string());
    bool ok = fm.contains("dye_density") && fm["dye_density"].is_array() && fm["dye_density"].size() == 3 &&
              fm.contains("neutral_dye_amounts") && fm["neutral_dye_amounts"].is_array() &&
              fm["neutral_dye_amounts"].size() == 3 && fm.contains("base_density") &&
              numbers(fm["base_density"], film.base) && film.base.size() == (size_t)kSpectralSamples &&
              fm.contains("log_exposure") && numbers(fm["log_exposure"], film.log_exposure) &&
              film.log_exposure.size() >= 2 && fm.contains("wavelength_nm") && fm["wavelength_nm"].is_object() &&
              fm["wavelength_nm"].value("start", 0.0) == kSpectralFirstNm &&
              fm["wavelength_nm"].value("step", 0.0) == kSpectralStepNm;
    for (int c = 0; ok && c < 3; ++c) {
      ok = numbers(fm["dye_density"][c], film.dye[c]) && film.dye[c].size() == film.base.size() &&
           numbers(fm["neutral_dye_amounts"][c], film.neutral[c]) &&
           film.neutral[c].size() == film.log_exposure.size();
    }
    film.present = ok;
    if (ok) bundle.film_model = std::move(film);
  }

  // The colour path the bundle was measured in. Bundles from before the field are camera_raw_rgb.
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
  } else {
    // The correction is the inverse of the light's mixing (DESIGN.md 2.4), so it is invertible, and it is applied as
    // floats: the float matrix's determinant is a normal float. A matrix given (not null) that is not such a 3x3
    // refuses the bundle: left out, the scan would be converted without its correction.
    const nlohmann::json* m = nullptr;
    if (j.contains("crosstalk_profile") && !j["crosstalk_profile"].is_object() && !j["crosstalk_profile"].is_null()) {
      refuse("crosstalk_profile", j["crosstalk_profile"], "is not an object");
    }
    if (j.contains("crosstalk_profile") && j["crosstalk_profile"].is_object()) {
      if (j["crosstalk_profile"].contains("crosstalk_correction_matrix")) {
        m = &j["crosstalk_profile"]["crosstalk_correction_matrix"];
      }
    } else if (j.contains("crosstalk_correction_matrix")) {
      m = &j["crosstalk_correction_matrix"];
    }
    if (m && !m->is_null()) {
      bool ok = parse_3x3_matrix(*m, bundle.crosstalk_matrix, bundle.crosstalk_matrix_d);
      for (int k = 0; ok && k < 9; ++k) {
        ok = std::fabs(bundle.crosstalk_matrix_d[k]) <= (double)std::numeric_limits<float>::max();
      }
      ok = ok && crosstalk_matrix_invertible(bundle.crosstalk_matrix);
      if (!ok) {
        throw std::runtime_error("profile bundle " + raw_path + ": the crosstalk correction matrix is not an "
                                 "invertible 3x3 of numbers within float range (its float determinant a normal "
                                 "float)");
      }
      bundle.has_crosstalk = true;
    }
  }

  // An exposure (the film base capture's, a bracket's) is shutter x ISO / 100 in seconds, taken as a float and divided
  // by: the ISO is above 0 and both are within float range, and the product neither overflows nor rounds to 0.
  static const char* const kBadExposure = "is not a positive float (shutter x ISO / 100, in seconds)";
  auto exposure_ok = [](const std::string& shutter, double iso) {
    const double t = shutter_to_seconds_d(shutter);
    const double fmax = (double)std::numeric_limits<float>::max();
    if (!(iso > 0.0 && iso <= fmax && t <= fmax)) return false;
    const float e = (float)t * (float)iso / 100.0f;
    return e > 0.0f && e <= std::numeric_limits<float>::max();
  };

  // A shutter speed is a string ("1/8s") or a number of seconds, read as written (dump() keeps every digit).
  auto shutter_text = [&](const std::string& field, const nlohmann::json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_number()) return v.dump();
    refuse(field, v, "is not a shutter speed (a string or a number of seconds)");
    return std::string();
  };

  if (j.contains("film_base") && !j["film_base"].is_object()) refuse("film_base", j["film_base"], "is not an object");
  if (j.contains("film_base")) {
    const auto& fb = j["film_base"];
    // Each channel's base is a mean of light through the film: every channel is divided by it, and it is rounded to
    // an int (profile_film_base_rgb) that is divided by too, so it rounds to 1 at least and fits an int.
    auto base = [&](const char* c) {
      const bool given = fb.contains(c) && fb[c].is_object() && fb[c].contains("avg") && fb[c]["avg"].is_number();
      const double v = given ? fb[c]["avg"].get<double>() : 0.0;
      if (!(v >= 0.5 && v <= (double)std::numeric_limits<int>::max())) {
        throw std::runtime_error("profile bundle " + raw_path + ": film_base." + c + ".avg " +
                                 (given ? fb[c]["avg"].dump() : std::string("(not a number)")) +
                                 " is not in [0.5, 2147483647]");
      }
      return v;
    };
    const double r_avg = base("r"), g_avg = base("g"), b_avg = base("b");
    bundle.film_base_avg[0] = (float)r_avg;
    bundle.film_base_avg[1] = (float)g_avg;
    bundle.film_base_avg[2] = (float)b_avg;
    bundle.film_base_avg_d[0] = r_avg;
    bundle.film_base_avg_d[1] = g_avg;
    bundle.film_base_avg_d[2] = b_avg;
    if (fb.contains("shutter")) {
      bundle.film_base_shutter = shutter_text("film_base.shutter", fb["shutter"]);
      (void)shutter_to_seconds_d(bundle.film_base_shutter);
    }
    double iso = bundle.film_base_iso;
    if (fb.contains("iso")) {
      if (!fb["iso"].is_number()) refuse("film_base.iso", fb["iso"], "is not a number");
      iso = fb["iso"].get<double>();
    }
    if (!exposure_ok(bundle.film_base_shutter, iso)) {
      throw std::runtime_error("profile bundle " + raw_path + ": the film base's exposure (shutter " +
                               bundle.film_base_shutter + ", ISO " + std::to_string(iso) + ") " + kBadExposure);
    }
    bundle.film_base_iso = (float)iso;
    bundle.profile_film_base_rgb[0] = (int)std::round(r_avg);
    bundle.profile_film_base_rgb[1] = (int)std::round(g_avg);
    bundle.profile_film_base_rgb[2] = (int)std::round(b_avg);
    bundle.has_film_base = true;
  }

  if (!j.contains("targets") || !j["targets"].is_array()) {
    throw std::runtime_error("profile bundle " + raw_path + " has no targets array");
  }

  bundle.targets.clear();
  for (size_t ti = 0; ti < j["targets"].size(); ++ti) {
    const nlohmann::json& tj = j["targets"][ti];
    const std::string at = "targets[" + std::to_string(ti) + "]";
    if (!tj.is_object()) refuse(at, tj, "is not an object");
    BundleTarget tgt;
    tgt.index = (int)bundle.targets.size();
    tgt.name = "Target " + std::to_string(tgt.index + 1);
    if (tj.contains("name")) {
      if (!tj["name"].is_string()) refuse(at + ".name", tj["name"], "is not a string");
      tgt.name = tj["name"].get<std::string>();
    }
    if (tj.contains("iso")) {
      if (!tj["iso"].is_number()) refuse(at + ".iso", tj["iso"], "is not a number");
      tgt.iso = tj["iso"].get<double>();
    }
    if (tj.contains("shutter")) tgt.shutter = shutter_text(at + ".shutter", tj["shutter"]);

    for (const char* key : {"target_ev", "ev"}) {
      if (!tj.contains(key)) continue;
      if (!tj[key].is_number()) refuse(at + "." + key, tj[key], "is not a number");
      if (!tgt.has_target_ev) {
        tgt.has_target_ev = true;
        tgt.target_ev = tj[key].get<double>();
      }
    }

    if (!exposure_ok(tgt.shutter, tgt.iso)) {
      throw std::runtime_error("profile bundle " + raw_path + ": the exposure of target " + tgt.name + " (shutter " +
                               tgt.shutter + ", ISO " + std::to_string(tgt.iso) + ") " + kBadExposure);
    }
    tgt.exposure_s = std::max(1e-6f, shutter_to_seconds(tgt.shutter) * float(tgt.iso) / 100.0f);

    if (tj.contains("patches") && !tj["patches"].is_object()) {
      refuse(at + ".patches", tj["patches"], "is not an object");
    }
    if (tj.contains("patches")) {
      std::vector<float> gs_g;
      for (auto it = tj["patches"].begin(); it != tj["patches"].end(); ++it) {
        const std::string pat = at + ".patches." + it.key();
        if (!it.value().is_object()) refuse(pat, it.value(), "is not an object");
        // A patch without a green is not counted. Kept as a float (the grey wedge below reads these same entries).
        if (!it.value().contains("g")) continue;
        const nlohmann::json& gv = it.value()["g"];
        if (!gv.is_number() || !(std::fabs(gv.get<double>()) <= (double)std::numeric_limits<float>::max())) {
          refuse(pat + ".g", gv, "is not a number within float range");
        }
        tgt.patches_g.push_back((float)gv.get<double>());
      }
      // Tonal capacity of the grey wedge gs0..gs23, or of all patches without one.
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
    if (tj.contains("profcheck_output")) {
      const nlohmann::json& v = tj["profcheck_output"];
      if (!v.is_string()) refuse(at + ".profcheck_output", v, "is not a string");
      tgt.profcheck_output = v.get<std::string>();
    }
    for (const char* key : {"icc_profile_base64", "icc_profile"}) {
      if (tj.contains(key) && !tj[key].is_string()) refuse(at + "." + key, tj[key], "is not a string");
    }

    // A bracket whose ICC is given (inline, or as a file) but cannot be read as a film profile refuses the bundle, as
    // an ICC file that cannot be read does: a bundle short of a bracket would search the others. Base64 that does not
    // decode, an empty profile and a refused layout are refused alike. A target given no ICC is skipped below.
    bool have_icc = false;
    if (tj.contains("icc_profile_base64") && tj["icc_profile_base64"].is_string()) {
      have_icc = true;
      if (!base64_decode(tj["icc_profile_base64"].get<std::string>(), tgt.icc_bytes)) {
        throw std::runtime_error("profile bundle " + raw_path + ": the inline ICC profile of target " + tgt.name +
                                 " is not valid base64");
      }
    } else if (tj.contains("icc_profile") && tj["icc_profile"].is_string()) {
      const std::string icc_name = tj["icc_profile"].get<std::string>();
      if (!read_file || !read_file(icc_name, tgt.icc_bytes)) {
        throw std::runtime_error("profile bundle " + raw_path + ": cannot read ICC profile " + icc_name +
                                 " of target " + tgt.name);
      }
      tgt.icc_file = icc_name;
      have_icc = true;
    }
    if (have_icc) {
      cmsHPROFILE hprof = tgt.icc_bytes.empty()
                              ? nullptr
                              : cmsOpenProfileFromMem(tgt.icc_bytes.data(), (cmsUInt32Number)tgt.icc_bytes.size());
      const bool parsed = hprof && parse_film_profile_pipeline(hprof, tgt.film_prof);
      if (hprof) cmsCloseProfile(hprof);
      if (!parsed) {
        throw std::runtime_error("profile bundle " + raw_path + ": the ICC profile of target " + tgt.name +
                                 (tgt.icc_bytes.empty() ? " is empty" : " cannot be read as a film profile"));
      }
      tgt.film_prof.rebind_pointers();
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

// --------------------------------------------------------------------------------------------------------------------
// Profiles
// --------------------------------------------------------------------------------------------------------------------

bool is_directory(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_directory(path, ec);
}

bool is_regular_file(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

static bool path_exists_file_or_dir(const std::string& path) {
  return is_regular_file(path) || is_directory(path);
}

std::string resolve_profile_path(const std::string& input_path) {
  if (path_exists_file_or_dir(input_path)) {
    return input_path;
  }
  std::string alt = "profiles/" + input_path;
  if (path_exists_file_or_dir(alt)) {
    return alt;
  }
  if (input_path.rfind("icc_out/", 0) == 0) {
    std::string remapped = "profiles/" + input_path.substr(8);
    if (path_exists_file_or_dir(remapped)) {
      return remapped;
    }
  }
  size_t last_slash = input_path.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    std::string base = "profiles/" + input_path.substr(last_slash + 1);
    if (path_exists_file_or_dir(base)) {
      return base;
    }
  }
  return input_path;
}

bool load_film_profile(const std::string& raw_path, ParsedFilmProfile& prof) {
  std::string resolved_path = resolve_profile_path(raw_path);
  cmsHPROFILE hprof = cmsOpenProfileFromFile(resolved_path.c_str(), "r");
  if (!hprof) {
    log_error("Cannot open film ICC profile: %s\n", raw_path.c_str());
    return false;
  }
  if (!parse_film_profile_pipeline(hprof, prof)) {
    log_error("Film ICC profile %s cannot be read as a film profile\n", resolved_path.c_str());
    cmsCloseProfile(hprof);
    return false;
  }
  cmsCloseProfile(hprof);
  log_info("Reading input ICC profile: %s\n", resolved_path.c_str());
  return true;
}

bool load_bundle_arg(const std::string& arg, ProfileBundle& bundle) {
  std::string resolved = resolve_profile_path(arg);
  if (is_directory(resolved)) resolved = (std::filesystem::path(resolved) / "bundle.json").string();
  if (!is_regular_file(resolved) || !is_json_bundle(resolved)) return false;
  load_profile_bundle(resolved, bundle);
  return true;
}

int read_profile(const std::string& prof_name, unsigned **prof_out, unsigned *size) {
  // The profile's size is the first field of its header (ICC.1 7.2.2): it must be the file's size, checked before
  // anything is allocated, and the bytes must open as a profile (Little CMS checks the header and the tag table).
  std::string resolved = resolve_profile_path(prof_name);
  std::error_code ec;
  const bool regular = std::filesystem::is_regular_file(resolved, ec);
  const uintmax_t file_size = regular ? std::filesystem::file_size(resolved, ec) : 0;
  FILE *fp = regular && !ec ? fopen(resolved.c_str(), "rb") : nullptr;
  if (fp) {
    uint32_t declared = 0;
    if (fread(&declared, 4, 1, fp) == 1 && (declared = ntohl(declared)) == file_size) {
      fseek(fp, 0, SEEK_SET);
      *size = declared;
      *prof_out = (unsigned *)malloc(*size);
      if (*prof_out && fread(*prof_out, 1, *size, fp) == *size) {
        cmsHPROFILE prof = cmsOpenProfileFromMem(*prof_out, *size);
        if (prof) {
          cmsCloseProfile(prof);
          fclose(fp);
          return 0;
        }
      }
      if (*prof_out) free(*prof_out);
      *prof_out = nullptr;
    }
    fclose(fp);
  }
  log_error("Cannot read ICC profile: %s\n", prof_name.c_str());
  return -1;
}

// --------------------------------------------------------------------------------------------------------------------
// Input matrix
// --------------------------------------------------------------------------------------------------------------------

template <class T>
static float dot_product(const std::vector<float>& v1, const std::vector<T>& v2) {
  float prod = 0;
  for (size_t i = 0; i < v1.size() && i < v2.size(); ++i) {
    prod += v1[i] * v2[i];
  }
  return prod;
}

void adjust_correction_matrix(const std::vector<float>& r_coef, 
                              const std::vector<float>& g_coef,
                              const std::vector<float>& b_coef,
                              float global_exposure_comp,
                              float gain_g,
                              float gain_b,
                              const std::vector<int>& profile_film_base_rgb,
                              const std::vector<int>& film_base_rgb,
                              float* merged_matrix) {
  // First convert the linear RGB into corrected RGB values that are proportional
  // to transmittance.
  float cc_average_r = dot_product(r_coef, film_base_rgb);
  float cc_average_g = dot_product(g_coef, film_base_rgb);
  float cc_average_b = dot_product(b_coef, film_base_rgb);
  float cc_profile_r = dot_product(r_coef, profile_film_base_rgb);
  float cc_profile_g = dot_product(g_coef, profile_film_base_rgb);
  float cc_profile_b = dot_product(b_coef, profile_film_base_rgb);
  log_info("Film base RGB (corrected): %f %f %f\n", cc_average_r, cc_average_g, cc_average_b);
  log_info("Profile film base RGB (corrected): %f %f %f\n", cc_profile_r, cc_profile_g, cc_profile_b);

  // The ICC profiles map the transmittance values after the correction matrix is
  // applied to XYZ. Transmittance is relative to the profile film base. For example
  // if the R value for profile film base is 10000, a R value of 5000 has
  // transmittance of 5000 / 10000 = 0.5. As the corrected RGB is not converted
  // into a floating point value first, the actual denominator is not important.
  // Suppose we record the transmittance as a 16-bit fixed point number, we will
  // lose precision for denser film. So the ICC profiles are generated with
  // transmittance values scaled given a fixed value for mid-grey patch (e.g.
  // GS14).
  //
  // The film base RGB values are only useful in relative terms. Here we are
  // compensating the difference in transmittance between the profile and the
  // scanned film.
  //
  // Suppose the mid-grey patch has R = 5000 and G = 10000. With film base
  // R = 10000 and film base G = 20000, the transmittance is 0.5 in both channels.
  //
  // A different captured film with R = 5000 (film base R = 10000) has a
  // transmittance of 0.5 in the R channel. Let's say the G layer is denser
  // because of fluctuation in the film development. G = 8000 (film base G = 16000)
  // will also have a transmittance of 0.5. However using the ICC profile which
  // assumes film base G is 20000 in the G channel, will treat the transmittance
  // as 8000 / 20000 = 0.4 and the color balance will be off.
  //
  // To compensate for this we will scale the G channel (with R as reference)
  // by (20000 / 10000) / (16000 / 10000) = 1.25.
  //
  // Of course the density even for R is not the same in the target and the
  // captured film. After adjusting the color balance, all channels can be
  // adjusted with a single scaling factor.
  float g_scale = 1.0f;
  float b_scale = 1.0f;
  if (cc_average_g > 0.0f && cc_average_r > 0.0f && cc_profile_r > 0.0f && cc_average_b > 0.0f) {
    g_scale = (cc_profile_g / cc_profile_r) / (cc_average_g / cc_average_r);
    b_scale = (cc_profile_b / cc_profile_r) / (cc_average_b / cc_average_r);
  }
  log_info("Scale channels to match film base: %f %f %f\n", 1.0, g_scale, b_scale);

  const float sr = global_exposure_comp;
  const float sg = g_scale * global_exposure_comp * gain_g;
  const float sb = b_scale * global_exposure_comp * gain_b;

  for (int i = 0; i < 3; ++i) {
    merged_matrix[i]     = r_coef[i] * sr;
    merged_matrix[3 + i] = g_coef[i] * sg;
    merged_matrix[6 + i] = b_coef[i] * sb;
  }
  log_info("R coefficients: %1.5f %1.5f %1.5f\n", merged_matrix[0], merged_matrix[1], merged_matrix[2]);
  log_info("G coefficients: %1.5f %1.5f %1.5f\n", merged_matrix[3], merged_matrix[4], merged_matrix[5]);
  log_info("B coefficients: %1.5f %1.5f %1.5f\n", merged_matrix[6], merged_matrix[7], merged_matrix[8]);
}
