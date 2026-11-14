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


#include "np_density.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#include <nlohmann/json.hpp>

#include "np_log.h"

void density_encode_in_place(uint16_t* pixels, int width, int height, int channels, const float merged[9],
                             double base_level, const double m[9], const DensityEncoding& enc, DensityClip& clip) {
  const double inv_base = 1.0 / base_level;
  const double offset = enc.offset, scale = enc.scale;
  const long max_code = enc.max_code();
  uint64_t low[3] = {0, 0, 0}, high[3] = {0, 0, 0}, floor[3] = {0, 0, 0};
  #pragma omp parallel for schedule(static) reduction(+ : low[:3], high[:3], floor[:3])
  for (int j = 0; j < height; ++j) {
    uint16_t* row = pixels + (size_t)j * width * channels;
    for (int i = 0; i < width; ++i) {
      uint16_t* px = row + (size_t)i * channels;
      const double r = px[0] / 65535.0, g = px[1] / 65535.0, b = px[2] / 65535.0;
      double d[3];
      for (int c = 0; c < 3; ++c) {
        const double v = (double)merged[c * 3] * r + (double)merged[c * 3 + 1] * g + (double)merged[c * 3 + 2] * b;
        const double t = v * inv_base;
        if (!(t > kDensityMinT)) ++floor[c];
        d[c] = -std::log10(std::max(t, kDensityMinT));
      }
      for (int c = 0; c < 3; ++c) {
        const double dens = m[c * 3] * d[0] + m[c * 3 + 1] * d[1] + m[c * 3 + 2] * d[2];
        long code = std::lround(offset + scale * enc.k[c] * dens);
        if (code < 0) {
          code = 0;
          ++low[c];
        } else if (code > max_code) {
          code = max_code;
          ++high[c];
        }
        px[c] = (uint16_t)code;
      }
    }
  }
  for (int c = 0; c < 3; ++c) {
    clip.low[c] = low[c];
    clip.high[c] = high[c];
    clip.floor[c] = floor[c];
  }
  clip.pixels = (uint64_t)width * (uint64_t)height;
}

std::string density_metadata_json(const DensityRecord& rec, int indent) {
  using nlohmann::ordered_json;
  ordered_json j;
  j["schema"] = "negicc.density/1";
  j["density"] = density_measure_name(rec.measure);
  if (rec.matrix && !rec.matrix->target_curves.empty()) j["target_curves"] = rec.matrix->target_curves;
  j["encoding"] = {{"container", rec.container},
                   {"bits", rec.encoding.bits},
                   {"offset", rec.encoding.offset},
                   {"scale", rec.encoding.scale},
                   {"channel_factors", {rec.encoding.k[0], rec.encoding.k[1], rec.encoding.k[2]}},
                   {"standard", rec.measure == kDensityApd ? "ADX, SMPTE ST 2065-3 (the channel factors applied)"
                                                           : (rec.measure == kDensityRp180 ? "Cineon scaling" : "none")},
                   {"formula", "code = offset + scale * k * D, D the density above the film base, k the channel factor"}};
  j["profile_bundle"] = rec.bundle_key;
  j["film_base"] = {{"rgb", {rec.film_base_rgb[0], rec.film_base_rgb[1], rec.film_base_rgb[2]}},
                    {"source", rec.film_base_source}};
  if (rec.matrix) {
    const double* m = rec.matrix->m;
    j["density_matrix"] = {{m[0], m[1], m[2]}, {m[3], m[4], m[5]}, {m[6], m[7], m[8]}};
  }
  if (rec.bundle) {
    if (!rec.bundle->density_fit_json.empty()) {
      j["density_fit"] = ordered_json::parse(rec.bundle->density_fit_json, nullptr, false);
    }
    if (!rec.bundle->light_source_json.empty()) {
      j["light_source"] = ordered_json::parse(rec.bundle->light_source_json, nullptr, false);
    }
  }
  if (rec.clip) {
    const DensityClip& c = *rec.clip;
    j["clipped_pixels"] = {{"below_code_0", {c.low[0], c.low[1], c.low[2]}},
                           {"above_max_code", {c.high[0], c.high[1], c.high[2]}},
                           {"at_density_floor", {c.floor[0], c.floor[1], c.floor[2]}},
                           {"pixels", c.pixels}};
  }
  double unused[3];
  if (density_balance_offset(rec, unused)) {
    // The picture conversion multiplies the light's green and blue channels by g and b to balance the frame; in
    // density that is an offset per channel. Through the measure's matrix, and with their mean taken out so that the
    // file's overall level stays, the offsets are the colour balance alone. The exposure E is recorded and left to
    // the grade: it is relative to the picture conversion's film profile, which a density file does not use.
    double measure[3];
    density_balance_offset(rec, measure);
    j["suggested_balance"] = {
        {"from", "the picture conversion's parameter search on this frame; not applied to the densities"},
        {"exposure", rec.balance_e},
        {"gain_g", rec.balance_g},
        {"gain_b", rec.balance_b},
        {"bracket", rec.balance_target},
        {"density_offset", {measure[0], measure[1], measure[2]}},
        {"code_offset",
         {measure[0] * rec.encoding.scale * rec.encoding.k[0], measure[1] * rec.encoding.scale * rec.encoding.k[1],
          measure[2] * rec.encoding.scale * rec.encoding.k[2]}},
        {"use", "D + density_offset (code + code_offset) has the colour balance the picture conversion gives this "
                "frame; the offsets sum to zero, so the overall level is unchanged and the exposure is left to the "
                "grade"}};
  }
  if (rec.bundle && !rec.bundle->print_slopes_json.empty() && rec.measure != kDensityStatusM) {
    // The film's slopes on each print material, for this measure, as the bundle records them. Not applied.
    const ordered_json all = ordered_json::parse(rec.bundle->print_slopes_json, nullptr, false);
    const char* key = density_measure_key(rec.measure);
    if (all.is_object() && all.contains(key) && all[key].is_object() && !all[key].empty()) {
      ordered_json ps;
      for (auto it = all.begin(); it != all.end(); ++it) {
        if (!it.value().is_object()) ps[it.key()] = it.value();      // what they are, the formula, how they were measured
      }
      ps["applied"] = "no: the densities are as read. The CLF written beside the file for each print material "
                      "(<output without its extension>.<print>.clf) applies them, with the suggested balance; apply "
                      "them once, there or by hand, not both";
      ps["prints"] = all[key];
      if (all.contains("bracket_grey_density") && all["bracket_grey_density"].is_object() &&
          all["bracket_grey_density"].contains(key)) {
        ps["bracket_grey_density"] = all["bracket_grey_density"][key];
      }
      j["print_slopes"] = ps;
    }
  }
  j["tool"] = rec.tool_version;
  return j.dump(indent);
}

bool density_balance_offset(const DensityRecord& rec, double offset[3]) {
  offset[0] = offset[1] = offset[2] = 0.0;
  if (!(rec.has_balance && rec.balance_e > 0.0 && rec.balance_g > 0.0 && rec.balance_b > 0.0)) return false;
  const double light[3] = {0.0, -std::log10(rec.balance_g), -std::log10(rec.balance_b)};
  for (int c = 0; c < 3; ++c) offset[c] = light[c];
  if (rec.matrix) {
    const double* m = rec.matrix->m;
    for (int c = 0; c < 3; ++c) offset[c] = m[c * 3] * light[0] + m[c * 3 + 1] * light[1] + m[c * 3 + 2] * light[2];
  }
  const double mean = (offset[0] + offset[1] + offset[2]) / 3.0;
  for (int c = 0; c < 3; ++c) offset[c] -= mean;
  return true;
}

std::vector<PrintSlopes> density_print_slopes(const ProfileBundle& bundle, DensityMeasure measure) {
  std::vector<PrintSlopes> out;
  if (bundle.print_slopes_json.empty()) return out;
  const nlohmann::ordered_json all = nlohmann::ordered_json::parse(bundle.print_slopes_json, nullptr, false);
  const char* key = density_measure_key(measure);
  if (!all.is_object() || !all.contains(key) || !all[key].is_object()) return out;
  for (auto it = all[key].begin(); it != all[key].end(); ++it) {
    const nlohmann::ordered_json& n = it.value();
    PrintSlopes p;
    p.id = it.key();
    bool ok = n.is_object() && n.contains("to_print_density") && n["to_print_density"].is_array() &&
              n["to_print_density"].size() == 3 && n.contains("slopes") && n["slopes"].is_array() &&
              n["slopes"].size() == 3 && n.contains("about_density") && n["about_density"].is_array() &&
              n["about_density"].size() == 3;
    for (int r = 0; ok && r < 3; ++r) {
      const nlohmann::ordered_json& row = n["to_print_density"][r];
      ok = row.is_array() && row.size() == 3 && n["slopes"][r].is_number() && n["about_density"][r].is_number();
      for (int c = 0; ok && c < 3; ++c) {
        ok = row[c].is_number();
        if (ok) p.to_print[r * 3 + c] = row[c].get<double>();
      }
      if (ok) {
        p.slopes[r] = n["slopes"][r].get<double>();
        p.about[r] = n["about_density"][r].get<double>();
        ok = std::isfinite(p.slopes[r]) && p.slopes[r] > 0.0 && std::isfinite(p.about[r]);
      }
    }
    if (!ok) continue;
    p.name = n.value("print", p.id);
    out.push_back(p);
  }
  return out;
}

bool density_frame_grey(const DensityRecord& rec, double grey[3]) {
  grey[0] = grey[1] = grey[2] = 0.0;
  if (!rec.bundle || rec.bundle->print_slopes_json.empty() || !rec.has_balance || !(rec.balance_e > 0.0)) return false;
  const nlohmann::ordered_json all = nlohmann::ordered_json::parse(rec.bundle->print_slopes_json, nullptr, false);
  const char* key = density_measure_key(rec.measure);
  if (!all.is_object() || !all.contains("bracket_grey_density")) return false;
  const nlohmann::ordered_json& table = all["bracket_grey_density"];
  if (!table.is_object() || !table.contains(key) || !table[key].is_object() ||
      !table[key].contains(rec.balance_target)) {
    return false;
  }
  const nlohmann::ordered_json& g = table[key][rec.balance_target];
  if (!g.is_array() || g.size() != 3 || !g[0].is_number() || !g[1].is_number() || !g[2].is_number()) return false;
  // The exposure E multiplies the light's channels before the bracket's profile reads them: the film that the
  // profile reads as its mid grey is denser by log10(E) in every light channel, through the measure's matrix.
  const double shift = std::log10(rec.balance_e);
  for (int c = 0; c < 3; ++c) {
    double row = 1.0;
    if (rec.matrix) row = rec.matrix->m[c * 3] + rec.matrix->m[c * 3 + 1] + rec.matrix->m[c * 3 + 2];
    grey[c] = g[c].get<double>() + shift * row;
  }
  return true;
}

std::string density_print_clf(const DensityRecord& rec, const PrintSlopes& print, const std::string& print_clf) {
  // x: the file's code / max code.  D_j = (x_j * max - offset) / (scale * k_j);  with the balance bal and the frame's
  // mid grey,  pd_i = kPrintGreyDensity + s_i * sum_j P_ij (D_j + bal_j - grey_j).
  const size_t at = print_clf.find("  <Matrix");
  if (at == std::string::npos) return std::string();
  const DensityEncoding& e = rec.encoding;
  const double max_code = (double)e.max_code();
  double bal[3];
  const bool balanced = density_balance_offset(rec, bal);
  double grey[3];
  const bool exposed = density_frame_grey(rec, grey);
  if (!exposed) {
    for (int c = 0; c < 3; ++c) grey[c] = print.about[c];
  }
  double a[9], b[3];
  for (int i = 0; i < 3; ++i) {
    double shift = 0.0;
    for (int j = 0; j < 3; ++j) {
      const double p = print.slopes[i] * print.to_print[i * 3 + j];
      a[i * 3 + j] = p * max_code / (e.scale * e.k[j]);
      shift += p * (bal[j] - grey[j] - e.offset / (e.scale * e.k[j]));
    }
    b[i] = kPrintGreyDensity + shift;
  }
  char buf[1024];
  snprintf(buf, sizeof buf,
           "  <Description>This frame (%s, %s density in %d bits, code = %d + %d k D) printed on %s. The first Matrix "
           "takes the file's code values to the print's printing density with the frame's colour balance, density "
           "offsets %+.4f %+.4f %+.4f%s, and the film's slopes on this print material, %.4f %.4f %.4f about the "
           "frame's mid grey, density %.4f %.4f %.4f, which prints as mid grey: %s. The density file holds none of "
           "this. Add a constant to the three offsets of the first Matrix to print lighter (+) or darker (-), 0.30 a "
           "stop.</Description>\n",
           rec.bundle_key.c_str(), density_measure_name(rec.measure), e.bits, e.offset, e.scale, print.name.c_str(),
           bal[0], bal[1], bal[2], balanced ? "" : " (no balance was found: none applied)", print.slopes[0],
           print.slopes[1], print.slopes[2], grey[0], grey[1], grey[2],
           exposed ? "the chart's mid grey on the bracket and at the exposure the picture conversion's search gave "
                     "this frame"
                   : "the chart's mid grey at normal exposure (no exposure was found for the frame)");
  std::string s = print_clf.substr(0, at);
  s += buf;
  s += "  <Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <Array dim=\"3 4\">\n";
  for (int i = 0; i < 3; ++i) {
    snprintf(buf, sizeof buf, "      %.10g %.10g %.10g %.10g\n", a[i * 3], a[i * 3 + 1], a[i * 3 + 2], b[i]);
    s += buf;
  }
  s += "    </Array>\n  </Matrix>\n";
  s += print_clf.substr(at);
  return s;
}

namespace {

std::string date_time(long long timestamp, char sep) {
  time_t t = timestamp > 0 ? (time_t)timestamp : time(nullptr);
  struct tm tm_buf;
  localtime_r(&t, &tm_buf);
  char buf[80];
  snprintf(buf, sizeof buf, "%04d:%02d:%02d%c%02d:%02d:%02d", tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
           sep, tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
  return buf;
}

// ---- TIFF (little-endian, one strip) ---------------------------------------------------------------------------------
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

// ---- DPX (big-endian) ------------------------------------------------------------------------------------------------
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
  if (strip_bytes > 0xffffff00ull) {
    log_error("The image is too large for a TIFF file (%dx%d)\n", width, height);
    return -1;
  }
  // The values that do not fit an entry, laid out after the IFD.
  const std::string date = date_time(info.timestamp, ' ');
  std::vector<TiffEntry> entries;
  std::vector<uint8_t> extra;
  const int n_entries = 18 + (info.make.empty() ? 0 : 1) + (info.model.empty() ? 0 : 1);
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
  ascii(306, date);                                        // DateTime
  if ((int)entries.size() != n_entries) {
    log_error("Internal error: the TIFF directory has %d entries, expected %d\n", (int)entries.size(), n_entries);
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
  if (channels == 3) {
    const size_t n = (size_t)width * height;
    ok = ok && fwrite(pixels, 6, n, fp) == n;
  } else {
    std::vector<uint16_t> row((size_t)width * 3);
    for (int j = 0; ok && j < height; ++j) {
      const uint16_t* src = pixels + (size_t)j * width * channels;
      for (int i = 0; i < width; ++i) {
        row[(size_t)i * 3] = src[(size_t)i * channels];
        row[(size_t)i * 3 + 1] = src[(size_t)i * channels + 1];
        row[(size_t)i * 3 + 2] = src[(size_t)i * channels + 2];
      }
      ok = fwrite(row.data(), 6, (size_t)width, fp) == (size_t)width;
    }
  }
  ok = (fclose(fp) == 0) && ok;
  if (!ok) log_error("Failed to write %s\n", path.c_str());
  return ok ? 0 : -1;
}

int write_density_dpx(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                      const std::string& description, const DensityEncoding& enc, DensityMeasure measure,
                      const DensityFileInfo& info) {
  if (enc.bits != 10) {
    log_error("Internal error: a DPX holds 10-bit density codes\n");
    return -1;
  }
  // The user-defined data: a 32-byte identification, then the metadata JSON; the image starts on a multiple of 8.
  const uint32_t kHeader = 2048, kUserId = 32, kMaxUser = 1024 * 1024;
  uint32_t user_size = kUserId + (uint32_t)description.size() + 1;
  user_size = (user_size + 7u) & ~7u;
  if (user_size > kMaxUser) {
    log_error("The density metadata (%u bytes) does not fit a DPX file's user data (1 MB)\n", user_size);
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
  put_str(p + 136, 24, date_time(info.timestamp, ':'));   // the capture's time, as the TIFF's DateTime
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
  if (!ok) log_error("Failed to write %s\n", path.c_str());
  return ok ? 0 : -1;
}
