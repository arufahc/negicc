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


#include "np_print.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "cie_tables.h"
#include "embedded_data.h"
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
      // The bundle's three LED spectra (led_spectra) are left out when they are the all-on spectrum cut at its band
      // splits: the spectrum and the splits, both recorded, give them back. Spectra measured one LED at a time stay.
      ordered_json& light = j["light_source"];
      if (light.is_object() && light.contains("led_spectra") && light["led_spectra"].contains("band_splits_nm")) {
        light.erase("led_spectra");
      }
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
    double measure[3], mean = 0.0;
    density_balance_offset(rec, measure, &mean);
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
        {"density_offset_mean_removed", mean},
        {"use", "D + density_offset (code + code_offset) has the colour balance the picture conversion gives this "
                "frame; the offsets sum to zero, so the overall level is unchanged and the exposure is left to the "
                "grade. The gains also shift the level, by density_offset_mean_removed in every channel; the print "
                "CLF beside the file, which sets the exposure, applies density_offset + density_offset_mean_removed"}};
  }
  // How the frame prints on each print material (below), computed for this frame and not applied to it.
  if (!rec.print_json.empty()) j["print"] = ordered_json::parse(rec.print_json, nullptr, false);
  j["tool"] = rec.tool_version;
  return j.dump(indent);
}

bool density_balance_offset(const DensityRecord& rec, double offset[3], double* mean_out) {
  offset[0] = offset[1] = offset[2] = 0.0;
  if (mean_out) *mean_out = 0.0;
  if (!(rec.has_balance && rec.balance_e > 0.0 && rec.balance_g > 0.0 && rec.balance_b > 0.0)) return false;
  const double light[3] = {0.0, -std::log10(rec.balance_g), -std::log10(rec.balance_b)};
  for (int c = 0; c < 3; ++c) offset[c] = light[c];
  if (rec.matrix) {
    const double* m = rec.matrix->m;
    for (int c = 0; c < 3; ++c) offset[c] = m[c * 3] * light[0] + m[c * 3 + 1] * light[1] + m[c * 3 + 2] * light[2];
  }
  const double mean = (offset[0] + offset[1] + offset[2]) / 3.0;
  for (int c = 0; c < 3; ++c) offset[c] -= mean;
  if (mean_out) *mean_out = mean;
  return true;
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


// ---- The print of a density file (np_print.h) ----------------------------------------------------------------------

namespace {

bool read_vector(const nlohmann::json& j, std::vector<double>& out) {
  if (!j.is_array()) return false;
  out.clear();
  for (const auto& v : j) {
    if (!v.is_number()) return false;
    out.push_back(v.get<double>());
    if (!std::isfinite(out.back())) return false;
  }
  return true;
}

// numpy.interp: linear between the points, the end values outside them.
double interp(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
  if (x <= xs.front()) return ys.front();
  if (x >= xs.back()) return ys.back();
  const size_t hi = std::upper_bound(xs.begin(), xs.end(), x) - xs.begin();
  const double x0 = xs[hi - 1], x1 = xs[hi];
  return ys[hi - 1] + (ys[hi] - ys[hi - 1]) * (x - x0) / (x1 - x0);
}

void amounts_to_xyz(const PrintMaterial& m, const double amount[3], double xyz[3]) {
  xyz[0] = xyz[1] = xyz[2] = 0.0;
  const size_t n = m.base.size();
  for (size_t w = 0; w < n; ++w) {
    const double t =
        std::pow(10.0, -(amount[0] * m.dye[0][w] + amount[1] * m.dye[1][w] + amount[2] * m.dye[2][w] + m.base[w]));
    xyz[0] += t * m.observer[0][w];
    xyz[1] += t * m.observer[1][w];
    xyz[2] += t * m.observer[2][w];
  }
}

}  // namespace

namespace {

// A column of spektrafilm data (numbers and nulls) on its own wavelengths w, resampled to 380-780 nm at 1 nm. Nulls
// before the first and after the last number take the nearest number (hold) or `missing`; a null between numbers is
// an error (density_matrices.py hold_gaps).
bool resample_column(const nlohmann::json& rows, int column, const std::vector<double>& w, bool hold, double missing,
                     bool as_power_of_ten, std::vector<double>& out) {
  if (!rows.is_array() || rows.size() != w.size()) return false;
  std::vector<double> v(w.size(), std::nan(""));
  for (size_t i = 0; i < w.size(); ++i) {
    const nlohmann::json& cell = column < 0 ? rows[i] : (rows[i].is_array() && (int)rows[i].size() > column
                                                             ? rows[i][column] : nlohmann::json());
    if (cell.is_number() && std::isfinite(cell.get<double>())) v[i] = cell.get<double>();
  }
  size_t first = 0, last = w.size();
  while (first < w.size() && std::isnan(v[first])) ++first;
  while (last > first && std::isnan(v[last - 1])) --last;
  if (first == w.size()) return false;
  if (hold) {
    for (size_t i = first; i < last; ++i) {
      if (std::isnan(v[i])) return false;
    }
    for (size_t i = 0; i < first; ++i) v[i] = v[first];
    for (size_t i = last; i < w.size(); ++i) v[i] = v[last - 1];
  }
  for (double& x : v) x = std::isnan(x) ? missing : as_power_of_ten ? std::pow(10.0, x) : x;
  out.resize(kCieSamples);
  for (int i = 0; i < kCieSamples; ++i) out[i] = interp(w, v, kCieFirstNm + i);
  return true;
}

void multiply3(const double a[9], const double b[9], double out[9]) {
  double r[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  }
  std::copy(r, r + 9, out);
}

bool invert3(const double m[9], double out[9]) {
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                     m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (!std::isfinite(det) || std::fabs(det) < 1e-12) return false;
  const double r[9] = {m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
                       m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
                       m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3]};
  for (int k = 0; k < 9; ++k) out[k] = r[k] / det;
  return true;
}

// XYZ of the print for a log exposure of each layer.
void exposure_to_xyz(const PrintMaterial& m, const double log_e[3], double xyz[3]) {
  double amount[3];
  for (int c = 0; c < 3; ++c) amount[c] = interp(m.log_exposure, m.curves[c], log_e[c]);
  amounts_to_xyz(m, amount, xyz);
}

// Q: the 3x3 from a film's dye amounts to the density above the film base that response (rows R, G, B, 401 samples)
// reads, least squares with no offset on the film model's fixed sample colours.
bool film_to_print_density(const FilmModel& film, const std::vector<double> response[3], double q[9]) {
  const double kSpread = 0.15 * std::sqrt(1.5);   // each dye amount times exp(kSpread z), z = -1, 0, 1
  double sum[3] = {0, 0, 0}, base_density[3];
  for (int c = 0; c < 3; ++c) {
    double t = 0.0;
    for (int w = 0; w < kCieSamples; ++w) {
      sum[c] += response[c][w];
      t += std::pow(10.0, -film.base[w]) * response[c][w];
    }
    if (!(sum[c] > 0.0)) return false;
    base_density[c] = -std::log10(t / sum[c]);
  }
  double ata[9] = {0}, atd[9] = {0};            // sums of a a^T and of a d^T over the samples
  std::vector<double> transmittance(kCieSamples);
  for (size_t i = 0; i < film.log_exposure.size(); ++i) {
    if (film.log_exposure[i] < -1.5 || film.log_exposure[i] > 1.0) continue;
    for (int v = 0; v < 27; ++v) {
      const int z[3] = {v / 9 - 1, v / 3 % 3 - 1, v % 3 - 1};
      double a[3], d[3];
      for (int c = 0; c < 3; ++c) a[c] = std::max(0.0, film.neutral[c][i] * std::exp(kSpread * z[c]));
      for (int w = 0; w < kCieSamples; ++w) {
        transmittance[w] = std::pow(
            10.0, -(a[0] * film.dye[0][w] + a[1] * film.dye[1][w] + a[2] * film.dye[2][w] + film.base[w]));
      }
      for (int c = 0; c < 3; ++c) {
        double t = 0.0;
        for (int w = 0; w < kCieSamples; ++w) t += transmittance[w] * response[c][w];
        d[c] = -std::log10(t / sum[c]) - base_density[c];
      }
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          ata[r * 3 + c] += a[r] * a[c];
          atd[r * 3 + c] += a[r] * d[c];
        }
      }
    }
  }
  double inverse[9], solved[9];
  if (!invert3(ata, inverse)) return false;
  multiply3(inverse, atd, solved);              // solved[k][c]: the weight of amount k in density c
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) q[r * 3 + c] = solved[c * 3 + r];
  }
  return true;
}

}  // namespace

bool check_spektrafilm_print(const nlohmann::json& j, std::string& err) {
  if (!j.is_object() || !j.contains("data") || !j["data"].is_object()) return err = "not a spektrafilm profile (no data)", false;
  if (!j.contains("info") || !j["info"].is_object()) return err = "a spektrafilm profile without info", false;
  const nlohmann::json& info = j["info"];
  // A print material: spektrafilm also has camera films (stage "filming"), which would compute without complaint.
  const std::string stage = info.contains("stage") && info["stage"].is_string() ? info["stage"].get<std::string>() : "";
  if (stage != "printing") {
    return err = stage.empty() ? "a spektrafilm profile that does not say it is a print material (info.stage)"
                               : "a spektrafilm profile of a " + stage + " stock, not of a print material", false;
  }
  // The viewing light comes from the support: D50 for a paper, D55 for a print film.
  const std::string support = info.contains("support") && info["support"].is_string() ? info["support"].get<std::string>() : "";
  if (support != "paper" && support != "film") {
    return err = "a spektrafilm print profile whose info.support is '" + support +
                 "', not paper or film: the viewing light is chosen by it", false;
  }
  // Each layer's curve rises with exposure: by kMinCurveRise over the profile, never falling by more than
  // kMaxCurveDip between points (spektrafilm's curves wobble by about 0.0003 where the print is at its minimum).
  const nlohmann::json& d = j["data"];
  if (!d.contains("density_curves") || !d["density_curves"].is_array() || d["density_curves"].size() < 2) {
    return err = "a spektrafilm print profile without density_curves", false;
  }
  constexpr double kMinCurveRise = 0.5, kMaxCurveDip = 0.02;
  for (int c = 0; c < 3; ++c) {
    double first = 0.0, last = 0.0, prev = 0.0;
    for (size_t i = 0; i < d["density_curves"].size(); ++i) {
      const nlohmann::json& row = d["density_curves"][i];
      if (!row.is_array() || row.size() != 3 || !row[c].is_number() || !std::isfinite(row[c].get<double>())) {
        return err = "density_curves of the spektrafilm print profile has a missing value", false;
      }
      const double v = row[c].get<double>();
      if (i == 0) first = v;
      if (i > 0 && v < prev - kMaxCurveDip) {
        return err = "layer " + std::to_string(c) + "'s curve of the spektrafilm print profile falls with exposure", false;
      }
      prev = last = v;
    }
    if (!(last - first > kMinCurveRise)) {
      return err = "layer " + std::to_string(c) + "'s curve of the spektrafilm print profile does not rise with exposure", false;
    }
  }
  // Each layer is sensitive somewhere: its peak log sensitivity within kMaxSensitivityGap (log10) of the most
  // sensitive layer's, so no layer of the print is blind to the printer's light.
  constexpr double kMaxSensitivityGap = 6.0;
  if (!d.contains("log_sensitivity") || !d["log_sensitivity"].is_array()) {
    return err = "a spektrafilm print profile without log_sensitivity", false;
  }
  double peak[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (const nlohmann::json& row : d["log_sensitivity"]) {
    for (int c = 0; row.is_array() && row.size() == 3 && c < 3; ++c) {
      if (row[c].is_number() && std::isfinite(row[c].get<double>())) peak[c] = std::max(peak[c], row[c].get<double>());
    }
  }
  const double best = std::max(peak[0], std::max(peak[1], peak[2]));
  for (int c = 0; c < 3; ++c) {
    if (!(peak[c] > best - kMaxSensitivityGap)) {
      return err = "layer " + std::to_string(c) + " of the spektrafilm print profile has no spectral sensitivity", false;
    }
  }
  return true;
}

bool load_spektrafilm_print(const std::string& json_text, const std::string& id, const ProfileBundle& bundle,
                            PrintMaterial& out, std::string& err) {
  const nlohmann::json j = nlohmann::json::parse(json_text, nullptr, false);
  if (!check_spektrafilm_print(j, err)) return false;
  const nlohmann::json& d = j["data"];
  const nlohmann::json& info = j["info"];
  PrintMaterial m;
  m.id = info.value("stock", id);
  m.name = info.value("name", m.id);
  m.viewing_illuminant = info["support"] == "paper" ? "D50" : "D55";
  std::vector<double> w, sensitivity[3];
  bool ok = d.contains("wavelengths") && read_vector(d["wavelengths"], w) && w.size() >= 2 &&
            std::is_sorted(w.begin(), w.end()) && d.contains("log_sensitivity") && d.contains("channel_density") &&
            d.contains("base_density") && d.contains("log_exposure") && read_vector(d["log_exposure"], m.log_exposure) &&
            m.log_exposure.size() >= 2 && std::is_sorted(m.log_exposure.begin(), m.log_exposure.end()) &&
            d.contains("density_curves") && d["density_curves"].is_array() &&
            d["density_curves"].size() == m.log_exposure.size() &&
            resample_column(d["base_density"], -1, w, true, 0.0, false, m.base);
  for (int c = 0; ok && c < 3; ++c) {
    // A layer's sensitivity where the profile has none is zero; its dye spectrum is held at the nearest value.
    ok = resample_column(d["log_sensitivity"], c, w, false, 0.0, true, sensitivity[c]) &&
         resample_column(d["channel_density"], c, w, true, 0.0, false, m.dye[c]);
    m.curves[c].resize(m.log_exposure.size());
    for (size_t i = 0; ok && i < m.log_exposure.size(); ++i) {
      const nlohmann::json& row = d["density_curves"][i];
      ok = row.is_array() && row.size() == 3 && row[c].is_number() && std::isfinite(row[c].get<double>());
      if (ok) m.curves[c][i] = row[c].get<double>();
    }
  }
  if (!ok) {
    err = "a field of the spektrafilm print profile is missing or malformed";
    return false;
  }
  // What the print's own density is read with: each layer's sensitivity under the printer lamp, a 3200 K blackbody.
  std::vector<double> response[3];
  for (int c = 0; c < 3; ++c) {
    response[c].resize(kCieSamples);
    for (int i = 0; i < kCieSamples; ++i) {
      const double metres = (kCieFirstNm + i) * 1e-9;
      const double lamp = 1.0 / (std::pow(metres, 5) * (std::exp(1.4388e-2 / (metres * 3200.0)) - 1.0));
      response[c][i] = sensitivity[c][i] * lamp;
    }
  }
  // The observer under the viewing light, Y of the light 1; the print's white is the light itself.
  const double* light = m.viewing_illuminant == "D50" ? kCieD50 : kCieD55;
  const double* cmf[3] = {kCieXBar, kCieYBar, kCieZBar};
  double light_y = 0.0;
  for (int i = 0; i < kCieSamples; ++i) light_y += light[i] * kCieYBar[i];
  for (int c = 0; c < 3; ++c) {
    m.observer[c].resize(kCieSamples);
    m.white[c] = 0.0;
    for (int i = 0; i < kCieSamples; ++i) {
      m.observer[c][i] = light[i] * cmf[c][i] / light_y;
      m.white[c] += m.observer[c][i];
    }
  }
  // XYZ under the viewing light to linear sRGB, adapted to D65 with CAT02.
  const double cat02[9] = {0.7328, 0.4296, -0.1624, -0.7036, 1.6975, 0.0061, 0.0030, 0.0136, 0.9834};
  const double xyz_to_srgb[9] = {3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108,
                                 0.0415560, 0.0556434,  -0.2040259, 1.0572252};
  const double d65[3] = {0.3127 / 0.3290, 1.0, (1.0 - 0.3127 - 0.3290) / 0.3290};
  double cat_inverse[9], scaled[9];
  invert3(cat02, cat_inverse);
  for (int r = 0; r < 3; ++r) {
    const double to = cat02[r * 3] * d65[0] + cat02[r * 3 + 1] * d65[1] + cat02[r * 3 + 2] * d65[2];
    const double from = cat02[r * 3] * m.white[0] + cat02[r * 3 + 1] * m.white[1] + cat02[r * 3 + 2] * m.white[2];
    for (int c = 0; c < 3; ++c) scaled[r * 3 + c] = cat02[r * 3 + c] * to / from;
  }
  multiply3(cat_inverse, scaled, scaled);
  multiply3(xyz_to_srgb, scaled, m.to_srgb);
  // aim: the exposure of each layer at which the print is an 18 % grey under its light. Newton's method on the three
  // exposures, with the Jacobian by differences.
  double e[3] = {0.0, 0.0, 0.0};
  for (int iteration = 0; iteration < 50; ++iteration) {
    double xyz[3], f[3], jac[9], inverse[9];
    exposure_to_xyz(m, e, xyz);
    double worst = 0.0;
    for (int c = 0; c < 3; ++c) {
      f[c] = xyz[c] - 0.18 * m.white[c];
      worst = std::max(worst, std::fabs(f[c]));
    }
    if (worst < 1e-13) break;
    for (int k = 0; k < 3; ++k) {
      double moved[3] = {e[0], e[1], e[2]}, xyz2[3];
      moved[k] += 1e-4;
      exposure_to_xyz(m, moved, xyz2);
      for (int c = 0; c < 3; ++c) jac[c * 3 + k] = (xyz2[c] - xyz[c]) / 1e-4;
    }
    if (!invert3(jac, inverse)) {
      err = "the print's curves do not reach an 18 % grey";
      return false;
    }
    for (int k = 0; k < 3; ++k) e[k] -= inverse[k * 3] * f[0] + inverse[k * 3 + 1] * f[1] + inverse[k * 3 + 2] * f[2];
  }
  double check[3];
  exposure_to_xyz(m, e, check);
  for (int c = 0; c < 3; ++c) {
    if (!(std::fabs(check[c] - 0.18 * m.white[c]) < 1e-6)) {
      err = "the print's curves do not reach an 18 % grey";
      return false;
    }
    m.aim[c] = e[c];
  }
  // The print matrix of each measure, for this bundle's film: Q inv(F), with any light_density_factors of the measure
  // undone on the light's channels (Q inv(F) + Q L diag(1 / alpha - 1) inv(L) inv(F) = Q L inv(M)) so that P M = Q L
  // for every measure and the bracket's uncalibrated APD slopes apply once.
  // py-ref: negicc_station/src/print_emulation.py (print_matrix)
  auto print_matrix = [](const double q[9], const DensityMatrix& dm, double p[9]) {
    double inv_f[9];
    if (!dm.present || !invert3(dm.film_to_measure, inv_f)) return false;
    multiply3(q, inv_f, p);
    const double* alpha = dm.light_density_factors;
    if (alpha[0] == 1.0 && alpha[1] == 1.0 && alpha[2] == 1.0) return true;
    double inv_l[9];
    if (!invert3(dm.light_to_film, inv_l)) return false;
    double ql[9], inv_lf[9], scaled[9], corr[9];
    multiply3(q, dm.light_to_film, ql);
    multiply3(inv_l, inv_f, inv_lf);
    for (int r = 0; r < 3; ++r) {
      const double d = 1.0 / alpha[r] - 1.0;
      for (int c = 0; c < 3; ++c) scaled[r * 3 + c] = d * inv_lf[r * 3 + c];
    }
    multiply3(ql, scaled, corr);
    for (int k = 0; k < 9; ++k) p[k] += corr[k];
    return true;
  };
  double q[9];
  if (bundle.film_model.present && film_to_print_density(bundle.film_model, response, q)) {
    for (int k = 0; k < kDensityMeasures; ++k) {
      PrintMaterial::ToPrint t;
      if (!print_matrix(q, bundle.density_matrix[k], t.m.data())) continue;
      t.film = bundle.film_model.name;
      m.to_print[k].push_back(t);
    }
  }
  out = std::move(m);
  return true;
}

std::string print_clf_nodes(const PrintMaterial& m, int curve_points, int cube) {
  char buf[256];
  std::string s;
  s.reserve((size_t)cube * cube * cube * 40 + (size_t)curve_points * 40 + 4096);
  // log E = aim - (D - grey)
  s += "  <Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <Array dim=\"3 4\">\n";
  for (int i = 0; i < 3; ++i) {
    snprintf(buf, sizeof buf, "      %d %d %d %.10g\n", i == 0 ? -1 : 0, i == 1 ? -1 : 0, i == 2 ? -1 : 0,
             m.aim[i] + m.grey_density);
    s += buf;
  }
  s += "    </Array>\n  </Matrix>\n";
  const double lo = m.log_exposure.front(), hi = m.log_exposure.back();
  snprintf(buf, sizeof buf,
           "  <Range inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <minInValue>%.10g</minInValue>\n"
           "    <maxInValue>%.10g</maxInValue>\n    <minOutValue>0</minOutValue>\n    <maxOutValue>1</maxOutValue>\n"
           "  </Range>\n",
           lo, hi);
  s += buf;
  snprintf(buf, sizeof buf, "  <LUT1D inBitDepth=\"32f\" outBitDepth=\"32f\" interpolation=\"linear\">\n"
                            "    <Array dim=\"%d 3\">\n", curve_points);
  s += buf;
  double top = 0.0;
  for (int c = 0; c < 3; ++c) top = std::max(top, *std::max_element(m.curves[c].begin(), m.curves[c].end()));
  top = std::ceil(top * 10.0) / 10.0;
  for (int i = 0; i < curve_points; ++i) {
    const double x = lo + (hi - lo) * i / (curve_points - 1);
    snprintf(buf, sizeof buf, "      %.6g %.6g %.6g\n", interp(m.log_exposure, m.curves[0], x),
             interp(m.log_exposure, m.curves[1], x), interp(m.log_exposure, m.curves[2], x));
    s += buf;
  }
  s += "    </Array>\n  </LUT1D>\n";
  snprintf(buf, sizeof buf, "  <Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <Array dim=\"3 3\">\n"
                            "      %.10g 0 0\n      0 %.10g 0\n      0 0 %.10g\n    </Array>\n  </Matrix>\n",
           1.0 / top, 1.0 / top, 1.0 / top);
  s += buf;
  snprintf(buf, sizeof buf, "  <LUT3D inBitDepth=\"32f\" outBitDepth=\"32f\" interpolation=\"tetrahedral\">\n"
                            "    <Array dim=\"%d %d %d 3\">\n", cube, cube, cube);
  s += buf;
  // The cube: dye amounts 0..top on each axis, blue changing fastest, to linear sRGB.
  const size_t points = (size_t)cube * cube * cube;
  std::vector<double> rgb(points * 3);
  #pragma omp parallel for schedule(static)
  for (long long idx = 0; idx < (long long)points; ++idx) {
    const int b = (int)(idx % cube), g = (int)((idx / cube) % cube), r = (int)(idx / ((long long)cube * cube));
    const double amount[3] = {top * r / (cube - 1), top * g / (cube - 1), top * b / (cube - 1)};
    double xyz[3];
    amounts_to_xyz(m, amount, xyz);
    for (int c = 0; c < 3; ++c) {
      rgb[idx * 3 + c] = m.to_srgb[c * 3] * xyz[0] + m.to_srgb[c * 3 + 1] * xyz[1] + m.to_srgb[c * 3 + 2] * xyz[2];
    }
  }
  for (size_t idx = 0; idx < points; ++idx) {
    snprintf(buf, sizeof buf, "      %.6g %.6g %.6g\n", rgb[idx * 3], rgb[idx * 3 + 1], rgb[idx * 3 + 2]);
    s += buf;
  }
  s += "    </Array>\n  </LUT3D>\n";
  s += "  <Exponent inBitDepth=\"32f\" outBitDepth=\"32f\" style=\"monCurveRev\">\n"
       "    <ExponentParams exponent=\"2.4\" offset=\"0.055\" />\n  </Exponent>\n";
  return s;
}

// The soft knee the white-light conversion renders through (multi_target_optimizer.cc AUTO_KNEE).
constexpr double kCurveKnee = 0.95;

std::vector<std::array<double, 3>> fitted_grey_scale(const DensityRecord& rec,
                                                     const std::vector<std::array<double, 3>>& steps) {
  const double n0 = (rec.bundle ? rec.bundle->normalization_target : 55000.0) / 65535.0;
  const double gains[3] = {rec.balance_e, rec.balance_e * rec.balance_g, rec.balance_e * rec.balance_b};
  // color_pipeline.h knee_compress inverted: the identity up to the knee, then knee + delta tanh((w - knee) / delta).
  auto unknee = [](double v) {
    if (v <= kCurveKnee) return v;
    const double delta = 1.0 - kCurveKnee;
    return kCurveKnee + delta * std::atanh(std::min((v - kCurveKnee) / delta, 0.998));   // knee_compress caps at 0.9999
  };
  std::vector<std::array<double, 3>> out(steps.size());
  for (size_t i = 0; i < steps.size(); ++i) {
    for (int c = 0; c < 3; ++c) {
      double v = n0 * std::pow(10.0, -steps[i][c]);      // what the profile renders as this step
      const int ch = c - 1;                              // the record's curves: green, blue
      if (ch >= 0 && rec.has_curves && (rec.curve_a[ch] != 0.0 || rec.curve_k[ch] != 0.0) && v < rec.curve_ref[ch]) {
        // input_curves.h curve_stops, in double, inverted by bisection.
        const double a = rec.curve_a[ch], k = rec.curve_k[ch], med = rec.curve_u_med[ch], hold = rec.curve_u_hold[ch];
        auto f = [&](double x) {
          if (x < hold) return hold * (1.0 + a + k * std::tanh(hold - med)) + (x - hold);
          return x * (1.0 + a + k * std::tanh(x - med));
        };
        const double target = std::log2(v / rec.curve_ref[ch]);
        double lo = -64.0, hi = 0.0;
        for (int it = 0; it < 80; ++it) {
          const double mid = 0.5 * (lo + hi);
          if (f(mid) < target) lo = mid; else hi = mid;
        }
        v = rec.curve_ref[ch] * std::exp2(0.5 * (lo + hi));
      }
      out[i][c] = -std::log10(unknee(v) / (gains[c] * n0));
    }
  }
  return out;
}

bool frame_print(const DensityRecord& rec, const PrintMaterial& material, FramePrint& out, std::string& err) {
  if (!rec.bundle || !rec.matrix) return err = "no bundle", false;
  const ProfileBundle& bundle = *rec.bundle;
  // The film model the bundle's density matrices were fitted on names the material's matrix for this film.
  const nlohmann::json fit_record = nlohmann::json::parse(bundle.density_fit_json, nullptr, false);
  const std::string film = fit_record.is_object() ? fit_record.value("film_dye_spectra", std::string()) : std::string();
  const PrintMaterial::ToPrint* to_print = material.to_print_for((int)rec.measure, film);
  if (!to_print) return err = "no print matrix for this measure and this film", false;
  if (bundle.targets.empty()) return err = "the bundle has no brackets", false;
  FramePrint p;
  p.id = material.id;
  p.name = material.name;
  p.grey_density = material.grey_density;
  for (int k = 0; k < 9; ++k) p.to_print[k] = to_print->m[k];
  p.to_print_for = to_print->film;
  // The frame's bracket and exposure: the search's, else the middle bracket at its own exposure.
  p.searched = rec.has_balance && rec.balance_e > 0.0;
  p.bracket = p.searched ? rec.balance_target : bundle.targets[bundle.targets.size() / 2].name;
  p.exposure = p.searched ? rec.balance_e : 1.0;
  const DensityBracket* bracket = nullptr;
  for (const DensityBracket& b : bundle.density_brackets) {
    if (b.name == p.bracket) bracket = &b;
  }
  if (!bracket) {
    return err = "the bundle has no density_matrices.brackets entry for " + p.bracket +
                 ": rebuild it with the profile-building tools", false;
  }
  // The exposure E multiplies the light's channels before the bracket's profile reads them: the film that the
  // profile reads as its mid grey is denser by log10(E) in every light channel, through the measure's matrix.
  // py-ref: negicc_station/src/print_emulation.py (frame_grey)
  // A narrow-band bundle with the film's grey balance curve: the mid grey and the slopes are the curve's, as in its
  // print ICCs (curve_grey, curve_slopes).
  // py-ref: negicc_station/src/print_icc.py (curve_grey, curve_slopes)
  const double* m = rec.matrix->m;
  const double shift = std::log10(p.exposure);
  double light[3] = {bracket->grey_light[0], bracket->grey_light[1], bracket->grey_light[2]};
  for (int c = 0; c < 3; ++c) p.slopes[c] = bracket->slopes[c];
  if (bundle.grey_balance.present && bundle.density_matrix[kDensityApd].present) {
    curve_grey(bundle, *bracket, light);
    curve_slopes(bundle, *bracket, p.slopes);
    p.from_curve = true;
  } else if (!bracket->has_slopes) {
    return err = "the bundle has neither grey_balance_curves nor slopes for " + p.bracket +
                 ": rebuild it with the profile-building tools", false;
  }
  for (int c = 0; c < 3; ++c) {
    p.grey[c] = 0.0;
    for (int k = 0; k < 3; ++k) p.grey[c] += m[c * 3 + k] * (light[k] + shift);
  }
  // A white-light frame with fitted curves: the mid grey and the slopes of the grey scale those curves describe.
  // py-ref: negicc_station/src/print_emulation.py (frame_print)
  const DensityMatrix& slopes_in = bundle.density_matrix[kDensityApd];
  if (p.searched && rec.has_curves && !bracket->grey_steps.empty() && slopes_in.present) {
    const std::vector<std::array<double, 3>> scale = fitted_grey_scale(rec, bracket->grey_steps);
    const std::array<double, 3>& mid = scale[bracket->grey_step];
    auto through = [](const double* mat, const double v[3], int row) {
      return mat[row * 3] * v[0] + mat[row * 3 + 1] * v[1] + mat[row * 3 + 2] * v[2];
    };
    // Slopes: 1 / (least-squares ratio of the red, or blue, to the green density of the steps, each less the mid
    // grey's), in Academy Printing Density, as the profile builder computes a bracket's.
    double num[3] = {0, 0, 0};
    for (const std::array<double, 3>& step : scale) {
      const double rel[3] = {step[0] - mid[0], step[1] - mid[1], step[2] - mid[2]};
      const double green = through(slopes_in.m, rel, 1);
      for (int c = 0; c < 3; ++c) num[c] += through(slopes_in.m, rel, c) * green;
    }
    if (num[0] > 0.0 && num[1] > 0.0 && num[2] > 0.0) {
      // The mid grey: the fitted mid step in the file's measure, less the gains' share, so that D + balance - grey is
      // zero for the film the conversion renders as mid grey (the balance is M [0, -log10 g, -log10 b]).
      const double gain_part[3] = {0.0, -std::log10(rec.balance_g), -std::log10(rec.balance_b)};
      for (int c = 0; c < 3; ++c) {
        p.grey[c] = through(m, mid.data(), c) + through(m, gain_part, c);
        p.slopes[c] = num[1] / num[c];
      }
      p.fitted = true;
    }
  }
  out = p;
  return true;
}

std::string frame_print_clf(const DensityRecord& rec, const FramePrint& print, const std::string& nodes) {
  // x: the file's code / max code.  D_j = (x_j * max - offset) / (scale * k_j);  with the balance bal and the frame's
  // mid grey,  pd_i = grey_density + s_i * sum_j P_ij (D_j + bal_j - grey_j).
  //
  // bal is the whole of what the search's gains g and b do to the density, level included (the zero-sum offsets and
  // the mean taken out of them). The conversion renders as the chart's mid grey the film that is denser than it by
  // log10 E, log10 E g and log10 E b in the light's channels; with grey carrying log10 E and bal the rest, that film
  // comes out at grey_density in all three channels.
  const DensityEncoding& e = rec.encoding;
  const double max_code = (double)e.max_code();
  double bal[3], bal_mean = 0.0;
  const bool balanced = density_balance_offset(rec, bal, &bal_mean);
  for (double& v : bal) v += bal_mean;
  double a[9], b[3];
  for (int i = 0; i < 3; ++i) {
    double shift = 0.0;
    for (int j = 0; j < 3; ++j) {
      const double p = print.slopes[i] * print.to_print[i * 3 + j];
      a[i * 3 + j] = p * max_code / (e.scale * e.k[j]);
      shift += p * (bal[j] - print.grey[j] - e.offset / (e.scale * e.k[j]));
    }
    b[i] = print.grey_density + shift;
  }
  // Names come from a bundle and a print material file: escaped, and the text built without a fixed buffer.
  auto xml = [](const std::string& text) {
    std::string out;
    for (char ch : text) {
      switch (ch) {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '"': out += "&quot;"; break;
        default: out += ch;
      }
    }
    return out;
  };
  auto num = [](const char* format, double a, double b, double c) {
    char text[96];
    snprintf(text, sizeof text, format, a, b, c);
    return std::string(text);
  };
  char buf[256];
  const std::string print_name = xml(print.name);
  std::string s = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  s += "<ProcessList id=\"negicc.print." + xml(print.id) + "\" name=\"" + print_name + "\" compCLFversion=\"3.0\">\n";
  snprintf(buf, sizeof buf, "%s density in %d bits, code = %d + %d k D", density_measure_name(rec.measure), e.bits,
           e.offset, e.scale);
  s += "  <Description>This frame (" + xml(rec.bundle_key) + ", " + buf + ") printed on " + print_name +
       ", as sRGB. The first Matrix takes the file's code values to the print's printing density with the frame's "
       "colour balance, density offsets " + num("%+.4f %+.4f %+.4f", bal[0], bal[1], bal[2]) +
       (balanced ? "" : " (no balance was found: none applied)") + ", and slopes " +
       num("%.4f %.4f %.4f", print.slopes[0], print.slopes[1], print.slopes[2]) +
       " about the frame's mid grey, density " + num("%.4f %.4f %.4f", print.grey[0], print.grey[1], print.grey[2]) +
       ". " +
       (print.fitted
            ? "The mid grey and the slopes are those of the grey scale the picture conversion renders on the bracket " +
                  xml(print.bracket) + " with this frame's exposure, gains and fitted input curves: the mid grey "
                  "prints as the print's mid grey, and the slopes give that grey scale equal contrast in the three "
                  "channels"
            : "The mid grey is the chart's on the bracket " + xml(print.bracket) +
                  (print.searched ? " and at the exposure the picture conversion's search gave this frame"
                                  : " (no search: the middle bracket)") +
                  "; balanced, it prints as the print's mid grey. The slopes are the bracket's in the film's profile: "
                  "they give the film's grey scale equal contrast in the three channels") +
       ". The density file holds none of this. Add a constant to the three offsets of the first Matrix to print "
       "lighter (+) or darker (-), 0.30 a stop.</Description>\n";
  s += "  <Description>The 1D and 3D tables are derived by neg_process from the print profile of this material in "
       "spektrafilm (Andrea Volpato, https://github.com/andreavolpato/spektrafilm): its curve per layer resampled, "
       "and its dye spectra computed into colours. CC BY-SA 4.0, "
       "https://github.com/andreavolpato/spektrafilm/blob/main/SPEKTRAFILM_LICENSE.txt. The matrices are computed "
       "by neg_process and are not spektrafilm's data. A model, not compared with a real print.</Description>\n";
  s += "  <InputDescriptor>density code values</InputDescriptor>\n  <OutputDescriptor>sRGB</OutputDescriptor>\n";
  s += "  <Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <Array dim=\"3 4\">\n";
  for (int i = 0; i < 3; ++i) {
    snprintf(buf, sizeof buf, "      %.10g %.10g %.10g %.10g\n", a[i * 3], a[i * 3 + 1], a[i * 3 + 2], b[i]);
    s += buf;
  }
  s += "    </Array>\n  </Matrix>\n";
  s += nodes;
  s += "</ProcessList>\n";
  return s;
}

bool is_print_file(const std::string& name) {
  return name.find_first_of("/\\") != std::string::npos ||
         (name.size() > 5 && name.substr(name.size() - 5) == ".json");
}

std::vector<PrintMaterial> builtin_print_materials(const ProfileBundle* bundle) {
  static const ProfileBundle no_bundle;
  const std::string folder = "3rd_party/spektrafilm/profiles/";
  std::vector<PrintMaterial> out;
  for (const EmbeddedFile& file : list_embedded_files()) {
    const std::string name = file.name;
    if (name.rfind(folder, 0) != 0 || name.size() < folder.size() + 6 || name.substr(name.size() - 5) != ".json") {
      continue;
    }
    PrintMaterial material;
    std::string err;
    if (!load_spektrafilm_print(std::string((const char*)file.data, file.size),
                                name.substr(folder.size(), name.size() - folder.size() - 5),
                                bundle ? *bundle : no_bundle, material, err)) {
      log_warn("Print profile %s cannot be read: %s\n", name.c_str(), err.c_str());
      continue;
    }
    out.push_back(std::move(material));
  }
  return out;
}

DensityPrints density_prints(const DensityRecord& rec, const std::string& output,
                             const std::vector<std::string>& print_ids) {
  DensityPrints out;
  if (print_ids.empty()) return out;
  // The materials asked for: built-in ones by id (or all of them), and spektrafilm print profiles by path, whose
  // matrices are computed here for the bundle's film.
  const bool all = std::find(print_ids.begin(), print_ids.end(), "all") != print_ids.end();
  std::vector<PrintMaterial> materials;
  for (PrintMaterial& material : builtin_print_materials(rec.bundle)) {
    if (all || std::find(print_ids.begin(), print_ids.end(), material.id) != print_ids.end()) {
      materials.push_back(std::move(material));
    }
  }
  for (const std::string& id : print_ids) {
    if (!is_print_file(id)) continue;
    std::ifstream file(id, std::ios::binary);
    std::stringstream text;
    text << file.rdbuf();
    const size_t slash = id.find_last_of("/\\");
    std::string stem = slash == std::string::npos ? id : id.substr(slash + 1);
    if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".json") stem.resize(stem.size() - 5);
    PrintMaterial material;
    std::string err;
    if (!file || !rec.bundle || !load_spektrafilm_print(text.str(), stem, *rec.bundle, material, err)) {
      log_warn("No print CLF for %s: %s\n", id.c_str(), !file ? "cannot be read" : err.c_str());
      continue;
    }
    materials.push_back(std::move(material));
  }
  if (!materials.empty() && (!rec.bundle || !rec.bundle->film_model.present)) {
    log_warn("No print CLF: the profile bundle has no film_model, which a print's matrix is computed from: rebuild "
             "it with the profile-building tools\n");
    return out;
  }
  for (PrintMaterial& material : materials) {
    FramePrint print;
    std::string err;
    // Two materials of one id would share a CLF name: the first is kept.
    const std::string path = output + "." + material.id + ".clf";
    if (std::find(out.paths.begin(), out.paths.end(), path) != out.paths.end()) {
      log_warn("--print names the print material %s twice: its CLF is written once, for the first\n",
               material.id.c_str());
      continue;
    }
    if (!frame_print(rec, material, print, err)) {
      log_warn("No print CLF for %s: %s\n", material.id.c_str(), err.c_str());
      continue;
    }
    out.paths.push_back(output + "." + material.id + ".clf");
    out.materials.push_back(std::move(material));
    out.frames.push_back(print);
  }
  if (out.frames.empty()) return out;
  using nlohmann::ordered_json;
  ordered_json j;
  const std::vector<FramePrint>* prints = &out.frames;
  {
    // How the frame prints on each print material: computed for this frame, applied by the CLF beside the file.
    ordered_json materials = ordered_json::object();
    for (const FramePrint& p : *prints) {
      const double* t = p.to_print;
      ordered_json one = {
          {"print", p.name},
          {"to_print_density", {{t[0], t[1], t[2]}, {t[3], t[4], t[5]}, {t[6], t[7], t[8]}}},
          {"to_print_density_for", p.to_print_for},
          {"bracket", p.bracket},
          {"bracket_from", p.searched ? "the picture conversion's search on this frame" : "the middle bracket (no search)"},
          {"exposure", p.exposure},
          {"frame_grey_density", {p.grey[0], p.grey[1], p.grey[2]}},
          {"slopes", {p.slopes[0], p.slopes[1], p.slopes[2]}},
          {"grey_and_slopes_from", p.fitted       ? "the grey scale this frame's fitted input curves describe"
                                   : p.from_curve ? "the film's grey balance curve at the bracket's mid grey"
                                                  : "the bracket's entry in the profile set"}};
      materials[p.id] = one;
    }
    j["print"] = {
        {"what", "how this frame prints on each print material: the frame's mid grey lands on the print's mid grey, "
                 "and the slopes give the film's grey scale equal contrast in the three channels. The bracket and the "
                 "exposure are the search's; the mid grey and the slopes are the film's grey balance curve's at the bracket's mid grey "
                 "(density_matrices.grey_balance_curves), or the bracket's in the profile set "
                 "(density_matrices.brackets), or, for a white-light frame whose input curves were fitted, those of "
                 "the grey scale the fitted curves describe (grey_and_slopes_from)"},
        {"formula", "print density = 0.70 + slopes * (to_print_density @ (D + balance - frame_grey_density)), "
                    "balance = suggested_balance.density_offset + density_offset_mean_removed"},
        {"applied", "no: the densities are as read. The CLF beside the file for each print material "
                    "(<output>.<print>.clf) applies the balance, the exposure and the slopes; apply them once, there "
                    "or by hand, not both"},
        {"materials", materials}};
  }
  out.metadata_json = j["print"].dump();
  return out;
}

void remove_print_clfs(const DensityPrints& prints) {
  for (const std::string& path : prints.paths) remove(path.c_str());
}

int write_print_clfs(const DensityRecord& rec, const DensityPrints& prints) {
  for (size_t k = 0; k < prints.frames.size(); ++k) {
    const FramePrint& print = prints.frames[k];
    const std::string clf = frame_print_clf(rec, print, print_clf_nodes(prints.materials[k]));
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

// ---- The print ICC --------------------------------------------------------------------------------------------------

namespace {

// The value of a table on start + step i at x, linear, the end values outside it (numpy.interp on a uniform grid).
double on_grid(const std::vector<double>& ys, double start, double step, double x) {
  const double t = (x - start) / step;
  if (t <= 0.0) return ys.front();
  if (t >= (double)(ys.size() - 1)) return ys.back();
  const size_t i = (size_t)t;
  return ys[i] + (t - (double)i) * (ys[i + 1] - ys[i]);
}

std::vector<double> grey_grid(const GreyBalanceCurves& g) {
  std::vector<double> x(g.apd[0].size());
  for (size_t i = 0; i < x.size(); ++i) x[i] = g.start + g.step * (double)i;
  return x;
}

// The print matrix for APD of a material, for the film of the bundle's density matrices; null when it has none.
const double* apd_print_matrix(const ProfileBundle& bundle, const PrintMaterial& m) {
  const nlohmann::json fit = nlohmann::json::parse(bundle.density_fit_json, nullptr, false);
  const std::string film = fit.is_object() ? fit.value("film_dye_spectra", std::string()) : std::string();
  const PrintMaterial::ToPrint* t = m.to_print_for(kDensityApd, film);
  return t ? t->m.data() : nullptr;
}

void put_u16(std::vector<uint8_t>& b, double v01) {
  const long q = std::lround(std::min(std::max(v01, 0.0), 1.0) * 65535.0);
  b.push_back((uint8_t)(q >> 8));
  b.push_back((uint8_t)(q & 255));
}
void put_u32(std::vector<uint8_t>& b, uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) b.push_back((uint8_t)(v >> s));
}
void put_s15(std::vector<uint8_t>& b, double v) { put_u32(b, (uint32_t)(int32_t)std::lround(v * 65536.0)); }
uint32_t get_u32(const std::vector<uint8_t>& b, size_t o) {
  return (uint32_t)b[o] << 24 | (uint32_t)b[o + 1] << 16 | (uint32_t)b[o + 2] << 8 | b[o + 3];
}
// The offset of an ICC's A2B0 lut16 tag, or 0.
size_t a2b0_lut16(const std::vector<uint8_t>& icc) {
  if (icc.size() < 132) return 0;
  const uint32_t n = get_u32(icc, 128);
  for (uint32_t i = 0; i < n && 144 + 12 * i <= icc.size(); ++i) {
    if (std::memcmp(icc.data() + 132 + 12 * i, "A2B0", 4) != 0) continue;
    const size_t off = get_u32(icc, 136 + 12 * i);
    if (off + 52 > icc.size() || std::memcmp(icc.data() + off, "mft2", 4) != 0 || icc[off + 8] != 3) return 0;
    return off;
  }
  return 0;
}

constexpr double kD50[3] = {0.9642, 1.0, 0.8249};
constexpr double kPrintKnee = 0.95;   // the conversion's soft knee (profile_bundle.h kKnee), as the station writes it
constexpr double kPcsMax = 1.0 + 32767.0 / 32768.0;   // the largest XYZ a lut16 holds

}  // namespace

double bracket_mid_apd(const ProfileBundle& bundle, const DensityBracket& bracket) {
  const double* m = bundle.density_matrix[kDensityApd].m;
  return m[3] * bracket.grey_light[0] + m[4] * bracket.grey_light[1] + m[5] * bracket.grey_light[2];
}

void curve_grey(const ProfileBundle& bundle, const DensityBracket& bracket, double light[3]) {
  const GreyBalanceCurves& g = bundle.grey_balance;
  const std::vector<double> x = grey_grid(g);
  const double a_mid = bracket_mid_apd(bundle, bracket);
  for (int c = 0; c < 3; ++c) light[c] = interp(g.apd[c], x, a_mid);
}

void curve_slopes(const ProfileBundle& bundle, const DensityBracket& bracket, double slopes[3]) {
  const GreyBalanceCurves& g = bundle.grey_balance;
  double n[3], inv[3], k[3] = {0, 0, 0};
  curve_grey(bundle, bracket, n);
  for (int c = 0; c < 3; ++c) {
    const double d = (on_grid(g.apd[c], g.start, g.step, n[c] + g.step) -
                      on_grid(g.apd[c], g.start, g.step, n[c] - g.step)) / (2.0 * g.step);
    inv[c] = 1.0 / d;
  }
  const double* m = bundle.density_matrix[kDensityApd].m;
  for (int c = 0; c < 3; ++c) k[c] = m[c * 3] * inv[0] + m[c * 3 + 1] * inv[1] + m[c * 3 + 2] * inv[2];
  slopes[0] = k[1] / k[0];
  slopes[1] = 1.0;
  slopes[2] = k[1] / k[2];
}

PrintCurves paper_curve(const PrintMaterial& m, const double p_apd[9]) {
  double top = 0.0;
  for (int c = 0; c < 3; ++c)
    for (double v : m.curves[c]) top = std::max(top, v);
  top = std::ceil(top * 10.0) / 10.0;
  PrintCurves out;
  for (int c = 0; c < 3; ++c) {
    const double k = p_apd[c * 3] + p_apd[c * 3 + 1] + p_apd[c * 3 + 2];
    out[c].resize(kPaperAlphaPoints);
    double run = INFINITY;
    for (int i = 0; i < kPaperAlphaPoints; ++i) {
      const double alpha = kPaperAlphaStart + kPaperAlphaStep * i;
      run = std::min(run, interp(m.log_exposure, m.curves[c], m.aim[c] - k * alpha) / top);
      out[c][i] = run - 1e-6 * i;
    }
  }
  return out;
}

PrintCurves bracket_print_curves(const GreyBalanceCurves& g, const PrintCurves& paper, double a_mid, double n0) {
  const double d = 1.0 - kPrintKnee;
  PrintCurves out;
  for (int c = 0; c < 3; ++c) {
    out[c].resize(kPrintIccPoints);
    double run = -INFINITY;
    for (int i = 0; i < kPrintIccPoints; ++i) {
      const double v = (double)i / (kPrintIccPoints - 1);
      const double u = v <= kPrintKnee ? v
                                       : kPrintKnee + d * std::atanh(std::min(std::max((v - kPrintKnee) / d, 0.0), 0.998));
      const double dl = -std::log10(std::max(u, 1e-7) / n0);
      const double alpha = on_grid(g.apd[c], g.start, g.step, dl) - a_mid;
      run = std::max(run, on_grid(paper[c], kPaperAlphaStart, kPaperAlphaStep, alpha));
      out[c][i] = run + 1e-9 * i;
    }
  }
  return out;
}

std::vector<double> print_table(const PrintMaterial& m, const double p_apd[9], const double m_apd[9],
                                const PrintCurves& paper) {
  // Q L = P M and w = M^-1 (1, 1, 1).
  double ql[9], inv[9], w[3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      ql[r * 3 + c] = p_apd[r * 3] * m_apd[c] + p_apd[r * 3 + 1] * m_apd[3 + c] + p_apd[r * 3 + 2] * m_apd[6 + c];
  {
    const double* a = m_apd;
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                       a[2] * (a[3] * a[7] - a[4] * a[6]);
    inv[0] = (a[4] * a[8] - a[5] * a[7]) / det;  inv[1] = (a[2] * a[7] - a[1] * a[8]) / det;
    inv[2] = (a[1] * a[5] - a[2] * a[4]) / det;  inv[3] = (a[5] * a[6] - a[3] * a[8]) / det;
    inv[4] = (a[0] * a[8] - a[2] * a[6]) / det;  inv[5] = (a[2] * a[3] - a[0] * a[5]) / det;
    inv[6] = (a[3] * a[7] - a[4] * a[6]) / det;  inv[7] = (a[1] * a[6] - a[0] * a[7]) / det;
    inv[8] = (a[0] * a[4] - a[1] * a[3]) / det;
  }
  for (int r = 0; r < 3; ++r) w[r] = inv[r * 3] + inv[r * 3 + 1] + inv[r * 3 + 2];
  // The PCS: the identity for a paper (D50), Bradford from the light's white to D50 for a print film.
  double pcs[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  if (m.viewing_illuminant != "D50") {
    static const double B[9] = {0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296};
    double src[3] = {m.white[0] / m.white[1], 1.0, m.white[2] / m.white[1]}, gain[3];
    for (int r = 0; r < 3; ++r) {
      const double to = B[r * 3] * kD50[0] + B[r * 3 + 1] * kD50[1] + B[r * 3 + 2] * kD50[2];
      const double from = B[r * 3] * src[0] + B[r * 3 + 1] * src[1] + B[r * 3 + 2] * src[2];
      gain[r] = to / from;
    }
    // inv(B) diag(gain) B, with inv(B) computed exactly rather than taken from the rounded table above.
    double bi[9];
    {
      const double* a = B;
      const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                         a[2] * (a[3] * a[7] - a[4] * a[6]);
      bi[0] = (a[4] * a[8] - a[5] * a[7]) / det;  bi[1] = (a[2] * a[7] - a[1] * a[8]) / det;
      bi[2] = (a[1] * a[5] - a[2] * a[4]) / det;  bi[3] = (a[5] * a[6] - a[3] * a[8]) / det;
      bi[4] = (a[0] * a[8] - a[2] * a[6]) / det;  bi[5] = (a[2] * a[3] - a[0] * a[5]) / det;
      bi[6] = (a[3] * a[7] - a[4] * a[6]) / det;  bi[7] = (a[1] * a[6] - a[0] * a[7]) / det;
      bi[8] = (a[0] * a[4] - a[1] * a[3]) / det;
    }
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) {
        double v = 0.0;
        for (int k = 0; k < 3; ++k) v += bi[r * 3 + k] * gain[k] * B[k * 3 + c];
        pcs[r * 3 + c] = v;
      }
  }
  // The nodes' alpha: Paper_c inverted (it is decreasing, so read reversed).
  std::vector<double> alpha_grid(kPaperAlphaPoints), node_alpha[3];
  for (int i = 0; i < kPaperAlphaPoints; ++i) alpha_grid[kPaperAlphaPoints - 1 - i] = kPaperAlphaStart + kPaperAlphaStep * i;
  for (int c = 0; c < 3; ++c) {
    const std::vector<double> rev(paper[c].rbegin(), paper[c].rend());
    for (int i = 0; i < kPrintIccCube; ++i)
      node_alpha[c].push_back(interp(rev, alpha_grid, (double)i / (kPrintIccCube - 1)));
  }
  std::vector<double> out((size_t)kPrintIccCube * kPrintIccCube * kPrintIccCube * 3);
  size_t o = 0;
  for (int i = 0; i < kPrintIccCube; ++i)
    for (int j = 0; j < kPrintIccCube; ++j)
      for (int k = 0; k < kPrintIccCube; ++k) {
        const double wa[3] = {node_alpha[0][i] * w[0], node_alpha[1][j] * w[1], node_alpha[2][k] * w[2]};
        double amount[3], xyz[3];
        for (int c = 0; c < 3; ++c) {
          const double pd = m.grey_density + ql[c * 3] * wa[0] + ql[c * 3 + 1] * wa[1] + ql[c * 3 + 2] * wa[2];
          amount[c] = interp(m.log_exposure, m.curves[c], m.aim[c] - (pd - m.grey_density));
        }
        amounts_to_xyz(m, amount, xyz);
        for (int c = 0; c < 3; ++c) out[o++] = pcs[c * 3] * xyz[0] + pcs[c * 3 + 1] * xyz[1] + pcs[c * 3 + 2] * xyz[2];
      }
  return out;
}

std::vector<uint8_t> write_print_icc(const PrintCurves& curves, const std::vector<double>& table,
                                     const std::string& description, const std::string& copyright) {
  const int n = (int)curves[0].size();
  std::vector<uint8_t> a2b0 = {'m', 'f', 't', '2', 0, 0, 0, 0, 3, 3, (uint8_t)kPrintIccCube, 0};
  for (int i = 0; i < 9; ++i) put_s15(a2b0, i % 4 == 0 ? 1.0 : 0.0);
  a2b0.push_back((uint8_t)(n >> 8));
  a2b0.push_back((uint8_t)(n & 255));
  a2b0.push_back(0);
  a2b0.push_back(2);
  for (int c = 0; c < 3; ++c)
    for (double v : curves[c]) put_u16(a2b0, v);
  for (double v : table) put_u16(a2b0, v / kPcsMax);
  for (int c = 0; c < 3; ++c) {
    put_u16(a2b0, 0.0);
    put_u16(a2b0, 1.0);
  }
  std::vector<uint8_t> desc = {'d', 'e', 's', 'c', 0, 0, 0, 0};
  put_u32(desc, (uint32_t)description.size() + 1);
  desc.insert(desc.end(), description.begin(), description.end());
  desc.insert(desc.end(), 1 + 8 + 3 + 67, 0);
  std::vector<uint8_t> cprt = {'t', 'e', 'x', 't', 0, 0, 0, 0};
  cprt.insert(cprt.end(), copyright.begin(), copyright.end());
  cprt.push_back(0);
  std::vector<uint8_t> wtpt = {'X', 'Y', 'Z', ' ', 0, 0, 0, 0};
  for (double v : kD50) put_s15(wtpt, v);
  const std::vector<std::pair<const char*, const std::vector<uint8_t>*>> tags = {
      {"desc", &desc}, {"cprt", &cprt}, {"wtpt", &wtpt}, {"A2B0", &a2b0}};
  std::vector<uint8_t> table_bytes, body;
  const size_t start = 128 + 4 + 12 * tags.size();
  for (const auto& t : tags) {
    table_bytes.insert(table_bytes.end(), t.first, t.first + 4);
    put_u32(table_bytes, (uint32_t)(start + body.size()));
    put_u32(table_bytes, (uint32_t)t.second->size());
    body.insert(body.end(), t.second->begin(), t.second->end());
    while (body.size() % 4) body.push_back(0);
  }
  std::vector<uint8_t> icc;
  put_u32(icc, (uint32_t)(start + body.size()));
  put_u32(icc, 0);
  put_u32(icc, 0x02400000);
  for (const char* s : {"scnr", "RGB ", "XYZ "}) icc.insert(icc.end(), s, s + 4);
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  for (int v : {utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec}) {
    icc.push_back((uint8_t)(v >> 8));
    icc.push_back((uint8_t)(v & 255));
  }
  icc.insert(icc.end(), {'a', 'c', 's', 'p'});
  icc.insert(icc.end(), 24, 0);
  put_u32(icc, 0);
  for (double v : kD50) put_s15(icc, v);
  icc.insert(icc.end(), 48, 0);
  put_u32(icc, (uint32_t)tags.size());
  icc.insert(icc.end(), table_bytes.begin(), table_bytes.end());
  icc.insert(icc.end(), body.begin(), body.end());
  return icc;
}

bool set_print_icc_curves(std::vector<uint8_t>& icc, const PrintCurves& curves) {
  const size_t off = a2b0_lut16(icc);
  if (!off) return false;
  const size_t n = (size_t)icc[off + 48] << 8 | icc[off + 49];
  if (n != curves[0].size() || off + 52 + 6 * n > icc.size()) return false;
  std::vector<uint8_t> codes;
  for (int c = 0; c < 3; ++c)
    for (double v : curves[c]) put_u16(codes, v);
  std::copy(codes.begin(), codes.end(), icc.begin() + (long)(off + 52));
  return true;
}

bool bracket_print_profiles(const ProfileBundle& bundle, const std::string& print, BracketPrints& out,
                            std::string& err) {
  if (!bundle.grey_balance.present) {
    return err = "the profile bundle has no grey_balance_curves (a print ICC is computed from them): rebuild it with "
                 "negicc-station", false;
  }
  if (!bundle.density_matrix[kDensityApd].present || !bundle.film_model.present) {
    return err = "the profile bundle has no APD matrix or film_model: rebuild it with negicc-station", false;
  }
  // The material: a built-in one by id, or a spektrafilm print profile by path.
  PrintMaterial material;
  bool found = false;
  if (is_print_file(print)) {
    std::ifstream file(print, std::ios::binary);
    std::stringstream text;
    text << file.rdbuf();
    if (!file) return err = "cannot be read", false;
    const size_t slash = print.find_last_of("/\\");
    std::string stem = slash == std::string::npos ? print : print.substr(slash + 1);
    if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".json") stem.resize(stem.size() - 5);
    if (!load_spektrafilm_print(text.str(), stem, bundle, material, err)) return false;
    found = true;
  } else {
    const EmbeddedFile* f = find_embedded_file("3rd_party/spektrafilm/profiles/" + print + ".json");
    if (f && !load_spektrafilm_print(std::string((const char*)f->data, f->size), print, bundle, material, err)) {
      return false;
    }
    found = f != nullptr;
  }
  if (!found) return err = "no such print material", false;
  const double* p = apd_print_matrix(bundle, material);
  if (!p) return err = "no print matrix for the bundle's film", false;
  const PrintCurves paper = paper_curve(material, p);
  const double n0 = bundle.normalization_target / 65535.0;
  auto bracket_of = [&](const std::string& name) -> const DensityBracket* {
    for (const DensityBracket& b : bundle.density_brackets)
      if (b.name == name) return &b;
    return nullptr;
  };
  // The table: the bundle's print ICC's when its own curves are those computed here for its bracket (the same film,
  // print material and math, to one 16-bit step), else computed here.
  std::vector<uint8_t> base;
  out = BracketPrints{};
  out.id = material.id;
  out.name = material.name;
  for (const BundlePrint& bp : bundle.prints) {
    if (is_print_file(print) || bp.id != print) continue;
    const DensityBracket* ref = bracket_of(bp.reference_bracket);
    const size_t off = a2b0_lut16(bp.icc_bytes);
    if (!ref || !off) break;
    const PrintCurves mine = bracket_print_curves(bundle.grey_balance, paper, bracket_mid_apd(bundle, *ref), n0);
    const size_t n = (size_t)bp.icc_bytes[off + 48] << 8 | bp.icc_bytes[off + 49];
    long worst = n == mine[0].size() ? 0 : 65535;
    for (int c = 0; c < 3 && worst <= 1; ++c)
      for (size_t i = 0; i < n; ++i) {
        const size_t at = off + 52 + 2 * (c * n + i);
        const long code = (long)bp.icc_bytes[at] << 8 | bp.icc_bytes[at + 1];
        worst = std::max(worst, std::labs(code - std::lround(std::min(std::max(mine[c][i], 0.0), 1.0) * 65535.0)));
      }
    if (worst <= 1) {
      base = bp.icc_bytes;
      out.from_bundle = true;
    } else {
      log_warn("The bundle's print ICC for %s does not match its film and print material as computed now (curves "
               "differ by %ld): its table is computed here\n", print.c_str(), worst);
    }
    break;
  }
  if (base.empty()) {
    const DensityBracket* mid = bracket_of(bundle.targets.empty() ? "" : bundle.targets[bundle.targets.size() / 2].name);
    if (!mid) return err = "the profile bundle has no density_matrices.brackets entry for its middle bracket", false;
    base = write_print_icc(bracket_print_curves(bundle.grey_balance, paper, bracket_mid_apd(bundle, *mid), n0),
                           print_table(material, p, bundle.density_matrix[kDensityApd].m, paper),
                           bundle.key + " printed on " + material.name + " (spektrafilm)",
                           "Tables derived from spektrafilm's print profile (Andrea Volpato), CC BY-SA 4.0. Computed "
                           "by neg_process.");
  }
  for (const BundleTarget& t : bundle.targets) {
    const DensityBracket* b = bracket_of(t.name);
    if (!b) return err = "the profile bundle has no density_matrices.brackets entry for " + t.name, false;
    std::vector<uint8_t> icc = base;
    if (!set_print_icc_curves(icc, bracket_print_curves(bundle.grey_balance, paper, bracket_mid_apd(bundle, *b), n0))) {
      return err = "the print ICC has no lut16 A2B0 to write curves into", false;
    }
    out.icc.push_back(std::move(icc));
  }
  return true;
}
