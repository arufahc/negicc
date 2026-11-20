// Test vectors for the Lean specification in formal/: inputs and the values neg_process's own functions compute from
// them, written as JSON. The Lean executable `check` (formal/Check.lean) evaluates the specification on the same
// inputs and compares. See formal/README.md for how to build and run this program.
//
// Every value is computed by the library's functions (color_pipeline.h, flat_field.h, profile_bundle.h, rgb_view.h,
// dinov3_auto_solver.h, multi_target_optimizer.h, tone_stage.h, input_curves.h, white_light_balance.h, and through
// gen_internal.cc the functions of internal linkage of multi_target_optimizer.cc and film_base_detector.cc); nothing is
// re-implemented here. The inputs are synthetic (fixed formulas or a seeded generator) or the bundles' own data.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "color_pipeline.h"
#include "dinov3_auto_solver.h"
#include "flat_field.h"
#include "input_curves.h"
#include "multi_target_optimizer.h"
#include "profile_bundle.h"
#include "rgb_view.h"
#include "tone_stage.h"
#include "white_light_balance.h"

// gen_internal.cc
namespace genint {
void enclosure(float g_min, float g_max, float exposure_s, float g_lo, float g_hi, float out[10]);
void lab_to_srgb_u8(const float lab[3], float out[3]);
void highlight_ceiling(const std::string& report, float out[3]);
float percentile_u16_linear(const std::vector<uint64_t>& hist, uint64_t total, float q);
float e_norm(const ProfileBundle& bundle, float scan_shutter_s, float scan_iso);
std::vector<size_t> picture_cells(const std::vector<uint16_t>& grid, const float film_base[3], bool rebate);
float percentile32(std::vector<float> a, double q);
double median16(std::vector<uint16_t> a);
}  // namespace genint

using nlohmann::json;
using negicc::Rgb;

static json arr3(const Rgb& v) { return json::array({v.r, v.g, v.b}); }
static json arr3(const negicc::Lab& v) { return json::array({v.L, v.a, v.b}); }

// JSON with one line per object member and per array element, except that an array of plain values (numbers,
// strings, booleans) stays on one line: readable in a diff, and a table of numbers is one line.
static void emit(std::ostream& os, const json& j, int depth) {
  const std::string pad(depth + 1, ' '), pad_close(depth, ' ');
  if (j.is_object()) {
    os << "{";
    bool first = true;
    for (auto it = j.begin(); it != j.end(); ++it) {
      os << (first ? "\n" : ",\n") << pad << json(it.key()).dump() << ": ";
      emit(os, it.value(), depth + 1);
      first = false;
    }
    os << "\n" << pad_close << "}";
  } else if (j.is_array()) {
    bool plain = true;
    for (const auto& v : j) plain = plain && v.is_primitive();
    if (plain) {
      os << j.dump();
    } else {
      os << "[";
      bool first = true;
      for (const auto& v : j) {
        os << (first ? "\n" : ",\n") << pad;
        emit(os, v, depth + 1);
        first = false;
      }
      os << "\n" << pad_close << "]";
    }
  } else {
    os << j.dump();
  }
}

static void write_json(const std::string& path, const json& j) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  emit(f, j, 0);
  f << "\n";
  std::cout << "wrote " << path << "\n";
}

// ---- Base primitives --------------------------------------------------------------------------------------
static json base_vectors() {
  json j;
  // No input lies on a threshold constant: there the single precision constant and the decimal one fall on different
  // sides (formal/README.md). The neighbours on both sides are included.
  for (float x : {0.0f, 0.001f, 0.003f, 0.0031307f, 0.0031309f, 0.0032f, 0.01f, 0.18f, 0.5f, 0.9f, 1.0f, 1.2f}) {
    j["srgb_encode"].push_back({{"x", x}, {"y", negicc::srgb_encode(x)}});
  }
  for (float c : {0.0f, 0.02f, 0.04044f, 0.04046f, 0.041f, 0.2f, 0.5f, 242.0f / 255.0f, 0.988f, 1.0f}) {
    j["srgb_decode"].push_back({{"x", c}, {"y", negicc::srgb_decode(c)}});
  }
  for (float t : {0.0f, 0.004f, 0.00885f, 0.00886f, 0.009f, 0.18f, 1.0f, 1.3f}) {
    j["lab_f"].push_back({{"x", t}, {"y", negicc::lab_f(t)}});
  }
  const float cols[][3] = {{0, 0, 0},          {1, 1, 1},          {0.18f, 0.18f, 0.18f}, {0.005f, 0.004f, 0.006f},
                           {0.8f, 0.1f, 0.05f}, {0.1f, 0.6f, 0.2f}, {0.05f, 0.1f, 0.9f},   {0.4f, 0.35f, 0.3f},
                           {0.001f, 0.002f, 0.0005f}, {0.0088f, 0.0089f, 0.0087f}};
  for (const auto& c : cols) {
    const Rgb v{c[0], c[1], c[2]};
    const Rgb enc{negicc::srgb_encode(c[0]), negicc::srgb_encode(c[1]), negicc::srgb_encode(c[2])};
    j["lab"].push_back({{"rgb", arr3(v)},
                        {"lab", arr3(negicc::linear_srgb_to_lab(v))},
                        {"L", negicc::linear_srgb_to_L(v)},
                        {"srgb", arr3(enc)},
                        {"lab_of_srgb", arr3(negicc::srgb_to_lab(enc))}});
  }
  const std::vector<float> table = {0.0f, 0.1f, 0.15f, 0.4f, 0.9f, 1.0f};
  j["trc"]["table"] = table;
  for (float v : {-0.1f, 0.0f, 0.05f, 0.2f, 0.33f, 0.5f, 0.77f, 0.999f, 1.0f, 1.1f}) {
    j["trc"]["cases"].push_back({{"x", v}, {"y", negicc::apply_trc(v, table.data(), (int)table.size())}});
  }
  for (float knee : {0.95f, 1.0f}) {
    for (float v : {-0.5f, 0.0f, 0.3f, 0.94f, 0.96f, 1.0f, 1.2f, 3.0f}) {
      j["knee"].push_back({{"knee", knee}, {"x", v}, {"y", negicc::knee_compress(v, knee)}});
    }
  }
  j["auto_knee"] = kKnee;
  for (float v : {0.0f, 0.1f, 0.5f, 0.70196f, 0.99999f, 1.0f}) {
    j["quantize"].push_back({{"x", v}, {"u8", negicc::quantize_u8(v)}, {"u16", negicc::quantize_u16(v)}});
  }
  for (float l : {0.0f, 0.0002f, 0.00025f, 0.5f, 0.995f, 0.9954f, 1.2f}) {
    j["clip"].push_back({{"x", l}, {"white", negicc::linear_channel_white_clipped(l)},
                         {"black", negicc::linear_channel_black_clipped(l)}});
  }
  // The tone curve's pieces on a pixel: f' on both sides of the break, the table lookup below 0, inside and above 1.
  for (float t : {0.0f, 0.005f, 0.00885f, 0.00886f, 0.2f, 1.0f, 1.5f}) {
    j["tone_fprime"].push_back({{"x", t}, {"y", negicc::tone_lab_fprime(t)}});
  }
  return j;
}

// ---- Flat field -------------------------------------------------------------------------------------------
static json flat_vectors() {
  // A 4 x 3 gain grid (3 rows, 4 columns) with block 8 on a 32 x 24 half-size decode.
  const int gw = 4, gh = 3, block = 8, half_w = 32, half_h = 24;
  std::vector<float> gain(gw * gh * 3);
  for (int y = 0; y < gh; ++y)
    for (int x = 0; x < gw; ++x)
      for (int c = 0; c < 3; ++c) gain[(y * gw + x) * 3 + c] = 0.80f + 0.05f * x + 0.03f * y + 0.02f * c * (x + 1);
  json j;
  j["grid_w"] = gw;
  j["grid_h"] = gh;
  j["block"] = block;
  j["gain"] = gain;
  const negicc::FlatField ff(gain, gh, gw, block, half_w, half_h, "test");
  for (int half = 0; half < 2; ++half) {
    const int w = half ? half_w : half_w * 2, h = half ? half_h : half_h * 2;
    const auto tables = ff.tables(w, h, half != 0);
    const negicc::FlatSampleArgs f = tables->host();
    const int pts[][2] = {{0, 0}, {3, 2}, {4, 4}, {11, 9}, {12, 12}, {20, 13}, {27, 20}, {31, 23}, {w - 1, h - 1}};
    for (const auto& p : pts) {
      const uint16_t in[3] = {12345, 40000, 65000};
      uint16_t px[3] = {in[0], in[1], in[2]};
      negicc::flat_divide(f, p[0], p[1], px);
      j["cases"].push_back({{"half", half != 0}, {"x", p[0]}, {"y", p[1]}, {"in", {in[0], in[1], in[2]}},
                            {"out", {px[0], px[1], px[2]}}});
    }
  }
  return j;
}

// ---- The per-pixel conversion through a real bracket profile -------------------------------------------
static json shoulder_json(const negicc::PipelineConfig& c) {
  return {{"x_hi", c.shoulder_x_hi}, {"x_k", c.shoulder_x_k}, {"t_k", c.shoulder_t_k}, {"t_hi", c.shoulder_t_hi},
          {"s_k", c.shoulder_s_k},   {"c", c.shoulder_c},     {"y_c", c.shoulder_y_c}};
}

static json pixel_vectors(const ProfileBundle& bundle, const std::string& bundle_dir, int target_idx) {
  json j;
  const BundleTarget& tgt = bundle.targets[target_idx];
  const ProfileData& prof = tgt.film_prof.data;
  j["bundle_dir"] = bundle_dir;
  j["target_idx"] = target_idx;
  j["target_name"] = tgt.name;
  j["icc"] = "Target_" + std::to_string(target_idx + 1) + ".icc";
  // What the Lean side's reading of the ICC file is checked against: sizes and a few entries of the parsed tables.
  j["profile_check"] = {{"in_size", prof.in_trc_size[0]},
                        {"out_size", prof.out_trc_size[0]},
                        {"clut_dim", prof.clut_dim[0]},
                        {"in_r_1000", prof.in_trc[0][1000]},
                        {"in_g_2000", prof.in_trc[1][2000]},
                        {"in_b_4095", prof.in_trc[2][4095]},
                        {"clut_12345", prof.clut[12345]},
                        {"clut_200000", prof.clut[200000]},
                        {"out_g_1", prof.out_trc[1][1]}};
  // The frame's merged matrix: the bundle's crosstalk correction and film base, a scan at 1/8 s and ISO 100.
  const double scan_shutter = 0.125, scan_iso = 100.0;
  float merged[9];
  double merged_d[9], row_scale[3];
  compute_station_merged_matrix(bundle, scan_shutter, scan_iso, nullptr, false, merged, merged_d, row_scale);
  j["scan_shutter"] = scan_shutter;
  j["scan_iso"] = scan_iso;
  j["film_base_shutter"] = shutter_to_seconds_d(bundle.film_base_shutter);
  j["film_base_iso"] = bundle.film_base_iso;
  j["film_base"] = {bundle.film_base_avg_d[0], bundle.film_base_avg_d[1], bundle.film_base_avg_d[2]};
  j["normalization_target"] = bundle.normalization_target;
  j["crosstalk"] = std::vector<double>(bundle.crosstalk_matrix_d, bundle.crosstalk_matrix_d + 9);
  j["merged"] = std::vector<double>(merged_d, merged_d + 9);
  j["merged_f32"] = std::vector<float>(merged, merged + 9);

  // Pixels: the film base times a transmission per channel, plus a few extremes.
  const double fb[3] = {9342.5, 5433.0, 7411.5};  // roughly the bundle's raw film base
  std::vector<std::array<uint16_t, 3>> pixels;
  const double trans[][3] = {{1.0, 1.0, 1.0},    {0.8, 0.8, 0.8},    {0.5, 0.5, 0.5},    {0.3, 0.3, 0.3},
                             {0.18, 0.18, 0.18}, {0.1, 0.1, 0.1},    {0.05, 0.05, 0.05}, {0.02, 0.02, 0.02},
                             {0.6, 0.35, 0.2},   {0.2, 0.45, 0.6},   {0.4, 0.15, 0.3},   {0.12, 0.3, 0.1},
                             {0.07, 0.05, 0.11}, {0.9, 0.6, 0.75},   {0.03, 0.08, 0.04}, {0.25, 0.22, 0.31}};
  for (const auto& t : trans) {
    pixels.push_back({(uint16_t)std::lround(fb[0] * t[0]), (uint16_t)std::lround(fb[1] * t[1]),
                      (uint16_t)std::lround(fb[2] * t[2])});
  }
  pixels.push_back({0, 0, 0});
  pixels.push_back({65535, 65535, 65535});
  pixels.push_back({200, 30000, 150});
  pixels.push_back({30000, 100, 20000});
  for (const auto& p : pixels) j["pixels"].push_back({p[0], p[1], p[2]});

  // (E, g, b, x_hi, tone curve index or -1, encoding). The tone curves are those of `tones` below; the last two cases
  // run the tone stage, the one before them the linear encoding (srgb-g10), which takes neither shoulder nor tone.
  const float cases[][6] = {{0.9f, 1.22f, 1.15f, 1.0f, -1, 0},   {0.45f, 1.10f, 1.15f, 1.0f, -1, 0},
                            {1.6f, 1.16f, 1.05f, 1.35f, -1, 0},  {2.4f, 1.28f, 1.25f, 1.9f, -1, 0},
                            {0.25f, 1.0f, 1.0f, 1.08f, -1, 0},   {4.0f, 0.9f, 1.3f, 2.5f, -1, 0},
                            {1.2f, 1.22f, 1.15f, 1.0002f, -1, 0}, {1.6f, 1.16f, 1.05f, 1.35f, 0, 1},
                            {1.6f, 1.16f, 1.05f, 1.35f, 0, 0},   {0.6f, 1.2f, 1.1f, 1.2f, 1, 0}};
  const ToneCurve tones[2] = {{3.0, 104.0, 0.9, 1.2}, {-8.0, 92.0, 1.3, 0.8}};
  for (const auto& cs : cases) {
    const float e = cs[0], g = cs[1], b = cs[2], x_hi = cs[3];
    const int tone_idx = (int)cs[4], enc = (int)cs[5];
    float m[9];
    negicc::candidate_matrix(merged, e, g, b, m);
    negicc::PipelineConfig cfg = make_pipeline_config(prof, kKnee, x_hi, enc ? ENCODE_SRGB_LINEAR : ENCODE_SRGB);
    std::vector<float> lut;
    json c;
    if (tone_idx >= 0) {
      const ToneCurve& q = tones[tone_idx];
      lut = tone_table(q);
      cfg.tone.lut = lut.data();
      cfg.tone.n = kToneTableSize;
      c["tone_curve"] = {q.black, q.white, q.gamma, q.contrast};
    } else {
      c["tone_curve"] = nullptr;
    }
    c["encoding"] = enc;
    c["e"] = e;
    c["g"] = g;
    c["b"] = b;
    c["x_hi"] = x_hi;
    c["knee"] = kKnee;
    c["matrix"] = std::vector<float>(m, m + 9);
    c["shoulder"] = shoulder_json(cfg);
    for (const auto& p : pixels) {
      const uint16_t px[3] = {p[0], p[1], p[2]};
      const Rgb lin = negicc::convert_pixel_linear(cfg, m, px);
      const Rgb disp = negicc::convert_pixel_display(cfg, m, px);
      const Rgb out = negicc::convert_pixel(cfg, m, px);
      c["out"].push_back({{"corrected", arr3(negicc::convert_pixel_corrected(cfg, m, px))},
                          {"pcs", arr3(negicc::convert_pixel_pcs(cfg, m, px))},
                          {"linear", arr3(lin)},
                          {"display", arr3(disp)},
                          {"encoded", arr3(out)},
                          {"u16", {negicc::quantize_u16(out.r), negicc::quantize_u16(out.g), negicc::quantize_u16(out.b)}},
                          {"lab", arr3(negicc::linear_srgb_to_lab(disp))},
                          {"white_clipped", negicc::linear_channel_white_clipped(lin.r) ||
                                                negicc::linear_channel_white_clipped(lin.g) ||
                                                negicc::linear_channel_white_clipped(lin.b)},
                          {"black_clipped", negicc::linear_channel_black_clipped(lin.r) &&
                                                negicc::linear_channel_black_clipped(lin.g) &&
                                                negicc::linear_channel_black_clipped(lin.b)}});
    }
    j["cases"].push_back(c);
  }
  // The exposure normalisation of scans at other exposures than the reference (frame_scale's e_norm).
  for (const auto& sc : std::vector<std::array<float, 2>>{{0.25f, 100.0f}, {0.125f, 200.0f}, {1.0f / 15.0f, 100.0f}}) {
    j["e_norm"].push_back({{"shutter", sc[0]}, {"iso", sc[1]}, {"e_norm", genint::e_norm(bundle, sc[0], sc[1])}});
  }
  return j;
}

// ---- The shoulder ---------------------------------------------------------------------------------------
static json shoulder_vectors() {
  json j;
  for (float x_hi : {1.0f, 1.00005f, 1.0002f, 1.01f, 1.05f, 1.1f, 1.3f, 1.7f, 2.2f, 2.5f, 3.0f}) {
    negicc::PipelineConfig c{};
    c.encoding = negicc::ENCODE_SRGB;
    negicc::set_shoulder(c, x_hi);
    json s = shoulder_json(c);
    s["x_hi_in"] = x_hi;
    for (float x : {0.2f, 0.5f, 0.7f, 0.85f, 0.95f, 1.0f, 1.1f, 1.4f, 2.0f, 2.6f}) {
      s["channel"].push_back({{"x", x}, {"y", negicc::apply_shoulder_channel(c, x)}});
    }
    j["set"].push_back(s);
  }
  for (float q : {0.0f, 0.5f, 0.88f, 0.9f, 0.95f, 1.0f, 1.05f, 1.2f, 1.5f, 1.9f, 2.3f}) {
    j["anchored"].push_back({{"q", q}, {"x_hi", negicc::shoulder_x_hi_anchored(q)}});
  }
  // A histogram of the largest channel: a smooth bulk and a highlight tail.
  std::vector<uint64_t> hist(negicc::kMaxHistBins, 0);
  for (int i = 0; i < negicc::kMaxHistBins; ++i) {
    const double x = (i + 0.5) * (negicc::kMaxHistRange / negicc::kMaxHistBins);
    hist[i] = (uint64_t)std::llround(4000.0 * std::exp(-0.5 * std::pow((x - 0.45) / 0.22, 2.0)) +
                                     25.0 * std::exp(-0.5 * std::pow((x - 1.25) / 0.08, 2.0)));
  }
  json h;
  h["hist"] = hist;
  for (double q : {0.005, 0.5, 0.9, 0.995, 0.9999}) {
    h["quantiles"].push_back({{"q", q}, {"value", negicc::max_hist_quantile(hist.data(), q)}});
  }
  float q995 = 0.0f;
  const float x_hi = shoulder_x_hi_from_hist(hist.data(), &q995);
  h["q995"] = q995;
  h["x_hi"] = x_hi;
  for (float v : {-0.2f, 0.0f, 0.3f, 0.9995f, 1.0f, 1.9999f, 2.0f, 7.0f}) {
    h["bins"].push_back({{"v", v}, {"bin", negicc::max_hist_bin(v)}});
  }
  j["hist"] = h;
  return j;
}

// ---- The tone stage -------------------------------------------------------------------------------------
static json tone_curve_json(const ToneCurve& q) { return json::array({q.black, q.white, q.gamma, q.contrast}); }

static json tone_vectors(const ProfileBundle& bundle, int target_idx) {
  json j;
  const ToneCurve curves[] = {{0.0, 100.0, 1.0, 1.0},  {3.0, 104.0, 0.9, 1.2},    {-8.0, 92.0, 1.3, 0.8},
                              {15.0, 80.0, 0.7, 1.5}, {-15.0, 110.0, 1.45, 0.7}};
  const int idx[] = {0, 1, 7, 100, 1000, 4096, 6000, 8191, 8192};
  for (const ToneCurve& q : curves) {
    const std::vector<float> lut = tone_table(q);
    json t;
    t["curve"] = tone_curve_json(q);
    for (int i : idx) t["entries"].push_back({{"i", i}, {"y", lut[i]}});
    j["tables"].push_back(t);
  }
  for (const ToneCurve& q : {ToneCurve{0.0, 100.0, 1.0, 1.0}, ToneCurve{20.0, 100.0, 1.0, 1.0},
                             ToneCurve{0.0, 100.0, 0.5, 1.0}, ToneCurve{0.0, 79.0, 1.0, 1.0},
                             ToneCurve{-15.0, 110.0, 1.45, 1.5}}) {
    j["ok"].push_back({{"curve", tone_curve_json(q)}, {"ok", tone_curve_ok(q)}});
  }
  // The table lookup and the curve on a pixel, with curve 1's table.
  const std::vector<float> lut = tone_table(curves[1]);
  const negicc::ToneTable tt{lut.data(), kToneTableSize};
  j["pixel_curve"] = tone_curve_json(curves[1]);
  for (float Y : {-0.1f, 0.0f, 1e-5f, 0.003f, 0.18f, 0.5f, 0.999f, 1.0f, 1.3f}) {
    j["table_Y"].push_back({{"x", Y}, {"y", negicc::tone_table_Y(tt, Y)}});
  }
  const float px[][3] = {{0.2f, 0.2f, 0.2f}, {0.5f, 0.3f, 0.1f},  {0.01f, 0.02f, 0.005f},
                         {0.9f, 0.95f, 1.0f}, {0.001f, 0.0f, 0.003f}, {0.05f, 0.6f, 0.02f}, {0.0f, 0.0f, 0.4f}};
  for (const auto& v : px) {
    j["apply"].push_back({{"x", {v[0], v[1], v[2]}}, {"y", arr3(negicc::apply_tone(tt, {v[0], v[1], v[2]}))}});
  }
  // A darkening curve on saturated pixels: a channel's y' + sc (v - y) falls below 0 before the encoding clamps it.
  {
    const std::vector<float> dark = tone_table(curves[2]);
    const negicc::ToneTable td{dark.data(), kToneTableSize};
    j["dark_curve"] = tone_curve_json(curves[2]);
    for (const auto& v : std::vector<std::array<float, 3>>{{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.9f, 0.05f, 0.02f}}) {
      j["apply_dark"].push_back({{"x", {v[0], v[1], v[2]}}, {"y", arr3(negicc::apply_tone(td, {v[0], v[1], v[2]}))}});
    }
  }
  // The descriptors of a synthetic 8-bit image, and the plans for predictions around them.
  const int W8 = 37, H8 = 23;
  std::vector<uint8_t> img8(W8 * H8 * 3);
  for (int y = 0; y < H8; ++y)
    for (int x = 0; x < W8; ++x) {
      uint8_t* p = &img8[(y * W8 + x) * 3];
      p[0] = (uint8_t)((x * 7 + y * 3) % 256);
      p[1] = (uint8_t)std::min(255, x * 5 + y * 6);
      p[2] = (uint8_t)((x * x + 2 * y * y) % 230 + 10);
    }
  j["image8"] = {{"w", W8}, {"h", H8}, {"rgb", img8}};
  const ToneDescriptors d = tone_descriptors(img8.data(), (size_t)W8 * H8);
  j["descriptors"] = {d.L2, d.L10, d.L25, d.L50, d.L75, d.L90, d.L98};
  const double preds[][3] = {{d.L2 + 3.0, d.L98 - 4.0, d.range() + 6.0}, {d.L2 - 2.0, d.L98 + 5.0, d.range() - 8.0},
                             {d.L2 + 6.0, d.L98 + 1.0, d.range() + 15.0}};
  for (const auto& pr : preds) {
    for (double share : {1.0, 0.5, 0.0}) {
      ToneTargets t;
      const ToneCurve q = tone_plan(d, pr[0], pr[1], pr[2], share, &t);
      j["plans"].push_back({{"pred", {pr[0], pr[1], pr[2]}}, {"share", share},
                            {"targets", {t.L2, t.L50, t.L98, t.range}}, {"curve", tone_curve_json(q)}});
    }
  }
  // The choice of the share on given clipping values.
  const double afters[][5] = {{0.012, 0.0, 0.0, 0.0, 0.0},    {0.03, 0.02, 0.014, 0.0, 0.0},
                              {0.05, 0.04, 0.035, 0.03, 0.03}, {0.05, 0.06, 0.07, 0.04, 0.04}};
  for (const auto& a : afters) {
    const int k = tone_pick_clip_share(0.01, [&](int i) { return a[i]; });
    j["pick"].push_back({{"before", 0.01}, {"after", std::vector<double>(a, a + 5)}, {"k", k}});
  }
  // The clip check on a 16-bit image through the bracket's conversion, and the plan for that image.
  const BundleTarget& tgt = bundle.targets[target_idx];
  float merged[9];
  double merged_d[9], row_scale[3];
  compute_station_merged_matrix(bundle, 0.125, 100.0, nullptr, false, merged, merged_d, row_scale);
  float m[9];
  negicc::candidate_matrix(merged, 0.5f, 1.16f, 1.1f, m);
  const negicc::PipelineConfig pc = make_pipeline_config(tgt.film_prof.data, kKnee, 1.3f, ENCODE_SRGB);
  const int W = 41, H = 27;
  std::vector<uint16_t> img(W * H * 3);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      uint16_t* p = &img[(y * W + x) * 3];
      const double t = 0.01 + 0.99 * (double)(x + W * y) / (W * H);
      p[0] = (uint16_t)std::lround(9342.5 * t * (1.0 + 0.2 * std::sin(x * 0.7)));
      p[1] = (uint16_t)std::lround(5433.0 * t);
      p[2] = (uint16_t)std::lround(7411.5 * t * (1.0 + 0.2 * std::cos(y * 0.9)));
    }
  j["clip"]["w"] = W;
  j["clip"]["h"] = H;
  j["clip"]["img"] = img;
  j["clip"]["matrix"] = std::vector<float>(m, m + 9);
  j["clip"]["x_hi"] = 1.3f;
  double wb = 0.0, kb = 0.0;
  tone_clip_sample(pc, m, img.data(), W, H, 3, nullptr, &wb, &kb);
  j["clip"]["none"] = {wb, kb};
  for (const ToneCurve& q : curves) {
    double wa = 0.0, ka = 0.0;
    tone_clip_sample(pc, m, img.data(), W, H, 3, &q, &wa, &ka);
    j["clip"]["curves"].push_back({{"curve", tone_curve_json(q)}, {"white", wa}, {"black", ka}});
  }
  {
    ToneTargets dt, t;
    const double pr[3] = {d.L2 + 3.0, d.L98 + 12.0, d.range() + 20.0};
    const ToneCurve designed = tone_plan(d, pr[0], pr[1], pr[2], 1.0, &dt);
    double share = 0.0;
    const ToneCurve q = tone_plan_for_image(d, pr[0], pr[1], pr[2], pc, m, img.data(), W, H, 3, wb, designed, dt, &t,
                                            &share);
    j["clip"]["plan"] = {{"pred", {pr[0], pr[1], pr[2]}}, {"share", share}, {"targets", {t.L2, t.L50, t.L98, t.range}},
                         {"curve", tone_curve_json(q)}};
  }
  return j;
}

// ---- The white-light fit --------------------------------------------------------------------------------
static json curve_json(const negicc::InputCurves& c) {
  json j;
  for (int i = 0; i < 2; ++i) {
    j[i ? "b" : "g"] = {{"a", c.a[i]}, {"k", c.k[i]}, {"ref", c.ref[i]}, {"u_med", c.u_med[i]}, {"u_lo", c.u_lo[i]}};
  }
  j["hold"] = c.hold;
  return j;
}

static json white_vectors(const ProfileBundle& bundle, const std::string& bundle_dir, int target_idx) {
  using negicc::candidate_matrix;
  using negicc::CombinedProfile;
  using negicc::compose_input_table;
  using negicc::curve_shape_ok;
  using negicc::curve_stops;
  using negicc::CurveFitOptions;
  using negicc::CurveFrame;
  using negicc::fit_input_curves_joint;
  using negicc::green_guard_cell_a;
  using negicc::grid_median_L;
  using negicc::input_curve;
  using negicc::input_curve_reference;
  using negicc::InputCurves;
  using negicc::joint_moments;
  using negicc::JointCurveFit;
  using negicc::JointCurveModel;
  using negicc::JointFitField;
  using negicc::JointMoments;
  using negicc::Lab;
  using negicc::PipelineConfig;
  using negicc::solve_e_for_median_L;
  json j;
  const ProfileData& prof = bundle.targets[target_idx].film_prof.data;
  j["bundle_dir"] = bundle_dir;
  j["icc"] = "Target_" + std::to_string(target_idx + 1) + ".icc";
  // The curve's pieces.
  for (float u : {-6.0f, -2.5f, -1.0f, -0.3f, 0.0f}) {
    for (const auto& p : std::vector<std::array<float, 4>>{{-1.5f, 0.2f, 0.0f, -INFINITY}, {-1.0f, -0.25f, 0.1f, -2.0f},
                                                           {-2.0f, 0.3f, -0.2f, -4.0f}}) {
      j["stops"].push_back({{"u", u}, {"u_med", p[0]}, {"a", p[1]}, {"k", p[2]},
                            {"hold", std::isfinite(p[3]) ? json(p[3]) : json(nullptr)},
                            {"y", curve_stops(u, p[0], p[1], p[2], p[3])}});
    }
  }
  for (float v : {0.0f, 1e-13f, 0.001f, 0.05f, 0.3f, 0.69f, 0.7f, 0.9f}) {
    for (const auto& p : std::vector<std::array<float, 5>>{{0.7f, -1.5f, 0.2f, 0.0f, -INFINITY},
                                                           {0.7f, -1.0f, -0.25f, 0.1f, -3.0f},
                                                           {0.6f, -2.0f, 0.0f, 0.0f, -INFINITY}}) {
      j["input"].push_back({{"v", v}, {"ref", p[0]}, {"u_med", p[1]}, {"a", p[2]}, {"k", p[3]},
                            {"hold", std::isfinite(p[4]) ? json(p[4]) : json(nullptr)},
                            {"y", input_curve(v, p[0], p[1], p[2], p[3], p[4])}});
    }
  }
  {
    const int n = prof.in_trc_size[1];
    std::vector<float> out(n);
    compose_input_table(prof.in_trc[1], n, 0.7f, -1.5f, 0.2f, 0.0f, out.data(), -4.0f);
    json c = {{"ref", 0.7f}, {"u_med", -1.5f}, {"a", 0.2f}, {"k", 0.0f}, {"hold", -4.0f}, {"n", n}};
    for (int i : {0, 1, 50, 1000, 2000, 2866, 2867, 3500, 4095}) c["entries"].push_back({{"i", i}, {"y", out[i]}});
    j["compose"] = c;
  }
  for (const auto& p : std::vector<std::array<float, 6>>{{0.2f, 0.0f, -1.5f, 0.7f, 4096, -INFINITY},
                                                         {0.35f, 0.0f, -1.5f, 0.7f, 4096, -3.0f},
                                                         {-1.2f, 0.0f, -1.5f, 0.7f, 4096, -INFINITY},
                                                         {0.9f, 0.0f, -1.0f, 0.5f, 4096, -INFINITY},
                                                         {0.2f, 0.4f, -1.0f, 0.6f, 64, -2.0f},
                                                         {0.0f, 0.0f, -1.0f, 0.6f, 4096, -INFINITY}}) {
    j["shape"].push_back({{"a", p[0]}, {"k", p[1]}, {"u_med", p[2]}, {"ref", p[3]}, {"n", (int)p[4]},
                          {"hold", std::isfinite(p[5]) ? json(p[5]) : json(nullptr)},
                          {"ok", curve_shape_ok(p[0], p[1], p[2], p[3], (int)p[4], 2.2f, p[5])}});
  }
  // A frame: the bundle's merged matrix, its film base, a grid of cells (film base times a transmission per channel),
  // the neutral head's p, an intent field.
  float merged[9];
  double merged_d[9], row_scale[3];
  compute_station_merged_matrix(bundle, 0.125, 100.0, nullptr, false, merged, merged_d, row_scale);
  j["merged_f32"] = std::vector<float>(merged, merged + 9);
  CurveFrame f;
  f.merged = merged;
  f.knee = kKnee;
  for (int c = 0; c < 3; ++c) f.film_base[c] = bundle.film_base_avg[c];
  j["film_base"] = {f.film_base[0], f.film_base[1], f.film_base[2]};
  const int gw = 8, gh = 6, n = gw * gh;
  std::vector<uint16_t> grid(n * 3);
  std::vector<float> p_neutral(n), mu(n * 3), sigma(n * 3);
  JointFitField fi;
  fi.nbx = 2;
  for (int i = 0; i < n; ++i) {
    const int x = i % gw, y = i / gw;
    const double t = std::pow(2.0, -0.1 - 0.075 * i);
    const double tint[3] = {1.0 + 0.08 * std::sin(i * 1.3), 1.0 + 0.06 * std::cos(i * 0.7), 1.0 + 0.1 * std::sin(i * 0.4)};
    for (int c = 0; c < 3; ++c)
      grid[i * 3 + c] = (uint16_t)std::lround(std::min(65535.0, f.film_base[c] * t * tint[c]));
    p_neutral[i] = (float)(0.5 + 0.45 * std::sin(i * 0.9));
    mu[i * 3] = 50.0f;
    mu[i * 3 + 1] = (float)(1.0 + 1.5 * std::sin(i * 0.5));
    mu[i * 3 + 2] = (float)(2.0 * std::cos(i * 0.3));
    sigma[i * 3] = 8.0f;
    sigma[i * 3 + 1] = (float)(2.0 + 0.5 * std::cos(i * 1.1));
    sigma[i * 3 + 2] = 3.0f;
    fi.bx.push_back(x / 4);
    fi.by.push_back(y / 4);
  }
  f.grid = grid.data();
  f.n = n;
  f.neutral_p = p_neutral.data();
  fi.mu = mu.data();
  fi.sigma = sigma.data();
  json fr;
  fr["grid"] = grid;
  fr["neutral_p"] = p_neutral;
  fr["mu"] = mu;
  fr["sigma"] = sigma;
  fr["bx"] = fi.bx;
  fr["by"] = fi.by;
  fr["nbx"] = fi.nbx;
  j["frame"] = fr;
  const PipelineConfig cfg = make_pipeline_config(prof, kKnee, 1.2f, ENCODE_SRGB);
  j["x_hi"] = 1.2f;
  const float e0 = 0.9f, g0 = 1.12f, b0 = 1.2f;
  j["auto"] = {e0, g0, b0};
  InputCurves shape;
  shape.hold = true;
  shape.a[0] = 0.12f;
  shape.a[1] = -0.08f;
  {
    InputCurves c = shape;
    input_curve_reference(merged, e0, g0, b0, kKnee, f.film_base, grid.data(), n, c);
    j["reference"] = {{"shape", curve_json(shape)}, {"curves", curve_json(c)}};
  }
  JointCurveModel model(cfg, f);
  {
    const InputCurves c = model.anchored(shape, e0, g0, b0);
    json a = {{"curves", curve_json(c)}};
    for (int i = 0; i < n; ++i) a["labs"].push_back(arr3(model.cell_lab(i, e0, g0, b0, c)));
    a["median_L"] = model.median_L(e0, g0, b0, shape);
    j["model"] = a;
  }
  const float target_L = model.median_L(e0, g0, b0, InputCurves{}) + 2.0f;
  j["target_L"] = target_L;
  {
    int renders = 0;
    float e = 0.0f;
    const bool ok = model.solve_e(g0, b0, InputCurves{}, target_L, e0, e0 * 0.25f, e0 * 4.0f, 2e-5f, renders, e);
    j["solve_e"] = {{"ok", ok}, {"e", e}};
  }
  {
    std::vector<Lab> labs(n);
    const InputCurves c = model.anchored(shape, e0, g0, b0);
    for (int i = 0; i < n; ++i) labs[i] = model.cell_lab(i, e0, g0, b0, c);
    std::vector<int> idx;
    std::vector<double> w;
    for (int i = 0; i < n; i += 3) {
      idx.push_back(i);
      w.push_back(0.1 + 0.05 * (i % 7));
    }
    const JointMoments mo = joint_moments(labs, idx, w);
    j["moments"] = {{"idx", idx}, {"w", w}, {"m", {mo.L, mo.a, mo.b, mo.vL, mo.ka, mo.kb}}};
  }
  for (const auto& c : std::vector<std::array<double, 3>>{{1.0, 2.0, 30.0}, {-1.0, 2.0, 30.0}, {-1.0, 2.0, 5.0},
                                                         {-1.0, 2.0, 15.0}, {-3.0, -1.0, 12.0}, {-3.0, -1.0, 19.0}}) {
    j["guard_cell"].push_back({{"a", c[0]}, {"a_auto", c[1]}, {"L_auto", c[2]},
                               {"y", green_guard_cell_a(c[0], c[1], c[2])}});
  }
  const int table_n[2] = {prof.in_trc_size[1], prof.in_trc_size[2]};
  const JointCurveFit out = fit_input_curves_joint(cfg, f, fi, e0, g0, b0, target_L, table_n, CurveFitOptions(),
                                                   e0 * 0.25f, e0 * 4.0f);
  j["fit"] = {{"ok", out.ok},
              {"e", out.e},
              {"g", out.g},
              {"b", out.b},
              {"curves", curve_json(out.curves)},
              {"loss_before", out.loss_before},
              {"loss_after", out.loss_after},
              {"evidence", out.evidence},
              {"shrink", out.shrink},
              {"slope_share", out.slope_share},
              {"span_L", out.span_L},
              {"anchor_sd", out.anchor_sd},
              {"intent_se", out.intent_se},
              {"intent_auto", out.intent_auto},
              {"refit", out.guard.refit}};
  {
    float m[9];
    candidate_matrix(merged, out.e, out.g, out.b, m);
    CombinedProfile comb;
    comb.build(prof, out.curves);
    PipelineConfig cc = cfg;
    cc.profile = comb.data;
    j["combined"] = {{"median_L", grid_median_L(cc, merged, out.e, out.g, out.b, grid.data(), n)},
                     {"remeter", solve_e_for_median_L(cfg, f, out.curves, out.e, e0 * 0.25f, e0 * 4.0f, out.g, out.b,
                                                      target_L)}};
  }
  for (bool rebate : {true, false}) {
    std::vector<uint16_t> g2 = grid;
    for (int i = 0; i < 6; ++i)
      for (int c = 0; c < 3; ++c) g2[i * 3 + c] = (uint16_t)std::lround(f.film_base[c] * 0.95);
    for (int i = 6; i < 9; ++i)
      for (int c = 0; c < 3; ++c) g2[i * 3 + c] = 3;
    j["picture_cells"].push_back({{"grid", g2}, {"rebate", rebate}, {"kept", genint::picture_cells(g2, f.film_base,
                                                                                                    rebate)}});
  }
  // The film-base balance of the decode, for a camera matrix and two bases.
  {
    LibRaw proc;
    const float cam[3][3] = {{1.72f, -0.62f, -0.10f}, {-0.18f, 1.45f, -0.27f}, {0.02f, -0.48f, 1.46f}};
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) proc.imgdata.color.rgb_cam[r][c] = cam[r][c];
    const int profile_base[3] = {41000, 23000, 11000};
    for (const auto& b : std::vector<std::array<double, 3>>{{38000, 22000, 12000}, {41000, 23000, 11000},
                                                           {30000, 25000, 15000}, {30000, -100, 15000}}) {
      float k[3] = {-1.0f, -1.0f, -1.0f};
      negicc::film_base_balance_gains(proc, b.data(), profile_base, k);
      j["balance"].push_back({{"rgb_cam", {cam[0][0], cam[0][1], cam[0][2], cam[1][0], cam[1][1], cam[1][2], cam[2][0],
                                           cam[2][1], cam[2][2]}},
                              {"base", {b[0], b[1], b[2]}},
                              {"profile_base", {profile_base[0], profile_base[1], profile_base[2]}},
                              {"k", {k[0], k[1], k[2]}}});
    }
  }
  return j;
}

// ---- Reading the scan, the film base, a bracket's range and ceiling ----------------------------------------
static json read_vectors(const ProfileBundle& tri, const ProfileBundle& wl) {
  using negicc::compute_inner_crop;
  using negicc::GrainAwareDebayerArgs;
  using negicc::InnerCrop;
  using negicc::PixelShiftMergeArgs;
  using negicc::PreviewDebayerArgs;
  using negicc::RgbView;
  json j;
  // A 16 x 12 RGGB mosaic with a white level of 15000, some photosites near it.
  const int W = 16, H = 12;
  const unsigned filters = 0x94949494u;  // RGGB in LibRaw's encoding
  std::mt19937 rng(20261120);
  std::uniform_int_distribution<int> dist(200, 14000);
  std::vector<uint16_t> mos[4];
  for (auto& m : mos) {
    m.resize(W * H);
    for (auto& v : m) v = (uint16_t)dist(rng);
  }
  for (int k = 0; k < 6; ++k) mos[0][(k * 37) % (W * H)] = (uint16_t)(14950 + k * 20);
  for (int k = 0; k < 4; ++k)
    for (int i = 0; i < 3; ++i) mos[k][(i * 53 + k * 11) % (W * H)] = 15000;
  j["mosaic"] = {{"w", W}, {"h", H}, {"white", 15000}, {"f0", mos[0]}, {"f1", mos[1]}, {"f2", mos[2]}, {"f3", mos[3]}};
  {
    std::vector<uint16_t> out(W / 2 * H / 2 * 3);
    PreviewDebayerArgs a;
    a.cfa_bayer = mos[0].data();
    a.src_w = W;
    a.src_h = H;
    a.rw = W;
    a.rh = H;
    a.dst_w = W / 2;
    a.dst_h = H / 2;
    a.filters = filters;
    a.rgb_out = out.data();
    negicc::cpu_preview_debayer(a);
    j["half"] = out;
  }
  {
    std::vector<uint16_t> out(W * H * 3);
    GrainAwareDebayerArgs a;
    a.cfa_bayer = mos[0].data();
    a.src_w = W;
    a.src_h = H;
    a.rw = W;
    a.rh = H;
    a.dst_w = W;
    a.dst_h = H;
    a.filters = filters;
    a.white_level = 15000.0f;
    a.rgb_out = out.data();
    negicc::cpu_grain_aware_debayer(a);
    j["grain_aware"] = out;
  }
  {
    std::vector<uint16_t> out(W * H * 3);
    PixelShiftMergeArgs a;
    a.bayer_f0 = mos[0].data();
    a.bayer_f1 = mos[1].data();
    a.bayer_f2 = mos[2].data();
    a.bayer_f3 = mos[3].data();
    a.src_w = W;
    a.src_h = H;
    a.rw = W;
    a.rh = H;
    a.dst_w = W;
    a.dst_h = H;
    a.filters = filters;
    a.white_level = 15000.0f;
    a.rgb_out = out.data();
    negicc::cpu_pixel_shift_merge(a);
    j["pixel_shift"] = out;
  }
  // The film base's statistics.
  for (int n : {1, 2, 7, 64, 513}) {
    std::vector<float> v(n);
    std::vector<uint16_t> u(n);
    for (int i = 0; i < n; ++i) {
      v[i] = (float)(1000.0 + 37.0 * ((i * 7919) % 101) + 0.25 * i);
      u[i] = (uint16_t)((i * 7919) % 60000);
    }
    json c = {{"f32", v}, {"u16", u}, {"median_u16", genint::median16(u)}};
    for (double q : {0.0, 50.0, 99.5, 100.0}) c["pct"].push_back({{"q", q}, {"y", genint::percentile32(v, q)}});
    j["stats"].push_back(c);
  }
  // A bracket's range: direct cases, the inner crop, the percentile, and the ranges neg_process meters on a frame.
  for (const auto& c : std::vector<std::array<float, 5>>{{2000.0f, 50000.0f, 0.125f, 300.0f, 9000.0f},
                                                         {2000.0f, 50000.0f, 0.5f, 300.0f, 9000.0f},
                                                         {8000.0f, 30000.0f, 0.125f, 100.0f, 30000.0f},
                                                         {8000.0f, 12000.0f, 0.25f, 50.0f, 60000.0f},
                                                         {1000.0f, 60000.0f, 0.03125f, 2000.0f, 3000.0f}}) {
    float r[10];
    genint::enclosure(c[0], c[1], c[2], c[3], c[4], r);
    j["enclosure"].push_back({{"in", {c[0], c[1], c[2], c[3], c[4]}}, {"out", std::vector<float>(r, r + 10)}});
  }
  for (const auto& c : std::vector<std::array<float, 3>>{{60, 40, 0.9f}, {61, 41, 0.9f}, {3, 2, 0.5f}, {100, 1, 0.9f},
                                                         {64, 48, 1.0f}}) {
    const InnerCrop ic = compute_inner_crop((int)c[0], (int)c[1], c[2]);
    j["inner_crop"].push_back({{"in", {(int)c[0], (int)c[1], c[2]}}, {"out", {ic.x0, ic.y0, ic.w, ic.h}}});
  }
  {
    std::vector<uint64_t> hist(65536, 0);
    uint64_t total = 0;
    for (int v = 100; v < 40000; v += 97) {
      hist[v] = 1 + (v % 5);
      total += hist[v];
    }
    json c = {{"total", total}};
    json nz;
    for (int v = 0; v < 65536; ++v)
      if (hist[v]) nz.push_back({v, hist[v]});
    c["hist"] = nz;
    for (float q : {0.0f, 0.005f, 0.5f, 0.995f, 1.0f}) {
      c["pct"].push_back({{"q", q}, {"y", genint::percentile_u16_linear(hist, total, q)}});
    }
    j["u16_percentile"] = c;
  }
  {
    const int FW = 60, FH = 40;
    std::vector<uint16_t> frame(FW * FH * 3);
    for (int y = 0; y < FH; ++y)
      for (int x = 0; x < FW; ++x) {
        const double t = std::pow(2.0, -7.0 * (double)(x + FW * y) / (FW * FH));
        uint16_t* p = &frame[(y * FW + x) * 3];
        p[0] = (uint16_t)std::lround(30000.0 * t * (1.0 + 0.1 * std::sin(x)));
        p[1] = (uint16_t)std::lround(20000.0 * t);
        p[2] = (uint16_t)std::lround(25000.0 * t * (1.0 + 0.1 * std::cos(y)));
      }
    j["meter"]["w"] = FW;
    j["meter"]["h"] = FH;
    j["meter"]["frame"] = frame;
    const RgbView view = RgbView::from_interleaved_rgb(frame.data(), FW, FH);
    for (const ProfileBundle* b : {&tri, &wl}) {
      std::vector<BracketExposure> out;
      meter_bracket_exposures(*b, view, 0.25, 100.0, 0.9f, nullptr, out);
      json bj = {{"white_light", !b->has_crosstalk}, {"e_norm", genint::e_norm(*b, 0.25f, 100.0f)}};
      for (size_t i = 0; i < out.size(); ++i) {
        const BundleTarget& t = b->targets[i];
        bj["targets"].push_back({{"g_min", t.tgt_min_g}, {"g_max", t.tgt_max_g}, {"exposure", t.exposure_s},
                                 {"valid", out[i].valid}, {"feasible", out[i].feasible},
                                 {"range", {out[i].e_min, out[i].e_max, out[i].e_center, out[i].search_lo,
                                            out[i].search_hi}}});
      }
      j["meter"]["bundles"].push_back(bj);
    }
    // The ranges kept in the bundle for given values (compute_bundle_enclosures: nearest rank over the whole frame).
    {
      ProfileBundle b = tri;
      float g_lo = 0.0f, g_hi = 0.0f, scene_ev = 0.0f;
      int num_feasible = 0;
      compute_bundle_enclosures(b, view, g_lo, g_hi, scene_ev, num_feasible);
      json bj = {{"g_lo", g_lo}, {"g_hi", g_hi}};
      for (const BundleTarget& t : b.targets) {
        bj["targets"].push_back({{"g_min", t.tgt_min_g}, {"g_max", t.tgt_max_g}, {"exposure", t.exposure_s},
                                 {"feasible", t.is_feasible}, {"range", {t.e_min, t.e_max, t.e_center}}});
      }
      j["meter"]["bundle_enclosures"] = bj;
    }
    // The frame's highlight roll-off through bracket 3 (the histogram of the pre-shoulder largest channel over the
    // inner crop, its 99.5 % point, x_hi), at two exposures.
    for (float e : {0.2f, 0.6f}) {
      float q995 = 0.0f;
      const float x_hi = measure_rolloff_x_hi(tri, 2, view, 0.25, 100.0, e, 1.16f, 1.1f, 0.9f, &q995, nullptr, nullptr);
      j["meter"]["rolloff"].push_back({{"e", e}, {"g", 1.16f}, {"b", 1.1f}, {"q995", q995}, {"x_hi", x_hi}});
    }
  }
  // The highlight ceiling: the sRGB codes of predicted PCS colours, and the ceiling of a report holding them.
  const float labs[][3] = {{50.0f, 2.0f, -3.0f}, {85.0f, -1.0f, 4.0f}, {80.0f, 10.0f, 30.0f}, {5.0f, 1.0f, 1.0f},
                           {70.0f, -40.0f, 20.0f}, {101.0f, 0.0f, 0.0f}};
  std::string report = "Profile check\n";
  for (const auto& l : labs) {
    float out[3];
    genint::lab_to_srgb_u8(l, out);
    j["lab_to_srgb"].push_back({{"lab", {l[0], l[1], l[2]}}, {"rgb", {out[0], out[1], out[2]}}});
    char line[200];
    std::snprintf(line, sizeof line, "[1.5] A1: 0.1 0.2 0.3 -> %.4f %.4f %.4f should be 50 0 0\n", l[0], l[1], l[2]);
    report += line;
  }
  std::string report5 = report.substr(0, report.rfind("[1.5]"));  // without the last row, which clips
  for (const std::string& r : {report, report5, std::string("no rows\n")}) {
    float ceil[3];
    genint::highlight_ceiling(r, ceil);
    j["ceiling"].push_back({{"rows", r == report ? 6 : r == report5 ? 5 : 0}, {"ceil", {ceil[0], ceil[1], ceil[2]}}});
  }
  return j;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: gen_vectors <TriLED bundle, e.g. profiles/Portra400-A7R4-TriLED> <white-light bundle, e.g. "
                 "profiles/Portra400-A7R4-D65> <output directory>\n";
    return 2;
  }
  const std::string bundle_dir = argv[1], wl_dir = argv[2], out_dir = argv[3];
  ProfileBundle bundle, wl;
  load_profile_bundle(bundle_dir + "/bundle.json", bundle);
  load_profile_bundle(wl_dir + "/bundle.json", wl);

  json scene;
  scene["base"] = base_vectors();
  scene["flat"] = flat_vectors();
  scene["pixel"] = pixel_vectors(bundle, bundle_dir, 2);
  scene["shoulder"] = shoulder_vectors();
  write_json(out_dir + "/scene.json", scene);
  write_json(out_dir + "/scene_tone.json", tone_vectors(bundle, 2));
  write_json(out_dir + "/scene_white.json", white_vectors(wl, wl_dir, 3));
  write_json(out_dir + "/scene_read.json", read_vectors(bundle, wl));
  return 0;
}
