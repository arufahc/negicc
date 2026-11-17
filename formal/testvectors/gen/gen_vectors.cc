// Test vectors for the Lean specification in formal/: inputs and the values neg_process's own functions compute from
// them, written as JSON. The Lean executable `check` (formal/Check.lean) evaluates the specification on the same
// inputs and compares. See formal/README.md for how to build and run this program.
//
// Every value is computed by the library's functions (color_pipeline.h, flat_field.h, profile_bundle.h, rgb_view.h,
// dinov3_auto_solver.h, multi_target_optimizer.h, tone_stage.h); nothing is re-implemented here.

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
#include "multi_target_optimizer.h"
#include "profile_bundle.h"
#include "rgb_view.h"
#include "tone_stage.h"

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
  for (float x : {0.0f, 0.001f, 0.003f, 0.0031308f, 0.0032f, 0.01f, 0.18f, 0.5f, 0.9f, 1.0f, 1.2f}) {
    j["srgb_encode"].push_back({{"x", x}, {"y", negicc::srgb_encode(x)}});
  }
  for (float c : {0.0f, 0.02f, 0.04045f, 0.041f, 0.2f, 0.5f, 242.0f / 255.0f, 0.988f, 1.0f}) {
    j["srgb_decode"].push_back({{"x", c}, {"y", negicc::srgb_decode(c)}});
  }
  for (float t : {0.0f, 0.004f, 0.008856f, 0.009f, 0.18f, 1.0f, 1.3f}) {
    j["lab_f"].push_back({{"x", t}, {"y", negicc::lab_f(t)}});
  }
  const float cols[][3] = {{0, 0, 0},          {1, 1, 1},          {0.18f, 0.18f, 0.18f}, {0.005f, 0.004f, 0.006f},
                           {0.8f, 0.1f, 0.05f}, {0.1f, 0.6f, 0.2f}, {0.05f, 0.1f, 0.9f},   {0.4f, 0.35f, 0.3f},
                           {0.001f, 0.002f, 0.0005f}};
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
    for (float v : {-0.5f, 0.0f, 0.3f, 0.95f, 0.96f, 1.0f, 1.2f, 3.0f}) {
      j["knee"].push_back({{"knee", knee}, {"x", v}, {"y", negicc::knee_compress(v, knee)}});
    }
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

  const float cases[][4] = {{0.9f, 1.22f, 1.15f, 1.0f},   {0.45f, 1.10f, 1.15f, 1.0f},  {1.6f, 1.16f, 1.05f, 1.35f},
                            {2.4f, 1.28f, 1.25f, 1.9f},   {0.25f, 1.0f, 1.0f, 1.08f},   {4.0f, 0.9f, 1.3f, 2.5f},
                            {1.2f, 1.22f, 1.15f, 1.0002f}};
  for (const auto& cs : cases) {
    const float e = cs[0], g = cs[1], b = cs[2], x_hi = cs[3];
    float m[9];
    negicc::candidate_matrix(merged, e, g, b, m);
    const negicc::PipelineConfig cfg = make_pipeline_config(prof, kKnee, x_hi, ENCODE_SRGB);
    json c;
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
      c["out"].push_back({{"corrected", arr3(negicc::convert_pixel_corrected(cfg, m, px))},
                          {"pcs", arr3(negicc::convert_pixel_pcs(cfg, m, px))},
                          {"linear", arr3(lin)},
                          {"display", arr3(disp)},
                          {"encoded", arr3(negicc::convert_pixel(cfg, m, px))},
                          {"lab", arr3(negicc::linear_srgb_to_lab(disp))},
                          {"white_clipped", negicc::linear_channel_white_clipped(lin.r) ||
                                                negicc::linear_channel_white_clipped(lin.g) ||
                                                negicc::linear_channel_white_clipped(lin.b)}});
    }
    j["cases"].push_back(c);
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

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: gen_vectors <bundle directory, e.g. profiles/Portra400-A7R4-TriLED> <output directory>\n";
    return 2;
  }
  const std::string bundle_dir = argv[1], out_dir = argv[2];
  ProfileBundle bundle;
  load_profile_bundle(bundle_dir + "/bundle.json", bundle);

  json scene;
  scene["base"] = base_vectors();
  scene["flat"] = flat_vectors();
  scene["pixel"] = pixel_vectors(bundle, bundle_dir, 2);
  scene["shoulder"] = shoulder_vectors();
  write_json(out_dir + "/scene.json", scene);
  return 0;
}
