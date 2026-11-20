// Test vectors for the Lean specification of the Auto search (formal/Negicc/Auto): inputs and the values the
// library's own functions compute from them, written as JSON. The Lean executable `checkAuto` (formal/CheckAuto.lean)
// evaluates the specification on the same inputs and compares. See formal/README.md for how to build and run it.
//
// Nothing is re-implemented here. The search's internal functions (bracket_enclosure, filter_admissible_targets,
// compute_frame_stats, u16_percentile_linear, GreyPoint, lstsq3, exposure_for_lightness, measure_highlight_clip,
// frame_scale) live in an unnamed namespace of multi_target_optimizer.cc, so this file includes that source file and
// calls them where they are; the library's own multi_target_optimizer.o is then not linked.
//
// The intent network is replaced by a stand-in (StandInRunner): a fixed, stated function of the network's input, so
// the whole search runs as neg_process runs it, with every rule of the C++ applied, and the Lean side can compute the
// same field. The engine's members are reached by compiling its header with private access opened; the library's
// code is unchanged.

#include <algorithm>
#include <cstring>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#define private public
#include "dinov3_engine.h"
#undef private

#include "../../../multi_target_optimizer.cc"

using nlohmann::json;
using negicc::Rgb;

// JSON with one line per object member and per array element, except that an array of plain values stays on one line.
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

static json lab3(const float* v) { return json::array({v[0], v[1], v[2]}); }

// ---- Inputs ------------------------------------------------------------------------------------------------

// A negative of a scene: film base fb (16-bit) times 10^-D per channel, D a smooth field of density with a few
// patches, plus a fixed pseudo-random grain. d_lo .. d_hi is the green density range.
static std::vector<uint16_t> negative(int w, int h, const float fb[3], float d_lo, float d_hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> grain(-0.02f, 0.02f);
  std::vector<uint16_t> px((size_t)w * h * 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float u = (x + 0.5f) / w, v = (y + 0.5f) / h;
      float t = 0.5f + 0.35f * std::sin(6.0f * u + 1.3f) * std::cos(4.0f * v - 0.4f) + 0.15f * (u - v);
      if (std::fabs(u - 0.3f) < 0.08f && std::fabs(v - 0.35f) < 0.12f) t = 0.95f;  // a bright patch (dense)
      if (std::fabs(u - 0.75f) < 0.1f && std::fabs(v - 0.7f) < 0.1f) t = 0.05f;   // a dark patch (thin)
      t = std::clamp(t, 0.0f, 1.0f);
      const float dg = d_lo + (d_hi - d_lo) * t;
      const float d[3] = {dg * 0.92f + 0.04f * u, dg, dg * 1.08f - 0.05f * v};
      for (int c = 0; c < 3; ++c) {
        const float val = fb[c] * std::pow(10.0f, -(d[c] + grain(rng)));
        px[((size_t)y * w + x) * 3 + c] = (uint16_t)std::clamp(std::lround(val), 0L, 65535L);
      }
    }
  }
  return px;
}

static std::vector<uint16_t> random_u16(int n, uint32_t seed, int lo = 200, int hi = 60000) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = (uint16_t)d(rng);
  return v;
}

// A smooth 8-bit picture with grain, so that resampling and CIELAB see structure rather than noise alone.
static std::vector<uint8_t> picture_u8(int w, int h, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> g(-12, 12);
  std::vector<uint8_t> v((size_t)w * h * 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c) {
        const float base = 128.0f + 100.0f * std::sin(0.21f * x + 0.9f * c) * std::cos(0.17f * y - 0.5f * c);
        v[((size_t)y * w + x) * 3 + c] = (uint8_t)std::clamp((int)std::lround(base) + g(rng), 0, 255);
      }
  return v;
}

// The bundle's merged matrix for a scan at the reference exposure (1/8 s, ISO 100) in the search's units
// (frame_scale; on TriLED divided by e_norm, which is 1 there).
static void reference_input_matrix(const ProfileBundle& bundle, float m[9]) {
  ParamSearchOptions opts;
  const FrameScale s = frame_scale(bundle, 0.125f, 100.0f, opts);
  for (int i = 0; i < 9; ++i) m[i] = s.frame_matrix[i] / (bundle.has_crosstalk ? s.e_norm : 1.0f);
}

// ---- Geometry and resampling -------------------------------------------------------------------------------

static json geometry_vectors() {
  json j;
  const float crops[][3] = {{120, 80, 0.9f}, {3000, 2000, 0.9f}, {7, 5, 0.9f}, {1, 1, 0.9f}, {10, 3, 0.5f},
                            {100, 100, 1.0f}, {33, 17, 0.95f}, {2, 9, 0.1f}};
  for (const auto& c : crops) {
    const InnerCrop r = compute_inner_crop((int)c[0], (int)c[1], c[2]);
    j["inner_crop"].push_back({{"W", (int)c[0]}, {"H", (int)c[1]}, {"ratio", c[2]},
                               {"out", {r.x0, r.y0, r.w, r.h}}});
  }
  const int shapes[][4] = {{108, 72, 512, 16}, {2700, 1800, 512, 16}, {1800, 2700, 512, 16}, {108, 72, 64, 8},
                           {5, 3, 512, 16},    {100, 100, 96, 16},    {1000, 333, 512, 16},  {0, 7, 512, 16},
                           {640, 427, 512, 16}, {36, 24, 512, 16},    {300, 200, 8, 16}};
  for (const auto& s : shapes) {
    const negicc::AspectPreservedShape r = negicc::compute_aspect_preserved_shape(s[0], s[1], s[2], s[3]);
    j["aspect"].push_back({{"w", s[0]}, {"h", s[1]}, {"max", s[2]}, {"patch", s[3]},
                           {"out", {r.target_w, r.target_h, r.grid_w, r.grid_h}}});
  }
  return j;
}

static json resample_vectors() {
  json j;
  // resample_view_rgb16: area average (shrinking), bilinear (enlarging), copy (same size).
  const int cases[][4] = {{37, 23, 8, 5}, {20, 13, 32, 21}, {16, 16, 16, 16}, {50, 9, 7, 9}};
  uint32_t seed = 11;
  for (const auto& c : cases) {
    const std::vector<uint16_t> src = random_u16(c[0] * c[1] * 3, seed++);
    std::vector<uint16_t> dst((size_t)c[2] * c[3] * 3);
    negicc::resample_view_rgb16(negicc::RgbView::from_interleaved_rgb(src.data(), c[0], c[1], 3), dst.data(), c[2],
                                c[3]);
    j["resample_u16"].push_back({{"sw", c[0]}, {"sh", c[1]}, {"dw", c[2]}, {"dh", c[3]}, {"src", src}, {"out", dst}});
  }
  // Pillow's antialiased bilinear resize: both passes shrinking, enlarging, one pass only.
  const int aa[][4] = {{37, 23, 16, 10}, {13, 9, 32, 24}, {40, 8, 40, 5}, {40, 8, 20, 8}, {64, 43, 64, 43}};
  for (const auto& c : aa) {
    const std::vector<uint8_t> src = picture_u8(c[0], c[1], seed++);
    std::vector<uint8_t> dst((size_t)c[2] * c[3] * 3);
    antialiased_bilinear_resize_rgb(src.data(), c[0], c[1], c[0] * 3, dst.data(), c[2], c[3]);
    j["aa_resize"].push_back({{"sw", c[0]}, {"sh", c[1]}, {"dw", c[2]}, {"dh", c[3]},
                              {"src", std::vector<int>(src.begin(), src.end())},
                              {"out", std::vector<int>(dst.begin(), dst.end())}});
  }
  // Tone weights on a 4 x 3 grid of 8-pixel patches.
  {
    const int gw = 4, gh = 3, p = 8;
    std::vector<uint8_t> img = picture_u8(gw * p, gh * p, seed++);
    for (int i = 0; i < 3 * p * p; ++i) img[i] = 250;  // a bright corner (highlight bin)
    std::vector<float> w((size_t)gw * gh * 3);
    compute_tone_weights(img.data(), gw, gh, p, w.data());
    j["tone_weights"] = {{"gw", gw}, {"gh", gh}, {"patch", p}, {"img", std::vector<int>(img.begin(), img.end())},
                         {"out", w}};
  }
  return j;
}

// ---- Frame statistics, trained cells, residual -------------------------------------------------------------

static json stats_vectors(const ProfileBundle& bundle) {
  json j;
  const int sizes[][4] = {{120, 80, 64, 8}, {40, 30, 512, 16}, {75, 110, 96, 16}};
  uint32_t seed = 101;
  for (const auto& s : sizes) {
    const std::vector<uint16_t> px = negative(s[0], s[1], bundle.film_base_avg, 0.2f, 1.7f, seed++);
    FrameStats st;
    compute_frame_stats(negicc::RgbView::from_interleaved_rgb(px.data(), s[0], s[1], 3), 0.9f, s[2], s[3], st);
    j["frame_stats"].push_back({{"w", s[0]}, {"h", s[1]}, {"crop_ratio", 0.9f}, {"max", s[2]}, {"patch", s[3]},
                                {"px", px},
                                {"g_lo", st.g_lo}, {"g_hi", st.g_hi},
                                {"log_avg", {st.log_avg_rgb[0], st.log_avg_rgb[1], st.log_avg_rgb[2]}},
                                {"grid_w", st.grid_w}, {"grid_h", st.grid_h}, {"grid", st.grid}});
  }
  // The percentile on small histograms: one, two, ten pixels; q at 0, the tails and 1.
  const std::vector<std::vector<int>> samples = {{500}, {100, 9000}, {3, 3, 70, 70, 70, 1000, 2, 65535, 40000, 12}};
  for (const auto& smp : samples) {
    std::vector<uint64_t> hist(65536, 0);
    for (int v : smp) hist[v]++;
    for (float q : {0.0f, 0.005f, 0.25f, 0.5f, 0.995f, 1.0f}) {
      j["percentile"].push_back({{"values", smp}, {"q", q},
                                 {"out", u16_percentile_linear(hist, smp.size(), q)}});
    }
  }
  // Trained cells: a preview larger than the network's size (shrinking) and one smaller (enlarging).
  const int tc[][5] = {{45, 30, 4, 3, 8}, {20, 14, 4, 3, 8}, {64, 40, 8, 5, 8}};
  for (const auto& c : tc) {
    const std::vector<uint8_t> pv = picture_u8(c[0], c[1], seed++);
    const int n = c[2] * c[3];
    std::vector<float> lab((size_t)n * 3), mu((size_t)n * 3), sigma((size_t)n * 3);
    trained_cells_lab(pv.data(), c[0], c[1], c[2], c[3], c[4], lab.data());
    std::mt19937 rng(seed++);
    std::uniform_real_distribution<float> dm(-8.0f, 8.0f), ds(2.0f, 9.0f);
    for (int i = 0; i < n * 3; ++i) {
      mu[i] = lab[i] + dm(rng);
      sigma[i] = ds(rng);
    }
    const float res = self_consistency_residual(pv.data(), c[0], c[1], c[2], c[3], c[4], mu.data(), sigma.data());
    j["trained_cells"].push_back({{"w", c[0]}, {"h", c[1]}, {"gw", c[2]}, {"gh", c[3]}, {"patch", c[4]},
                                  {"preview", std::vector<int>(pv.begin(), pv.end())}, {"lab", lab}, {"mu", mu},
                                  {"sigma", sigma}, {"residual", res}});
  }
  return j;
}

// ---- The optimiser's cells and loss, and the optimiser -----------------------------------------------------

// A grid of n cells: a negative's cells at the reference exposure.
static std::vector<uint16_t> grid_cells(const ProfileBundle& bundle, int n, uint32_t seed) {
  const std::vector<uint16_t> px = negative(n, 1, bundle.film_base_avg, 0.15f, 1.8f, seed);
  return px;
}

static json optimiser_vectors(const ProfileBundle& bundle, int target_idx) {
  json j;
  const ProfileData& prof = bundle.targets[target_idx].film_prof.data;
  float merged[9];
  reference_input_matrix(bundle, merged);
  j["target_idx"] = target_idx;
  j["merged"] = std::vector<float>(merged, merged + 9);
  const int n = 12;
  const std::vector<uint16_t> grid = grid_cells(bundle, n, 7);
  j["grid"] = grid;
  // A field: the cells at "true" gains, perturbed, with spreads.
  SolverConfig cfg;
  cfg.prof = &prof;
  cfg.knee = 0.95f;
  cfg.shoulder = 1.0f;
  std::vector<float> truth((size_t)n * 3), mu((size_t)n * 3), sigma((size_t)n * 3);
  intent_cells(cfg, merged, grid.data(), n, {1.1f, 1.05f, 0.98f}, truth.data());
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> dm(-3.0f, 3.0f), ds(3.0f, 8.0f);
  for (int i = 0; i < n * 3; ++i) {
    mu[i] = truth[i] + dm(rng);
    sigma[i] = ds(rng);
  }
  j["mu"] = mu;
  j["sigma"] = sigma;
  // intent_cells and intent_loss at several candidates and shoulders.
  for (float shoulder : {1.0f, 1.3f, 2.2f}) {
    cfg.shoulder = shoulder;
    for (const auto& c : std::vector<IntentGains>{{1.0f, 1.0f, 1.0f}, {0.6f, 1.2f, 1.1f}, {2.4f, 0.9f, 1.3f},
                                                  {0.25f, 1.1f, 1.15f}}) {
      std::vector<float> lab((size_t)n * 3);
      intent_cells(cfg, merged, grid.data(), n, c, lab.data());
      IntentLoss l;
      intent_loss(cfg, merged, grid.data(), n, &c, 1, mu.data(), sigma.data(), &l);
      j["loss"].push_back({{"shoulder", shoulder}, {"cand", {c.e, c.g, c.b}}, {"lab", lab},
                           {"out", {l.L, l.a, l.b, l.clip}}});
    }
  }
  // The optimiser: windows, starts, pins, the pullback's enclosure.
  struct Case {
    const char* name;
    bool has_init;
    IntentGains init;
    bool has_bounds;
    float bounds[3];
    unsigned pins;
    float enc_min, enc_max;
    float shoulder;
  };
  const Case cases[] = {
      {"window", false, {}, true, {0.5f, 2.0f, 1.1f}, 0, 0, 0, 1.0f},
      {"window, start", true, {1.0f, 1.2f, 1.1f}, true, {0.7f, 1.6f, 1.1f}, 0, 0, 0, 1.0f},
      {"window, start, shoulder", true, {1.3f, 1.0f, 0.95f}, true, {0.9f, 1.8f, 1.2f}, 0, 0, 0, 1.6f},
      {"no pull", false, {}, true, {0.4f, 3.0f, 0.0f}, 0, 0, 0, 1.0f},
      {"pinned E", false, {}, true, {0.5f, 2.0f, 1.1f}, PIN_E, 0, 0, 1.0f},
      {"pinned E, start", true, {1.2f, 1.1f, 1.0f}, true, {0.5f, 2.0f, 1.1f}, PIN_E, 0, 0, 1.0f},
      {"pinned g b", false, {}, true, {0.5f, 2.0f, 1.1f}, PIN_G | PIN_B, 0, 0, 1.0f},
      {"pinned g", true, {1.0f, 1.2f, 1.1f}, true, {0.5f, 2.0f, 1.1f}, PIN_G, 0, 0, 1.0f},
      {"pullback screening", false, {}, true, {0.5f, 2.5f, 1.1f}, 0, 0.9f, 1.4f, 1.0f},
      {"pullback screening, infeasible", false, {}, true, {0.3f, 3.0f, 0.0f}, 0, 1.6f, 0.7f, 1.0f},
      {"pullback loop", true, {1.0f, 1.1f, 1.0f}, true, {0.6f, 1.8f, 0.0f}, 0, 0.9f, 1.4f, 1.0f},
      {"no window, start", true, {1.0f, 1.2f, 1.1f}, false, {0, 0, 0}, 0, 0, 0, 1.0f},
      {"no window, no start", false, {}, false, {0, 0, 0}, 0, 0, 0, 1.0f},
  };
  for (const Case& c : cases) {
    SolverConfig sc;
    sc.prof = &prof;
    sc.knee = 0.95f;
    sc.shoulder = c.shoulder;
    sc.enc_e_min = c.enc_min;
    sc.enc_e_max = c.enc_max;
    sc.pin_e = 1.05f;
    sc.pin_g = 1.12f;
    sc.pin_b = 1.08f;
    IntentGains out;
    const float score = optimize_target_for_intent(sc, merged, grid.data(), n, 1, mu.data(), sigma.data(),
                                                   c.has_init ? &c.init : nullptr, c.has_bounds ? c.bounds : nullptr,
                                                   c.pins, out);
    json row = {{"name", c.name}, {"shoulder", c.shoulder}, {"enc", {c.enc_min, c.enc_max}},
                {"pin_values", {sc.pin_e, sc.pin_g, sc.pin_b}},
                {"pins", {(c.pins & PIN_E) != 0, (c.pins & PIN_G) != 0, (c.pins & PIN_B) != 0}},
                {"out", {out.e, out.g, out.b}}, {"score", score}};
    row["init"] = c.has_init ? json::array({c.init.e, c.init.g, c.init.b}) : json(nullptr);
    row["bounds"] = c.has_bounds ? json::array({c.bounds[0], c.bounds[1], c.bounds[2]}) : json(nullptr);
    j["optimize"].push_back(row);
  }
  return j;
}

// ---- Brackets, metering, highlight statistics, frame scale -------------------------------------------------

static json bracket_vectors(const ProfileBundle& bundle) {
  json j;
  for (size_t i = 0; i < bundle.targets.size(); ++i) {
    for (const auto& g : std::vector<std::array<float, 2>>{{900.0f, 30000.0f}, {3000.0f, 9000.0f},
                                                           {150.0f, 60000.0f}, {20.0f, 64000.0f}}) {
      Enclosure r;
      if (!bracket_enclosure(bundle.targets[i], (int)i, g[0], g[1], r)) continue;
      j["enclosure"].push_back({{"idx", i}, {"g_lo", g[0]}, {"g_hi", g[1]}, {"feasible", r.feasible},
                                {"admissible", r.admissible}, {"inverted", r.usable_inverted},
                                {"e_min", r.e_min}, {"e_max", r.e_max}, {"search_lo", r.search_lo},
                                {"search_hi", r.search_hi}, {"e_center", r.e_center}, {"slack_ev", r.slack_ev},
                                {"fill_ratio", r.fill_ratio}});
    }
  }
  const std::vector<std::vector<float>> caps = {{5.0f, 6.0f, 7.0f},
                                                {3.0f, 6.0f, 6.5f, 7.0f, 2.0f},
                                                {8.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
                                                {6.0f, 7.0f, 7.0f, 5.0f, 5.2f, 6.9f, 7.1f},
                                                {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
  for (const auto& c : caps) {
    ProfileBundle b;
    b.targets.resize(c.size());
    for (size_t i = 0; i < c.size(); ++i) b.targets[i].tonal_capacity_ev = c[i];
    j["filter"].push_back({{"caps", c}, {"out", filter_admissible_targets(b)}});
  }
  return j;
}

static json metering_vectors(const ProfileBundle& bundle) {
  json j;
  std::mt19937 rng(17);
  std::uniform_real_distribution<float> d(-2.0f, 2.0f);
  for (int k = 0; k < 6; ++k) {
    float J[3][3], r[3], x[3];
    for (auto& row : J)
      for (float& v : row) v = d(rng);
    if (k == 4) {  // a rank-deficient matrix: the third row repeats the first
      for (int c = 0; c < 3; ++c) J[2][c] = J[0][c];
    }
    if (k == 5) {  // a zero matrix
      for (auto& row : J)
        for (float& v : row) v = 0.0f;
    }
    for (float& v : r) v = d(rng);
    lstsq3(J, r, 1e-6f, x);
    j["lstsq3"].push_back({{"J", {J[0][0], J[0][1], J[0][2], J[1][0], J[1][1], J[1][2], J[2][0], J[2][1], J[2][2]}},
                           {"r", {r[0], r[1], r[2]}}, {"x", {x[0], x[1], x[2]}}});
  }
  const float scenes[][3] = {{0.25f, 0.30f, 0.28f}, {0.05f, 0.06f, 0.07f}, {0.9f, 0.7f, 0.8f}};
  for (size_t i = 0; i < bundle.targets.size(); ++i) {
    const GreyPoint gp(bundle.targets[i].film_prof.data);
    float log_t[3];
    const bool ok = gp.reference_grey(log_t);
    json row = {{"idx", i}, {"ok", ok}, {"log_t", {log_t[0], log_t[1], log_t[2]}}};
    const float batch[3][3] = {{0.1f, 0.12f, 0.09f}, {0.5f, 0.4f, 0.45f}, {0.02f, 0.03f, 0.025f}};
    float lab[3][3];
    gp.lab(batch, 3, lab);
    row["batch"] = {{batch[0][0], batch[0][1], batch[0][2]}, {batch[1][0], batch[1][1], batch[1][2]},
                    {batch[2][0], batch[2][1], batch[2][2]}};
    row["batch_lab"] = {lab3(lab[0]), lab3(lab[1]), lab3(lab[2])};
    for (const auto& p : scenes) {
      float e = 0.0f;
      const bool m = exposure_for_lightness(gp, p, BOOTSTRAP_G, BOOTSTRAP_B, e);
      row["meter"].push_back({{"p", {p[0], p[1], p[2]}}, {"ok", m}, {"e", e}});
    }
    j["grey"].push_back(row);
  }
  return j;
}

static json knob_vectors(const ProfileBundle& bundle, int target_idx) {
  json j;
  for (float L : {5.0f, 8.0f, 30.0f, 50.0f, 90.0f}) {
    for (float st : {-2.0f, -0.5f, 0.0f, 0.7f, 3.0f}) {
      j["lstar_after_stops"].push_back({{"L", L}, {"stops", st}, {"out", lstar_after_stops(L, st)}});
      j["miss_stops"].push_back({{"brightness", st}, {"L", L}, {"out", knob_grey_miss_stops(st, L)}});
    }
  }
  const ProfileData& prof = bundle.targets[target_idx].film_prof.data;
  j["target_idx"] = target_idx;
  const float cases[][6] = {{1.0f, 1.1f, 1.15f, 0.0f, 0.0f, 0.0f},  {1.0f, 1.1f, 1.15f, 0.5f, 0.0f, 0.0f},
                            {2.0f, 1.2f, 0.9f, -0.7f, 3.0f, -4.0f}, {0.8f, 1.05f, 1.2f, 0.0f, -6.0f, 8.0f},
                            {1.0f, 1.1f, 1.15f, 4.0f, 0.0f, 0.0f},  {1.0f, 1.1f, 1.15f, 0.3f, 60.0f, -70.0f},
                            {1.0f, 1.1f, 1.15f, -6.0f, 0.0f, 0.0f}};
  for (const auto& c : cases) {
    KnobSolveResult r;
    const bool ok = solve_grey_knobs(prof, c[0], c[1], c[2], c[3], c[4], c[5], r);
    j["solve"].push_back({{"base", {c[0], c[1], c[2]}}, {"knobs", {c[3], c[4], c[5]}}, {"ok", ok},
                          {"egb", {r.e, r.g, r.b}}, {"limited", r.limited},
                          {"grey_lab", {r.grey_lab[0], r.grey_lab[1], r.grey_lab[2]}}});
  }
  return j;
}

static json clip_vectors(const ProfileBundle& bundle, int target_idx) {
  json j;
  const int w = 60, h = 40;
  const std::vector<uint16_t> px = negative(w, h, bundle.film_base_avg, 0.05f, 2.2f, 31);
  const negicc::RgbView view = negicc::RgbView::from_interleaved_rgb(px.data(), w, h, 3);
  const InnerCrop crop = compute_inner_crop(w, h, 0.9f);
  float merged[9];
  reference_input_matrix(bundle, merged);
  j["w"] = w;
  j["h"] = h;
  j["crop_ratio"] = 0.9f;
  j["px"] = px;
  j["merged"] = std::vector<float>(merged, merged + 9);
  j["target_idx"] = target_idx;
  const ProfileData& prof = bundle.targets[target_idx].film_prof.data;
  for (float shoulder : {1.0f, 1.5f}) {
    for (const auto& c : std::vector<std::array<float, 3>>{{1.0f, 1.1f, 1.15f}, {0.3f, 1.0f, 1.0f}}) {
      for (const auto& ceil : std::vector<std::array<float, 3>>{{242, 242, 242}, {235, 235, 235}, {255, 255, 255},
                                                                {250.3f, 248.9f, 252.0f}, {0.9f, 0.95f, 1.0f}}) {
        float m[9];
        negicc::candidate_matrix(merged, c[0], c[1], c[2], m);
        std::vector<uint64_t> hist(negicc::kMaxHistBins, 0);
        const ClipStats s =
            measure_highlight_clip(view, crop, std::vector<float>(m, m + 9), 0.95f, prof, ceil.data(), shoulder,
                                   hist.data())[0];
        float q = 0.0f;
        const float x_hi = shoulder_x_hi_from_hist(hist.data(), &q);
        j["cases"].push_back({{"shoulder", shoulder}, {"cand", {c[0], c[1], c[2]}},
                              {"ceil", {ceil[0], ceil[1], ceil[2]}}, {"exc", {s.exc[0], s.exc[1], s.exc[2]}},
                              {"white_clip", s.white_clip}, {"median_L", s.median_L}, {"x_hi", x_hi}, {"q995", q}});
      }
    }
  }
  return j;
}

static json bundle_data(const ProfileBundle& bundle, const std::string& dir) {
  json j;
  j["dir"] = dir;
  j["has_crosstalk"] = bundle.has_crosstalk;
  j["crosstalk"] = std::vector<float>(bundle.crosstalk_matrix, bundle.crosstalk_matrix + 9);
  j["normalization_target"] = bundle.normalization_target;
  j["film_base"] = {bundle.film_base_avg[0], bundle.film_base_avg[1], bundle.film_base_avg[2]};
  const float fb_t = bundle.film_base_shutter.empty() ? 0.125f : shutter_to_seconds(bundle.film_base_shutter);
  j["film_base_exposure"] = fb_t * (bundle.film_base_iso / 100.0f);
  for (const auto& t : bundle.targets) {
    j["targets"].push_back({{"name", t.name}, {"has_patches", !t.patches_g.empty()}, {"tgt_min_g", t.tgt_min_g},
                            {"tgt_max_g", t.tgt_max_g}, {"exposure", t.exposure_s},
                            {"tonal_capacity", t.tonal_capacity_ev}});
  }
  // frame_scale for a few scans.
  for (const auto& s : std::vector<std::array<float, 2>>{{0.125f, 100.0f}, {0.1f, 100.0f}, {0.4f, 200.0f}}) {
    ParamSearchOptions opts;
    const FrameScale fs = frame_scale(bundle, s[0], s[1], opts);
    j["frame_scale"].push_back({{"shutter", s[0]}, {"iso", s[1]}, {"custom", nullptr}, {"from_scan", false},
                                {"ratio", fs.ratio_frame}, {"e_norm", fs.e_norm},
                                {"matrix", std::vector<float>(fs.frame_matrix, fs.frame_matrix + 9)}});
  }
  for (bool from_scan : {false, true}) {
    ParamSearchOptions opts;
    const double fb[3] = {bundle.film_base_avg[0] * 1.1, bundle.film_base_avg[1] * 0.9, bundle.film_base_avg[2]};
    opts.film_base_rgb_d = fb;
    opts.film_base_from_scan = from_scan;
    const FrameScale fs = frame_scale(bundle, 0.2f, 100.0f, opts);
    j["frame_scale"].push_back({{"shutter", 0.2f}, {"iso", 100.0f},
                                {"custom", {(float)fb[0], (float)fb[1], (float)fb[2]}}, {"from_scan", from_scan},
                                {"ratio", fs.ratio_frame}, {"e_norm", fs.e_norm},
                                {"matrix", std::vector<float>(fs.frame_matrix, fs.frame_matrix + 9)}});
  }
  return j;
}

// ---- The whole search, with a stand-in for the intent network ----------------------------------------------

// The stand-in network: per cell, the mean m_c of the input's three planes over the cell's patch and the cell's tone
// weights (w_sh, w_mid, w_hi) give
//   mu    = (46 + 9 (m_r + m_g + m_b) / 3 + 4 (w_hi - w_sh),  1.5 + 3 (m_r - m_g),  4 + 3 (m_g - m_b))
//   sigma = (7 + 2 |m_g|,  4 + w_mid,  5)
// a fixed function of what the network is given, so the field follows the render as the real one does.
struct StandInRunner : DinoV3Runner {
  int patch = 16;
  const char* name() const override { return "stand-in"; }
  bool has_pool_weights() const override { return true; }
  void run(const float* pixel_values, int height, int width, const float* pool_weights, int patches,
           const DinoV3RunnerOutputs& outputs) override {
    const int gw = width / patch, gh = height / patch;
    if (gw * gh != patches) throw std::runtime_error("stand-in: grid does not match");
    const size_t plane = (size_t)width * height;
    for (int gy = 0; gy < gh; ++gy) {
      for (int gx = 0; gx < gw; ++gx) {
        float m[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
          for (int py = 0; py < patch; ++py)
            for (int px = 0; px < patch; ++px)
              m[c] += pixel_values[c * plane + (size_t)(gy * patch + py) * width + gx * patch + px];
          m[c] /= (float)(patch * patch);
        }
        const int i = gy * gw + gx;
        const float* w = pool_weights + i * 3;
        outputs.mu[i * 3 + 0] = 46.0f + 9.0f * (m[0] + m[1] + m[2]) / 3.0f + 4.0f * (w[2] - w[0]);
        outputs.mu[i * 3 + 1] = 1.5f + 3.0f * (m[0] - m[1]);
        outputs.mu[i * 3 + 2] = 4.0f + 3.0f * (m[1] - m[2]);
        outputs.sigma[i * 3 + 0] = 7.0f + 2.0f * std::fabs(m[1]);
        outputs.sigma[i * 3 + 1] = 4.0f + w[1];
        outputs.sigma[i * 3 + 2] = 5.0f;
      }
    }
  }
};

static const char* fb_status(FixedPointStatus s) {
  return s == FixedPointStatus::kConverged ? "converged" : s == FixedPointStatus::kDiverged ? "diverged" : "max_iters";
}

static json search_case(const ProfileBundle& bundle, const char* name, int w, int h, float d_lo, float d_hi,
                        uint32_t seed, float shutter, float iso, int pin_target, unsigned pin_mask, float pin_e,
                        float pin_g, float pin_b, int max_size, int patch) {
  DinoV3Engine engine;
  auto runner = std::make_unique<StandInRunner>();
  runner->patch = patch;
  engine.runner_ = std::move(runner);
  engine.max_size_ = max_size;
  engine.patch_size_ = patch;
  // The frame at the scan's exposure: the film base scaled from its capture's exposure.
  const float fb_t = bundle.film_base_shutter.empty() ? 0.125f : shutter_to_seconds(bundle.film_base_shutter);
  const float k = (shutter * iso / 100.0f) / (fb_t * bundle.film_base_iso / 100.0f);
  const float fb[3] = {bundle.film_base_avg[0] * k, bundle.film_base_avg[1] * k, bundle.film_base_avg[2] * k};
  const std::vector<uint16_t> px = negative(w, h, fb, d_lo, d_hi, seed);
  ParamSearchOptions opts;
  opts.verbose = false;
  opts.pin_target = pin_target;
  opts.pin_mask = pin_mask;
  opts.pin_e = pin_e;
  opts.pin_g = pin_g;
  opts.pin_b = pin_b;
  ParamSearchResult out;
  const bool ok = search_conversion_params(bundle, engine, negicc::RgbView::from_interleaved_rgb(px.data(), w, h, 3),
                                           shutter, iso, opts, out);
  json j = {{"name", name}, {"w", w}, {"h", h}, {"crop_ratio", opts.crop_ratio}, {"px", px}, {"shutter", shutter},
            {"iso", iso},
            {"pin_target", pin_target}, {"pins", {(pin_mask & PIN_E) != 0, (pin_mask & PIN_G) != 0,
                                                  (pin_mask & PIN_B) != 0}},
            {"pin_values", {pin_e, pin_g, pin_b}}, {"max_size", max_size}, {"patch", patch}, {"ok", ok}};
  if (!ok) return j;
  for (const auto& s : out.screening) {
    j["screening"].push_back({{"idx", s.target_idx}, {"params", {s.params.e, s.params.g, s.params.b}},
                              {"loss", s.intent_loss}, {"e_lo", s.e_lo}, {"e_hi", s.e_hi},
                              {"e_metered", s.e_metered}});
  }
  for (const auto& t : out.tournament) {
    j["tournament"].push_back({{"idx", t.target_idx},
                               {"params", {t.params.e, t.params.g, t.params.b, t.params.shoulder}},
                               {"residual", t.residual}, {"status", fb_status(t.status)}, {"steps", t.steps},
                               {"shoulder_guard", t.shoulder_guard}});
  }
  j["result"] = {{"idx", out.target_idx},
                 {"params", {out.params.e, out.params.g, out.params.b, out.params.shoulder}},
                 {"residual", out.residual},
                 {"status", fb_status(out.status)},
                 {"clip_fallback", to_string(out.clip_fallback)},
                 {"clip_fallback_white_clip", out.clip_fallback_white_clip},
                 {"highlight_pullback", out.highlight_pullback},
                 {"shoulder_guard", out.shoulder_guard},
                 {"e_range", {out.e_range_lo, out.e_range_hi}},
                 {"intent_mu", out.intent_mu},
                 {"e_norm", out.e_norm},
                 {"g_lo", out.g_lo},
                 {"g_hi", out.g_hi},
                 {"inferences", out.inferences}};
  return j;
}

// The neutral and tone heads' forward passes on weights of the files' layout, drawn at random (the trained weights are
// data; their formulas are what is checked).
static std::vector<uint8_t> head_file(const char* magic, const std::vector<uint32_t>& dims, const std::vector<float>& v) {
  std::vector<uint8_t> b(4 + 4 * dims.size() + 4 * v.size());
  std::memcpy(b.data(), magic, 4);
  std::memcpy(b.data() + 4, dims.data(), 4 * dims.size());
  std::memcpy(b.data() + 4 + 4 * dims.size(), v.data(), 4 * v.size());
  return b;
}

static json head_vectors() {
  json j;
  std::mt19937 rng(61);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f), pos(0.5f, 2.0f);
  auto draw = [&](size_t n, bool positive = false) {
    std::vector<float> v(n);
    for (float& x : v) x = positive ? pos(rng) : u(rng);
    return v;
  };
  {
    const uint32_t d = 6, h = 4;
    const std::vector<float> mean = draw(d), sd = draw(d, true), w1 = draw(h * d), b1 = draw(h), w2 = draw(h),
                             b2 = draw(1);
    std::vector<float> all;
    for (const std::vector<float>* p : {&mean, &sd, &w1, &b1, &w2, &b2}) all.insert(all.end(), p->begin(), p->end());
    const std::vector<uint8_t> file = head_file("NPH1", {d, h}, all);
    NeutralHead head;
    std::string err;
    if (!head.load(file.data(), file.size(), err)) throw std::runtime_error(err);
    const int n = 4;
    std::vector<float> tokens = draw(n * d);
    for (float& t : tokens) t *= 3.0f;
    std::vector<float> p(n);
    head.run(tokens.data(), n, p.data());
    j["neutral"] = {{"in_dim", d}, {"hidden", h}, {"mean", mean}, {"std", sd}, {"w1", w1}, {"b1", b1}, {"w2", w2},
                    {"b2", b2[0]}, {"tokens", tokens}, {"p", p}};
  }
  {
    const uint32_t d = ToneHead::kInDim, h1 = 3, h2 = 2, n = ToneHead::kTargets;
    const std::vector<float> xm = draw(d), xs = draw(d, true), w1 = draw(h1 * d), b1 = draw(h1), w2 = draw(h2 * h1),
                             b2 = draw(h2), w3 = draw(2 * n * h2), b3 = draw(2 * n), ym = draw(n), ys = draw(n, true);
    std::vector<float> w1s = w1;
    for (float& x : w1s) x *= 0.05f;  // keeps the first layer's sums of 1920 terms of order one
    std::vector<float> all;
    for (const std::vector<float>* p : std::initializer_list<const std::vector<float>*>{&xm, &xs, &w1s, &b1, &w2, &b2, &w3, &b3, &ym, &ys}) all.insert(all.end(), p->begin(), p->end());
    const std::vector<uint8_t> file = head_file("TNH1", {d, h1, h2, n}, all);
    ToneHead head;
    std::string err;
    if (!head.load(file.data(), file.size(), err)) throw std::runtime_error(err);
    for (int k = 0; k < 2; ++k) {
      const std::vector<float> f = draw(d);
      double mean[ToneHead::kTargets], sd[ToneHead::kTargets];
      head.run(f.data(), mean, sd);
      j["tone"]["cases"].push_back({{"features", f}, {"mean", std::vector<double>(mean, mean + n)},
                                    {"sd", std::vector<double>(sd, sd + n)}});
    }
    j["tone"]["dims"] = {d, h1, h2, n};
    j["tone"]["weights"] = {{"xm", xm}, {"xs", xs}, {"w1", w1s}, {"b1", b1}, {"w2", w2}, {"b2", b2}, {"w3", w3},
                            {"b3", b3}, {"ym", ym}, {"ys", ys}};
  }
  return j;
}

// The single-profile solver with the stand-in network, the brackets' exposures and the roll-off at given values.
static json given_vectors(const ProfileBundle& tri, const ProfileBundle& white) {
  json j;
  const int w = 48, h = 32;
  const std::vector<uint16_t> px = negative(w, h, tri.film_base_avg, 0.25f, 1.5f, 41);
  j["solver_image"] = {{"w", w}, {"h", h}, {"px", px}, {"target_idx", 2}, {"max_size", 32}, {"patch", 8}};
  float merged[9];
  reference_input_matrix(tri, merged);
  struct Case {
    const char* name;
    float shoulder;
    IntentGains user;
    unsigned pins;
  };
  const Case cases[] = {{"free", 1.0f, {1, 1, 1}, 0},
                        {"free, roll-off", 1.4f, {1.3f, 1.05f, 0.95f}, 0},
                        {"pinned E", 1.0f, {1.2f, 1, 1}, PIN_E},
                        {"pinned g b", 1.0f, {1, 1.1f, 0.9f}, PIN_G | PIN_B},
                        {"all pinned", 1.0f, {1.1f, 1.1f, 1.1f}, PIN_E | PIN_G | PIN_B}};
  for (const Case& c : cases) {
    DinoV3Engine engine;
    auto runner = std::make_unique<StandInRunner>();
    runner->patch = 8;
    engine.runner_ = std::move(runner);
    engine.max_size_ = 32;
    engine.patch_size_ = 8;
    SolverConfig cfg;
    cfg.prof = &tri.targets[2].film_prof.data;
    cfg.knee = 0.95f;
    cfg.shoulder = c.shoulder;
    float m[9];
    negicc::candidate_matrix(merged, c.user.e, c.user.g, c.user.b, m);
    DinoSolveResult out;
    const bool ok = solve_dinov3_intent(engine, cfg, m, negicc::RgbView::from_interleaved_rgb(px.data(), w, h, 3),
                                        c.user, c.pins, 8, out);
    j["solver"].push_back({{"name", c.name}, {"shoulder", c.shoulder}, {"user", {c.user.e, c.user.g, c.user.b}},
                           {"merged", std::vector<float>(m, m + 9)},
                           {"pins", {(c.pins & PIN_E) != 0, (c.pins & PIN_G) != 0, (c.pins & PIN_B) != 0}},
                           {"ok", ok}, {"status", out.status}, {"residual", out.best_residual},
                           {"relative", {out.solved_relative.e, out.solved_relative.g, out.solved_relative.b}},
                           {"absolute", {out.solved_absolute.e, out.solved_absolute.g, out.solved_absolute.b}},
                           {"e_center", out.e_center}, {"steps", (int)out.iterations.size()}});
  }
  // The brackets' exposures and the roll-off, on a frame of each bundle.
  for (int bi = 0; bi < 2; ++bi) {
    const ProfileBundle& b = bi == 0 ? tri : white;
    const int fw = 60, fh = 40;
    const float shutter = bi == 0 ? 0.1f : 0.0667f;
    const std::vector<uint16_t> f = negative(fw, fh, b.film_base_avg, 0.2f, 1.6f, 51 + bi);
    const negicc::RgbView view = negicc::RgbView::from_interleaved_rgb(f.data(), fw, fh, 3);
    std::vector<BracketExposure> ex;
    meter_bracket_exposures(b, view, shutter, 100.0, 0.9f, nullptr, ex);
    json row = {{"bundle", bi}, {"w", fw}, {"h", fh}, {"crop_ratio", 0.9f}, {"px", f}, {"shutter", shutter},
                {"iso", 100.0f}};
    for (const auto& e : ex) {
      row["exposures"].push_back({{"valid", e.valid}, {"feasible", e.feasible}, {"e_min", e.e_min},
                                  {"e_max", e.e_max}, {"e_center", e.e_center}, {"search_lo", e.search_lo},
                                  {"search_hi", e.search_hi}, {"has_metered", e.has_metered},
                                  {"e_metered", e.e_metered}});
    }
    for (const auto& g : std::vector<std::array<float, 3>>{{1.0f, 1.1f, 1.15f}, {0.25f, 1.0f, 1.0f}, {3.0f, 0.9f, 1.2f}}) {
      float q = 0.0f;
      const float x = measure_rolloff_x_hi(b, 3, view, shutter, 100.0, g[0], g[1], g[2], 0.9f, &q);
      row["rolloff"].push_back({{"target_idx", 3}, {"params", {g[0], g[1], g[2]}}, {"x_hi", x}});
    }
    j["frames"].push_back(row);
  }
  return j;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: gen_auto <TriLED bundle directory> <white-light bundle directory> <output directory>\n";
    return 2;
  }
  const std::string tri_dir = argv[1], white_dir = argv[2], out_dir = argv[3];
  ProfileBundle tri, white;
  load_profile_bundle(tri_dir + "/bundle.json", tri);
  load_profile_bundle(white_dir + "/bundle.json", white);

  json a;
  a["bundle"] = bundle_data(tri, tri_dir);
  a["geometry"] = geometry_vectors();
  a["resample"] = resample_vectors();
  a["stats"] = stats_vectors(tri);
  a["optimiser"] = optimiser_vectors(tri, 2);
  a["brackets"] = bracket_vectors(tri);
  a["metering"] = metering_vectors(tri);
  a["clip"] = clip_vectors(tri, 3);
  a["knobs"] = knob_vectors(tri, 2);
  a["given"] = given_vectors(tri, white);
  a["heads"] = head_vectors();
  a["bundle_white"] = bundle_data(white, white_dir);
  write_json(out_dir + "/auto.json", a);

  json s;
  s["bundles"] = {bundle_data(tri, tri_dir), bundle_data(white, white_dir)};
  struct Run {
    int bundle;
    const char* name;
    float d_lo, d_hi;
    uint32_t seed;
    float shutter;
    int pin_target;
    unsigned pins;
    float pe, pg, pb;
  };
  const Run runs[] = {
      {0, "TriLED, mid contrast", 0.25f, 1.6f, 1, 0.1f, -1, 0, 1, 1, 1},
      {0, "TriLED, dense highlights", 0.35f, 2.7f, 2, 0.1f, -1, 0, 1, 1, 1},
      {0, "TriLED, thin and flat", 0.08f, 0.55f, 3, 0.2f, -1, 0, 1, 1, 1},
      {0, "TriLED, pinned bracket", 0.25f, 1.6f, 1, 0.1f, 4, 0, 1, 1, 1},
      {0, "TriLED, pinned E", 0.25f, 1.6f, 1, 0.1f, -1, PIN_E, 0.9f, 1, 1},
      {0, "TriLED, pinned gains", 0.25f, 1.6f, 1, 0.1f, -1, PIN_G | PIN_B, 1, 1.05f, 1.2f},
      {1, "white light, mid contrast", 0.25f, 1.6f, 4, 0.0667f, -1, 0, 1, 1, 1},
      {1, "white light, dense highlights", 0.35f, 2.7f, 5, 0.0667f, -1, 0, 1, 1, 1},
      {0, "TriLED, highlight pullback (a)", 0.4f, 2.0f, 9, 0.1f, -1, 0, 1, 1, 1},
      {0, "TriLED, highlight pullback (b)", 0.4f, 2.8f, 1, 0.1f, -1, 0, 1, 1, 1},
      {1, "white light, clip fallback taken", 0.05f, 1.2f, 1, 0.05f, -1, 0, 1, 1, 1},
      {1, "white light, highlight pullback", 0.2f, 2.8f, 1, 0.05f, -1, 0, 1, 1, 1},
  };
  for (const Run& r : runs) {
    const ProfileBundle& b = r.bundle == 0 ? tri : white;
    json row = search_case(b, r.name, 72, 48, r.d_lo, r.d_hi, r.seed, r.shutter, 100.0f, r.pin_target, r.pins, r.pe,
                           r.pg, r.pb, 64, 8);
    row["bundle"] = r.bundle;
    s["searches"].push_back(row);
  }
  write_json(out_dir + "/auto_search.json", s);
  return 0;
}
