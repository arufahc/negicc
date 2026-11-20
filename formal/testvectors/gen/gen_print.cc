// Test vectors of the print emulation for the Lean specification in formal/ (Negicc.Print): inputs and the values
// neg_process's own functions compute from them, written as JSON. The Lean executable `checkPrint`
// (formal/CheckPrint.lean) evaluates the specification on the same inputs and compares. See formal/README.md for how
// to build and run this program.
//
// Every value is computed by the library's functions (np_print.h, profile_bundle.h, color_pipeline.h,
// multi_target_optimizer.h, and np_pipeline.cc's to_paper_white through gen_print_white.cc); nothing is
// re-implemented here. The CLF texts are read back with a small number reader, which only parses.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <lcms2.h>
#include <nlohmann/json.hpp>

#include "color_pipeline.h"
#include "embedded_data.h"
#include "multi_target_optimizer.h"
#include "np_print.h"
#include "profile_bundle.h"
#include "rgb_view.h"

using nlohmann::json;

// np_pipeline.cc's to_paper_white (file-local there), exposed by gen_print_white.cc.
void gen_to_paper_white(ParsedFilmProfile& p, double* paper_xyz);

// JSON with one line per object member and per array element, except that an array of plain values stays on one
// line (as gen_vectors.cc writes it).
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

static json arr(const double* v, int n) {
  json a = json::array();
  for (int i = 0; i < n; ++i) a.push_back(v[i]);
  return a;
}
static json arr(const std::vector<double>& v) { return json(v); }
static json arr9(const double* m) { return arr(m, 9); }

// The indices of a table of n entries at which values are written: every `step`-th, and the last.
static std::vector<int> sample_indices(int n, int step) {
  std::vector<int> ix;
  for (int i = 0; i < n; i += step) ix.push_back(i);
  if (ix.back() != n - 1) ix.push_back(n - 1);
  return ix;
}

// The numbers of the k-th <Array ...> ... </Array> block of a CLF text (k from 0), in order.
static std::vector<double> clf_array(const std::string& clf, int k) {
  size_t at = 0;
  for (int i = 0; i <= k; ++i) {
    at = clf.find("<Array", at);
    if (at == std::string::npos) throw std::runtime_error("CLF: no Array block " + std::to_string(k));
    if (i < k) at += 6;
  }
  const size_t open = clf.find('>', at) + 1, close = clf.find("</Array>", open);
  std::vector<double> out;
  const char* p = clf.c_str() + open;
  const char* end = clf.c_str() + close;
  while (p < end) {
    char* next = nullptr;
    const double v = std::strtod(p, &next);
    if (next == p) {
      ++p;
      continue;
    }
    out.push_back(v);
    p = next;
  }
  return out;
}
// The text between <tag> and </tag>, as a number.
static double clf_value(const std::string& clf, const std::string& tag) {
  const size_t at = clf.find("<" + tag + ">");
  if (at == std::string::npos) throw std::runtime_error("CLF: no " + tag);
  return std::strtod(clf.c_str() + at + tag.size() + 2, nullptr);
}

static const char* kMeasureKey[kDensityMeasures] = {"statusm", "rp180", "apd"};

// ---- What the bundle gives the print: its film model, density matrices, grey balance curve and brackets ----
static json bundle_inputs(const ProfileBundle& bundle, const std::string& bundle_dir) {
  json j;
  j["bundle_dir"] = bundle_dir;
  j["key"] = bundle.key;
  j["has_crosstalk"] = bundle.has_crosstalk;
  j["normalization_target"] = bundle.normalization_target;
  j["crosstalk"] = arr9(bundle.crosstalk_matrix_d);
  j["film_base"] = arr(bundle.film_base_avg_d, 3);
  for (int k = 0; k < kDensityMeasures; ++k) {
    const DensityMatrix& d = bundle.density_matrix[k];
    j["density_matrices"][kMeasureKey[k]] = {{"present", d.present},
                                             {"m", arr9(d.m)},
                                             {"film_to_measure", arr9(d.film_to_measure)},
                                             {"light_to_film", arr9(d.light_to_film)},
                                             {"light_density_factors", arr(d.light_density_factors, 3)}};
  }
  const GreyBalanceCurves& g = bundle.grey_balance;
  j["grey_balance"] = {{"start", g.start}, {"step", g.step}, {"apd", {g.apd[0], g.apd[1], g.apd[2]}}};
  for (const BundleTarget& t : bundle.targets) j["targets"].push_back(t.name);
  for (const DensityBracket& b : bundle.density_brackets) {
    j["brackets"].push_back({{"name", b.name}, {"grey_light", arr(b.grey_light, 3)}, {"tilt", arr(b.tilt, 3)},
                             {"mid_apd", bracket_mid_apd(bundle, b)}});
  }
  const FilmModel& f = bundle.film_model;
  j["film_model"] = {{"name", f.name},
                     {"dye", {f.dye[0], f.dye[1], f.dye[2]}},
                     {"base", f.base},
                     {"log_exposure", f.log_exposure},
                     {"neutral", {f.neutral[0], f.neutral[1], f.neutral[2]}}};
  return j;
}

// ---- The grey balance curve's grey, slopes and the brackets' tilt at a mid grey A ----
static json grey_vectors(const ProfileBundle& bundle) {
  json j = json::array();
  double lo = INFINITY, hi = -INFINITY;
  for (double v : bundle.grey_balance.apd[1]) lo = std::min(lo, v), hi = std::max(hi, v);
  std::vector<double> as;
  for (int i = 0; i <= 24; ++i) as.push_back(lo - 0.05 + (hi - lo + 0.1) * i / 24.0);
  for (const DensityBracket& b : bundle.density_brackets) as.push_back(bracket_mid_apd(bundle, b));
  for (double a : as) {
    double light[3], slopes[3], tilt[3];
    curve_grey_at(bundle, a, light);
    curve_slopes_at(bundle, a, slopes);
    print_tilt_at(bundle, a, tilt);
    j.push_back({{"a", a}, {"grey_light", arr(light, 3)}, {"slopes", arr(slopes, 3)}, {"tilt", arr(tilt, 3)}});
  }
  return j;
}

// The bundle's merged matrix for a scan at 1/8 s and ISO 100 with the bundle's film base (profile_bundle.h).
static void reference_merged(const ProfileBundle& bundle, float merged[9]) {
  double merged_d[9], row_scale[3];
  compute_station_merged_matrix(bundle, 0.125, 100.0, nullptr, false, merged, merged_d, row_scale);
}

// Pixels around the film base: thinner and denser film, bare film, and a pixel of zeros.
static std::vector<std::array<uint16_t, 3>> test_pixels(const ProfileBundle& bundle) {
  std::vector<std::array<uint16_t, 3>> px;
  const double* fb = bundle.film_base_avg_d;
  for (double f : {1.2, 1.0, 0.9, 0.5, 0.3, 0.12, 0.05, 0.01, 0.002}) {
    px.push_back({(uint16_t)std::lround(std::min(65535.0, fb[0] * f)), (uint16_t)std::lround(std::min(65535.0, fb[1] * f)),
                  (uint16_t)std::lround(std::min(65535.0, fb[2] * f))});
  }
  std::mt19937 rng(20261120);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int k = 0; k < 16; ++k) {
    std::array<uint16_t, 3> p;
    for (int c = 0; c < 3; ++c) p[c] = (uint16_t)std::lround(std::min(65535.0, fb[c] * std::pow(10.0, -2.2 * u(rng)) * (0.85 + 0.3 * u(rng))));
    px.push_back(p);
  }
  px.push_back({0, 0, 0});
  return px;
}

// ---- One print material on one bundle ----
static json material_vectors(const ProfileBundle& bundle, const std::string& bundle_dir, const std::string& id,
                             bool full) {
  json j;
  const EmbeddedFile* f = find_embedded_file("3rd_party/spektrafilm/profiles/" + id + ".json");
  if (!f) throw std::runtime_error("no built-in print material " + id);
  PrintMaterial m;
  std::string err;
  if (!load_spektrafilm_print(std::string((const char*)f->data, f->size), id, bundle, m, err)) {
    throw std::runtime_error(id + ": " + err);
  }
  j["id"] = id;
  j["profile"] = "3rd_party/spektrafilm/profiles/" + id + ".json";
  j["viewing_illuminant"] = m.viewing_illuminant;
  j["log_exposure"] = m.log_exposure;
  j["curves"] = {m.curves[0], m.curves[1], m.curves[2]};
  j["dye"] = {m.dye[0], m.dye[1], m.dye[2]};
  j["base"] = m.base;
  j["observer"] = {m.observer[0], m.observer[1], m.observer[2]};
  j["white"] = arr(m.white, 3);
  j["aim"] = arr(m.aim, 3);
  j["to_srgb"] = arr9(m.to_srgb);
  for (int k = 0; k < kDensityMeasures; ++k) {
    if (!m.to_print[k].empty()) j["to_print"][kMeasureKey[k]] = json(m.to_print[k].front().m);
  }
  double paper[3];
  print_paper_xyz(m, paper);
  j["paper_xyz"] = arr(paper, 3);

  // The print's curves and table.
  const double* p_apd = m.to_print[kDensityApd].front().m.data();
  const PrintCurves paper_c = paper_curve(m, p_apd);
  j["paper_curve"] = {paper_c[0], paper_c[1], paper_c[2]};
  const double n0 = bundle.normalization_target / 65535.0;
  const std::vector<int> ix = sample_indices(kPrintIccPoints, 15);
  j["curve_indices"] = ix;
  auto sampled = [&](const PrintCurves& c) {
    json s = json::array();
    for (int ch = 0; ch < 3; ++ch) {
      json one = json::array();
      for (int i : ix) one.push_back(c[ch][i]);
      s.push_back(one);
    }
    return s;
  };
  for (const DensityBracket& b : bundle.density_brackets) {
    j["bracket_curves"].push_back({{"bracket", b.name}, {"a_mid", bracket_mid_apd(bundle, b)},
                                   {"tilt", arr(b.tilt, 3)},
                                   {"curves", sampled(bracket_print_curves(bundle.grey_balance, paper_c,
                                                                           bracket_mid_apd(bundle, b), n0, b.tilt))}});
  }
  BracketPrints prints;
  if (!bracket_print_profiles(bundle, id, prints, err)) throw std::runtime_error(id + ": " + err);
  const PrintContinuum& pc = prints.continuum;
  j["from_bundle"] = prints.from_bundle;
  j["a_lo"] = pc.a_lo();
  j["a_hi"] = pc.a_hi();
  for (const BundlePrint& bp : bundle.prints) {
    if (bp.id == id) {
      j["icc"] = bundle_dir + "/print_" + id + ".icc";
      j["reference_bracket"] = bp.reference_bracket;
    }
  }
  struct Case { double a, g, b; bool gains; };
  std::vector<Case> cases;
  const double span = pc.a_hi() - pc.a_lo();
  cases.push_back({pc.a_lo() + 0.45 * span, 1.0, 1.0, false});
  cases.push_back({pc.a_lo() + 0.3 * span, 0.93, 1.08, true});
  cases.push_back({pc.a_lo() + 0.62 * span, 1.12, 0.9, true});
  cases.push_back({pc.a_lo() - 0.2, 1.0, 1.0, false});
  for (const Case& c : cases) {
    const double gains[3] = {1.0, c.g, c.b};
    double t[3];
    pc.tilt_at(c.a, t);
    std::vector<float> trc[3];
    pc.trc_at(c.a, c.gains ? gains : nullptr, trc);
    json trc_s = json::array();
    for (int ch = 0; ch < 3; ++ch) {
      json one = json::array();
      for (int i : ix) one.push_back(trc[ch][i]);
      trc_s.push_back(one);
    }
    j["continuum"].push_back({{"a", c.a}, {"gains", c.gains ? json::array({1.0, c.g, c.b}) : json(nullptr)},
                              {"tilt", arr(t, 3)}, {"curves", sampled(pc.curves_at(c.a, c.gains ? gains : nullptr))},
                              {"trc", trc_s}});
  }
  if (!full) return j;

  // The table at every fourth node along each axis.
  const std::vector<double> table = print_table(m, p_apd, bundle.density_matrix[kDensityApd].m, paper_c);
  json nodes = json::array();
  for (int r = 0; r < kPrintIccCube; r += 4)
    for (int g = 0; g < kPrintIccCube; g += 4)
      for (int b = 0; b < kPrintIccCube; b += 4) {
        const size_t o = (((size_t)r * kPrintIccCube + g) * kPrintIccCube + b) * 3;
        nodes.push_back({r, g, b, table[o], table[o + 1], table[o + 2]});
      }
  j["table_nodes"] = nodes;

  // The print profile as the conversion renders it: the first bracket's print ICC, read by Little-CMS, made
  // relative to the paper's white, its curves at A with gains, through the conversion's pixel stages.
  {
    std::vector<uint8_t> icc = prints.icc[0];
    cmsHPROFILE h = cmsOpenProfileFromMem(icc.data(), (cmsUInt32Number)icc.size());
    ParsedFilmProfile prof;
    if (!h || !parse_film_profile_pipeline(h, prof)) throw std::runtime_error(id + ": print ICC");
    cmsCloseProfile(h);
    prof.rebind_pointers();
    double paper_xyz[3];
    gen_to_paper_white(prof, paper_xyz);
    j["paper_white"] = {{"paper_xyz", arr(paper_xyz, 3)}};
    json clut_s = json::array();
    for (size_t i = 0; i < prof.clut.size(); i += 997) clut_s.push_back({i, prof.clut[i]});
    j["paper_white"]["clut"] = clut_s;
    // The embedded profile's table, scaled the same way (np_pipeline.cc prepare: scale_print_icc_table).
    double k[3];
    for (int c = 0; c < 3; ++c) k[c] = negicc::kPcsWhite[c] / paper_xyz[c];
    std::vector<uint8_t> scaled = prints.icc[0];
    scale_print_icc_table(scaled, k);
    const size_t off = [&] {
      const uint32_t n = (uint32_t)scaled[128] << 24 | (uint32_t)scaled[129] << 16 | (uint32_t)scaled[130] << 8 | scaled[131];
      for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* t = scaled.data() + 132 + 12 * i;
        if (std::string((const char*)t, 4) == "A2B0") {
          return (size_t)((uint32_t)t[4] << 24 | (uint32_t)t[5] << 16 | (uint32_t)t[6] << 8 | t[7]);
        }
      }
      return (size_t)0;
    }();
    const size_t n_in = (size_t)scaled[off + 48] << 8 | scaled[off + 49];
    const size_t at = off + 52 + 6 * n_in;
    json codes = json::array();
    for (size_t i = 0; i < (size_t)kPrintIccCube * kPrintIccCube * kPrintIccCube * 3; i += 991) {
      codes.push_back({i, (int)scaled[at + 2 * i] << 8 | scaled[at + 2 * i + 1]});
    }
    j["paper_white"]["scaled_codes"] = codes;
    j["paper_white"]["icc_bracket"] = bundle.targets[0].name;

    float merged[9];
    reference_merged(bundle, merged);
    j["pixels"]["merged"] = json(std::vector<float>(merged, merged + 9));
    const auto px = test_pixels(bundle);
    for (const auto& p : px) j["pixels"]["in"].push_back({p[0], p[1], p[2]});
    for (const Case& c : cases) {
      const double gains[3] = {1.0, c.g, c.b};
      pc.set_curves_at(c.a, prof, c.gains ? gains : nullptr);
      const negicc::PipelineConfig cfg = make_pipeline_config(prof.data, kKnee, 1.0f);
      for (float e : {1.0f, 0.8f}) {
        float mat[9];
        negicc::candidate_matrix(merged, e, 1.0f, 1.0f, mat);
        json outs = json::array(), disp = json::array(), lab = json::array();
        for (const auto& p : px) {
          const negicc::Rgb v = negicc::convert_pixel_pcs(cfg, mat, p.data());
          outs.push_back({v.r, v.g, v.b});
          // The print as the conversion shows and scores it: no shoulder (x_hi 1), no tone stage.
          const negicc::Rgb d = negicc::convert_pixel_display(cfg, mat, p.data());
          disp.push_back({d.r, d.g, d.b});
          const negicc::Lab l = negicc::linear_srgb_to_lab(d);
          lab.push_back({l.L, l.a, l.b});
        }
        j["pixels"]["cases"].push_back({{"a", c.a}, {"gains", c.gains ? json::array({1.0, c.g, c.b}) : json(nullptr)},
                                         {"e", e}, {"pcs", outs}, {"display", disp}, {"lab", lab}});
      }
    }
  }
  return j;
}

// ---- The density file ----
static json density_vectors(const ProfileBundle& bundle) {
  json j;
  float merged[9];
  reference_merged(bundle, merged);
  const double base_level = bundle.normalization_target / 65535.0;
  j["merged"] = json(std::vector<float>(merged, merged + 9));
  j["base_level"] = base_level;
  const auto px = test_pixels(bundle);
  for (const auto& p : px) j["in"].push_back({p[0], p[1], p[2]});
  for (int k = 0; k < kDensityMeasures; ++k) {
    for (bool dpx : {false, true}) {
      const DensityEncoding enc = density_encoding(dpx, (DensityMeasure)k);
      std::vector<uint16_t> buf;
      for (const auto& p : px) buf.insert(buf.end(), p.begin(), p.end());
      DensityClip clip;
      density_encode_in_place(buf.data(), (int)px.size(), 1, 3, merged, base_level, bundle.density_matrix[k].m, enc,
                              clip);
      j["encode"].push_back({{"measure", kMeasureKey[k]},
                             {"dpx", dpx},
                             {"bits", enc.bits},
                             {"offset", enc.offset},
                             {"scale", enc.scale},
                             {"k", arr(enc.k, 3)},
                             {"codes", buf},
                             {"floor", {clip.floor[0], clip.floor[1], clip.floor[2]}},
                             {"low", {clip.low[0], clip.low[1], clip.low[2]}},
                             {"high", {clip.high[0], clip.high[1], clip.high[2]}}});
    }
    DensityRecord rec;
    rec.measure = (DensityMeasure)k;
    rec.matrix = &bundle.density_matrix[k];
    rec.has_balance = true;
    for (const auto& gb : std::vector<std::array<double, 2>>{{1.0, 1.0}, {0.91, 1.13}, {1.21, 0.84}}) {
      rec.balance_e = 1.3;
      rec.balance_g = gb[0];
      rec.balance_b = gb[1];
      double off[3], mean = 0.0;
      density_balance_offset(rec, off, &mean);
      j["balance"].push_back({{"measure", kMeasureKey[k]}, {"g", gb[0]}, {"b", gb[1]}, {"offset", arr(off, 3)},
                              {"mean", mean}});
    }
  }
  return j;
}

// ---- A frame's print of a density file: the CLF's first matrix and the material's nodes ----
static json clf_vectors(const ProfileBundle& bundle, const std::string& id) {
  json j;
  const EmbeddedFile* f = find_embedded_file("3rd_party/spektrafilm/profiles/" + id + ".json");
  PrintMaterial m;
  std::string err;
  if (!f || !load_spektrafilm_print(std::string((const char*)f->data, f->size), id, bundle, m, err)) {
    throw std::runtime_error(id + ": " + err);
  }
  j["id"] = id;
  // The material's nodes: the exposure matrix's offsets, the range, the curve table, the scale and the colour cube.
  const std::string nodes = print_clf_nodes(m);
  {
    const std::vector<double> m0 = clf_array(nodes, 0), lut1 = clf_array(nodes, 1), diag = clf_array(nodes, 2),
                              cube = clf_array(nodes, 3);
    j["nodes"]["exposure_matrix"] = m0;
    j["nodes"]["range"] = {clf_value(nodes, "minInValue"), clf_value(nodes, "maxInValue")};
    // The LUT1D: 4096 rows of three; the LUT3D: 45^3 rows of three, blue fastest (print_clf_nodes' defaults).
    const int n1 = (int)(lut1.size() / 3);
    json rows = json::array();
    for (int i : sample_indices(n1, 97)) rows.push_back({i, lut1[3 * i], lut1[3 * i + 1], lut1[3 * i + 2]});
    j["nodes"]["lut1d_points"] = n1;
    j["nodes"]["lut1d"] = rows;
    j["nodes"]["scale"] = diag[0];
    const int n3 = (int)std::lround(std::cbrt((double)(cube.size() / 3)));
    json crows = json::array();
    for (int i : sample_indices(n3 * n3 * n3, 997)) {
      crows.push_back({i, cube[3 * i], cube[3 * i + 1], cube[3 * i + 2]});
    }
    j["nodes"]["cube_points"] = n3;
    j["nodes"]["lut3d"] = crows;
  }
  // The frame's first matrix for each measure and encoding: without a print search (the picture search's bracket and
  // exposure), and with the material's own print search (its A, E and gains).
  for (int k = 0; k < kDensityMeasures; ++k) {
    for (bool dpx : {false, true}) {
      for (int own = 0; own < 2; ++own) {
        DensityRecord rec;
        rec.measure = (DensityMeasure)k;
        rec.matrix = &bundle.density_matrix[k];
        rec.bundle = &bundle;
        rec.encoding = density_encoding(dpx, (DensityMeasure)k);
        rec.bundle_key = bundle.key;
        rec.has_balance = true;
        rec.balance_e = 1.4;
        rec.balance_g = 0.94;
        rec.balance_b = 1.07;
        rec.balance_target = bundle.targets[2].name;
        if (own) {
          DensityRecord::PrintSearch s;
          const double a_mid = bracket_mid_apd(bundle, bundle.density_brackets[3]);
          s.a = a_mid + 0.137;
          s.e = 1.0;
          s.g = 1.05;
          s.b = 0.96;
          s.bracket = bundle.density_brackets[3].name;
          rec.print_search[m.id] = s;
        }
        FramePrint fp;
        if (!frame_print(rec, m, fp, err)) throw std::runtime_error("frame_print: " + err);
        const std::string clf = frame_print_clf(rec, fp, "");
        j["frames"].push_back({{"measure", kMeasureKey[k]},
                               {"dpx", dpx},
                               {"own_search", (bool)own},
                               {"balance_e", rec.balance_e},
                               {"balance_g", rec.balance_g},
                               {"balance_b", rec.balance_b},
                               {"bracket", fp.bracket},
                               {"a", own ? json(fp.a) : json(nullptr)},
                               {"search_e", fp.exposure},
                               {"search_g", fp.gain_g},
                               {"search_b", fp.gain_b},
                               {"to_print", arr9(fp.to_print)},
                               {"grey", arr(fp.grey, 3)},
                               {"slopes", arr(fp.slopes, 3)},
                               {"matrix", clf_array(clf, 0)}});
      }
    }
  }
  return j;
}

// ---- The film base held no darker than the picture's thinnest film ----
static json floor_vectors(const ProfileBundle& bundle) {
  json j;
  const int w = 37, h = 23;
  std::vector<uint16_t> img((size_t)w * h * 3);
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double* fb = bundle.film_base_avg_d;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c) {
        // Mostly denser than the base, a few thinner (a base estimate darker than the thinnest film).
        const double f = u(rng) < 0.02 ? 1.0 + 0.3 * u(rng) : std::pow(10.0, -2.0 * u(rng));
        img[((size_t)y * w + x) * 3 + c] = (uint16_t)std::lround(std::min(65535.0, fb[c] * f));
      }
  negicc::RgbView view;
  view.width = w;
  view.height = h;
  view.r_ptr = img.data();
  view.g_ptr = img.data() + 1;
  view.b_ptr = img.data() + 2;
  view.pitch = 3;
  view.stride = (ptrdiff_t)w * 3;
  j["width"] = w;
  j["height"] = h;
  j["pixels"] = img;
  for (double scale : {1.0, 1.15, 0.8}) {
    for (float crop : {1.0f, 0.8f}) {
      const double base[3] = {fb[0] * scale, fb[1] * scale, fb[2] * scale};
      const negicc::InnerCrop c = negicc::compute_inner_crop(w, h, crop);
      j["cases"].push_back({{"base", arr(base, 3)},
                            {"crop_ratio", crop},
                            {"crop", {c.x0, c.y0, c.w, c.h}},
                            {"factor", print_floor_factor(bundle, view, crop, base)}});
    }
  }
  // The base before and after the floor (raise_base_to_print_floor): the bundle's base scaled by the exposures, or a
  // base found on the scan, on a scan at 1/4 s and ISO 100.
  for (int which = 0; which < 2; ++which) {
    const double found[3] = {fb[0] * 1.1, fb[1] * 1.05, fb[2] * 0.97};
    double base_in[3], base_out[3];
    const float k = raise_base_to_print_floor(bundle, view, 1.0f, which ? found : nullptr, 0.25, 100.0, base_in,
                                              base_out);
    j["raise"].push_back({{"found", which ? arr(found, 3) : json(nullptr)},
                          {"scan_shutter", 0.25},
                          {"scan_iso", 100.0},
                          {"film_base_shutter", shutter_to_seconds_d(bundle.film_base_shutter)},
                          {"film_base_iso", bundle.film_base_iso},
                          {"base_in", arr(base_in, 3)},
                          {"base_out", arr(base_out, 3)},
                          {"factor", k}});
  }
  return j;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: gen_print <TriLED bundle dir> <output dir> [white-light bundle dir]\n";
    return 2;
  }
  try {
    const std::string dir = argv[1];
    ProfileBundle bundle;
    load_profile_bundle(dir + "/bundle.json", bundle);
    json j;
    j["bundle"] = bundle_inputs(bundle, dir);
    j["grey"] = grey_vectors(bundle);
    j["materials"].push_back(material_vectors(bundle, dir, "kodak_portra_endura", true));
    j["materials"].push_back(material_vectors(bundle, dir, "kodak_2383", true));
    j["density"] = density_vectors(bundle);
    j["clf"] = clf_vectors(bundle, "kodak_portra_endura");
    j["floor"] = floor_vectors(bundle);
    if (argc > 3) {
      const std::string wdir = argv[3];
      ProfileBundle white;
      load_profile_bundle(wdir + "/bundle.json", white);
      j["white_light"]["bundle"] = bundle_inputs(white, wdir);
      j["white_light"]["grey"] = grey_vectors(white);
      j["white_light"]["materials"].push_back(material_vectors(white, wdir, "fujifilm_crystal_archive_typeii", false));
    }
    const std::string path = std::string(argv[2]) + "/print.json";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    emit(out, j, 0);
    out << "\n";
    std::cout << "wrote " << path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "gen_print: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
