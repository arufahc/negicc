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
#include <fstream>
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
  FILE* fp = fopen(raw_path.c_str(), "rb");
  if (!fp) {
    std::string alt = "profiles/" + raw_path;
    fp = fopen(alt.c_str(), "rb");
    if (!fp) {
      size_t last_slash = raw_path.find_last_of("/\\");
      if (last_slash != std::string::npos) {
        std::string base = "profiles/" + raw_path.substr(last_slash + 1);
        fp = fopen(base.c_str(), "rb");
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

static bool parse_3x3_matrix(const nlohmann::json& node, float out_m[9]) {
  if (!node.is_array() || node.size() != 3) return false;
  for (int r = 0; r < 3; ++r) {
    if (!node[r].is_array() || node[r].size() != 3) return false;
    for (int c = 0; c < 3; ++c) {
      if (!node[r][c].is_number()) return false;
      out_m[r * 3 + c] = node[r][c].get<float>();
    }
  }
  return true;
}

bool load_profile_bundle(const std::string& raw_path, ProfileBundle& bundle) {
  std::ifstream f(raw_path);
  if (!f.is_open()) {
    std::string alt = "profiles/" + raw_path;
    f.open(alt);
    if (!f.is_open()) {
      size_t last_slash = raw_path.find_last_of("/\\");
      if (last_slash != std::string::npos) {
        std::string base = "profiles/" + raw_path.substr(last_slash + 1);
        f.open(base);
      }
    }
  }
  if (!f.is_open()) return false;

  try {
    nlohmann::json j;
    f >> j;

    bundle.path = raw_path;
    bundle.camera_name = j.value("camera_name", j.value("camera_model", "Unknown"));
    bundle.normalization_target = j.value("normalization_target", 55000.0);

    if (j.contains("crosstalk_profile") && j["crosstalk_profile"].is_object()) {
      const auto& ct = j["crosstalk_profile"];
      if (ct.contains("crosstalk_correction_matrix")) {
        bundle.has_crosstalk = parse_3x3_matrix(ct["crosstalk_correction_matrix"], bundle.crosstalk_matrix);
      }
    } else if (j.contains("crosstalk_correction_matrix")) {
      bundle.has_crosstalk = parse_3x3_matrix(j["crosstalk_correction_matrix"], bundle.crosstalk_matrix);
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
      if (r_avg > 0.0 && g_avg > 0.0 && b_avg > 0.0) {
        bundle.profile_film_base_rgb[0] = (int)std::round(r_avg);
        bundle.profile_film_base_rgb[1] = (int)std::round(g_avg);
        bundle.profile_film_base_rgb[2] = (int)std::round(b_avg);
        bundle.has_film_base = true;
      }
    }

    if (!j.contains("targets") || !j["targets"].is_array()) {
      fprintf(stderr, "ERROR! Profile bundle has no targets array: %s\n", raw_path.c_str());
      return false;
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

      if (tj.contains("patches") && tj["patches"].is_object()) {
        for (auto it = tj["patches"].begin(); it != tj["patches"].end(); ++it) {
          if (it.value().is_object() && it.value().contains("g") && it.value()["g"].is_number()) {
            tgt.patches_g.push_back(it.value()["g"].get<float>());
          }
        }
        if (!tgt.patches_g.empty()) {
          tgt.tgt_min_g = *std::min_element(tgt.patches_g.begin(), tgt.patches_g.end());
          tgt.tgt_max_g = *std::max_element(tgt.patches_g.begin(), tgt.patches_g.end());
        }
      }

      if (tj.contains("icc_profile_base64") && tj["icc_profile_base64"].is_string()) {
        std::string b64 = tj["icc_profile_base64"].get<std::string>();
        if (base64_decode(b64, tgt.icc_bytes) && !tgt.icc_bytes.empty()) {
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
        fprintf(stderr, "Warning: Target '%s' has no valid ICC profile; skipping\n", tgt.name.c_str());
        continue;
      }

      tgt.index = (int)bundle.targets.size();
      bundle.targets.push_back(std::move(tgt));
    }

    if (bundle.targets.empty()) {
      fprintf(stderr, "ERROR! Bundle has 0 valid targets: %s\n", raw_path.c_str());
      return false;
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

    return true;
  } catch (const std::exception& e) {
    fprintf(stderr, "ERROR! Profile bundle loading failed for %s: %s\n", raw_path.c_str(), e.what());
    return false;
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

int run_bundle_tournament_with_dino(ProfileBundle& bundle, DinoV3Engine& engine,
                                   const SolverConfig& base_cfg, const float* base_merged,
                                   const uint16_t (*image)[4], int width, int height,
                                   IntentGains user_gains, unsigned pin_mask, int max_iters,
                                   int min_target_idx, DinoSolveResult& winner_res) {
  if (bundle.targets.empty()) return -1;
  const int n = (int)bundle.targets.size();
  const int start_idx = (n >= 5) ? std::min(min_target_idx, n - 1) : 0;

  std::vector<int> cand_indices;
  for (int i = start_idx; i < n; ++i) {
    if (!bundle.targets[i].patches_g.empty()) {
      cand_indices.push_back(i);
    }
  }
  if (cand_indices.empty()) {
    for (int i = 0; i < n; ++i) cand_indices.push_back(i);
  }

  auto rank_fn = [&](int idx) -> float {
    const auto& r = bundle.targets[idx];
    if (!r.is_feasible || r.slack_ev < 0.0f) {
      return 1000.0f + std::abs(r.slack_ev) * 100.0f;
    }
    return std::abs(r.fill_ratio - 0.85f);
  };

  std::sort(cand_indices.begin(), cand_indices.end(),
            [&](int a, int b) { return rank_fn(a) < rank_fn(b); });

  std::vector<int> finalists;
  for (int idx : cand_indices) {
    if (bundle.targets[idx].is_feasible && bundle.targets[idx].slack_ev >= 0.10f) {
      finalists.push_back(idx);
      if (finalists.size() == 2) break;
    }
  }
  if (finalists.size() < 2) {
    for (int idx : cand_indices) {
      if (bundle.targets[idx].is_feasible &&
          std::find(finalists.begin(), finalists.end(), idx) == finalists.end()) {
        finalists.push_back(idx);
        if (finalists.size() == 2) break;
      }
    }
  }
  if (finalists.empty()) {
    finalists.push_back(cand_indices[0]);
    if (cand_indices.size() > 1) finalists.push_back(cand_indices[1]);
  }

  std::sort(finalists.begin(), finalists.end());

  int num_feasible = 0;
  for (const auto& t : bundle.targets) if (t.is_feasible) num_feasible++;

  std::string bname = bundle.path;
  size_t last_slash = bname.find_last_of("/\\");
  if (last_slash != std::string::npos) bname = bname.substr(last_slash + 1);

  printf("Profile bundle: %s (%zu targets, %d feasible)\n",
         bname.c_str(), bundle.targets.size(), num_feasible);

  std::vector<int> valid_finalists;
  std::vector<DinoSolveResult> valid_results;

  for (size_t f = 0; f < finalists.size(); ++f) {
    const int idx = finalists[f];
    const auto& tgt = bundle.targets[idx];
    SolverConfig cfg = base_cfg;
    set_target_bounds(cfg, tgt);

    DinoSolveResult res;
    bool ok = solve_dinov3_intent(engine, cfg, base_merged, image, width, height,
                                  user_gains, pin_mask, max_iters, res);
    if (!ok) {
      fprintf(stderr, "Warning: DINOv3 intent solver failed for finalist target %s\n",
              tgt.display_name.c_str());
      continue;
    }

    printf("  %s: resid %.3f %s\n",
           tgt.display_name.c_str(), res.best_residual, res.status.c_str());
    valid_finalists.push_back(idx);
    valid_results.push_back(std::move(res));
  }

  if (valid_finalists.empty()) {
    fprintf(stderr, "ERROR! All DINOv3 tournament finalists failed.\n");
    return -1;
  }

  size_t best_f = 0;
  if (valid_finalists.size() > 1) {
    const bool conv0 = (valid_results[0].status == "CONVERGED");
    const bool conv1 = (valid_results[1].status == "CONVERGED");
    if (conv0 && !conv1) {
      best_f = 0;
    } else if (conv1 && !conv0) {
      best_f = 1;
    } else {
      best_f = (valid_results[0].best_residual <= valid_results[1].best_residual) ? 0 : 1;
    }
  }

  const int winner_idx = valid_finalists[best_f];
  winner_res = std::move(valid_results[best_f]);
  printf("Selected target: %s\n", bundle.targets[winner_idx].display_name.c_str());
  return winner_idx;
}
