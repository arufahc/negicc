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
#include <fstream>
#include <functional>
#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

#include "embedded_data.h"
#include "np_log.h"
#include "profile_bundle.h"

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

bool read_fixed(const nlohmann::json& j, double* out, size_t n) {
  std::vector<double> v;
  if (!read_vector(j, v) || v.size() != n) return false;
  std::copy(v.begin(), v.end(), out);
  return true;
}

bool read_3x3(const nlohmann::json& j, double out[9]) {
  if (!j.is_array() || j.size() != 3) return false;
  for (int r = 0; r < 3; ++r) {
    if (!read_fixed(j[r], out + r * 3, 3)) return false;
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

// Least squares over up to 4 parameters within bounds: Levenberg-Marquardt with a forward-difference Jacobian, each
// step clamped to the bounds.
using Residuals = std::function<void(const double* p, std::vector<double>& r)>;
void least_squares(const Residuals& fn, int n, double* p, const double* lo, const double* hi) {
  std::vector<double> r, r1;
  fn(p, r);
  auto cost_of = [](const std::vector<double>& v) {
    double c = 0.0;
    for (double x : v) c += x * x;
    return c;
  };
  double cost = cost_of(r), lambda = 1e-3;
  for (int iter = 0; iter < 200; ++iter) {
    const size_t rows = r.size();
    std::vector<double> jac(rows * n);
    for (int k = 0; k < n; ++k) {
      double q[4];
      std::copy(p, p + n, q);
      double h = 1e-6;
      if (hi && q[k] + h > hi[k]) h = -h;
      q[k] += h;
      fn(q, r1);
      for (size_t i = 0; i < rows; ++i) jac[i * n + k] = (r1[i] - r[i]) / h;
    }
    double a[16] = {0}, g[4] = {0};
    for (size_t i = 0; i < rows; ++i) {
      for (int k = 0; k < n; ++k) {
        g[k] += jac[i * n + k] * r[i];
        for (int l = 0; l < n; ++l) a[k * 4 + l] += jac[i * n + k] * jac[i * n + l];
      }
    }
    bool improved = false;
    double change = 0.0;
    for (int tries = 0; tries < 30 && !improved; ++tries) {
      // Solve (J'J + lambda diag(J'J)) d = -g by Gaussian elimination with pivoting.
      double mtx[4][5];
      for (int k = 0; k < n; ++k) {
        for (int l = 0; l < n; ++l) mtx[k][l] = a[k * 4 + l];
        mtx[k][k] += lambda * std::max(a[k * 4 + k], 1e-12);
        mtx[k][n] = -g[k];
      }
      bool singular = false;
      for (int c = 0; c < n && !singular; ++c) {
        int piv = c;
        for (int k = c + 1; k < n; ++k) {
          if (std::fabs(mtx[k][c]) > std::fabs(mtx[piv][c])) piv = k;
        }
        if (std::fabs(mtx[piv][c]) < 1e-300) {
          singular = true;
          break;
        }
        if (piv != c) std::swap_ranges(mtx[c], mtx[c] + n + 1, mtx[piv]);
        for (int k = 0; k < n; ++k) {
          if (k == c) continue;
          const double f = mtx[k][c] / mtx[c][c];
          for (int l = c; l <= n; ++l) mtx[k][l] -= f * mtx[c][l];
        }
      }
      if (singular) {
        lambda *= 10.0;
        continue;
      }
      double q[4];
      for (int k = 0; k < n; ++k) {
        q[k] = p[k] + mtx[k][n] / mtx[k][k];
        if (lo) q[k] = std::max(q[k], lo[k]);
        if (hi) q[k] = std::min(q[k], hi[k]);
      }
      fn(q, r1);
      const double c1 = cost_of(r1);
      if (c1 < cost) {
        change = cost - c1;
        std::copy(q, q + n, p);
        r = r1;
        cost = c1;
        lambda = std::max(lambda * 0.2, 1e-12);
        improved = true;
      } else {
        lambda *= 10.0;
      }
    }
    if (!improved || change <= 1e-14 * std::max(cost, 1e-30)) break;
  }
}

}  // namespace

bool load_print_material(const std::string& json_text, PrintMaterial& out, std::string& err) {
  const nlohmann::json j = nlohmann::json::parse(json_text, nullptr, false);
  if (!j.is_object() || j.value("schema", "") != "negicc.print_material/1") {
    err = "not a negicc.print_material/1 JSON";
    return false;
  }
  PrintMaterial m;
  m.id = j.value("id", "");
  m.name = j.value("name", m.id);
  m.viewing_illuminant = j.value("viewing_illuminant", "");
  m.grey_density = j.value("print_grey_density", 0.70);
  if (j.contains("to_print_density") && j["to_print_density"].is_object()) {
    const char* keys[2] = {"rp180_printing_density", "academy_printing_density"};
    for (int k = 0; k < 2; ++k) {
      if (!j["to_print_density"].contains(keys[k]) || !j["to_print_density"][keys[k]].is_object()) continue;
      for (auto it = j["to_print_density"][keys[k]].begin(); it != j["to_print_density"][keys[k]].end(); ++it) {
        PrintMaterial::ToPrint t;
        t.film = it.key();
        if (read_3x3(it.value(), t.m.data())) m.to_print[k].push_back(t);
      }
    }
  }
  bool ok = !m.id.empty() && j.contains("log_exposure") && read_vector(j["log_exposure"], m.log_exposure) &&
            j.contains("curves") && j["curves"].is_array() && j["curves"].size() == 3 && j.contains("dye_density") &&
            j["dye_density"].is_array() && j["dye_density"].size() == 3 && j.contains("base_density") &&
            read_vector(j["base_density"], m.base) && j.contains("observer") && j["observer"].is_array() &&
            j["observer"].size() == 3 && j.contains("white_xyz") && read_fixed(j["white_xyz"], m.white, 3) &&
            j.contains("aim_log_exposure") && read_fixed(j["aim_log_exposure"], m.aim, 3) &&
            j.contains("xyz_to_linear_srgb") && read_3x3(j["xyz_to_linear_srgb"], m.to_srgb);
  for (int c = 0; ok && c < 3; ++c) {
    ok = read_vector(j["curves"][c], m.curves[c]) && m.curves[c].size() == m.log_exposure.size() &&
         read_vector(j["dye_density"][c], m.dye[c]) && m.dye[c].size() == m.base.size() &&
         read_vector(j["observer"][c], m.observer[c]) && m.observer[c].size() == m.base.size();
  }
  ok = ok && m.log_exposure.size() >= 2 && !m.base.empty() &&
       std::is_sorted(m.log_exposure.begin(), m.log_exposure.end()) && m.white[0] > 0 && m.white[1] > 0 && m.white[2] > 0;
  if (!ok) {
    err = "a field of the print material is missing or malformed";
    return false;
  }
  double pair[2];
  if (j.contains("judged_lightness") && read_fixed(j["judged_lightness"], pair, 2)) {
    m.judged_lo = pair[0];
    m.judged_hi = pair[1];
  }
  if (j.contains("slope_bounds") && read_fixed(j["slope_bounds"], pair, 2)) {
    m.slope_lo = pair[0];
    m.slope_hi = pair[1];
  }
  m.min_patches = j.value("bracket_min_patches", 12);
  out = std::move(m);
  return true;
}

void print_lab(const PrintMaterial& m, const double log_e[3], double lab[3]) {
  double amount[3], xyz[3], f[3];
  for (int c = 0; c < 3; ++c) amount[c] = interp(m.log_exposure, m.curves[c], log_e[c]);
  amounts_to_xyz(m, amount, xyz);
  for (int c = 0; c < 3; ++c) {
    const double t = std::max(xyz[c], 1e-7) / m.white[c];
    f[c] = t > 0.008856 ? std::cbrt(t) : 7.787 * t + 16.0 / 116.0;
  }
  lab[0] = 116.0 * f[1] - 16.0;
  lab[1] = 500.0 * (f[0] - f[1]);
  lab[2] = 200.0 * (f[1] - f[2]);
}

PrintSlopeFit fit_print_slopes(const PrintMaterial& m, const std::vector<std::array<double, 3>>& pd, int grey_at) {
  PrintSlopeFit fit;
  const int n = (int)pd.size();
  if (grey_at < 0 || grey_at >= n) return fit;
  const std::array<double, 3> grey = pd[grey_at];
  // p: slope red, slope blue, light red, light blue.
  auto printed = [&](const double p[4], int i, double lab[3]) {
    const double slopes[3] = {p[0], 1.0, p[1]}, lights[3] = {p[2], 0.0, p[3]};
    double log_e[3];
    for (int c = 0; c < 3; ++c) log_e[c] = m.aim[c] - slopes[c] * (pd[i][c] - grey[c]) + lights[c];
    print_lab(m, log_e, lab);
  };
  std::vector<char> ok(n, 1);
  auto chroma_residuals = [&](const double p[4], std::vector<double>& r) {
    r.clear();
    for (int i = 0; i < n; ++i) {
      if (!ok[i]) continue;
      double lab[3];
      printed(p, i, lab);
      r.push_back(lab[1]);
      r.push_back(lab[2]);
    }
  };
  auto lights_only = [&](double lights[2]) {
    lights[0] = lights[1] = 0.0;
    least_squares([&](const double* q, std::vector<double>& r) {
      const double p[4] = {1.0, 1.0, q[0], q[1]};
      chroma_residuals(p, r);
    }, 2, lights, nullptr, nullptr);
  };
  auto mean_chroma = [&](const double p[4]) {
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < n; ++i) {
      if (!ok[i]) continue;
      double lab[3];
      printed(p, i, lab);
      sum += std::hypot(lab[1], lab[2]);
      ++count;
    }
    return count ? sum / count : 0.0;
  };
  // The steps judged: those that print between near black and near white with the lights that balance all of them.
  double lights[2];
  lights_only(lights);
  {
    const double p[4] = {1.0, 1.0, lights[0], lights[1]};
    std::vector<char> judged(n, 0);
    for (int i = 0; i < n; ++i) {
      double lab[3];
      printed(p, i, lab);
      judged[i] = lab[0] > m.judged_lo && lab[0] < m.judged_hi;
      fit.judged += judged[i];
    }
    ok = judged;
  }
  if (fit.judged < m.min_patches) return fit;
  lights_only(lights);
  double p[4] = {1.0, 1.0, lights[0], lights[1]};
  const double lo[4] = {m.slope_lo, m.slope_lo, -1.0, -1.0}, hi[4] = {m.slope_hi, m.slope_hi, 1.0, 1.0};
  least_squares([&](const double* q, std::vector<double>& r) { chroma_residuals(q, r); }, 4, p, lo, hi);
  // The fit's own lights are not part of a CLF (the frame's balance takes their place): the record is of the steps
  // printed with the slopes alone.
  const double plain[4] = {1.0, 1.0, 0.0, 0.0}, applied[4] = {p[0], p[1], 0.0, 0.0};
  fit.chroma_before = mean_chroma(plain);
  fit.chroma_after = mean_chroma(applied);
  for (int k = 0; k < 2; ++k) fit.at_bound |= p[k] <= m.slope_lo + 1e-9 || p[k] >= m.slope_hi - 1e-9;
  fit.slopes[0] = p[0];
  fit.slopes[2] = p[1];
  fit.ok = true;
  return fit;
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

namespace {

// The bundle's JSON as it was loaded: the file at its path (a split bundle's directory holds bundle.json), else the
// one built in under its key.
std::string bundle_json_text(const ProfileBundle& bundle) {
  for (const std::string& path : {bundle.path, bundle.path + "/bundle.json"}) {
    std::ifstream file(path, std::ios::binary);
    std::stringstream text;
    if (file && text << file.rdbuf() && !text.str().empty() && text.str()[0] == '{') return text.str();
  }
  std::string dir;
  const EmbeddedFile* file = find_embedded_bundle(bundle.key, dir);
  return file ? std::string((const char*)file->data, file->size) : std::string();
}

// The chart's grey steps of one bracket as the profile read them (R, G, B after the crosstalk correction, at the
// bracket's exposure): every patch gs<N>, in the order of N, and the position of the mid grey gs10 among them.
// py-ref: negicc_station/src/print_slopes.py (grey_scale)
bool bracket_grey_steps(const ProfileBundle& bundle, const std::string& bracket,
                        std::vector<std::array<double, 3>>& rgb, int& mid) {
  const nlohmann::json j = nlohmann::json::parse(bundle_json_text(bundle), nullptr, false);
  if (!j.is_object() || !j.contains("targets") || !j["targets"].is_array()) return false;
  for (const auto& target : j["targets"]) {
    if (!target.is_object() || target.value("name", "") != bracket || !target.contains("patches")) continue;
    std::vector<std::pair<int, std::array<double, 3>>> steps;
    for (auto it = target["patches"].begin(); it != target["patches"].end(); ++it) {
      const std::string& key = it.key();
      if (key.size() < 3 || key.compare(0, 2, "gs") != 0 || key.find_first_not_of("0123456789", 2) != std::string::npos) {
        continue;
      }
      const auto& p = it.value();
      if (!p.is_object() || !p.contains("r") || !p.contains("g") || !p.contains("b") || !p["r"].is_number() ||
          !p["g"].is_number() || !p["b"].is_number()) {
        return false;
      }
      steps.push_back({std::stoi(key.substr(2)), {p["r"].get<double>(), p["g"].get<double>(), p["b"].get<double>()}});
    }
    std::sort(steps.begin(), steps.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    rgb.clear();
    mid = -1;
    for (const auto& step : steps) {
      if (step.first == 10) mid = (int)rgb.size();
      rgb.push_back(step.second);
    }
    return !rgb.empty() && mid >= 0;
  }
  return false;
}

}  // namespace

bool frame_print(const DensityRecord& rec, const PrintMaterial& material, FramePrint& out, std::string& err) {
  if (!rec.bundle || !rec.matrix || rec.measure == kDensityStatusM) return err = "not a printing density", false;
  const ProfileBundle& bundle = *rec.bundle;
  // The film model the bundle's density matrices were fitted on names the material's matrix for this film.
  const nlohmann::json fit_record = nlohmann::json::parse(bundle.density_fit_json, nullptr, false);
  const std::string film = fit_record.is_object() ? fit_record.value("film_dye_spectra", std::string()) : std::string();
  const PrintMaterial::ToPrint* to_print = material.to_print_for(rec.measure == kDensityRp180 ? 0 : 1, film);
  if (!to_print) return err = "the print material has no matrix for this measure", false;
  if (bundle.targets.empty()) return err = "the bundle has no brackets", false;
  FramePrint p;
  p.id = material.id;
  p.name = material.name;
  p.grey_density = material.grey_density;
  for (int k = 0; k < 9; ++k) p.to_print[k] = to_print->m[k];
  p.to_print_for = to_print->film;
  // The frame's bracket and exposure: the search's, else the middle bracket at its own exposure.
  const BundleTarget* target = nullptr;
  p.searched = rec.has_balance && rec.balance_e > 0.0;
  if (p.searched) {
    for (const BundleTarget& t : bundle.targets) {
      if (t.name == rec.balance_target) target = &t;
    }
  }
  if (!target) {
    p.searched = false;
    target = &bundle.targets[bundle.targets.size() / 2];
  }
  p.bracket = target->name;
  p.exposure = p.searched ? rec.balance_e : 1.0;
  std::vector<std::array<double, 3>> grey_rgb;
  int grey_mid = -1;
  if (!bracket_grey_steps(bundle, target->name, grey_rgb, grey_mid)) {
    return err = "the bundle's bracket " + target->name + " has no grey steps", false;
  }
  // The steps as light densities above the profile's film base, at the base's exposure, then as the measure's and as
  // the print material's density.
  // py-ref: negicc_station/src/print_slopes.py (grey_scale, bracket_slopes)
  const double base_exposure = shutter_to_seconds_d(bundle.film_base_shutter.empty() ? "1/8s" : bundle.film_base_shutter) *
                               (double)bundle.film_base_iso / 100.0;
  const double target_exposure = shutter_to_seconds_d(target->shutter) * target->iso / 100.0;
  if (!(base_exposure > 0.0) || !(target_exposure > 0.0)) return err = "the bundle's exposures cannot be read", false;
  const double ratio = base_exposure / target_exposure;
  const double* m = rec.matrix->m;
  std::vector<std::array<double, 3>> pd;
  std::array<double, 3> mid = {0.0, 0.0, 0.0};
  for (size_t i = 0; i < grey_rgb.size(); ++i) {
    double light[3], measure[3];
    for (int c = 0; c < 3; ++c) {
      const double v = grey_rgb[i][c] * ratio, base = bundle.film_base_avg_d[c];
      if (!(v > 0.0) || !(base > 0.0)) return err = "a grey step or the film base of the bundle is not positive", false;
      light[c] = -std::log10(v / base);
    }
    for (int c = 0; c < 3; ++c) measure[c] = m[c * 3] * light[0] + m[c * 3 + 1] * light[1] + m[c * 3 + 2] * light[2];
    if ((int)i == grey_mid) mid = {measure[0], measure[1], measure[2]};
    std::array<double, 3> row;
    for (int c = 0; c < 3; ++c) {
      row[c] = p.to_print[c * 3] * measure[0] + p.to_print[c * 3 + 1] * measure[1] + p.to_print[c * 3 + 2] * measure[2];
    }
    pd.push_back(row);
  }
  // The exposure E multiplies the light's channels before the bracket's profile reads them: the film that the
  // profile reads as its mid grey is denser by log10(E) in every light channel, through the measure's matrix.
  const double shift = std::log10(p.exposure);
  for (int c = 0; c < 3; ++c) p.grey[c] = mid[c] + shift * (m[c * 3] + m[c * 3 + 1] + m[c * 3 + 2]);
  const PrintSlopeFit fit = fit_print_slopes(material, pd, grey_mid);
  p.steps = (int)pd.size();
  p.steps_judged = fit.judged;
  p.fitted = fit.ok;
  if (fit.ok) {
    for (int c = 0; c < 3; ++c) p.slopes[c] = fit.slopes[c];
    p.at_bound = fit.at_bound;
    p.chroma_before = fit.chroma_before;
    p.chroma_after = fit.chroma_after;
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
       ". The mid grey is the chart's on the bracket " + xml(print.bracket) +
       (print.searched ? " and at the exposure the picture conversion's search gave this frame"
                       : " (no search: the middle bracket)") +
       "; balanced, it prints as the print's mid grey. The slopes are " +
       (print.fitted ? "fitted on that bracket's grey steps so that they print neutral"
                     : "1: the bracket has too few usable grey steps") +
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

DensityPrints density_prints(const DensityRecord& rec, const std::string& output) {
  DensityPrints out;
  if (rec.measure == kDensityStatusM) return out;
  for (const EmbeddedFile& file : list_embedded_files()) {
    const std::string name = file.name;
    if (name.rfind("prints/", 0) != 0 || name.size() < 6 || name.substr(name.size() - 5) != ".json") continue;
    PrintMaterial material;
    FramePrint print;
    std::string err;
    if (!load_print_material(std::string((const char*)file.data, file.size), material, err) ||
        !frame_print(rec, material, print, err)) {
      log_warn("No print CLF for %s: %s\n", name.c_str(), err.c_str());
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
          {"slopes_fitted", p.fitted},
          {"slopes_at_bound", p.at_bound},
          {"grey_steps", p.steps},
          {"grey_steps_judged", p.steps_judged}};
      if (p.fitted) {
        one["grey_scale_chroma_before"] = p.chroma_before;
        one["grey_scale_chroma_after"] = p.chroma_after;
      }
      materials[p.id] = one;
    }
    j["print"] = {
        {"what", "how this frame prints on each print material: the slopes are fitted, when the file is written, on "
                 "the grey steps of the frame's exposure bracket in the film's profile, as the print material reads "
                 "them, so that they print neutral"},
        {"formula", "print density = 0.70 + slopes * (to_print_density @ (D + balance - frame_grey_density)), "
                    "balance = suggested_balance.density_offset + density_offset_mean_removed"},
        {"grey_scale_chroma", "mean C*ab of the bracket's judged grey steps as the CLF prints them, about the "
                              "bracket's mid grey: _before with no slopes, _after with the slopes"},
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
    log_info("Wrote '%s': this file as a print on %s, sRGB (bracket %s, slopes %.3f %.3f %.3f%s)\n",
             prints.paths[k].c_str(), print.name.c_str(), print.bracket.c_str(), print.slopes[0], print.slopes[1],
             print.slopes[2], print.fitted ? "" : ", not fitted");
  }
  return 0;
}
