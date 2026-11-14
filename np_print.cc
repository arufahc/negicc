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
#include <functional>

#include <nlohmann/json.hpp>

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
  fit.chroma_before = mean_chroma(p);
  const double lo[4] = {m.slope_lo, m.slope_lo, -1.0, -1.0}, hi[4] = {m.slope_hi, m.slope_hi, 1.0, 1.0};
  least_squares([&](const double* q, std::vector<double>& r) { chroma_residuals(q, r); }, 4, p, lo, hi);
  fit.chroma_after = mean_chroma(p);
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
