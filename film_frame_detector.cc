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

#include "film_frame_detector.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <omp.h>

namespace {

enum PixelClass : uint8_t {
  CLASS_NONE = 0,
  CLASS_K    = 1,   // Carrier / mechanical black
  CLASS_L    = 2,   // Light box / bare illumination / flare
  CLASS_F    = 4,   // Film area (not K and not L)
  CLASS_R    = 8    // Rebate-like (unexposed film mask)
};

struct PreviewPixel {
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  float d = 0.0f;
  float c1 = 0.0f;
  float c2 = 0.0f;
  uint8_t cls = CLASS_NONE;
  bool is_clipped = false;
};

struct ValidatedGap {
  int x_a = 0;
  int x_b = 0;
  float d_mean = 0.0f;
  float std_d = 0.0f;
};

struct StripSegment {
  int x1 = 0;
  int x2 = 0;
  int gap_left = -1;   // index into val_gaps, or -1
  int gap_right = -1;  // index into val_gaps, or -1
  bool complete = false;
};

static float compute_median(std::vector<float>& vals) {
  if (vals.empty()) return 0.0f;
  const size_t mid = vals.size() / 2;
  std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
  return vals[mid];
}

static float compute_percentile(std::vector<float>& vals, double p) {
  if (vals.empty()) return 0.0f;
  p = std::clamp(p, 0.0, 1.0);
  const size_t idx = std::min(vals.size() - 1, (size_t)(p * (double)vals.size()));
  std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
  return vals[idx];
}

static void gaussian_smooth_1d(const std::vector<float>& in, float sigma, std::vector<float>& out) {
  const int n = (int)in.size();
  out.resize(n);
  if (n == 0) return;
  const int radius = std::max(1, (int)std::ceil(3.0f * sigma));
  std::vector<float> kernel(2 * radius + 1);
  float sum_k = 0.0f;
  for (int i = -radius; i <= radius; ++i) {
    const float v = std::exp(-0.5f * (float)(i * i) / (sigma * sigma));
    kernel[i + radius] = v;
    sum_k += v;
  }
  for (int i = 0; i <= 2 * radius; ++i) kernel[i] /= sum_k;

  for (int i = 0; i < n; ++i) {
    float acc = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
      const int idx = std::clamp(i + k, 0, n - 1);
      acc += in[idx] * kernel[k + radius];
    }
    out[i] = acc;
  }
}

} // namespace

bool detect_film_frame(const uint16_t (*image)[4], int width, int height, unsigned filters, bool half_size,
                       char axis_hint, FrameDetection& out) {
  if (!image || width <= 0 || height <= 0) return false;

  // 1. Empirical Saturation & Preview Dimensions (L_p = 1024)
  int pw = 0, ph = 0;
  if (width >= height) {
    pw = 1024;
    ph = std::max(1, (int)std::round((double)height * 1024.0 / (double)width));
  } else {
    ph = 1024;
    pw = std::max(1, (int)std::round((double)width * 1024.0 / (double)height));
  }

  // Measure per-channel maximums
  uint32_t global_max_r = 0, global_max_g = 0, global_max_b = 0;
  #pragma omp parallel
  {
    uint32_t tr = 0, tg = 0, tb = 0;
    #pragma omp for schedule(static)
    for (int y = 0; y < height; ++y) {
      const uint16_t* row = &image[(size_t)y * width][0];
      for (int x = 0; x < width; ++x) {
        tr = std::max<uint32_t>(tr, row[x * 4 + 0]);
        tg = std::max<uint32_t>(tg, row[x * 4 + 1]);
        tb = std::max<uint32_t>(tb, row[x * 4 + 2]);
      }
    }
    #pragma omp critical
    {
      global_max_r = std::max(global_max_r, tr);
      global_max_g = std::max(global_max_g, tg);
      global_max_b = std::max(global_max_b, tb);
    }
  }

  // Check for saturation plateau: >= 0.05% of pixels within 0.5% of max
  const uint64_t total_px = (uint64_t)width * height;
  const uint64_t plateau_thresh = std::max<uint64_t>(1, (uint64_t)(0.0005 * (double)total_px));
  const uint32_t r_plat_min = (uint32_t)(0.995 * (double)global_max_r);
  const uint32_t g_plat_min = (uint32_t)(0.995 * (double)global_max_g);
  const uint32_t b_plat_min = (uint32_t)(0.995 * (double)global_max_b);

  uint64_t count_r = 0, count_g = 0, count_b = 0;
  #pragma omp parallel
  {
    uint64_t cr = 0, cg = 0, cb = 0;
    #pragma omp for schedule(static)
    for (int y = 0; y < height; ++y) {
      const uint16_t* row = &image[(size_t)y * width][0];
      for (int x = 0; x < width; ++x) {
        if (row[x * 4 + 0] >= r_plat_min) cr++;
        if (row[x * 4 + 1] >= g_plat_min) cg++;
        if (row[x * 4 + 2] >= b_plat_min) cb++;
      }
    }
    #pragma omp atomic
    count_r += cr;
    #pragma omp atomic
    count_g += cg;
    #pragma omp atomic
    count_b += cb;
  }

  const float S_R = (count_r >= plateau_thresh) ? (float)global_max_r : 1e9f;
  const float S_G = (count_g >= plateau_thresh) ? (float)global_max_g : 1e9f;
  const float S_B = (count_b >= plateau_thresh) ? (float)global_max_b : 1e9f;
  const float S = std::max({(float)global_max_r, (float)global_max_g, (float)global_max_b, 1.0f});

  // Area average to preview buffer
  std::vector<PreviewPixel> preview((size_t)pw * ph);
  #pragma omp parallel for schedule(static)
  for (int py = 0; py < ph; ++py) {
    const int y0 = (int)(((int64_t)py * height) / ph);
    const int y1 = std::max(y0 + 1, std::min((int)(((int64_t)(py + 1) * height) / ph), height));
    for (int px = 0; px < pw; ++px) {
      const int x0 = (int)(((int64_t)px * width) / pw);
      const int x1 = std::max(x0 + 1, std::min((int)(((int64_t)(px + 1) * width) / pw), width));

      double sr = 0.0, sg = 0.0, sb = 0.0;
      bool clipped = false;
      const int n_box = (x1 - x0) * (y1 - y0);

      for (int sy = y0; sy < y1; ++sy) {
        const uint16_t* in_row = &image[(size_t)sy * width][0];
        for (int sx = x0; sx < x1; ++sx) {
          const uint16_t r = in_row[sx * 4 + 0];
          const uint16_t g = in_row[sx * 4 + 1];
          const uint16_t b = in_row[sx * 4 + 2];
          sr += r;
          sg += g;
          sb += b;
          if ((float)r >= 0.98f * S_R || (float)g >= 0.98f * S_G || (float)b >= 0.98f * S_B) {
            clipped = true;
          }
        }
      }

      PreviewPixel& pix = preview[(size_t)py * pw + px];
      pix.r = (float)(sr / n_box);
      pix.g = (float)(sg / n_box);
      pix.b = (float)(sb / n_box);
      pix.is_clipped = clipped;
    }
  }

  // 2. Classify Preview Pixels (d, c1, c2, K, L_clip, L_chroma, F, R)
  const float eps = 1e-3f;
  std::vector<int> l_clip_indices;

  for (size_t i = 0; i < preview.size(); ++i) {
    auto& pix = preview[i];
    pix.d = -std::log10(std::max(pix.g, eps) / S);
    pix.c1 = std::log((pix.r + eps) / (pix.g + eps));
    pix.c2 = std::log((pix.b + eps) / (pix.g + eps));

    if (pix.g < 0.02f * S) {
      pix.cls = CLASS_K;
    } else if (pix.is_clipped) {
      pix.cls = CLASS_L;
      l_clip_indices.push_back((int)i);
    }
  }

  // Light chroma c2^L estimation
  float c2_L = -999.0f;
  if (!l_clip_indices.empty()) {
    std::vector<float> ring_c2;
    for (int idx : l_clip_indices) {
      const int cx = idx % pw;
      const int cy = idx / pw;
      for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
          if (dx == 0 && dy == 0) continue;
          const int nx = cx + dx;
          const int ny = cy + dy;
          if (nx >= 0 && nx < pw && ny >= 0 && ny < ph) {
            const auto& np = preview[(size_t)ny * pw + nx];
            if (np.cls != CLASS_L && np.g >= 0.50f * S) {
              ring_c2.push_back(np.c2);
            }
          }
        }
      }
    }
    if (!ring_c2.empty()) {
      c2_L = compute_median(ring_c2);
    }
  }

  // If c2_L is still undefined, split thinnest 10% of ~K pixels
  if (c2_L < -900.0f) {
    std::vector<size_t> non_k_indices;
    for (size_t i = 0; i < preview.size(); ++i) {
      if (preview[i].cls != CLASS_K) non_k_indices.push_back(i);
    }
    if (!non_k_indices.empty()) {
      const size_t n_top = std::max<size_t>(10, non_k_indices.size() / 10);
      std::nth_element(non_k_indices.begin(), non_k_indices.begin() + n_top, non_k_indices.end(),
                       [&](size_t a, size_t b) { return preview[a].d < preview[b].d; });
      std::vector<float> c2_thin;
      for (size_t k = 0; k < n_top; ++k) c2_thin.push_back(preview[non_k_indices[k]].c2);
      std::sort(c2_thin.begin(), c2_thin.end());
      const float m_lo = compute_percentile(c2_thin, 0.25);
      const float m_hi = compute_percentile(c2_thin, 0.75);
      if (m_hi - m_lo >= 0.40f) {
        c2_L = m_hi;
        for (size_t k = 0; k < n_top; ++k) {
          if (preview[non_k_indices[k]].c2 >= 0.5f * (m_lo + m_hi)) {
            preview[non_k_indices[k]].cls = CLASS_L;
          }
        }
      }
    }
  }

  // Iterative base estimate
  float d_b = 0.0f;
  float c1_b = 0.0f, c2_b = 0.0f;
  for (int iter = 0; iter < 2; ++iter) {
    std::vector<float> d_f;
    for (const auto& pix : preview) {
      if (!(pix.cls & (CLASS_K | CLASS_L))) d_f.push_back(pix.d);
    }
    if (d_f.empty()) break;
    d_b = compute_percentile(d_f, 0.02);

    std::vector<float> c1_base_vec, c2_base_vec;
    for (const auto& pix : preview) {
      if (!(pix.cls & (CLASS_K | CLASS_L)) && pix.d <= d_b + 0.05f) {
        c1_base_vec.push_back(pix.c1);
        c2_base_vec.push_back(pix.c2);
      }
    }
    c1_b = compute_median(c1_base_vec);
    c2_b = compute_median(c2_base_vec);

    if (c2_L > -900.0f && c2_b > c2_L - 0.35f) {
      for (auto& pix : preview) {
        if (!(pix.cls & CLASS_K) && pix.d <= d_b + 0.05f) {
          pix.cls = CLASS_L;
        }
      }
    } else {
      break;
    }
  }

  // Mark L_chroma, F, and R
  for (auto& pix : preview) {
    if (pix.cls & (CLASS_K | CLASS_L)) continue;
    if (c2_L > -900.0f && pix.d < d_b - 0.10f && pix.c2 > c2_L - 0.25f) {
      pix.cls = CLASS_L;
      continue;
    }
    pix.cls = CLASS_F;
    if (std::abs(pix.d - d_b) <= 0.06f &&
        std::abs(pix.c1 - c1_b) <= 0.05f &&
        std::abs(pix.c2 - c2_b) <= 0.05f) {
      pix.cls |= CLASS_R;
    }
  }

  // 3. Strip Envelope & Axis Scoring
  auto find_run = [](const std::vector<float>& vec, float thresh) -> std::pair<int, int> {
    int best_s = 0, best_e = 0, cur_s = -1;
    for (int i = 0; i < (int)vec.size(); ++i) {
      if (vec[i] >= thresh) {
        if (cur_s < 0) cur_s = i;
        if (i - cur_s > best_e - best_s) {
          best_s = cur_s;
          best_e = i;
        }
      } else {
        cur_s = -1;
      }
    }
    return {best_s, std::min<int>((int)vec.size(), best_e + 1)};
  };

  // Horizontal candidate envelope
  std::vector<float> col_f(pw, 0.0f);
  for (int x = 0; x < pw; ++x) {
    int cnt = 0;
    for (int y = 0; y < ph; ++y) if (preview[(size_t)y * pw + x].cls & CLASS_F) cnt++;
    col_f[x] = (float)cnt / (float)ph;
  }
  auto [c1_h, c2_h] = find_run(col_f, 0.40f);
  if (c2_h - c1_h < pw / 8) std::tie(c1_h, c2_h) = find_run(col_f, 0.20f);

  const int span_c = std::max(1, c2_h - c1_h);
  std::vector<float> row_f_in(ph, 0.0f);
  for (int y = 0; y < ph; ++y) {
    int cnt = 0;
    for (int x = c1_h; x < c2_h; ++x) if (preview[(size_t)y * pw + x].cls & CLASS_F) cnt++;
    row_f_in[y] = (float)cnt / (float)span_c;
  }
  auto [r1_h, r2_h] = find_run(row_f_in, 0.40f);
  if (r2_h - r1_h < ph / 8) std::tie(r1_h, r2_h) = find_run(row_f_in, 0.20f);

  // Vertical candidate envelope: each row has pw pixels, so divide by pw
  std::vector<float> row_f_all(ph, 0.0f);
  for (int y = 0; y < ph; ++y) {
    int cnt = 0;
    for (int x = 0; x < pw; ++x) if (preview[(size_t)y * pw + x].cls & CLASS_F) cnt++;
    row_f_all[y] = (float)cnt / (float)pw;
  }
  auto [r1_v, r2_v] = find_run(row_f_all, 0.40f);
  if (r2_v - r1_v < ph / 8) std::tie(r1_v, r2_v) = find_run(row_f_all, 0.20f);

  const int span_r = std::max(1, r2_v - r1_v);
  std::vector<float> col_f_in(pw, 0.0f);
  for (int x = 0; x < pw; ++x) {
    int cnt = 0;
    for (int y = r1_v; y < r2_v; ++y) if (preview[(size_t)y * pw + x].cls & CLASS_F) cnt++;
    col_f_in[x] = (float)cnt / (float)span_r;
  }
  auto [c1_v, c2_v] = find_run(col_f_in, 0.40f);
  if (c2_v - c1_v < pw / 8) std::tie(c1_v, c2_v) = find_run(col_f_in, 0.20f);

  char chosen_axis = 'h';
  if (axis_hint == 'h' || axis_hint == 'H') {
    chosen_axis = 'h';
  } else if (axis_hint == 'v' || axis_hint == 'V') {
    chosen_axis = 'v';
  } else {
    const int len_h = c2_h - c1_h;
    const int len_v = r2_v - r1_v;
    const int trans_h = std::max(1, r2_h - r1_h);
    const int trans_v = std::max(1, c2_v - c1_v);

    // Border touching cues
    const bool touch_h = (c1_h <= std::max(2, (int)(0.04f * (float)pw))) &&
                         (c2_h >= pw - std::max(2, (int)(0.04f * (float)pw)));
    const bool touch_v = (r1_v <= std::max(2, (int)(0.04f * (float)ph))) &&
                         (r2_v >= ph - std::max(2, (int)(0.04f * (float)ph)));

    // Rebate bands in transverse margins
    int r_band_h = 0, f_band_h = 0;
    const int m_h = std::max(1, (int)(0.15f * (float)trans_h));
    for (int x = c1_h; x < c2_h; ++x) {
      for (int y = r1_h; y < r1_h + m_h; ++y) {
        const uint8_t c = preview[(size_t)y * pw + x].cls;
        if (c & CLASS_F) { f_band_h++; if (c & CLASS_R) r_band_h++; }
      }
      for (int y = r2_h - m_h; y < r2_h; ++y) {
        const uint8_t c = preview[(size_t)y * pw + x].cls;
        if (c & CLASS_F) { f_band_h++; if (c & CLASS_R) r_band_h++; }
      }
    }
    const float frac_band_h = f_band_h > 0 ? (float)r_band_h / (float)f_band_h : 0.0f;

    int r_band_v = 0, f_band_v = 0;
    const int m_v = std::max(1, (int)(0.15f * (float)trans_v));
    for (int y = r1_v; y < r2_v; ++y) {
      for (int x = c1_v; x < c1_v + m_v; ++x) {
        const uint8_t c = preview[(size_t)y * pw + x].cls;
        if (c & CLASS_F) { f_band_v++; if (c & CLASS_R) r_band_v++; }
      }
      for (int x = c2_v - m_v; x < c2_v; ++x) {
        const uint8_t c = preview[(size_t)y * pw + x].cls;
        if (c & CLASS_F) { f_band_v++; if (c & CLASS_R) r_band_v++; }
      }
    }
    const float frac_band_v = f_band_v > 0 ? (float)r_band_v / (float)f_band_v : 0.0f;

    float score_h = (touch_h ? 2.0f : 0.0f) + (frac_band_h >= 0.50f ? 1.0f : 0.0f) + (len_h >= len_v ? 1.0f : 0.0f);
    float score_v = (touch_v ? 2.0f : 0.0f) + (frac_band_v >= 0.50f ? 1.0f : 0.0f) + (len_v > len_h ? 1.0f : 0.0f);

    if (len_h >= 1.25f * (float)len_v) score_h += 1.5f;
    if (len_v >= 1.25f * (float)len_h) score_v += 1.5f;

    if (score_h > score_v) chosen_axis = 'h';
    else if (score_v > score_h) chosen_axis = 'v';
    else chosen_axis = (pw >= ph) ? 'h' : 'v';
  }

  // Work with horizontal normalization: if axis is 'v', transpose the preview buffer
  const bool transposed = (chosen_axis == 'v');
  const int cur_w = transposed ? ph : pw;
  const int cur_h = transposed ? pw : ph;
  int c1 = transposed ? r1_v : c1_h;
  int c2 = transposed ? r2_v : c2_h;
  int r1 = transposed ? c1_v : r1_h;
  int r2 = transposed ? c2_v : r2_h;

  c1 = std::clamp(c1, 0, cur_w - 1);
  c2 = std::clamp(c2, c1 + 1, cur_w);
  r1 = std::clamp(r1, 0, cur_h - 1);
  r2 = std::clamp(r2, r1 + 1, cur_h);

  const int T = std::max(1, r2 - r1);
  const int r_top = r1 + (int)(0.20f * (float)T);
  const int r_bot = r2 - (int)(0.20f * (float)T);

  auto get_pix = [&](int x, int y) -> const PreviewPixel& {
    if (transposed) return preview[(size_t)x * pw + y];
    return preview[(size_t)y * pw + x];
  };

  // 4. Strip-Axis Profiles & Gap Validation
  std::vector<float> rho(cur_w, 0.0f);
  std::vector<float> m(cur_w, 0.0f);

  for (int x = c1; x < c2; ++x) {
    int cnt_r = 0, cnt_f = 0;
    std::vector<float> d_vals;
    for (int y = r_top; y < r_bot; ++y) {
      const auto& pix = get_pix(x, y);
      if (pix.cls & CLASS_F) {
        cnt_f++;
        if (pix.cls & CLASS_R) cnt_r++;
        d_vals.push_back(pix.d);
      }
    }
    rho[x] = cnt_f > 0 ? (float)cnt_r / (float)cnt_f : 0.0f;
    m[x] = cnt_f > 0 ? compute_median(d_vals) : d_b;
  }

  const float sigma_s = std::max(1.0f, 0.004f * (float)std::max(pw, ph));
  std::vector<float> rho_smooth, m_smooth;
  gaussian_smooth_1d(rho, sigma_s, rho_smooth);
  gaussian_smooth_1d(m, sigma_s, m_smooth);

  // Central difference m'(x)
  std::vector<float> m_prime(cur_w, 0.0f);
  for (int x = 1; x < cur_w - 1; ++x) {
    m_prime[x] = 0.5f * (m_smooth[x + 1] - m_smooth[x - 1]);
  }

  // Compute MAD(m')
  std::vector<float> abs_grad;
  for (int x = c1; x < c2; ++x) abs_grad.push_back(std::abs(m_prime[x]));
  const float mad = compute_median(abs_grad);
  const float tau = std::max(0.002f, 6.0f * mad);
  const int w_e = std::max(2, (int)(0.01f * (float)T));

  // Find gap candidates where rho_smooth >= 0.85
  std::vector<ValidatedGap> raw_val_gaps;
  int gap_start = -1;
  for (int x = c1; x <= c2; ++x) {
    if (x < c2 && rho_smooth[x] >= 0.85f) {
      if (gap_start < 0) gap_start = x;
    } else {
      if (gap_start >= 0) {
        const int xa = gap_start, xb = x;
        gap_start = -1;

        // V1 width: 0.015T <= width <= 0.25T, >= 3 px
        const int gw = xb - xa;
        if (gw < 3 || (float)gw < 0.015f * (float)T || (float)gw > 0.25f * (float)T) {
          continue;
        }

        // V2 spans strip: R fraction over full [r1, r2] >= 0.70
        int full_f = 0, full_r = 0;
        for (int gx = xa; gx < xb; ++gx) {
          for (int gy = r1; gy < r2; ++gy) {
            const auto& pix = get_pix(gx, gy);
            if (pix.cls & CLASS_F) {
              full_f++;
              if (pix.cls & CLASS_R) full_r++;
            }
          }
        }
        float r_frac = full_f > 0 ? (float)full_r / (float)full_f : 0.0f;
        if (full_f == 0 || r_frac < 0.70f) {
          continue;
        }

        // V3 edge polarity
        float min_drop = 0.0f, max_rise = 0.0f;
        for (int k = -w_e; k <= w_e; ++k) {
          const int ix = std::clamp(xa + k, 0, cur_w - 1);
          min_drop = std::min(min_drop, m_prime[ix]);
        }
        for (int k = -w_e; k <= w_e; ++k) {
          const int ix = std::clamp(xb + k, 0, cur_w - 1);
          max_rise = std::max(max_rise, m_prime[ix]);
        }
        const bool drop_ok = (min_drop <= -tau) || (xa <= c1 + 2 * w_e);
        const bool rise_ok = (max_rise >= +tau) || (xb >= c2 - 2 * w_e);
        if (!drop_ok || !rise_ok) {
          continue;
        }

        // V4 flatness inside gap (d, c1, c2)
        double sum_d = 0.0, sum_d2 = 0.0;
        double sum_c1 = 0.0, sum_c1_2 = 0.0;
        double sum_c2 = 0.0, sum_c2_2 = 0.0;
        int d_cnt = 0;
        for (int gx = xa + w_e; gx < xb - w_e; ++gx) {
          for (int gy = r_top; gy < r_bot; ++gy) {
            const auto& pix = get_pix(gx, gy);
            if (pix.cls & CLASS_F) {
              sum_d += pix.d; sum_d2 += pix.d * pix.d;
              sum_c1 += pix.c1; sum_c1_2 += pix.c1 * pix.c1;
              sum_c2 += pix.c2; sum_c2_2 += pix.c2 * pix.c2;
              d_cnt++;
            }
          }
        }
        float std_d = 0.0f, std_c1 = 0.0f, std_c2 = 0.0f;
        if (d_cnt > 1) {
          const double mean_d = sum_d / d_cnt;
          std_d = (float)std::sqrt(std::max(0.0, (sum_d2 / d_cnt) - mean_d * mean_d));
          const double mean_c1 = sum_c1 / d_cnt;
          std_c1 = (float)std::sqrt(std::max(0.0, (sum_c1_2 / d_cnt) - mean_c1 * mean_c1));
          const double mean_c2 = sum_c2 / d_cnt;
          std_c2 = (float)std::sqrt(std::max(0.0, (sum_c2_2 / d_cnt) - mean_c2 * mean_c2));
        }
        if (std_d > 0.035f || std_c1 > 0.035f || std_c2 > 0.035f) {
          continue;
        }

        ValidatedGap vg;
        vg.x_a = xa;
        vg.x_b = xb;
        vg.d_mean = d_cnt > 0 ? (float)(sum_d / d_cnt) : d_b;
        vg.std_d = std_d;
        raw_val_gaps.push_back(vg);
      }
    }
  }

  // V5 Gap Consistency (when >= 2 gaps)
  bool v5_pass = true;
  std::vector<ValidatedGap> val_gaps = raw_val_gaps;
  if (raw_val_gaps.size() >= 2) {
    std::vector<float> widths, d_levels;
    for (const auto& g : raw_val_gaps) {
      widths.push_back((float)(g.x_b - g.x_a));
      d_levels.push_back(g.d_mean);
    }
    const float med_w = compute_median(widths);
    const float med_d = compute_median(d_levels);
    std::vector<ValidatedGap> consistent_gaps;
    for (const auto& g : raw_val_gaps) {
      const float w = (float)(g.x_b - g.x_a);
      if (std::abs(w - med_w) <= 0.50f * med_w && std::abs(g.d_mean - med_d) <= 0.035f) {
        consistent_gaps.push_back(g);
      } else {
        v5_pass = false;
      }
    }
    if (!consistent_gaps.empty()) val_gaps = consistent_gaps;
  }

  // 5. Multi-Frame Segmentation & Selection
  std::vector<StripSegment> segments;
  if (val_gaps.empty()) {
    int sx1 = c1;
    for (int x = c1; x < c2 - (int)(0.1f * (float)T); ++x) {
      if (rho_smooth[x] < 0.50f) { sx1 = x; break; }
    }
    int sx2 = c2;
    for (int x = c2 - 1; x > sx1 + (int)(0.1f * (float)T); --x) {
      if (rho_smooth[x] < 0.50f) { sx2 = x; break; }
    }
    StripSegment seg;
    seg.x1 = sx1;
    seg.x2 = sx2;
    seg.gap_left = -1;
    seg.gap_right = -1;
    seg.complete = false;
    segments.push_back(seg);
  } else {
    // Segment before first gap
    StripSegment s0;
    s0.x1 = c1;
    s0.x2 = val_gaps[0].x_a;
    s0.gap_left = -1;
    s0.gap_right = 0;
    s0.complete = false;
    segments.push_back(s0);

    // Complete segments between validated gaps
    for (size_t i = 0; i < val_gaps.size() - 1; ++i) {
      StripSegment mid_seg;
      mid_seg.x1 = val_gaps[i].x_b;
      mid_seg.x2 = val_gaps[i + 1].x_a;
      mid_seg.gap_left = (int)i;
      mid_seg.gap_right = (int)(i + 1);
      mid_seg.complete = true;
      segments.push_back(mid_seg);
    }

    // Segment after last gap
    StripSegment s_last;
    s_last.x1 = val_gaps.back().x_b;
    s_last.x2 = c2;
    s_last.gap_left = (int)(val_gaps.size() - 1);
    s_last.gap_right = -1;
    s_last.complete = false;
    segments.push_back(s_last);
  }

  // Choose segment: complete segments preferred; closest to center; tie -> larger area
  const int mid_strip = (c1 + c2) / 2;
  int best_seg = 0;
  double best_score = -1e9;
  for (size_t i = 0; i < segments.size(); ++i) {
    const auto& seg = segments[i];
    const int sw = seg.x2 - seg.x1;
    if (sw <= 0) continue;
    const int smid = (seg.x1 + seg.x2) / 2;
    const double center_dist = std::abs(smid - mid_strip);
    double score = (seg.complete ? 1000.0 : 0.0) - center_dist + 0.1 * sw;
    if (score > best_score) {
      best_score = score;
      best_seg = (int)i;
    }
  }

  int cand_x1 = segments[best_seg].x1;
  int cand_x2 = segments[best_seg].x2;
  int bound_gaps[2][2] = {{-1, -1}, {-1, -1}};
  int sample_gap_idx = -1;

  if (segments[best_seg].gap_left >= 0) {
    const int gi = segments[best_seg].gap_left;
    bound_gaps[0][0] = val_gaps[gi].x_a;
    bound_gaps[0][1] = val_gaps[gi].x_b;
    sample_gap_idx = gi;
  }
  if (segments[best_seg].gap_right >= 0) {
    const int gi = segments[best_seg].gap_right;
    bound_gaps[1][0] = val_gaps[gi].x_a;
    bound_gaps[1][1] = val_gaps[gi].x_b;
    if (sample_gap_idx >= 0) {
      // Both bounding gaps exist: if difference > 0.03 D, choose thinner one (§3.9)
      const float d_left = val_gaps[segments[best_seg].gap_left].d_mean;
      const float d_right = val_gaps[gi].d_mean;
      if (d_right < d_left - 0.03f) {
        sample_gap_idx = gi;
      } else if (d_left < d_right - 0.03f) {
        sample_gap_idx = segments[best_seg].gap_left;
      } else {
        // Within 0.03 D: tie-break by lower std_d
        const float std_left = val_gaps[segments[best_seg].gap_left].std_d;
        const float std_right = val_gaps[gi].std_d;
        sample_gap_idx = (std_right < std_left) ? gi : segments[best_seg].gap_left;
      }
    } else {
      sample_gap_idx = gi;
    }
  }

  // 6. Transverse Bounds inside middle 50% columns of [cand_x1, cand_x2]
  const int mid_col_1 = cand_x1 + (cand_x2 - cand_x1) / 4;
  const int mid_col_2 = cand_x2 - (cand_x2 - cand_x1) / 4;

  std::vector<float> row_rho(cur_h, 0.0f);
  for (int y = r1; y < r2; ++y) {
    int cnt_r = 0, cnt_f = 0;
    for (int x = mid_col_1; x < mid_col_2; ++x) {
      const auto& pix = get_pix(x, y);
      if (pix.cls & CLASS_F) {
        cnt_f++;
        if (pix.cls & CLASS_R) cnt_r++;
      }
    }
    row_rho[y] = cnt_f > 0 ? (float)cnt_r / (float)cnt_f : 1.0f;
  }

  const int mid_y = (r1 + r2) / 2;
  int cand_y1 = r1;
  for (int y = mid_y; y >= r1; --y) {
    if (row_rho[y] >= 0.60f) { cand_y1 = y; break; }
  }
  int cand_y2 = r2;
  for (int y = mid_y; y < r2; ++y) {
    if (row_rho[y] >= 0.60f) { cand_y2 = y; break; }
  }

  // 7. Format & Confidence Calculation
  const int min_w = std::max(16, (int)(0.15f * (float)(c2 - c1)));
  const int min_h = std::max(16, (int)(0.20f * (float)T));
  const bool degenerate = ((cand_x2 - cand_x1) < min_w || (cand_y2 - cand_y1) < min_h ||
                           (cand_x2 - cand_x1) <= 10 || (cand_y2 - cand_y1) <= 10);

  const float cand_w = (float)std::max(1, cand_x2 - cand_x1);
  const float cand_h = (float)std::max(1, cand_y2 - cand_y1);
  const float aspect = std::max(cand_w, cand_h) / std::min(cand_w, cand_h);

  std::string fmt = "-";
  float fmt_factor = 0.70f;
  auto aspect_match = [&](float target) {
    return std::abs(aspect - target) <= 0.04f * target;
  };
  if (aspect_match(1.00f)) { fmt = "1:1"; fmt_factor = 1.0f; }
  else if (aspect_match(7.0f / 6.0f)) { fmt = "7:6"; fmt_factor = 1.0f; }
  else if (aspect_match(4.0f / 3.0f)) { fmt = "4:3"; fmt_factor = 1.0f; }
  else if (aspect_match(1.50f)) { fmt = "3:2"; fmt_factor = 1.0f; }
  else if (aspect_match(65.0f / 24.0f)) { fmt = "65:24"; fmt_factor = 1.0f; }

  float edge_factor = 0.50f;
  if (degenerate) edge_factor = 0.40f;
  else if (val_gaps.size() >= 2) edge_factor = 1.0f;
  else if (val_gaps.size() == 1) edge_factor = 0.75f;
  else edge_factor = 0.50f;

  const float v5_factor = v5_pass ? 1.0f : 0.80f;

  // Thin frame check: median density over selected frame area
  std::vector<float> frame_densities;
  for (int y = cand_y1; y < cand_y2; ++y) {
    for (int x = cand_x1; x < cand_x2; ++x) {
      const auto& pix = get_pix(x, y);
      if (pix.cls & CLASS_F) frame_densities.push_back(pix.d);
    }
  }
  const float med_frame_d = compute_median(frame_densities);
  const float thin_factor = (!frame_densities.empty() && med_frame_d < d_b + 0.10f) ? 0.50f : 1.0f;

  const float conf = std::clamp(edge_factor * fmt_factor * v5_factor * thin_factor, 0.0f, 1.0f);

  // Fallback to strip envelope when confidence is low or frame is degenerate (§3.8)
  const bool fallback_to_envelope = (conf < 0.50f || degenerate);
  int frame_x1 = 0, frame_x2 = 0, frame_y1 = 0, frame_y2 = 0;

  if (fallback_to_envelope) {
    frame_x1 = c1;
    frame_x2 = c2;
    frame_y1 = r1;
    frame_y2 = r2;
    bound_gaps[0][0] = bound_gaps[0][1] = -1;
    bound_gaps[1][0] = bound_gaps[1][1] = -1;
    sample_gap_idx = -1;

    const int fw = std::max(1, frame_x2 - frame_x1);
    const int fh = std::max(1, frame_y2 - frame_y1);
    const int dx = std::max(2, (int)(0.03f * (float)fw));
    const int dy = std::max(2, (int)(0.03f * (float)fh));
    frame_x1 += dx; frame_x2 -= dx;
    frame_y1 += dy; frame_y2 -= dy;
  } else {
    frame_x1 = cand_x1;
    frame_x2 = cand_x2;
    frame_y1 = cand_y1;
    frame_y2 = cand_y2;

    const int fw = std::max(1, frame_x2 - frame_x1);
    const int fh = std::max(1, frame_y2 - frame_y1);
    const int dx = std::max(2, (int)(0.015f * (float)fw));
    const int dy = std::max(2, (int)(0.015f * (float)fh));
    frame_x1 += dx; frame_x2 -= dx;
    frame_y1 += dy; frame_y2 -= dy;
  }

  // Reason codes
  std::vector<std::string> reasons;
  if (degenerate) reasons.push_back("degenerate_frame");
  if (val_gaps.empty()) reasons.push_back("no_gaps");
  else if (val_gaps.size() == 1) reasons.push_back("single_gap");
  if (fmt == "-") reasons.push_back("unknown_aspect");
  if (!v5_pass) reasons.push_back("v5_gap_inconsistent");
  if (thin_factor < 1.0f) reasons.push_back("thin_frame");
  if (fallback_to_envelope) reasons.push_back("envelope_fallback");
  if (conf < 0.50f) reasons.push_back("low_conf");

  std::string reason_str;
  for (size_t i = 0; i < reasons.size(); ++i) {
    if (i > 0) reason_str += ", ";
    reason_str += reasons[i];
  }

  // Un-transpose coordinates if vertical
  int px_x = frame_x1, px_y = frame_y1, px_w = frame_x2 - frame_x1, px_h = frame_y2 - frame_y1;
  if (transposed) {
    std::swap(px_x, px_y);
    std::swap(px_w, px_h);
  }

  // 8. Map to full-resolution image[] coordinates & CFA Period Alignment
  int src_x = (int)std::round((double)px_x * (double)width / (double)pw);
  int src_y = (int)std::round((double)px_y * (double)height / (double)ph);
  int src_w = (int)std::round((double)px_w * (double)width / (double)pw);
  int src_h = (int)std::round((double)px_h * (double)height / (double)ph);

  int period = (filters == 9) ? 6 : 2;
  if (half_size) {
    period = std::max(1, period / 2);
  }

  // Ceil start, floor end
  int x_end = src_x + src_w;
  int y_end = src_y + src_h;
  src_x = ((src_x + period - 1) / period) * period;
  src_y = ((src_y + period - 1) / period) * period;
  x_end = (x_end / period) * period;
  y_end = (y_end / period) * period;

  src_x = std::clamp(src_x, 0, width - period);
  src_y = std::clamp(src_y, 0, height - period);
  src_w = std::max(period, std::min(x_end - src_x, width - src_x));
  src_h = std::max(period, std::min(y_end - src_y, height - src_y));

  if (src_w < 32 || src_h < 32) {
    return false;
  }

  // Map bounding gaps to source coordinates
  for (int g = 0; g < 2; ++g) {
    if (bound_gaps[g][0] >= 0) {
      const double scale = transposed ? ((double)height / (double)ph) : ((double)width / (double)pw);
      out.gap[g][0] = (int)std::round((double)bound_gaps[g][0] * scale);
      out.gap[g][1] = (int)std::round((double)bound_gaps[g][1] * scale);
    } else {
      out.gap[g][0] = -1;
      out.gap[g][1] = -1;
    }
  }

  // Final mapped aspect
  const float final_aspect = (float)std::max(src_w, src_h) / (float)std::max(1, std::min(src_w, src_h));

  // 9. Rebate Film Base Sampling (§3.9)
  bool sampled_base = false;
  int base_rgb[3] = {0, 0, 0};
  float base_std_d = 0.0f;

  if (sample_gap_idx >= 0 && sample_gap_idx < (int)val_gaps.size()) {
    const auto& gap = val_gaps[sample_gap_idx];
    const int gw = gap.x_b - gap.x_a;
    const int shrink = std::max(1, (int)(0.15f * (float)gw));
    int g_x1 = gap.x_a + shrink;
    int g_x2 = gap.x_b - shrink;
    int g_y1 = r_top;
    int g_y2 = r_bot;

    if (transposed) {
      std::swap(g_x1, g_y1);
      std::swap(g_x2, g_y2);
    }

    const int img_gx1 = std::clamp((int)std::round((double)g_x1 * (double)width / (double)pw), 0, width - 1);
    const int img_gx2 = std::clamp((int)std::round((double)g_x2 * (double)width / (double)pw), img_gx1 + 1, width);
    const int img_gy1 = std::clamp((int)std::round((double)g_y1 * (double)height / (double)ph), 0, height - 1);
    const int img_gy2 = std::clamp((int)std::round((double)g_y2 * (double)height / (double)ph), img_gy1 + 1, height);

    std::vector<float> sample_r, sample_g, sample_b, sample_d;
    for (int sy = img_gy1; sy < img_gy2; ++sy) {
      const uint16_t* in_row = &image[(size_t)sy * width][0];
      for (int sx = img_gx1; sx < img_gx2; ++sx) {
        const float r = in_row[sx * 4 + 0];
        const float g = in_row[sx * 4 + 1];
        const float b = in_row[sx * 4 + 2];
        sample_r.push_back(r);
        sample_g.push_back(g);
        sample_b.push_back(b);
        sample_d.push_back(-std::log10(std::max(g, eps) / S));
      }
    }

    if (sample_r.size() >= 32) {
      double sum_d = 0.0, sum_d2 = 0.0;
      for (float d : sample_d) {
        sum_d += d;
        sum_d2 += d * d;
      }
      const double mean_d = sum_d / sample_d.size();
      base_std_d = (float)std::sqrt(std::max(0.0, (sum_d2 / sample_d.size()) - mean_d * mean_d));

      const float med_r = compute_median(sample_r);
      const float med_g = compute_median(sample_g);
      const float med_b = compute_median(sample_b);
      const float med_c2 = std::log((med_b + eps) / (med_g + eps));

      // Must be flat, not clipped, and distinct from bare light
      if (base_std_d <= 0.025f && (c2_L < -900.0f || med_c2 <= c2_L - 0.35f)) {
        base_rgb[0] = std::clamp((int)std::round(med_r), 1, 65535);
        base_rgb[1] = std::clamp((int)std::round(med_g), 1, 65535);
        base_rgb[2] = std::clamp((int)std::round(med_b), 1, 65535);
        sampled_base = true;
      }
    }
  }

  // Populate output struct
  out.x = src_x;
  out.y = src_y;
  out.w = src_w;
  out.h = src_h;
  out.axis = chosen_axis;
  out.n_gaps = (int)val_gaps.size();
  out.n_segments = (int)segments.size();
  out.selected = best_seg;
  out.aspect = final_aspect;
  out.conf = conf;
  out.format = fmt;
  out.reason = reason_str;
  out.has_base = sampled_base;
  out.base_rgb[0] = base_rgb[0];
  out.base_rgb[1] = base_rgb[1];
  out.base_rgb[2] = base_rgb[2];
  out.base_std_d = base_std_d;

  return (out.w > 0 && out.h > 0);
}
