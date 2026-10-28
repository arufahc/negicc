// Copyright 2024 Alpha Lam <arufa.hc@gmail.com>
//
// Regression test suite for film_frame_detector.cc:
// - 0.5 degree 16-phase sub-pixel sweep and -3 degree tilted strip gap widths (3.2)
// - Forced-6x6 relaid frame end does not report a gap inside the picture (3.2)
// - No-light carrier scenes (6x6 in tight holder, 6x7 base-to-border) at 0.15% noise (3.6)
// - Picture-fills-the-image scene yields conf < 0.5 with no_film_base reason (3.7)
// - Three-frame 6x6 strip (horizontal and vertical transposed) selection, secondaries, and ~60 px gaps (3.15, 5.3)
// - Low px/mm 35mm strip with sprocket perforations beats 6x9 (3.15)
// - Degenerate inputs and small-crop early return leave out.ok == false (3.15, 5.3)
// - OpenMP 1 vs 8 thread determinism (5.3)

#include "film_frame_detector.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

namespace {

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                \
      std::fprintf(stderr, __VA_ARGS__);                                       \
      std::fprintf(stderr, "\n");                                              \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct SceneRGB {
  int w = 0, h = 0;
  std::vector<uint16_t> data;
  FilmImage image() const {
    return film_image_from_interleaved(data.data(), w, h, 3, 0, 65535.0f);
  }
};

// Rotates image point (x, y) around (cx, cy) by -theta_deg into strip coordinates (sx, sy).
inline void to_strip_coords(float x, float y, float cx, float cy, float theta_deg, float& sx, float& sy) {
  const float rad = theta_deg * 3.14159265358979323846f / 180.0f;
  const float c = std::cos(rad), s = std::sin(rad);
  const float dx = x - cx, dy = y - cy;
  sx = cx + c * dx + s * dy;
  sy = cy - s * dx + c * dy;
}

// Builds a synthetic negative film strip scene with configurable holder/light/base/frames, tilt, and shot noise.
SceneRGB make_strip_scene(int W, int H, float ppm, float frame_along_mm, float frame_across_mm,
                          float film_width_mm, float gap_mm, int n_frames, float shift_x_px,
                          float theta_deg, bool has_holder, bool has_bare_light, bool sprockets,
                          bool vertical_strip, float noise_sigma, uint32_t seed = 12345) {
  SceneRGB sc;
  sc.w = W;
  sc.h = H;
  sc.data.resize((size_t)W * H * 3);

  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, noise_sigma);

  // Typical orange-masked negative levels (fraction of 65535):
  //   base: R=0.26, G=0.15, B=0.09 (orange mask: R > G > B, max ~0.26 < 0.90)
  //   picture: textured 0.03..0.14 in R, scaled proportionally in G/B
  //   holder: 0.0015 (dark carrier)
  //   light: 0.98 (clipped/bright backlight)
  const float base_rgb[3] = {0.26f, 0.15f, 0.09f};
  const float holder_rgb[3] = {0.0015f, 0.0012f, 0.0010f};
  const float light_rgb[3] = {0.985f, 0.985f, 0.985f};

  const float frame_len = frame_along_mm * ppm;
  const float frame_h = frame_across_mm * ppm;
  const float strip_h = film_width_mm * ppm;
  const float gap_px = gap_mm * ppm;
  const float pitch = frame_len + gap_px;

  const float cx = 0.5f * W, cy = 0.5f * H;

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float u = x + 0.5f, v = y + 0.5f;
      float sx, sy;
      to_strip_coords(u, v, cx, cy, theta_deg, sx, sy);
      if (vertical_strip) {
        std::swap(sx, sy);
      }
      const float scx = vertical_strip ? cy : cx;
      const float scy = vertical_strip ? cx : cy;
      const float du = sx - (scx + shift_x_px);
      const float dv = sy - scy;

      float rgb[3];
      if (std::fabs(dv) > 0.5f * strip_h) {
        for (int c = 0; c < 3; ++c) rgb[c] = has_bare_light ? light_rgb[c] : (has_holder ? holder_rgb[c] : base_rgb[c]);
      } else {
        // Inside the film strip: check sprocket holes first if 35mm.
        bool in_hole = false;
        if (sprockets && std::fabs(dv) > 0.5f * frame_h + 1.0f * ppm &&
            std::fabs(dv) < 0.5f * frame_h + (1.0f + 2.79f) * ppm) {
          const float hpitch = 4.75f * ppm;
          const float hw = 1.98f * ppm;
          float hpos = std::fmod(du + 100.0f * hpitch, hpitch);
          if (hpos < hw) in_hole = true;
        }
        if (in_hole) {
          for (int c = 0; c < 3; ++c) rgb[c] = light_rgb[c];
        } else if (std::fabs(dv) <= 0.5f * frame_h) {
          // Check if inside any of the n_frames centered around du = 0.
          bool in_pic = false;
          float local_u = 0.0f;
          for (int k = 0; k < n_frames; ++k) {
            const float fc = (k - 0.5f * (n_frames - 1)) * pitch;
            if ( std::fabs(du - fc) <= 0.5f * frame_len) {
              in_pic = true;
              local_u = (du - fc) / frame_len;
              break;
            }
          }
          if (in_pic) {
            const float local_v = dv / frame_h;
            const float wave = 0.5f + 0.3f * std::sin(local_u * 9.0f + local_v * 5.0f) +
                               0.2f * std::cos(local_u * 13.0f - local_v * 7.0f);
            for (int c = 0; c < 3; ++c) rgb[c] = base_rgb[c] * (0.18f + 0.42f * wave);
          } else {
            for (int c = 0; c < 3; ++c) rgb[c] = base_rgb[c];
          }
        } else {
          for (int c = 0; c < 3; ++c) rgb[c] = base_rgb[c];
        }
      }

      for (int c = 0; c < 3; ++c) {
        const float noisy = rgb[c] * (1.0f + nd(rng));
        sc.data[((size_t)y * W + x) * 3 + c] =
            (uint16_t)std::clamp(std::lround(noisy * 65535.0f), 0L, 65535L);
      }
    }
  }
  return sc;
}

void test_tilted_gap_phase_sweep() {
  // 3.2: 0.5 degree sweep across 16 sub-pixel shift phases, truth gap = 48 px (6 mm at 8 px/mm),
  // plus -3.0 degree tilt. Every gap width must be >= 0.6 * 48 = 28.8 px (never 1..2 px).
  const int W = 1200, H = 800;
  const float ppm = 8.0f;
  const float truth_gap = 6.0f * ppm;  // 48 px
  for (int phase = 0; phase < 16; ++phase) {
    const float shift = phase * 0.25f;
    SceneRGB sc = make_strip_scene(W, H, ppm, 56.0f, 56.0f, 61.5f, 6.0f, 3, shift, 0.5f,
                                   /*has_holder=*/true, /*has_bare_light=*/false, /*sprockets=*/false,
                                   /*vertical_strip=*/false, /*noise_sigma=*/0.005f, 100 + phase);
    FrameDetectionParams P;
    P.format = FILM_6X6;
    FrameDetection out;
    bool ok = detect_film_frame(sc.image(), P, out);
    CHECK(ok && out.ok, "phase %d: detect_film_frame failed", phase);
    CHECK(out.n_gaps == 2, "phase %d: expected 2 gaps, got %d", phase, out.n_gaps);
    for (int g = 0; g < 2; ++g) {
      const int gw = out.gaps[g][1] - out.gaps[g][0];
      CHECK(gw >= (int)(0.6f * truth_gap),
            "phase %d gap %d collapsed to %d px [%d..%d] (truth %.0f px)",
            phase, g, gw, out.gaps[g][0], out.gaps[g][1], truth_gap);
    }
  }

  // Also verify -3.0 degree tilt on both gaps.
  SceneRGB sc_neg3 = make_strip_scene(W, H, ppm, 56.0f, 56.0f, 61.5f, 6.0f, 3, 0.0f, -3.0f,
                                      /*has_holder=*/true, /*has_bare_light=*/false, /*sprockets=*/false,
                                      /*vertical_strip=*/false, /*noise_sigma=*/0.005f, 999);
  FrameDetectionParams P;
  P.format = FILM_6X6;
  FrameDetection out;
  bool ok = detect_film_frame(sc_neg3.image(), P, out);
  CHECK(ok && out.ok, "-3 deg: detect_film_frame failed");
  CHECK(out.n_gaps == 2, "-3 deg: expected 2 gaps, got %d", out.n_gaps);
  for (int g = 0; g < 2; ++g) {
    const int gw = out.gaps[g][1] - out.gaps[g][0];
    CHECK(gw >= (int)(0.6f * truth_gap),
          "-3 deg gap %d collapsed to %d px [%d..%d]", g, gw, out.gaps[g][0], out.gaps[g][1]);
  }
}

void test_forced_6x6_relaid_end_no_interior_gap() {
  // 3.2 second bug: a 6x7 frame (69.5 mm along x, 56 mm across y) bounded by gaps at both ends,
  // forced to FILM_6X6 (nominal 56 mm along x). One end is relaid to 56 mm inside the picture;
  // that relaid end must be marked open (not report a 1 px gap inside the picture).
  const int W = 1200, H = 800;
  const float ppm = 8.0f;
  SceneRGB sc = make_strip_scene(W, H, ppm, 69.5f, 56.0f, 61.5f, 6.0f, 3, 0.0f, 0.0f,
                                 /*has_holder=*/true, /*has_bare_light=*/false, /*sprockets=*/false,
                                 /*vertical_strip=*/false, /*noise_sigma=*/0.005f, 202);
  FrameDetectionParams P;
  P.format = FILM_6X6;
  FrameDetection out;
  bool ok = detect_film_frame(sc.image(), P, out);
  CHECK(ok && out.ok, "forced 6x6: detect_film_frame failed");
  CHECK(out.n_gaps <= 1, "forced 6x6: expected at most 1 gap after relaying one end, got %d", out.n_gaps);
  for (int g = 0; g < 2; ++g) {
    if (out.gaps[g][0] >= 0) {
      const int gw = out.gaps[g][1] - out.gaps[g][0];
      CHECK(gw >= 25, "forced 6x6: side %d reported bogus %d px gap [%d..%d] inside picture",
            g, gw, out.gaps[g][0], out.gaps[g][1]);
    }
  }
}

void test_no_light_carrier_low_noise() {
  // 3.6: At 0.15% noise with no bare light (mx ~ 0.26), the film base must not be mistaken for a
  // clipped-light plateau.
  const int W = 1200, H = 850;
  const float ppm = 10.0f;
  // Scene A: 6x6 in a tight holder (holder on top/bottom, base gaps on left/right).
  {
    SceneRGB sc = make_strip_scene(W, H, ppm, 56.0f, 56.0f, 61.5f, 6.0f, 3, 0.0f, 0.0f,
                                   /*has_holder=*/true, /*has_bare_light=*/false, /*sprockets=*/false,
                                   /*vertical_strip=*/false, /*noise_sigma=*/0.0015f, 301);
    FrameDetectionParams P;
    FrameDetection out;
    bool ok = detect_film_frame(sc.image(), P, out);
    CHECK(ok && out.ok, "no-light 6x6 holder: failed");
    CHECK(!out.light_clipped, "no-light 6x6 holder: light_clipped must be false");
    CHECK(out.format == FILM_6X6, "no-light 6x6 holder: expected 6x6, got %s", film_format_name(out.format));
    CHECK(std::fabs(out.primary.rect.w - 560.0f) < 35.0f && std::fabs(out.primary.rect.h - 560.0f) < 35.0f,
          "no-light 6x6 holder: wrong rect %.1fx%.1f", out.primary.rect.w, out.primary.rect.h);
  }
  // Scene B: 6x7 with film base reaching the top/bottom border (no holder, no bare light).
  {
    const int H_tight = 600;  // 60 mm height < 61.5 mm film width, so rebate reaches top/bottom border
    SceneRGB sc = make_strip_scene(W, H_tight, ppm, 69.5f, 56.0f, 61.5f, 6.0f, 3, 0.0f, 0.0f,
                                   /*has_holder=*/false, /*has_bare_light=*/false, /*sprockets=*/false,
                                   /*vertical_strip=*/false, /*noise_sigma=*/0.0015f, 302);
    FrameDetectionParams P;
    FrameDetection out;
    bool ok = detect_film_frame(sc.image(), P, out);
    CHECK(ok && out.ok, "no-light 6x7 base-to-border: failed");
    CHECK(!out.light_clipped, "no-light 6x7 base-to-border: light_clipped must be false");
    CHECK(out.format == FILM_6X7, "no-light 6x7 base-to-border: expected 6x7, got %s", film_format_name(out.format));
    CHECK(std::fabs(out.primary.rect.w - 695.0f) < 40.0f && std::fabs(out.primary.rect.h - 560.0f) < 35.0f,
          "no-light 6x7 base-to-border: wrong rect %.1fx%.1f", out.primary.rect.w, out.primary.rect.h);
  }
}

void test_picture_fills_image_low_confidence() {
  // 3.7: Picture filling the entire 1200x800 image with no base, light, or carrier must produce
  // conf < 0.50 with "no_film_base" in out.reason.
  const int W = 1200, H = 800;
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.0f, 0.01f);
  std::vector<uint16_t> rgb((size_t)W * H * 3);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float fx = (float)x / W, fy = (float)y / H;
      const float lum = 0.06f + 0.16f * (0.5f + 0.3f * std::sin(fx * 5.0f + fy * 3.0f) +
                                         0.2f * std::cos(fx * 11.0f - fy * 7.0f));
      const float n = 1.0f + nd(rng);
      rgb[((size_t)y * W + x) * 3 + 0] = (uint16_t)std::clamp(lum * 2.0f * n * 65535.0f, 0.0f, 65535.0f);
      rgb[((size_t)y * W + x) * 3 + 1] = (uint16_t)std::clamp(lum * 1.2f * n * 65535.0f, 0.0f, 65535.0f);
      rgb[((size_t)y * W + x) * 3 + 2] = (uint16_t)std::clamp(lum * 0.7f * n * 65535.0f, 0.0f, 65535.0f);
    }
  }
  FilmImage im = film_image_from_interleaved(rgb.data(), W, H, 3, 0, 65535.0f);
  FrameDetectionParams P;
  FrameDetection out;
  bool ok = detect_film_frame(im, P, out);
  if (ok) {
    CHECK(out.conf < 0.50f, "picture-only: expected conf < 0.50, got %.2f (reason=%s)",
          out.conf, out.reason.c_str());
    CHECK(out.reason.find("no_film_base") != std::string::npos,
          "picture-only: expected no_film_base in reason, got '%s'", out.reason.c_str());
  }
}

void test_three_frame_6x6_strip_and_transposed() {
  // 5.3 & 3.15: Three-frame 6x6 strip with ~60 px gaps, both horizontal and vertical (even when all 4
  // borders are holder so axis_ev == 0).
  const int W = 1200, H = 900;
  const float ppm = 10.0f;  // 56 mm = 560 px frame, 6 mm = 60 px gaps
  for (int vert = 0; vert < 2; ++vert) {
    const int sw = vert ? H : W;
    const int sh = vert ? W : H;
    SceneRGB sc = make_strip_scene(sw, sh, ppm, 56.0f, 56.0f, 61.5f, 6.0f, 3, 0.0f, 0.0f,
                                   /*has_holder=*/true, /*has_bare_light=*/false, /*sprockets=*/false,
                                   /*vertical_strip=*/vert != 0, /*noise_sigma=*/0.005f, 500 + vert);
    FrameDetectionParams P;
    FrameDetection out;
    bool ok = detect_film_frame(sc.image(), P, out);
    CHECK(ok && out.ok, "3-frame 6x6 (vert=%d): failed", vert);
    CHECK(out.format == FILM_6X6, "3-frame 6x6 (vert=%d): expected 6x6, got %s",
          vert, film_format_name(out.format));
    CHECK(out.axis == (vert ? 'v' : 'h'), "3-frame 6x6 (vert=%d): expected axis %c, got %c",
          vert, vert ? 'v' : 'h', out.axis);
    CHECK(out.secondaries.size() == 2, "3-frame 6x6 (vert=%d): expected 2 secondaries, got %zu",
          vert, out.secondaries.size());
    CHECK(out.secondaries[0].index == -1 && out.secondaries[1].index == 1,
          "3-frame 6x6 (vert=%d): expected secondary indices -1 and +1", vert);
    CHECK(out.n_gaps == 2, "3-frame 6x6 (vert=%d): expected 2 gaps, got %d", vert, out.n_gaps);
    for (int g = 0; g < 2; ++g) {
      const int gw = out.gaps[g][1] - out.gaps[g][0];
      CHECK(gw >= 45 && gw <= 75, "3-frame 6x6 (vert=%d): gap %d width %d not near 60 px", vert, g, gw);
    }
  }
}

void test_low_ppm_35mm_sprockets_vs_6x9() {
  // 3.15: 35mm vs 6x9 tie at low canvas px/mm (~2.5 px/mm) with sprockets must pick 35mm.
  const int W = 320, H = 220;
  const float ppm = 2.5f;  // 36x24 mm = 90x60 px, holes 5.0x7.0 px
  SceneRGB sc = make_strip_scene(W, H, ppm, 36.0f, 24.0f, 35.0f, 2.0f, 3, 0.0f, 0.0f,
                                 /*has_holder=*/true, /*has_bare_light=*/false, /*sprockets=*/true,
                                 /*vertical_strip=*/false, /*noise_sigma=*/0.005f, 601);
  FrameDetectionParams P;
  FrameDetection out;
  bool ok = detect_film_frame(sc.image(), P, out);
  CHECK(ok && out.ok, "low-ppm 35mm: failed");
  CHECK(!out.sprockets.empty(), "low-ppm 35mm: expected sprockets to be detected");
  CHECK(out.format == FILM_35MM, "low-ppm 35mm: expected 35mm, got %s", film_format_name(out.format));
}

void test_degenerate_inputs_and_crop_too_small() {
  // 3.15 & 5.3: Degenerate inputs and crop-size early return must set out.ok == false.
  FrameDetectionParams P;
  FrameDetection out;
  out.ok = true;
  FilmImage empty;
  CHECK(!detect_film_frame(empty, P, out) && !out.ok, "empty image must return false with out.ok == false");

  std::vector<uint16_t> small_buf(20 * 20 * 3, 10000);
  FilmImage small_im = film_image_from_interleaved(small_buf.data(), 20, 20, 3, 0, 65535.0f);
  P.crop_inset = 0.45f;  // forces inscribed crop < 16 px after Step 5 succeeds
  out.ok = true;
  bool ok = detect_film_frame(small_im, P, out);
  CHECK(!ok && !out.ok, "crop < 16 early return must return false and leave out.ok == false");
}

void test_omp_determinism() {
  // 5.3: OpenMP 1 vs 8 threads must produce identical FrameDetection outputs.
  SceneRGB sc = make_strip_scene(1000, 700, 8.0f, 56.0f, 56.0f, 61.5f, 6.0f, 3, 1.5f, 1.25f,
                                 /*has_holder=*/true, /*has_bare_light=*/false, /*sprockets=*/false,
                                 /*vertical_strip=*/false, /*noise_sigma=*/0.008f, 777);
  FrameDetectionParams P;
  FrameDetection out1, out8;
  omp_set_num_threads(1);
  bool ok1 = detect_film_frame(sc.image(), P, out1);
  omp_set_num_threads(8);
  bool ok8 = detect_film_frame(sc.image(), P, out8);
  CHECK(ok1 && ok8, "OMP determinism: detection failed");
  CHECK(out1.format == out8.format && out1.axis == out8.axis && out1.n_gaps == out8.n_gaps,
        "OMP 1 vs 8 format/axis/gaps mismatch");
  for (int i = 0; i < 4; ++i) {
    CHECK(out1.crop[i] == out8.crop[i], "OMP 1 vs 8 crop[%d] mismatch: %d vs %d", i, out1.crop[i], out8.crop[i]);
  }
  for (int g = 0; g < 2; ++g) {
    CHECK(out1.gaps[g][0] == out8.gaps[g][0] && out1.gaps[g][1] == out8.gaps[g][1],
          "OMP 1 vs 8 gap %d mismatch", g);
  }
  CHECK(out1.conf == out8.conf && out1.rotation_deg == out8.rotation_deg,
        "OMP 1 vs 8 conf/rotation mismatch");
}

}  // namespace

int main() {
  test_tilted_gap_phase_sweep();
  test_forced_6x6_relaid_end_no_interior_gap();
  test_no_light_carrier_low_noise();
  test_picture_fills_image_low_confidence();
  test_three_frame_6x6_strip_and_transposed();
  test_low_ppm_35mm_sprockets_vs_6x9();
  test_degenerate_inputs_and_crop_too_small();
  test_omp_determinism();
  std::printf("detector_test: all assertions passed.\n");
  return 0;
}
