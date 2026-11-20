// The functions of multi_target_optimizer.cc and film_base_detector.cc that have internal linkage (bracket_enclosure,
// pull_centre, u16_percentile_linear, compute_frame_stats, frame_scale, fallback_picture_cells, curve_frame_of,
// shape_scale_of, percentile_f32, median_u16), made callable for gen_vectors.cc. The two source files are compiled
// into this translation unit as they are, so the generator calls the library's own code; the library's objects of the
// same files are then not linked. Nothing here computes a value itself.

#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#include "multi_target_optimizer.cc"  // NOLINT(bugprone-suspicious-include)
#include "film_base_detector.cc"      // NOLINT(bugprone-suspicious-include)

namespace genint {

void enclosure(float g_min, float g_max, float exposure_s, float g_lo, float g_hi, float out[10]) {
  BundleTarget t;
  t.patches_g = {g_min, g_max};
  t.tgt_min_g = g_min;
  t.tgt_max_g = g_max;
  t.exposure_s = exposure_s;
  Enclosure r;
  bracket_enclosure(t, 0, g_lo, g_hi, r);
  const float v[10] = {r.feasible ? 1.0f : 0.0f, r.admissible ? 1.0f : 0.0f, r.e_min, r.e_max, r.search_lo,
                       r.search_hi, r.e_center, r.usable_inverted ? 1.0f : 0.0f, r.slack_ev, r.fill_ratio};
  std::copy(v, v + 10, out);
}

void enclosure_of(const BundleTarget& t, float g_lo, float g_hi, float out[10]) {
  Enclosure r;
  bracket_enclosure(t, 0, g_lo, g_hi, r);
  const float v[10] = {r.feasible ? 1.0f : 0.0f, r.admissible ? 1.0f : 0.0f, r.e_min, r.e_max, r.search_lo,
                       r.search_hi, r.e_center, r.usable_inverted ? 1.0f : 0.0f, r.slack_ev, r.fill_ratio};
  std::copy(v, v + 10, out);
}

float pull_centre_of(float g_min, float g_max, float exposure_s, float g_lo, float g_hi) {
  BundleTarget t;
  t.patches_g = {g_min, g_max};
  t.tgt_min_g = g_min;
  t.tgt_max_g = g_max;
  t.exposure_s = exposure_s;
  Enclosure r;
  bracket_enclosure(t, 0, g_lo, g_hi, r);
  return pull_centre(r);
}

void curve_frame_base(const std::vector<uint16_t>& grid, const double* film_base_d, float out[3]) {
  FrameStats st;
  st.grid = grid;
  ParamSearchOptions opts;
  opts.film_base_rgb_d = film_base_d;
  const float merged[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const negicc::CurveFrame cf = curve_frame_of(st, merged, opts);
  std::copy(cf.film_base, cf.film_base + 3, out);
}

float shape_scale(float a, float k, float u_med, float ref, int table_n, float max_dev, float u_hold) {
  return shape_scale_of(
      [&](float t) { return negicc::curve_shape_ok(t * a, t * k, u_med, ref, table_n, max_dev, u_hold); });
}

float percentile_u16_linear(const std::vector<uint64_t>& hist, uint64_t total, float q) {
  return u16_percentile_linear(hist, total, q);
}

void frame_green_range(const negicc::RgbView& frame, float crop_ratio, float out[2]) {
  FrameStats st;
  compute_frame_stats(frame, crop_ratio, 224, 16, st);
  out[0] = st.g_lo;
  out[1] = st.g_hi;
}

float e_norm(const ProfileBundle& bundle, float scan_shutter_s, float scan_iso) {
  ParamSearchOptions opts;
  return frame_scale(bundle, scan_shutter_s, scan_iso, opts).e_norm;
}

std::vector<size_t> picture_cells(const std::vector<uint16_t>& grid, const float film_base[3], bool rebate) {
  return fallback_picture_cells(grid.data(), grid.size() / 3, film_base, rebate);
}

float percentile32(std::vector<float> a, double q) { return negicc::percentile_f32(a, q); }

double median16(std::vector<uint16_t> a) { return negicc::median_u16(a); }

}  // namespace genint
