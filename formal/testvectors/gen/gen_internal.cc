// The functions of multi_target_optimizer.cc and film_base_detector.cc that have internal linkage (bracket_enclosure,
// pcs_lab_to_srgb_u8, profile_highlight_ceiling, u16_percentile_linear, frame_scale, fallback_picture_cells,
// percentile_f32, median_u16), made callable for gen_vectors.cc. The two source files are compiled into this
// translation unit as they are, so the generator calls the library's own code; the library's objects of the same files
// are then not linked. Nothing here computes a value itself.

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

void lab_to_srgb_u8(const float lab[3], float out[3]) { pcs_lab_to_srgb_u8(lab, out); }

void highlight_ceiling(const std::string& report, float out[3]) { profile_highlight_ceiling(report, out); }

float percentile_u16_linear(const std::vector<uint64_t>& hist, uint64_t total, float q) {
  return u16_percentile_linear(hist, total, q);
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
