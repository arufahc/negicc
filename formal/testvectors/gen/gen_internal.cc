// The functions of film_detector.cc that have internal linkage (percentile_f32, median_u16), made callable for
// gen_vectors.cc. The source file is compiled into this translation unit as it is, so the generator calls the
// library's own code; the library's object of the same file is then not linked. Nothing here computes a value itself.

#include "film_detector.cc"  // NOLINT(bugprone-suspicious-include)

namespace genint {

float percentile32(std::vector<float> a, double q) { return negicc::percentile_f32(a, q); }

double median16(std::vector<uint16_t> a) { return negicc::median_u16(a); }

}  // namespace genint
