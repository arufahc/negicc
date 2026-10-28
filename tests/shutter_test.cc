#include "profile_bundle.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::abort();                                                                 \
    }                                                                               \
  } while (0)

static void expect_throws(const std::string& s) {
  bool threw = false;
  try {
    (void)shutter_to_seconds_d(s);
  } catch (const std::invalid_argument&) {
    threw = true;
  } catch (const std::out_of_range&) {
    threw = true;
  }
  if (!threw) {
    std::fprintf(stderr, "Expected shutter_to_seconds_d(\"%s\") to throw\n", s.c_str());
    std::abort();
  }
}

int main() {
  CHECK(std::fabs(shutter_to_seconds_d("1/8s") - 0.125) < 1e-12);
  CHECK(std::fabs(shutter_to_seconds_d("1/8") - 0.125) < 1e-12);
  CHECK(std::fabs(shutter_to_seconds_d("0.4s") - 0.4) < 1e-12);
  CHECK(std::fabs(shutter_to_seconds_d("2") - 2.0) < 1e-12);
  CHECK(std::fabs(shutter_to_seconds_d("  1/125s \n") - 0.008) < 1e-12);
  CHECK(std::fabs(shutter_to_seconds("1/4s") - 0.25f) < 1e-6f);

  const char* bad_inputs[] = {
      "", "   ", "s", "ss", "0", "0s", "0.0", "-1", "-0.5s", "1/0", "1/0s",
      "0/8", "-1/8", "1/-8", "nan", "NaN", "inf", "-inf", "infinity",
      "1/8x", "abc", "1/", "/8", "1/2/3", "0x10", "1_000", "1e", "+",
  };
  for (const char* bad : bad_inputs) {
    expect_throws(bad);
  }

  // Verify parse_profile_bundle rejects invalid film_base.shutter and target.shutter.
  const std::string bad_fb_json = R"({
    "input_space": "camera_raw_rgb",
    "film_base": {"r": {"avg": 1000}, "g": {"avg": 1000}, "b": {"avg": 1000}, "shutter": "0", "iso": 100},
    "targets": [{"name": "Target 1", "shutter": "1/8s", "iso": 100}]
  })";
  bool fb_threw = false;
  try {
    ProfileBundle b;
    load_profile_bundle_from_memory(bad_fb_json.data(), bad_fb_json.size(), "bad_fb.json",
                                    [](const std::string&, std::vector<uint8_t>&) { return false; }, b);
  } catch (const std::exception&) {
    fb_threw = true;
  }
  CHECK(fb_threw);

  std::printf("shutter_test: all assertions passed.\n");
  return 0;
}
