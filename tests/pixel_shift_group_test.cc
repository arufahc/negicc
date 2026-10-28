// Unit test of check_pixel_shift_group: the group tags of four files accepted and rejected.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "pixel_shift_group.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)

static void group(PixelShiftTags t[4], unsigned id, int n, const int nums[4]) {
  for (int i = 0; i < 4; ++i) t[i] = {id, n, nums[i]};
}

int main() {
  std::string err;
  PixelShiftTags t[4];
  const int in_order[4] = {1, 2, 3, 4};

  // No tags at all (single shots from a camera that writes none): the caller falls back to the file counters.
  CHECK(check_pixel_shift_group(t, err) == PixelShiftGroupCheck::kNoTags);

  group(t, 77, 4, in_order);
  CHECK(check_pixel_shift_group(t, err) == PixelShiftGroupCheck::kOk);

  // A real group given out of order.
  const int swapped[4] = {1, 3, 2, 4};
  group(t, 77, 4, swapped);
  CHECK(check_pixel_shift_group(t, err) == PixelShiftGroupCheck::kBad);
  CHECK(err.find("out of frame order") != std::string::npos);

  // Four single shots in counter order: one shot per group, each its own group.
  const int ones[4] = {1, 1, 1, 1};
  for (int i = 0; i < 4; ++i) t[i] = {100u + i, 1, ones[i]};
  CHECK(check_pixel_shift_group(t, err) == PixelShiftGroupCheck::kBad);

  // Shots 1-4 of a 16-shot group.
  group(t, 77, 16, in_order);
  CHECK(check_pixel_shift_group(t, err) == PixelShiftGroupCheck::kBad);
  CHECK(err.find("16 shots") != std::string::npos);

  // Two groups' files interleaved in counter order.
  group(t, 77, 4, in_order);
  t[2].group_id = 78;
  CHECK(check_pixel_shift_group(t, err) == PixelShiftGroupCheck::kBad);
  CHECK(err.find("different") != std::string::npos);

  // Some files tagged, some not.
  group(t, 77, 4, in_order);
  t[3] = PixelShiftTags();
  CHECK(check_pixel_shift_group(t, err) == PixelShiftGroupCheck::kBad);

  printf("all assertions passed\n");
  return 0;
}
