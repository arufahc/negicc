// The print's self-checks, built and run by `make check_print`: the bracket term (np_print.h print_tilt_at, both
// forms, on brackets given out of order) and the floor on a film base that is not a capture
// (multi_target_optimizer.h print_floor_base, on synthetic frames). Prints each case; exits 1 if one fails.
#include <cstdio>

#include "multi_target_optimizer.h"
#include "np_print.h"

int main() {
  std::printf("print_tilt_at:\n");
  const int tilt = print_tilt_selfcheck();
  std::printf("print_floor_base:\n");
  const int floor = print_floor_selfcheck();
  std::printf("%s: %d failed\n", tilt + floor ? "FAIL" : "ok", tilt + floor);
  return tilt + floor ? 1 : 0;
}
