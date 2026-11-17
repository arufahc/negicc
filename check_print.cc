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
