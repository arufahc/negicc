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

// neg_process's messages: progress on stdout, warnings ("Warning: ...") and errors ("ERROR! ...") on stderr. The
// format strings are printf's and end their own lines.

#pragma once

#include <stdarg.h>
#include <stdio.h>

__attribute__((format(printf, 1, 2))) inline void log_info(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stdout, fmt, ap);
  va_end(ap);
}

__attribute__((format(printf, 1, 2))) inline void log_warn(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fputs("Warning: ", stderr);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

__attribute__((format(printf, 1, 2))) inline void log_error(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fputs("ERROR! ", stderr);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}
