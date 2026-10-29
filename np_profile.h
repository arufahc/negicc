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

// neg_process's profiles: resolving a profile path, and loading an ICC film profile, a profile bundle JSON or a
// built-in bundle.

#pragma once

#include <string>

#include "neg_pipeline.h"
#include "profile_bundle.h"

bool is_directory(const std::string& path);
bool is_regular_file(const std::string& path);
// The path of a profile argument: as given, else under profiles/ (and a few historical locations).
std::string resolve_profile_path(const std::string& input_path);
// Loads an ICC film profile (-p with a single ICC profile, for research).
bool load_film_profile(const std::string& raw_path, ParsedFilmProfile& prof);
// Loads the negicc-station profile bundle JSON a -p argument names (as given or under profiles/), or the bundle.json
// of a split-bundle directory (profiles/<Key>). False when the argument is not a bundle (an ICC profile).
bool load_bundle_arg(const std::string& arg, ProfileBundle& bundle);
// Loads the built-in profile bundle with this key (embedded_bundle_key). False when none has that key.
bool load_embedded_bundle(const std::string& key, ProfileBundle& bundle);
// Reads an ICC profile file into a malloc'ed buffer (*prof_out, *size): 0 on success, -1 otherwise.
int read_profile(const std::string& prof_name, unsigned** prof_out, unsigned* size);
