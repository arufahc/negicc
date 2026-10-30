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

#include "np_log.h"
#include "np_profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <sys/stat.h>

#include <filesystem>
#include <string>

#include "embedded_data.h"

bool is_directory(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_directory(path, ec);
}

bool is_regular_file(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

static bool path_exists_file_or_dir(const std::string& path) {
  return is_regular_file(path) || is_directory(path);
}

std::string resolve_profile_path(const std::string& input_path) {
  if (path_exists_file_or_dir(input_path)) {
    return input_path;
  }
  std::string alt = "profiles/" + input_path;
  if (path_exists_file_or_dir(alt)) {
    return alt;
  }
  if (input_path.rfind("icc_out/", 0) == 0) {
    std::string remapped = "profiles/" + input_path.substr(8);
    if (path_exists_file_or_dir(remapped)) {
      return remapped;
    }
  }
  size_t last_slash = input_path.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    std::string base = "profiles/" + input_path.substr(last_slash + 1);
    if (path_exists_file_or_dir(base)) {
      return base;
    }
  }
  return input_path;
}

bool load_film_profile(const std::string& raw_path, ParsedFilmProfile& prof) {
  std::string resolved_path = resolve_profile_path(raw_path);
  cmsHPROFILE hprof = cmsOpenProfileFromFile(resolved_path.c_str(), "r");
  if (!hprof) {
    log_error("Cannot open film ICC profile: %s\n", raw_path.c_str());
    return false;
  }
  if (!parse_film_profile_pipeline(hprof, prof)) {
    log_error("Film ICC profile %s has no AToB0 tag\n", resolved_path.c_str());
    cmsCloseProfile(hprof);
    return false;
  }
  cmsCloseProfile(hprof);
  log_info("Reading input ICC profile: %s\n", resolved_path.c_str());
  return true;
}

bool load_bundle_arg(const std::string& arg, ProfileBundle& bundle) {
  std::string resolved = resolve_profile_path(arg);
  if (is_directory(resolved)) resolved = (std::filesystem::path(resolved) / "bundle.json").string();
  if (!is_regular_file(resolved) || !is_json_bundle(resolved)) return false;
  load_profile_bundle(resolved, bundle);
  return true;
}

bool load_embedded_bundle(const std::string& key, ProfileBundle& bundle) {
  std::string dir;
  const EmbeddedFile* f = find_embedded_bundle(key, dir);
  if (!f) return false;
  const BundleFileReader read_file = [&dir](const std::string& name, std::vector<uint8_t>& out) {
    const EmbeddedFile* e = find_embedded_file(dir + "/" + name);
    if (!e) return false;
    out.assign(e->data, e->data + e->size);
    return true;
  };
  load_profile_bundle_from_memory((const char*)f->data, f->size, std::string("[built-in] ") + f->name, read_file,
                                  bundle);
  return true;
}

int read_profile(const std::string& prof_name, unsigned **prof_out, unsigned *size) {
  std::string resolved = resolve_profile_path(prof_name);
  FILE *fp = fopen(resolved.c_str(), "rb");
  if (fp) {
    if (fread(size, 4, 1, fp) != 1) {
      fclose(fp);
      return -1;
    }
    fseek(fp, 0, SEEK_SET);
    *prof_out = (unsigned *)malloc(*size = ntohl(*size));
    if (*prof_out && fread(*prof_out, 1, *size, fp) == *size) {
      fclose(fp);
      return 0;
    }
    if (*prof_out) free(*prof_out);
    fclose(fp);
  }
  log_error("Cannot read ICC profile: %s\n", prof_name.c_str());
  return -1;
}
