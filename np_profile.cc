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

// neg_process's profiles and the conversion set up from them (np_profile.h).

#include "np_log.h"
#include "np_profile.h"

#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <filesystem>
#include <string>

#include "embedded_data.h"

// --------------------------------------------------------------------------------------------------------------------
// Profiles
// --------------------------------------------------------------------------------------------------------------------

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

// --------------------------------------------------------------------------------------------------------------------
// Input matrix
// --------------------------------------------------------------------------------------------------------------------

template <class T>
static float dot_product(const std::vector<float>& v1, const std::vector<T>& v2) {
  float prod = 0;
  for (size_t i = 0; i < v1.size() && i < v2.size(); ++i) {
    prod += v1[i] * v2[i];
  }
  return prod;
}

void adjust_correction_matrix(const std::vector<float>& r_coef, 
                              const std::vector<float>& g_coef,
                              const std::vector<float>& b_coef,
                              float global_exposure_comp,
                              float gain_g,
                              float gain_b,
                              const std::vector<int>& profile_film_base_rgb,
                              const std::vector<int>& film_base_rgb,
                              float* merged_matrix) {
  // First convert the linear RGB into corrected RGB values that are proportional
  // to transmittance.
  float cc_average_r = dot_product(r_coef, film_base_rgb);
  float cc_average_g = dot_product(g_coef, film_base_rgb);
  float cc_average_b = dot_product(b_coef, film_base_rgb);
  float cc_profile_r = dot_product(r_coef, profile_film_base_rgb);
  float cc_profile_g = dot_product(g_coef, profile_film_base_rgb);
  float cc_profile_b = dot_product(b_coef, profile_film_base_rgb);
  log_info("Film base RGB (corrected): %f %f %f\n", cc_average_r, cc_average_g, cc_average_b);
  log_info("Profile film base RGB (corrected): %f %f %f\n", cc_profile_r, cc_profile_g, cc_profile_b);

  // The ICC profiles map the transmittance values after the correction matrix is
  // applied to XYZ. Transmittance is relative to the profile film base. For example
  // if the R value for profile film base is 10000, a R value of 5000 has
  // transmittance of 5000 / 10000 = 0.5. As the corrected RGB is not converted
  // into a floating point value first, the actual denominator is not important.
  // Suppose we record the transmittance as a 16-bit fixed point number, we will
  // lose precision for denser film. So the ICC profiles are generated with
  // transmittance values scaled given a fixed value for mid-grey patch (e.g.
  // GS14).
  //
  // The film base RGB values are only useful in relative terms. Here we are
  // compensating the difference in transmittance between the profile and the
  // scanned film.
  //
  // Suppose the mid-grey patch has R = 5000 and G = 10000. With film base
  // R = 10000 and film base G = 20000, the transmittance is 0.5 in both channels.
  //
  // A different captured film with R = 5000 (film base R = 10000) has a
  // transmittance of 0.5 in the R channel. Let's say the G layer is denser
  // because of fluctuation in the film development. G = 8000 (film base G = 16000)
  // will also have a transmittance of 0.5. However using the ICC profile which
  // assumes film base G is 20000 in the G channel, will treat the transmittance
  // as 8000 / 20000 = 0.4 and the color balance will be off.
  //
  // To compensate for this we will scale the G channel (with R as reference)
  // by (20000 / 10000) / (16000 / 10000) = 1.25.
  //
  // Of course the density even for R is not the same in the target and the
  // captured film. After adjusting the color balance, all channels can be
  // adjusted with a single scaling factor.
  float g_scale = 1.0f;
  float b_scale = 1.0f;
  if (cc_average_g > 0.0f && cc_average_r > 0.0f && cc_profile_r > 0.0f && cc_average_b > 0.0f) {
    g_scale = (cc_profile_g / cc_profile_r) / (cc_average_g / cc_average_r);
    b_scale = (cc_profile_b / cc_profile_r) / (cc_average_b / cc_average_r);
  }
  log_info("Scale channels to match film base: %f %f %f\n", 1.0, g_scale, b_scale);

  const float sr = global_exposure_comp;
  const float sg = g_scale * global_exposure_comp * gain_g;
  const float sb = b_scale * global_exposure_comp * gain_b;

  for (int i = 0; i < 3; ++i) {
    merged_matrix[i]     = r_coef[i] * sr;
    merged_matrix[3 + i] = g_coef[i] * sg;
    merged_matrix[6 + i] = b_coef[i] * sb;
  }
  log_info("R coefficients: %1.5f %1.5f %1.5f\n", merged_matrix[0], merged_matrix[1], merged_matrix[2]);
  log_info("G coefficients: %1.5f %1.5f %1.5f\n", merged_matrix[3], merged_matrix[4], merged_matrix[5]);
  log_info("B coefficients: %1.5f %1.5f %1.5f\n", merged_matrix[6], merged_matrix[7], merged_matrix[8]);
}
