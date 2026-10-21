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

#include "embedded_data.h"

#include <cctype>
#include <cstring>

#include "profile_bundle.h"

// Defined by bin_out/embedded_data.S, which the Makefile writes; it passes the symbol names as EMBED_TABLE and
// EMBED_COUNT.
#if !defined(EMBED_TABLE) || !defined(EMBED_COUNT)
#error Build with -DEMBED_TABLE=... -DEMBED_COUNT=... (see the Makefile)
#endif
extern "C" const EmbeddedFile EMBED_TABLE[];
extern "C" const uint64_t EMBED_COUNT;

static std::string lower(std::string s) {
  for (char& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

static std::string base_name(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

static bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<EmbeddedFile> list_embedded_files() {
  return std::vector<EmbeddedFile>(EMBED_TABLE, EMBED_TABLE + EMBED_COUNT);
}

const EmbeddedFile* find_embedded_file(const std::string& name) {
  for (uint64_t i = 0; i < EMBED_COUNT; ++i) {
    if (name == EMBED_TABLE[i].name) return &EMBED_TABLE[i];
  }
  return nullptr;
}

const EmbeddedFile* find_embedded_bundle(const std::string& key, std::string& bundle_dir) {
  const std::string want = lower(key);
  if (want.empty()) return nullptr;
  for (uint64_t i = 0; i < EMBED_COUNT; ++i) {
    const EmbeddedFile& f = EMBED_TABLE[i];
    const std::string path = f.name;
    if (path.rfind("profiles/", 0) != 0 || !ends_with(path, "/bundle.json")) continue;
    const std::string dir = path.substr(0, path.size() - std::strlen("/bundle.json"));
    if (lower(base_name(dir)) == want) {
      bundle_dir = dir;
      return &f;
    }
  }
  return nullptr;
}

std::string embedded_bundle_key(const std::string& film_stock, const std::string& light_source, std::string& error) {
  std::string light;
  if (lower(light_source) == "tri") {
    light = "triled";
  } else if (lower(light_source) == "d65") {
    light = "d65";
  } else {
    error = "unknown light source '" + light_source + "' (tri or d65)";
    return "";
  }
  const std::string prefix = lower(film_stock) + "-", suffix = "-" + light;
  std::vector<std::string> found;
  for (const std::string& key : list_embedded_bundles()) {
    const std::string k = lower(key);
    if (k.size() > prefix.size() + suffix.size() && k.compare(0, prefix.size(), prefix) == 0 && ends_with(k, suffix)) {
      found.push_back(key);
    }
  }
  if (found.size() != 1) {
    error = std::string(found.empty() ? "no" : "more than one") + " built-in profile bundle for film stock '" +
            film_stock + "' and light source '" + light_source + "'";
    return "";
  }
  return found[0];
}

std::vector<std::string> list_embedded_bundles() {
  std::vector<std::string> keys;
  for (uint64_t i = 0; i < EMBED_COUNT; ++i) {
    const std::string path = EMBED_TABLE[i].name;
    if (path.rfind("profiles/", 0) == 0 && ends_with(path, "/bundle.json")) {
      keys.push_back(base_name(path.substr(0, path.size() - std::strlen("/bundle.json"))));
    }
  }
  return keys;
}

const EmbeddedFile* find_embedded_adapter(const std::string& profile_name, std::string& adapter_dir) {
  static const std::string kBackbone = "dinov3-small";
  std::string bundle_key, stock_key;
  parse_profile_stem(profile_name, bundle_key, stock_key);
  std::vector<std::string> dirs;
  if (!bundle_key.empty()) {
    dirs.push_back("models/" + kBackbone + "_" + bundle_key);
    dirs.push_back("models/" + kBackbone + "_" + lower(bundle_key));
  }
  if (!stock_key.empty()) dirs.push_back("models/" + kBackbone + "_" + stock_key);
  dirs.push_back("models/" + kBackbone + "_portra400");
  for (const std::string& d : dirs) {
    if (const EmbeddedFile* f = find_embedded_file(d + "/adapter.onnx")) {
      adapter_dir = d;
      return f;
    }
  }
  adapter_dir.clear();
  return nullptr;
}

const EmbeddedFile* find_embedded_backbone(bool& is_trt_engine) {
  is_trt_engine = true;
  if (const EmbeddedFile* f = find_embedded_file("models/dinov3-small/backbone.engine")) return f;
  is_trt_engine = false;
  return find_embedded_file("models/dinov3-small/backbone.onnx");
}
