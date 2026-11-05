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

#include <algorithm>
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

const EmbeddedFile* find_embedded_adapter(const std::string& profile_name, std::string& adapter_dir, bool* fell_back) {
  static const std::string kBackbone = "dinov3-small";
  static const std::string kPrefix = "models/" + kBackbone + "_";
  if (fell_back) *fell_back = false;
  auto normalised = [](const std::string& v) {
    std::string n;
    for (char c : v)
      if (std::isalnum((unsigned char)c)) n += (char)std::tolower((unsigned char)c);
    return n;
  };
  // The embedded adapters' stock names (models/dinov3-small_<stock>/adapter.onnx), longest first.
  std::vector<std::string> stocks;
  for (uint64_t i = 0; i < EMBED_COUNT; ++i) {
    const std::string path = EMBED_TABLE[i].name;
    if (path.rfind(kPrefix, 0) == 0 && ends_with(path, "/adapter.onnx")) {
      stocks.push_back(path.substr(kPrefix.size(), path.size() - kPrefix.size() - std::strlen("/adapter.onnx")));
    }
  }
  std::sort(stocks.begin(), stocks.end(), [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
  // The name, then each directory above it (a profile inside a bundle's directory names the stock there).
  std::vector<std::string> names = {profile_name};
  std::string p = profile_name;
  for (;;) {
    while (!p.empty() && (p.back() == '/' || p.back() == '\\')) p.pop_back();
    const size_t slash = p.find_last_of("/\\");
    if (slash == std::string::npos) break;
    p = p.substr(0, slash);
    const size_t s2 = p.find_last_of("/\\");
    const std::string comp = s2 == std::string::npos ? p : p.substr(s2 + 1);
    if (!comp.empty() && comp != "." && comp != "..") names.push_back(comp);
  }
  std::vector<std::string> dirs;
  for (const std::string& name : names) {
    std::string bundle_key, stock_key;
    parse_profile_stem(name, bundle_key, stock_key);
    if (!bundle_key.empty()) {
      dirs.push_back(kPrefix + bundle_key);
      dirs.push_back(kPrefix + lower(bundle_key));
    }
    if (!stock_key.empty()) dirs.push_back(kPrefix + stock_key);
    // The stock name before any truncation at '-' or '_' ("Portra_400-..." or "portra-400"), matched as a prefix of
    // the normalised name.
    const std::string n = normalised(bundle_key.empty() ? name : bundle_key);
    for (const std::string& st : stocks) {
      const std::string ns = normalised(st);
      if (!ns.empty() && n.rfind(ns, 0) == 0) dirs.push_back(kPrefix + st);
    }
  }
  for (const std::string& d : dirs) {
    if (const EmbeddedFile* f = find_embedded_file(d + "/adapter.onnx")) {
      adapter_dir = d;
      return f;
    }
  }
  if (const EmbeddedFile* f = find_embedded_file(kPrefix + "portra400/adapter.onnx")) {
    adapter_dir = kPrefix + "portra400";
    if (fell_back) *fell_back = true;
    return f;
  }
  adapter_dir.clear();
  return nullptr;
}

const EmbeddedFile* find_embedded_backbone(bool trt_engine) {
  return find_embedded_file(trt_engine ? "models/dinov3-small/backbone.engine" : "models/dinov3-small/backbone.onnx");
}
