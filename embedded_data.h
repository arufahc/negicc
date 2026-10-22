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

#ifndef EMBEDDED_DATA_H
#define EMBEDDED_DATA_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Data files built into neg_process (EMBED_FILES in the Makefile): the profile bundles under profiles/<Key>/ (bundle.json and
// its ICC profiles), the DINOv3 intent adapters under models/dinov3-small_<stock>/ and the DINOv3 backbone when it
// was exported before the build (models/dinov3-small/backbone.onnx, and backbone.engine in USE_CUDA=1 builds). Names
// are the files' paths in the
// repository, e.g. "profiles/Portra400-A7R4-TriLED/bundle.json" or "models/dinov3-small_portra400/adapter.onnx".
struct EmbeddedFile {
  const char* name;
  const uint8_t* data;  // followed by a NUL byte
  size_t size;
};

// All embedded files, in build order.
std::vector<EmbeddedFile> list_embedded_files();

// The embedded file with this exact name, or null.
const EmbeddedFile* find_embedded_file(const std::string& name);

// The embedded profile bundle with this key ("Portra400-A7R4-TriLED", ignoring case), or null. bundle_dir
// receives its directory ("profiles/Portra400-A7R4-TriLED"), which holds its ICC profiles.
const EmbeddedFile* find_embedded_bundle(const std::string& key, std::string& bundle_dir);

// The key of the embedded bundle of a film stock ("portra400") and scanning light ("tri" for the narrowband RGB
// LED, "d65" for white light): the one named "<Stock>-<camera>-<Light>", e.g. "Portra400-A7R4-TriLED". Empty, with
// the reason in error, when the light is unknown or no single bundle matches.
std::string embedded_bundle_key(const std::string& film_stock, const std::string& light_source, std::string& error);

// Keys of the embedded profile bundles.
std::vector<std::string> list_embedded_bundles();

// The embedded intent adapter for a profile bundle, or null. Resolution follows the station's
// resolve_model_for_profile: models/dinov3-small_<BundleKey>, then models/dinov3-small_<bundlekey>, then
// models/dinov3-small_<stock>, then models/dinov3-small_portra400. adapter_dir receives the directory used.
// py-ref: negicc_station/src/intent_model.py @ 93be1ec (resolve_model_for_profile)
const EmbeddedFile* find_embedded_adapter(const std::string& profile_name, std::string& adapter_dir);

// The DINOv3 backbone built in: its TensorRT engine (backbone.engine) when trt_engine, else its ONNX model
// (backbone.onnx). Null when the build has none.
const EmbeddedFile* find_embedded_backbone(bool trt_engine);

#endif  // EMBEDDED_DATA_H
