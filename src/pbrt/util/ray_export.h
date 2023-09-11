// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_RAY_EXPORT_H
#define PBRT_UTIL_RAY_EXPORT_H

#include <pbrt/pbrt.h>
#include "pbrt/base/camera.h"
#include "pbrt/ray.h"

#include <functional>
#include <string>

namespace pbrt {

// DisplayServer Function Declarations
void ConnectToExporter(const std::string &filename, Camera& camera);
void DisconnectFromExporter();

void ExportTile(Photon_Tile& tile);

}  // namespace pbrt

#endif  // PBRT_UTIL_RAY_EXPORT_H
