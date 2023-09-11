// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/ray_export.h>

#include <pbrt/cameras.h>
#include <pbrt/ray.h>
#include <pbrt/util/error.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/image.h>
#include <pbrt/util/print.h>
#include <pbrt/util/string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>
#include "pbrt/util/transform.h"

namespace pbrt {

static std::atomic<bool> exitThread{false};
static std::mutex mutex;
static std::thread updateThread;
static std::queue<Photon_Tile> dynamicItems;
static std::ofstream output_file;
static Transform cam_xform;

void export_raytree(Photon_Tile &photon_tile) {
    for (auto &photons : photon_tile) {
        output_file.write((char *)&photons.pixel, sizeof(Point2i));
        std::stringstream ss;
        short size = (short)(photons.segments.size());
        ss.write((char *)&size, sizeof(short));
        for (auto &segment : photons.segments) {
            ss.write((char *)&segment.seg_type, sizeof(char));
            Point3f pt = cam_xform.ApplyInverse(segment.start_pt);
            ss.write((char *)&pt, sizeof(Point3f));
            CHECK_NE(segment.end_pt, Point3f());
            pt = cam_xform.ApplyInverse(segment.end_pt);
            ss.write((char *)&pt, sizeof(Point3f));
        }
        std::string sss = ss.str();
        size = (short)(sss.size());
        output_file.write((char *)&size, sizeof(short));
        output_file.write(sss.c_str(), sss.size());
    }
}

static void updateDynamicItems() {
    while (!exitThread) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        while (!dynamicItems.empty()) {
            export_raytree(dynamicItems.front());
            std::lock_guard<std::mutex> lock(mutex);
            dynamicItems.pop();
        }
    }

    // One last time to get the last bits
    std::lock_guard<std::mutex> lock(mutex);
    while (!dynamicItems.empty()) {
        export_raytree(dynamicItems.front());
        std::lock_guard<std::mutex> lock(mutex);
        dynamicItems.pop();
    }
}

void ConnectToExporter(const std::string &filename, Camera &camera) {
    cam_xform = camera.GetCameraTransform().RenderFromWorld();

    output_file.open(filename, std::ios::out | std::ios::binary);
    // Portability info
    int size = sizeof(char);
    output_file.write((char *)&size, sizeof(int));
    size = sizeof(short);
    output_file.write((char *)&size, sizeof(int));
    size = sizeof(int);
    output_file.write((char *)&size, sizeof(int));
    size = sizeof(float);
    output_file.write((char *)&size, sizeof(int));

    updateThread = std::thread(updateDynamicItems);
}

void DisconnectFromExporter() {
    if (updateThread.get_id() != std::thread::id()) {
        exitThread = true;
        updateThread.join();
        updateThread = std::thread();
        exitThread = false;
    }
}

void ExportTile(Photon_Tile &tile) {
    std::lock_guard<std::mutex> lock(mutex);
    dynamicItems.push(tile);
}

}  // namespace pbrt
