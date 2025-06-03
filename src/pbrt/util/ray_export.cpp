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
        std::stringstream ss;
        short size = (short)(photons.segments.size());
        // 3. How many path types (up to 3, Light, Primary, and Shadow)
        ss.write((char *)(&size), sizeof(short));
        CHECK_NE(size, 0);
        for (const auto& [key, value] : photons.segments) {
            // 4. Light path type
            ss.write((char *)(&key), sizeof(char));
            size = (short)(value.size());
            // 5. How many paths of that type
            ss.write((char *)(&size), sizeof(short));
            CHECK_NE(size, 0);
            for (auto &segments : value) {
                size = (short)(segments.size());
                // 6. How many segments in the path
                ss.write((char *)(&size), sizeof(short));
                CHECK_NE(size, 0);
                for (auto &segment : segments) {
                    // 7. status (what happened to this segment)
                    ss.write((char *)&(segment.status), sizeof(char));
                    Point3f pt = cam_xform.ApplyInverse(segment.start_pt);
                    // 8. start point
                    ss.write((char *)&pt, sizeof(Point3f));
                    CHECK_NE(segment.end_pt, Point3f());
                    pt = cam_xform.ApplyInverse(segment.end_pt);
                    // 9. end point
                    ss.write((char *)&pt, sizeof(Point3f));
                }
            }
        }
        std::string sss = ss.str();
        int sizei = (int)(sss.size());
        // 1. pixel
        output_file.write((char *)&photons.pixel, sizeof(Point2i));
        // 2. size of binary blob
        output_file.write((char *)&sizei, sizeof(int));
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

    Point2i pt = camera.GetFilm().PixelBounds().pMin;
    output_file.write((char *)&pt, sizeof(Point2i));
    pt = camera.GetFilm().PixelBounds().pMax;
    output_file.write((char *)&pt, sizeof(Point2i));

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
