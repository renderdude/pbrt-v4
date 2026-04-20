#ifndef PBRT_UTIL_MEDIATRACKER_H
#define PBRT_UTIL_MEDIATRACKER_H

#include "pbrt/base/medium.h"
#include "pbrt/pbrt.h"
#include "pbrt/util/error.h"
#include "pbrt/util/log.h"
#include "pbrt/util/pstd.h"

namespace pbrt {

static constexpr int NNestedVolumes = 2;

class MediaTracker {
  public:
    MediaTracker() = default;

    PBRT_CPU_GPU
    bool valid() { return _index > 0; }

    PBRT_CPU_GPU
    void set(Medium medium, int index = 0) { _mediums[index] = medium; }

    PBRT_CPU_GPU
    Medium get(int index = 0) { return _mediums[index]; }

    PBRT_CPU_GPU
    Medium get(int index = 0) const { return _mediums[index]; }

    PBRT_CPU_GPU
    bool has(Medium medium) const {
        for (int i = 0; i < _index; i++) {
            if (_mediums[i] == medium) {
                return true;
            }
        }
        return false;
    }

    PBRT_CPU_GPU
    void push_back(Medium medium, Point3f pt) {
        // Only add non-empty volumes
        if (medium != nullptr) {
            if (_index + 1 > NNestedVolumes) {
                bool all_inside = true;
                for (int i = _index-1; i >= 0; i--) {
                    if (!_mediums[i].inside(pt)) {
                        all_inside = false;
                        remove(_mediums[i]);
                    }
                }
                if (all_inside) {
                    // All existing mediums are still active; no room for the new one.
                    // Discard it and warn — writing past the array end would be UB.
                    Warning("Pushed Volumes, %d, exceeds maximum allowed, %d; "
                            "new medium discarded\n", _index, NNestedVolumes);
                }
                else {
                    _mediums[_index++] = medium;
                }
            } else
                _mediums[_index++] = medium;
        }
    }

    PBRT_CPU_GPU
    void pop_back() {
        DCHECK(_index > 0);
        _mediums[--_index] = 0;
    }

    PBRT_CPU_GPU
    void remove(Medium medium) {
        DCHECK(_index > 0);
        int index = -1;
        for(int i = 0; i < _index; i++) {
            if (_mediums[i] == medium) {
                index = i;
                break;
            }
        }

        // This can potentially occur if there's an event close to the interface that
        // causes the ray to enter the volume, but the origin and hit point were within
        // error bounds causing the hit to be rejected
        if (index == -1) {
            LOG_VERBOSE("Unable to find requested medium in list");
        }
        else {
            while(index < _index-1) {
                _mediums[index] = _mediums[index+1];
                index++;
            }
            _mediums[index] = 0;
            _index--;
        }
    }

    PBRT_CPU_GPU
    Medium back() const { return _mediums[_index - 1]; }

    PBRT_CPU_GPU
    int count() const { return _index; }

    PBRT_CPU_GPU
    Medium operator[](int i) const {
        DCHECK(i >= 0 && i < NNestedVolumes);
        return _mediums[i];
    }
    PBRT_CPU_GPU
    Medium &operator[](int i) {
        DCHECK(i >= 0 && i < NNestedVolumes);
        return _mediums[i];
    }

    std::string ToString() const {
        if (count() == 0) {
            return "";
        }
        return StringPrintf("[ count = %d, current medium: %s  ]", _index,
                            back().ToString());
    }

  private:
    friend struct SOA<MediaTracker>;
    int _index = 0;
    pstd::array<Medium, NNestedVolumes> _mediums;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MEDIATRACKER_H
