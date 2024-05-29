#ifndef PBRT_UTIL_MEDIATRACKER_H
#define PBRT_UTIL_MEDIATRACKER_H

#include "pbrt/base/medium.h"
#include "pbrt/util/error.h"
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
    void push_back(Medium medium) {
        // Only add non-empty volumes
        if (medium != nullptr) {
            if (_index + 1 > NNestedVolumes) {
                Warning("Pushed Volumes, %d, exceeds maximum allowed, %d\n", _index,
                        NNestedVolumes);
                _mediums[_index] = medium;
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
    Medium back() const { return _mediums[_index - 1]; }

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
