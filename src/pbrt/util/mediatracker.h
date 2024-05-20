#ifndef PBRT_UTIL_MEDIATRACKER_H
#define PBRT_UTIL_MEDIATRACKER_H

#include "pbrt/base/medium.h"
#include "pbrt/util/pstd.h"

namespace pbrt {

static constexpr int NNestedVolumes = 1;

class MediaTracker {
  public:
    MediaTracker() = default;

    PBRT_CPU_GPU
    bool valid() { return _mediums[0] != nullptr; }

    PBRT_CPU_GPU
    void set(Medium medium, int index = 0) { _mediums[index] = medium; }

    PBRT_CPU_GPU
    Medium get(int index = 0) { return _mediums[index]; }

    PBRT_CPU_GPU
    Medium get(int index = 0) const { return _mediums[index]; }

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

    std::string ToString() const { return StringPrintf("[ medium: %s  ]", _mediums[0].ToString()); }

  private:
    friend struct SOA<MediaTracker>;
    pstd::array<Medium, NNestedVolumes> _mediums;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MEDIATRACKER_H
