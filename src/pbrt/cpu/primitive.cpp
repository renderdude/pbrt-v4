// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/cpu/primitive.h>

#include <pbrt/cpu/aggregates.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

Bounds3f Primitive::Bounds() const {
    auto bounds = [&](auto ptr) { return ptr->Bounds(); };
    return DispatchCPU(bounds);
}

pstd::optional<ShapeIntersection> Primitive::Intersect(const Ray &r, Float tMax) const {
    auto isect = [&](auto ptr) { return ptr->Intersect(r, tMax); };
    return DispatchCPU(isect);
}

bool Primitive::IntersectP(const Ray &r, Float tMax) const {
    auto isectp = [&](auto ptr) { return ptr->IntersectP(r, tMax); };
    return DispatchCPU(isectp);
}

// GeometricPrimitive Method Definitions
GeometricPrimitive::GeometricPrimitive(Shape shape, Material material, Light areaLight,
                                       const MediumInterface &mediumInterface,
                                       FloatTexture alpha)
    : shape(shape),
      material(material),
      areaLight(areaLight),
      mediumInterface(mediumInterface),
      alpha(alpha) {
    primitiveMemory += sizeof(*this);
}

Bounds3f GeometricPrimitive::Bounds() const {
    return shape.Bounds();
}

pstd::optional<ShapeIntersection> GeometricPrimitive::Intersect(const Ray &r,
                                                                Float tMax) const {
    pstd::optional<ShapeIntersection> si = shape.Intersect(r, tMax);
    if (!si)
        return {};
    CHECK_LT(si->tHit, 1.001 * tMax);
    // Test intersection against alpha texture, if present
    if (alpha) {
        if (Float a = alpha.Evaluate(si->intr); a < 1) {
            // Possibly ignore intersection based on stochastic alpha test
            Float u = (a <= 0) ? 1.f : HashFloat(r.o, r.d);
            if (u > a) {
                // Ignore this intersection and trace a new ray
                Ray rNext = si->intr.SpawnRay(r.d);
                pstd::optional<ShapeIntersection> siNext =
                    Intersect(rNext, tMax - si->tHit);
                if (siNext)
                    siNext->tHit += si->tHit;
                return siNext;
            }
        }
    }

    // Initialize _SurfaceInteraction_ after _Shape_ intersection
    si->intr.SetIntersectionProperties(material, areaLight, &mediumInterface, r.medium);

    return si;
}

bool GeometricPrimitive::IntersectP(const Ray &r, Float tMax) const {
    if (alpha)
        return Intersect(r, tMax).has_value();
    else
        return shape.IntersectP(r, tMax);
}

// SimplePrimitive Method Definitions
SimplePrimitive::SimplePrimitive(Shape shape, Material material)
    : shape(shape), material(material) {
    primitiveMemory += sizeof(*this);
}

Bounds3f SimplePrimitive::Bounds() const {
    return shape.Bounds();
}

bool SimplePrimitive::IntersectP(const Ray &r, Float tMax) const {
    return shape.IntersectP(r, tMax);
}

pstd::optional<ShapeIntersection> SimplePrimitive::Intersect(const Ray &r,
                                                             Float tMax) const {
    pstd::optional<ShapeIntersection> si = shape.Intersect(r, tMax);
    if (!si)
        return {};

    si->intr.SetIntersectionProperties(material, nullptr, nullptr, r.medium);

    return si;
}

// TransformedPrimitive Method Definitions
pstd::optional<ShapeIntersection> TransformedPrimitive::Intersect(const Ray &r,
                                                                  Float tMax) const {
    // Transform ray to primitive-space and intersect with primitive
    Ray ray = renderFromPrimitive->ApplyInverse(r, &tMax);
    pstd::optional<ShapeIntersection> si = primitive.Intersect(ray, tMax);
    if (!si)
        return {};
    CHECK_LT(si->tHit, 1.001 * tMax);

    // Return transformed instance's intersection information
    si->intr = (*renderFromPrimitive)(si->intr);
    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0);
    return si;
}

bool TransformedPrimitive::IntersectP(const Ray &r, Float tMax) const {
    Ray ray = renderFromPrimitive->ApplyInverse(r, &tMax);
    return primitive.IntersectP(ray, tMax);
}

// AnimatedPrimitive Method Definitions
AnimatedPrimitive::AnimatedPrimitive(Primitive p,
                                     const AnimatedTransform &renderFromPrimitive)
    : primitive(p), renderFromPrimitive(renderFromPrimitive) {
    primitiveMemory += sizeof(*this);
    CHECK(renderFromPrimitive.IsAnimated());
}

pstd::optional<ShapeIntersection> AnimatedPrimitive::Intersect(const Ray &r,
                                                               Float tMax) const {
    // Compute _ray_ after transformation by _renderFromPrimitive_
    Transform interpRenderFromPrimitive = renderFromPrimitive.Interpolate(r.time);
    Ray ray = interpRenderFromPrimitive.ApplyInverse(r, &tMax);
    pstd::optional<ShapeIntersection> si = primitive.Intersect(ray, tMax);
    if (!si)
        return {};

    // Transform instance's intersection data to rendering space
    si->intr = interpRenderFromPrimitive(si->intr);
    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0);
    return si;
}

bool AnimatedPrimitive::IntersectP(const Ray &r, Float tMax) const {
    Ray ray = renderFromPrimitive.ApplyInverse(r, &tMax);
    return primitive.IntersectP(ray, tMax);
}

// MultiVolumePrimitive Method Definitions

MultiVolumePrimitive::MultiVolumePrimitive(std::vector<SubShape> s)
    : subShapes(std::move(s)) {
    primitiveMemory += sizeof(*this) + subShapes.size() * sizeof(SubShape);
}

Bounds3f MultiVolumePrimitive::Bounds() const {
    Bounds3f b;
    for (const auto &s : subShapes)
        b = Union(b, s.shape.Bounds());
    return b;
}

pstd::optional<ShapeIntersection> MultiVolumePrimitive::Intersect(const Ray &ray,
                                                                   Float tMax) const {
    // Intersect all sub-shapes, keeping each result.
    std::vector<pstd::optional<ShapeIntersection>> hits(subShapes.size());
    int nearestIdx = -1;
    for (int i = 0; i < (int)subShapes.size(); ++i) {
        hits[i] = subShapes[i].shape.Intersect(ray, tMax);
        if (hits[i] && (nearestIdx < 0 || hits[i]->tHit < hits[nearestIdx]->tHit))
            nearestIdx = i;
    }
    if (nearestIdx < 0)
        return {};

    Float tNearest = hits[nearestIdx]->tHit;
    Float eps = 64.f * MachineEpsilon * (1.f + tNearest);

    // Accumulate medium transitions for every sub-shape hit within eps of nearest.
    // Each sub-shape is an interface surface; apply GetMedium in sequence so the
    // accumulated medium flows correctly from one transition to the next.
    MediaTracker acc = ray.medium;
    for (int i = 0; i < (int)subShapes.size(); ++i) {
        if (!hits[i] || hits[i]->tHit > tNearest + eps)
            continue;
        auto &intr = hits[i]->intr;
        intr.medium = acc;
        intr.mediumInterface = &subShapes[i].mediumInterface;
        acc = intr.GetMedium(ray.d, (Point3f)intr.pi);
    }

    // Return the nearest hit with the fully accumulated medium.
    // Setting mediumInterface=nullptr ensures SkipIntersection just advances
    // the ray origin without re-applying any transition (all already applied).
    auto &result = hits[nearestIdx].value();
    result.intr.medium = acc;
    result.intr.mediumInterface = nullptr;
    result.intr.material = nullptr;
    return result;
}

bool MultiVolumePrimitive::IntersectP(const Ray &ray, Float tMax) const {
    for (const auto &s : subShapes)
        if (s.shape.IntersectP(ray, tMax))
            return true;
    return false;
}

}  // namespace pbrt
