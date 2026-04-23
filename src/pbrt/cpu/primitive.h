// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_PRIMITIVE_H
#define PBRT_CPU_PRIMITIVE_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/medium.h>
#include <pbrt/base/shape.h>
#include <pbrt/base/texture.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/transform.h>

#include <memory>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Primitives", primitiveMemory);

class SimplePrimitive;
class GeometricPrimitive;
class TransformedPrimitive;
class AnimatedPrimitive;
class MultiVolumePrimitive;
class BVHAggregate;
class KdTreeAggregate;

// Primitive Definition
class Primitive
    : public TaggedPointer<SimplePrimitive, GeometricPrimitive, TransformedPrimitive,
                           AnimatedPrimitive, MultiVolumePrimitive,
                           BVHAggregate, KdTreeAggregate> {
  public:
    // Primitive Interface
    using TaggedPointer::TaggedPointer;

    Bounds3f Bounds() const;

    pstd::optional<ShapeIntersection> Intersect(const Ray &r,
                                                Float tMax = Infinity) const;
    bool IntersectP(const Ray &r, Float tMax = Infinity) const;
};

// GeometricPrimitive Definition
class GeometricPrimitive {
  public:
    // GeometricPrimitive Public Methods
    GeometricPrimitive(Shape shape, Material material, Light areaLight,
                       const MediumInterface &mediumInterface,
                       FloatTexture alpha = nullptr);
    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;

  private:
    // GeometricPrimitive Private Members
    Shape shape;
    Material material;
    Light areaLight;
    MediumInterface mediumInterface;
    FloatTexture alpha;
};

// SimplePrimitive Definition
class SimplePrimitive {
  public:
    // SimplePrimitive Public Methods
    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;
    SimplePrimitive(Shape shape, Material material);

  private:
    // SimplePrimitive Private Members
    Shape shape;
    Material material;
};

// TransformedPrimitive Definition
class TransformedPrimitive {
  public:
    // TransformedPrimitive Public Methods
    TransformedPrimitive(Primitive primitive, const Transform *renderFromPrimitive)
        : primitive(primitive), renderFromPrimitive(renderFromPrimitive) {
        primitiveMemory += sizeof(*this);
    }

    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;

    Bounds3f Bounds() const { return (*renderFromPrimitive)(primitive.Bounds()); }

  private:
    // TransformedPrimitive Private Members
    Primitive primitive;
    const Transform *renderFromPrimitive;
};

// AnimatedPrimitive Definition
class AnimatedPrimitive {
  public:
    // AnimatedPrimitive Public Methods
    Bounds3f Bounds() const {
        return renderFromPrimitive.MotionBounds(primitive.Bounds());
    }

    AnimatedPrimitive(Primitive primitive, const AnimatedTransform &renderFromPrimitive);
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;

  private:
    // AnimatedPrimitive Private Members
    Primitive primitive;
    AnimatedTransform renderFromPrimitive;
};

// MultiVolumePrimitive Definition
//
// Groups N interface shapes whose faces are spatially coincident (occupy the
// same position along the ray). On intersection it finds all sub-shapes hit
// within a floating-point epsilon of the nearest hit, accumulates all of their
// medium transitions in one step, then returns a synthetic surface interaction
// with mediumInterface=nullptr and medium=accumulated. The integrator's
// SkipIntersection call then simply advances the ray origin without applying any
// further transition, which is correct because all transitions were pre-applied.
class MultiVolumePrimitive {
  public:
    struct SubShape {
        Shape shape;
        MediumInterface mediumInterface;
    };

    MultiVolumePrimitive(std::vector<SubShape> subShapes);

    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax) const;

  private:
    std::vector<SubShape> subShapes;
};

}  // namespace pbrt

#endif  // PBRT_CPU_PRIMITIVE_H
