#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_AREA_H
#define PBRT_INTEGRATORS_AREA_H

#include "pbrt.h"
#include "integrator.h"
#include "scene.h"

namespace pbrt {

class AreaIntegrator : public SamplerIntegrator {
  public:
    AreaIntegrator(int numSamples, std::shared_ptr<const Camera> camera,
                      std::shared_ptr<Sampler> sampler,
                      const Bounds2i &pixelBounds)
        : SamplerIntegrator(camera, sampler, pixelBounds),
          num_shading_samples(numSamples) {}
    Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;
    //Vector3f random_on_area(const void *data) const;

  private:
    uint32_t num_shading_samples;
};

AreaIntegrator *CreateAreaIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_AREA_H