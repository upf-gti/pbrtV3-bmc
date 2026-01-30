
#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_DIRECT_H
#define PBRT_INTEGRATORS_DIRECT_H

#include "pbrt.h"
#include "integrator.h"
#include "scene.h"

// DirectIntegrator CMC implementation
namespace pbrt {

class DirectIntegrator : public SamplerIntegrator {
    public:
        DirectIntegrator(int numSamples, std::shared_ptr<const Camera> camera,
                      std::shared_ptr<Sampler> sampler,
                      const Bounds2i &pixelBounds)
          : SamplerIntegrator(camera, sampler, pixelBounds),
            num_shading_samples(numSamples) {}
        Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;

        Vector3f random_unit_vector() const;
        Vector3f random_on_hemisphere(void *data) const;

    private:
        uint32_t num_shading_samples;
};

DirectIntegrator* CreateDirectIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_DIRECT_H
