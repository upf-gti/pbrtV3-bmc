
#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_DIRECT_BMC_H
#define PBRT_INTEGRATORS_DIRECT_BMC_H

#include "pbrt.h"
#include "integrator.h"
#include "scene.h"
#include "../GPRender/src/bmc.h"

namespace pbrt {

class DirectBMCIntegrator : public SamplerIntegrator {
    public:
        DirectBMCIntegrator(int numSamples, int numBMC,
                        std::vector<BMC<Vector3f, Spectrum> *> bmcList,
                        std::shared_ptr<const Camera> camera,
                      std::shared_ptr<Sampler> sampler,
                      const Bounds2i &pixelBounds)
          : SamplerIntegrator(camera, sampler, pixelBounds),
            num_shading_samples(numSamples), num_bmcs(numBMC), bmc_list(bmcList) {}
        
        Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;
        void Preprocess(const Scene &scene, Sampler &sampler);

                
    private:
        uint32_t num_shading_samples;
        std::vector<BMC<Vector3f, Spectrum> *> bmc_list;
        uint32_t num_bmcs;
};

DirectBMCIntegrator* CreateDirectBMCIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_DIRECT_H