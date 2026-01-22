
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


#include "area_bmc.h"
#include "interaction.h"
#include "camera.h"
#include "film.h"
#include "paramset.h"
#include "lights/diffuse.h"
#include "cov_kernels.h"

// GTI

namespace pbrt {

struct sSamplingParams_area {
    Vector3f corner;
    Vector3f diagonal;
};

inline Vector3f random_on_area(const void *data) {
    const sSamplingParams_area *params =
        static_cast<const sSamplingParams_area *>(data);
    Vector3f corner = params->corner;
    Vector3f diagonal = params->diagonal;

    Float r1 = rand() / (RAND_MAX + 1.0);
    Float r2 = rand() / (RAND_MAX + 1.0);

    return Vector3f(corner.x + r1 * diagonal.x, corner.y,
                    corner.z + r2 * diagonal.z);
}

// DirectIntegrator Method Definitions
Spectrum AreaBMCIntegrator::Li(const RayDifferential &ray, const Scene &scene,
                               Sampler &sampler, MemoryArena &arena,
                               int depth) const {
    
    SurfaceInteraction isect;
    if (!scene.Intersect(ray, &isect)) {
        return Spectrum(0.0);
    }
    Spectrum L(0.0), f(0.0);
    std::shared_ptr<Light> light = scene.lights[0];

    if (!light->CanIlluminate(isect)) {
        return Spectrum(0.0);
    }

    // Initialize common variables for Direct integrator
    Vector3f wo = isect.wo;
    Spectrum Le = isect.Le(wo);

    isect.ComputeScatteringFunctions(ray, arena);
    if (Le.MaxComponentValue() > 0.0) return Le;
        
    Vector3f wi;
    Float pdf;
    VisibilityTester visibility;
    Le = light->Sample_Li(isect, sampler.Get2D(), &wi, &pdf, &visibility);
    Normal3f light_normal = visibility.P1().n;

    uint32_t randomGP = rand() % num_bmcs;
    BMC<Vector3f, Spectrum> *bmc = bmc_list[randomGP];
    // Add contribution of each light source
    Vector3f random_point;
    std::vector<Spectrum> radianceSamples;
    Point3f x = isect.p;

    for (uint32_t i = 0; i < num_shading_samples; ++i) {
        random_point = bmc->get_gaussian_process()->get_observation(i);
        Vector3f vector_to_light = random_point - Vector3f(x.x, x.y, x.z);
        Vector3f wi = Normalize(vector_to_light);
        Ray nextRay = isect.SpawnRay(wi);
        nextRay.tMax = vector_to_light.Length() - 1e-4;

        if (!scene.IntersectP(nextRay)) {
            Spectrum f = isect.bsdf->f(wo, wi) * Dot(wi, isect.shading.n);
            Float geo = Dot(-wi, light_normal) / vector_to_light.LengthSquared();

            radianceSamples.push_back(Le * f * geo);
        } else {
            radianceSamples.push_back(Spectrum(0.0));
        }
    }
    bmc->compute_integral(radianceSamples, L);   

    return L;
}

void AreaBMCIntegrator::Preprocess(const Scene &scene,
                                          Sampler &sampler) {
    std::shared_ptr<Light> light = scene.lights[0];
    auto *area_light = dynamic_cast<const DiffuseAreaLight *>(light.get());

    Vector3f wi;
    VisibilityTester visibility;
    Normal3f light_normal = visibility.P1().n;

    Bounds3f bounds = area_light->shape->WorldBound();
    sSamplingParams_area *samplingParams = new sSamplingParams_area();
    samplingParams->corner = Vector3f(bounds.pMin.x, bounds.pMin.y, bounds.pMin.z);
    samplingParams->diagonal = Vector3f(bounds.pMax.x - bounds.pMin.x, bounds.pMax.y - bounds.pMin.y,
                 bounds.pMax.z - bounds.pMin.z);

    Float pdf_area = 1.0 / area_light->area;
    bmc_list.resize(num_bmcs);
    std::vector<Vector3f> sample_points;

    for (uint32_t i = 0; i < num_bmcs; ++i) {
        pbrt_kernel::sAreaParams *area_params = new pbrt_kernel::sAreaParams();
        area_params->h = samplingParams->diagonal.Length();

        GaussianProcess<Vector3f, Spectrum>::sKernelInfo kernel_info;

        kernel_info.kernel = pbrt_kernel::kernel_area;
        kernel_info.kernel_params = area_params;

        GaussianProcess<Vector3f, Spectrum> *gaussian_process =
            new GaussianProcess<Vector3f, Spectrum>(kernel_info, 0.01);

        // Set x number of samples (observation/training points), in our case
        // directions
        sample_points.clear();
        sample_points.reserve(num_shading_samples);

        // Victor's birth year plus offset :)
        srand(1998 + i);

        for (uint32_t s_idx = 0; s_idx < num_shading_samples; s_idx++) {
            sample_points.push_back(random_on_area(samplingParams));
        }

        // Fill the GP instance with the array of directions (observation
        // points)
        gaussian_process->set_observations(sample_points, {});

        BMC<Vector3f, Spectrum>::sSamplingInfo sampling_info;
        sampling_info.sample = random_on_area;
        sampling_info.sample_params = samplingParams;

        bmc_list[i] = new BMC<Vector3f, Spectrum>(sampling_info, gaussian_process, pdf_area);
    }

}

AreaBMCIntegrator *CreateAreaBMCIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int numSamples = params.FindOneInt("numsamples", 32);

    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }

    std::vector<BMC<Vector3f, Spectrum> *> bmc_list;   

    printf("\nExecuting Area BMC Integrator\n");

    return new AreaBMCIntegrator(numSamples, bmc_list, camera, sampler,
                                 pixelBounds);
    
}

}  // namespace pbrt