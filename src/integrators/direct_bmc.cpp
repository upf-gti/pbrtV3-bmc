
#include "direct_bmc.h"
#include "interaction.h"
#include "camera.h"
#include "film.h"
#include "paramset.h"
#include "../GPRender/src/cov_kernels.h"

// DirectIntegrator implementation
namespace pbrt {

// Functions to generate random vectors on the hemisphere, using spherical coordinates
Vector3f random_unit_vector() {
    Float psi1 = rand() / (RAND_MAX + 1.0);
    Float psi2 = rand() / (RAND_MAX + 1.0);

    Float theta = std::acos(psi1);
    Float phi = 2 * Pi * psi2;

    Float x = std::sin(theta) * std::cos(phi);
    Float y = std::sin(theta) * std::sin(phi);
    Float z = std::cos(theta);

    return Vector3f(x, y, z);
}

// Normal axis as parameter
Vector3f random_on_hemisphere(const void *data) {
    const Vector3f *normal = static_cast<const Vector3f *>(data);
    Vector3f on_unit_sphere = random_unit_vector();
    if (Dot(on_unit_sphere, *normal) > 0.0) {  // In the same hemisphere as the normal
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

// Rotate a sample around Z by a random angle
Vector3f rotate_around_z(Vector3f dir, Float alpha) {
    Float sin_alpha = sin(alpha);
    Float cos_alpha = cos(alpha);
    return Vector3f(dir.x * cos_alpha - dir.y * sin_alpha,
                    dir.x * sin_alpha + dir.y * cos_alpha, dir.z);
}

struct sSamplingParams {
    Vector3f normal;
};

// DirectIntegrator Method Definitions
Spectrum DirectBMCIntegrator::Li(const RayDifferential &ray, const Scene &scene,
                               Sampler &sampler, MemoryArena &arena,
                               int depth) const {
    
    SurfaceInteraction isect, random_si;
    if (!scene.Intersect(ray, &isect)) {
        Spectrum L(0.0);
    }

    isect.ComputeScatteringFunctions(ray, arena);

    // Initialize common variables for Direct integrator
    Vector3f wo = isect.wo;

    // Choose a random Gaussian Process instance
    BMC<Vector3f, Spectrum> *bmc = bmc_list[rand() % num_bmcs];
    
    // Add contribution of each light source
    Vector3f wiLocal, wiWorld;
    std::vector<Spectrum> radianceSamples(num_shading_samples, Spectrum(0.0));

    // Random angle to rotate the GP directions
    Float alpha = TwoPi * sampler.Get1D();

    for (uint32_t i = 0; i < num_shading_samples; ++i) {

        // Get the i-th observation direction from the GP and rotate it
        wiLocal = bmc->get_gaussian_process()->get_observation(i);
        wiLocal = rotate_around_z(wiLocal, alpha);
        wiWorld = Normalize(isect.bsdf->LocalToWorld(wiLocal));

        // If ray points somewhere in the scene, continue
        Ray nextRay = isect.SpawnRay(wiWorld);
        if (!scene.Intersect(nextRay, &random_si)) continue;

        // Get emitted radiance from the intersected surface and sum contributions
        Spectrum Le = random_si.Le(-nextRay.d);
        Spectrum BRDF = isect.bsdf->f(wo, wiWorld);
        Float G_term = Dot(wiWorld, isect.shading.n);

        radianceSamples[i] = Le * BRDF * G_term;
    }

    // Integrate the samples using BMC/GPRender
    Spectrum L(0.0);
    bmc->compute_integral(radianceSamples, L);

    L += isect.Le(wo);

    return L;
}

// Preprocess method to initialize BMC instances
void DirectBMCIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {

    sSamplingParams *sampling_params = new sSamplingParams();
    sampling_params->normal = { 0.0, 0.0, 1.0 };

    Float pdf = Inv2Pi;  // Uniform hemisphere pdf
    bmc_list.resize(num_bmcs);
    std::vector<Vector3f> sample_directions(num_shading_samples);

    for (uint32_t i = 0; i < num_bmcs; ++i) {

        // Create Sobolev kernel parameters for GP
        pbrt_kernel::sSobolevParams *sobolev_params = new pbrt_kernel::sSobolevParams();
        sobolev_params->s = 1.5f;

        GaussianProcess<Vector3f, Spectrum>::sKernelInfo kernel_info;

        kernel_info.kernel = pbrt_kernel::sobolev;
        kernel_info.kernel_params = sobolev_params;

        GaussianProcess<Vector3f, Spectrum> *gaussian_process = new GaussianProcess<Vector3f, Spectrum>(kernel_info, 0.01);

        // Change seed for different random directions
        srand(1998 + i);

        // Generate random directions on the hemisphere
        for (uint32_t s_idx = 0; s_idx < num_shading_samples; s_idx++) {
            sample_directions[s_idx] = random_on_hemisphere(sampling_params);
        }

        // Fill the GP instance with the array of directions (observation points)
        gaussian_process->set_observations(sample_directions, {});

        // Create sampling info for BMC
        BMC<Vector3f, Spectrum>::sSamplingInfo sampling_info;
        sampling_info.sample = random_on_hemisphere;
        sampling_info.sample_params = sampling_params;

        bmc_list[i] = new BMC<Vector3f, Spectrum>(sampling_info, gaussian_process, pdf);
    }
}

DirectBMCIntegrator *CreateDirectBMCIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int numSamples = params.FindOneInt("numsamples", 256);
    int numBMCs = params.FindOneInt("numbmcs", 10);
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

    printf("\nExecuting Direct BMC Integrator\n");

    return new DirectBMCIntegrator(numSamples, numBMCs, bmc_list, camera, sampler, pixelBounds);
    
}

}  // namespace pbrt