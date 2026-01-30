
#include "direct.h"
#include "interaction.h"
#include "camera.h"
#include "film.h"
#include "paramset.h"
#include "eigen_utils.h"

namespace pbrt {

// Functions to generate random vectors on the hemisphere, using spherical coordinates
Vector3f DirectIntegrator::random_unit_vector() const {
    Float psi1 = rand() / (RAND_MAX + 1.0);
    Float psi2 = rand() / (RAND_MAX + 1.0);

    Float theta = std::acos(psi1);
    Float phi = 2 * Pi * psi2;

    Float x = std::sin(theta) * std::cos(phi);
    Float y = std::sin(theta) * std::sin(phi);
    Float z = std::cos(theta);

    return Vector3f(x, y, z);
}

// normal axis as parameter
Vector3f DirectIntegrator::random_on_hemisphere(void *data) const {
    const Vector3f *normal = static_cast<const Vector3f *>(data);
    Vector3f on_unit_sphere = random_unit_vector();
    if (Dot(on_unit_sphere, *normal) > 0.0) {  // In the same hemisphere as the normal
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

// Struct to be consistent with BMC implementation
struct sSamplingParams {
    Vector3f normal;
};

// DirectIntegrator Method
Spectrum DirectIntegrator::Li(const RayDifferential &ray, const Scene &scene,
                               Sampler &sampler, MemoryArena &arena,
                               int depth) const {
    
    SurfaceInteraction isect, random_si;
    if (!scene.Intersect(ray, &isect)) {
        Spectrum L(0.0);
    }

    isect.ComputeScatteringFunctions(ray, arena);

    // Parameters for Monte Carlo sum
    Vector3f wo = isect.wo;
    Float pdf = Inv2Pi;
    sSamplingParams *sampling_params = new sSamplingParams();
    sampling_params->normal = {0.0, 0.0, 1.0};

    Spectrum L(0.0);
    for (uint32_t i = 0; i < num_shading_samples; ++i) {
        // Generate random direction in hemisphere
        Vector3f wiLocal = random_on_hemisphere(sampling_params);
        Vector3f wiWorld = Normalize(isect.bsdf->LocalToWorld(wiLocal));

        // Check if the direction points to somewhere in the scene
        Ray nextRay = isect.SpawnRay(wiWorld);
        if (!scene.Intersect(nextRay, &random_si)) continue;

        // Evaluate Rendering Equation
        Spectrum Le = random_si.Le(-nextRay.d);
        Spectrum BRDF = isect.bsdf->f(wo, wiWorld);
        Float G_term = Dot(wiWorld, isect.shading.n);
        L += Le * BRDF * G_term;
    }
    
    // Average over the number of samples and account for the PDF
    L /= (num_shading_samples * pdf);

    // Compute emitted light if ray hit an area light source
    L += isect.Le(wo);

    return L;
}

DirectIntegrator* CreateDirectIntegrator(
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

    printf("\nExecuting Direct CMC Integrator\n");

    return new DirectIntegrator(numSamples, camera, sampler, pixelBounds);
}

}  // namespace pbrt