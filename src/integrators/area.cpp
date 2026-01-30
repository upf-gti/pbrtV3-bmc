
#include "integrators/area.h"
#include "interaction.h"
#include "camera.h"
#include "film.h"
#include "paramset.h"
#include "lights/diffuse.h"

// AreaIntegrator CMC implementation
namespace pbrt {

// Struct to be consistent with BMC implementation
struct sSamplingParams_area {
    Vector3f corner;
    Vector3f diagonal;
};

// Function to generate random point on area light source
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

// AreaIntegrator Method
Spectrum AreaIntegrator::Li(const RayDifferential &ray, const Scene &scene,
                               Sampler &sampler, MemoryArena &arena,
                               int depth) const {

    SurfaceInteraction isect;
    if (!scene.Intersect(ray, &isect)) {
        return Spectrum(0.0);
    }

    isect.ComputeScatteringFunctions(ray, arena);

    Vector3f wo = isect.wo;
    Spectrum Le = isect.Le(wo);
    
    // Check if light can illuminate the intersection point
    std::shared_ptr<Light> light = scene.lights[0];
    if (!light->CanIlluminate(isect)) {
        return Le;
    }

    // Initialize common variables for Area integrator
    // Light parameters
    auto *area_light = dynamic_cast<const DiffuseAreaLight *>(light.get());
    
    Vector3f wi;
    Float pdf;
    VisibilityTester visibility;
    Le = area_light->Sample_Li(isect, sampler.Get2D(), &wi, &pdf, &visibility);
    Normal3f light_normal = visibility.P1().n;
    
    // Sampling parameters for area light
    Bounds3f bounds = area_light->shape->WorldBound();
    sSamplingParams_area *samplingParams = new sSamplingParams_area(); // Using this struct to be consistent with BMC
    samplingParams->corner = Vector3f(bounds.pMin.x, bounds.pMin.y, bounds.pMin.z);
    samplingParams->diagonal = Vector3f(bounds.pMax.x - bounds.pMin.x,
                                        bounds.pMax.y - bounds.pMin.y,
                                        bounds.pMax.z - bounds.pMin.z);
            
    // Other variables for the Monte Carlo integration
    Float pdf_area = 1.0 / area_light->area;
    Point3f x = isect.p;
    Spectrum L(0.0);

    // Monte Carlo integration over the area light source
    for (int i = 0; i < num_shading_samples; ++i) {

        // Sample random point on area light and compute wi
        Vector3f random_point = random_on_area(samplingParams);
        Vector3f vector_to_light = random_point - Vector3f(x.x, x.y, x.z);
        wi = Normalize(vector_to_light);

        // Visibility check, if unoccluded, accumulate contribution
        Ray nextRay = isect.SpawnRay(wi);
        nextRay.tMax = vector_to_light.Length() - 1e-4;      
        if (scene.IntersectP(nextRay)) continue;
        
        // Compute BRDF and geometry term
        Spectrum BRDF = isect.bsdf->f(wo, wi);
        Float G_term = Dot(wi, isect.shading.n) * Dot(-wi, light_normal) / vector_to_light.LengthSquared();

        L += Le * BRDF * G_term;
    }
        
    L /= (num_shading_samples * pdf_area);
        
    return L;
}

AreaIntegrator *CreateAreaIntegrator(
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

    printf("\nExecuting Area CMC Integrator\n");

    return new AreaIntegrator(numSamples, camera, sampler, pixelBounds);
}

}  // namespace pbrt
