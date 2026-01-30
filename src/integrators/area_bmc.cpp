
#include "area_bmc.h"
#include "interaction.h"
#include "camera.h"
#include "film.h"
#include "paramset.h"
#include "lights/diffuse.h"
#include "cov_kernels.h"

// AreaIntegrator BMC implementation
namespace pbrt {

struct sSamplingParams_area {
    Vector3f corner;
    Vector3f diagonal;
};

// Function to sample a random point on the area light
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
    
    isect.ComputeScatteringFunctions(ray, arena);

    Vector3f wo = isect.wo;
    Spectrum Le = isect.Le(wo);
    std::shared_ptr<Light> light = scene.lights[0];

    // Check if light can illuminate the intersection point
    if (!light->CanIlluminate(isect)) {
        return Le;
    }
    
    // Initialize common variables for Area integrator
    // Light parameters
    Vector3f wi;
    Float pdf;
    VisibilityTester visibility;
    Le = light->Sample_Li(isect, sampler.Get2D(), &wi, &pdf, &visibility);
    Normal3f light_normal = visibility.P1().n;

    // Choose a random Gaussian Process instance
    BMC<Vector3f, Spectrum> *bmc = bmc_list[rand() % num_bmcs];

    // Other variables for the Monte Carlo integration
    std::vector<Spectrum> radianceSamples(num_shading_samples, Spectrum(0.0));
    Point3f x = isect.p;
    
    for (uint32_t i = 0; i < num_shading_samples; ++i) {

        // Get random point on area light from GP and compute wi
        Vector3f random_point = bmc->get_gaussian_process()->get_observation(i);
        Vector3f vector_to_light = random_point - Vector3f(x.x, x.y, x.z);
        wi = Normalize(vector_to_light);

        // Check visibility
        Ray nextRay = isect.SpawnRay(wi);
        nextRay.tMax = vector_to_light.Length() - 1e-4;
        if (scene.IntersectP(nextRay)) continue;

        // Evaluate BRDF and geometry term
        Spectrum BRDF = isect.bsdf->f(wo, wi);
        Float G_term = Dot(wi, isect.shading.n) * Dot(-wi, light_normal) /
                        vector_to_light.LengthSquared();
        radianceSamples[i] = (Le * BRDF * G_term);   
    }

    // Compute the integral using BMC
    Spectrum L(0.0);
    bmc->compute_integral(radianceSamples, L);

    return L;
}

void AreaBMCIntegrator::Preprocess(const Scene &scene,
                                          Sampler &sampler) {
    
    // Set up sampling parameters for area light
    std::shared_ptr<Light> light = scene.lights[0];
    auto *area_light = dynamic_cast<const DiffuseAreaLight *>(light.get());

    Bounds3f bounds = area_light->shape->WorldBound();
    sSamplingParams_area *samplingParams = new sSamplingParams_area();
    samplingParams->corner = Vector3f(bounds.pMin.x, bounds.pMin.y, bounds.pMin.z);
    samplingParams->diagonal = Vector3f(bounds.pMax.x - bounds.pMin.x, 
                                        bounds.pMax.y - bounds.pMin.y,
                                        bounds.pMax.z - bounds.pMin.z);

    Float pdf_area = 1.0 / area_light->area; // Uniform sampling PDF over area light
    bmc_list.resize(num_bmcs);
    std::vector<Vector3f> sample_points(num_shading_samples);

    for (uint32_t i = 0; i < num_bmcs; ++i) {

        // Define the kernel and Gaussian Process instance
        pbrt_kernel::sAreaParams *area_params = new pbrt_kernel::sAreaParams();
        area_params->h = samplingParams->diagonal.Length();

        GaussianProcess<Vector3f, Spectrum>::sKernelInfo kernel_info;
        kernel_info.kernel = pbrt_kernel::kernel_area;
        kernel_info.kernel_params = area_params;

        GaussianProcess<Vector3f, Spectrum> *gaussian_process = new GaussianProcess<Vector3f, Spectrum>(kernel_info, 0.01);

        // Change random seed
        srand(1998 + i);

        // Generate random sample points on area light
        for (uint32_t s_idx = 0; s_idx < num_shading_samples; s_idx++) {
            sample_points[s_idx] = random_on_area(samplingParams);
        }

        // Fill the GP instance with the array of points (observation points)
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

    printf("\nExecuting Area BMC Integrator\n");

    return new AreaBMCIntegrator(numSamples, numBMCs, std::vector<BMC<Vector3f, Spectrum> *>(), camera, sampler, pixelBounds);
    
}

}  // namespace pbrt