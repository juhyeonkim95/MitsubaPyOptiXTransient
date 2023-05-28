#include "optix/integrators/common.h"

using namespace optix;

rtBuffer<float, 2>              transient_radiance_histogram;
rtDeclareVariable(float,         transient_dist_min, , );
rtDeclareVariable(float,         transient_dist_max, , );
rtDeclareVariable(uint,         transient_bin_num, , );
rtDeclareVariable(float,         speed_of_wave, , );


namespace path_transient{
RT_FUNCTION uint path_length_to_index(float path_length)
{
    uint idx = static_cast<unsigned int>((path_length - transient_dist_min) / (transient_dist_max - transient_dist_min) * transient_bin_num);
    idx = max(min(idx, transient_bin_num-1), 0);
    return idx;
}

RT_FUNCTION float luminance(float3 color)
{
    return 0.299 * color.x + 0.587* color.y + 0.114 * color.z;
}

RT_FUNCTION void path_trace(Ray& ray, unsigned int& seed, PerPathData &ppd)
{
    float emission_weight = 1.0;
    float3 throughput = make_float3(1.0);
    float3 result = make_float3(0.0);
    float path_length = 0.0;

    BSDFSample3f bs;

    // ---------------------- First intersection ----------------------
    SurfaceInteraction si;
    si.emission = make_float3(0);
    si.is_valid = true;
    rtTrace(top_object, ray, si);

    path_length += si.t;

#if USE_NEXT_EVENT_ESTIMATION
    int num_lights = sysLightParameters.size();
    float num_lights_inv = 1.0 / (float) (num_lights);
    LightSample lightSample;
    PerRayData_pathtrace_shadow prd_shadow;
#endif
    int depth;
    for (depth = 1; ; depth++){
        // ---------------- Intersection with emitters ----------------
        result += emission_weight * throughput * si.emission;
        if(si.emission.x > 0){
            // add transient output
            uint path_length_idx = path_length_to_index(path_length);
            uint2 idx = make_uint2(depth-1, path_length_idx);
            atomicAdd(&transient_radiance_histogram[idx], luminance(emission_weight * throughput * si.emission));
        }

        // ---------------- Terminate ray tracing ----------------
        // (1) over max depth
        // (2) ray missed
        // (3) hit emission
        if(depth >= max_depth || ! si.is_valid || (si.emission.x + si.emission.y + si.emission.z) > 0){
            break;
        }
        // (4) Russian roulette termination
        if(depth >= rr_begin_depth)
        {
            float pcont = fmaxf(throughput);
            pcont = max(pcont, 0.05);
            if(rnd(seed) >= pcont)
                break;
            throughput /= pcont;
        }

        MaterialParameter &mat = sysMaterialParameters[si.material_id];
        optix::Onb onb(si.normal);

#if USE_NEXT_EVENT_ESTIMATION
        // --------------------- Emitter sampling ---------------------
        float3 L = make_float3(0.0f);

        // sample light index
        int index = (num_lights==1)?0:optix::clamp(static_cast<int>(floorf(rnd(seed) * num_lights)), 0, num_lights - 1);
        const LightParameter& light = sysLightParameters[index];

        // sample light
        sample_light(si.p, light, seed, lightSample);

        float lightDist = lightSample.lightDist;
        float3 wo = lightSample.wi;
        float3 Li = lightSample.Li;
        float lightPdf = lightSample.pdf * num_lights_inv;
        //return make_float3(1.0f);
        bool is_light_delta = (light.lightType == LIGHT_POINT) || (light.lightType == LIGHT_DIRECTIONAL) || (light.lightType == LIGHT_SPOT);
        if ((!is_light_delta && (dot(wo, si.normal) <= 0.0f) )|| length(Li) == 0)
            L = make_float3(0.0);

        float3 wo_local = to_local(onb, wo);

        // Check visibility
        prd_shadow.inShadow = false;
        prd_shadow.seed = seed;
        optix::Ray shadowRay = optix::make_Ray(si.p, wo, 1, scene_epsilon, lightDist - scene_epsilon);
        rtTrace(top_shadower, shadowRay, prd_shadow);
        seed = prd_shadow.seed;

        if (!prd_shadow.inShadow)
        {
            float scatterPdf = bsdf::Pdf(mat, si, wo_local);
            float3 f = bsdf::Eval(mat, si, wo_local);

            // Delta light
            if(is_light_delta){
                L = Li * f / lightPdf;
            } else {
                float weight = powerHeuristic(lightPdf, scatterPdf);    // MIS
                L =  weight * Li * f / lightPdf;
            }
            // L *= float(num_lights);

            float path_length_em = path_length + lightDist;

            // add transient output
            uint path_length_idx = path_length_to_index(path_length_em);
            uint2 idx = make_uint2(depth, path_length_idx);
            atomicAdd(&transient_radiance_histogram[idx], luminance(throughput * L));
        }

        result += throughput * L;

#endif

        // ----------------------- BSDF sampling ----------------------
        bsdf::Sample(mat, si, seed, bs);
        onb.inverse_transform(bs.wo);
        ray.direction = bs.wo;
        ray.origin = si.p;
        throughput *= bs.weight;

        if(dot(throughput, throughput) == 0){
            break;
        }

        // clear emission & trace again
        si.emission = make_float3(0.0);
        si.seed = seed;
        rtTrace(top_object, ray, si);
        seed = si.seed;
        path_length += si.t;

#if USE_NEXT_EVENT_ESTIMATION
        /* Determine probability of having sampled that same
        direction using emitter sampling. */
        if(si.emission.x > 0)
        {
            LightParameter& light = sysLightParameters[si.light_id];
            float lightPdfArea = pdf_light(si.hitTriIdx, ray.origin, ray.direction, light);
            float light_pdf = (si.t * si.t) / si.wi.z * lightPdfArea * num_lights_inv;
            emission_weight = powerHeuristic(bs.pdf, light_pdf);
        }
#endif
    }

    ppd.result = result;
    ppd.depth = depth;
    ppd.is_valid = si.is_valid;
}

}
