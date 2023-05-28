#include "optix/integrators/common.h"

using namespace optix;

// path tracer
rtDeclareVariable(float,         tof_path_w_g_mhz, , );
rtDeclareVariable(float,         tof_path_w_s_mhz, , );
rtDeclareVariable(float,         tof_path_w_g_w_s_difference_hz, , );

namespace tof_path{

RT_FUNCTION float eval_modulation_weight(float ray_time, float path_length) {
    //const float m_illumination_modulation_frequency_mhz = 30;
    //const float m_sensor_modulation_frequency_mhz = 30;
    //float w_g = 2 * M_PIf * m_illumination_modulation_frequency_mhz * 1e6;
    //float w_f = 2 * M_PIf * m_sensor_modulation_frequency_mhz * 1e6;
    //float w_delta = 2 * M_PIf * 1.0 / tof_path_exposure_time;
    float phi = (2 * M_PIf * tof_path_w_g_mhz) / 300 * path_length;
    float fg_t = 0.25 * cosf(2 * M_PIf * tof_path_w_g_w_s_difference_hz * ray_time + phi);
    return fg_t;
}

RT_FUNCTION void path_trace(Ray& ray, unsigned int& seed, PerPathData &ppd)
{
    float emission_weight = 1.0;
    float3 throughput = make_float3(1.0);
    float3 result = make_float3(0.0);
    BSDFSample3f bs;
    float path_length = 0;

    // ---------------------- First intersection ----------------------
    SurfaceInteraction si;
    si.emission = make_float3(0);
    si.is_valid = true;

    const float current_time = rnd(seed) * 0.0015;
    rtTrace(top_object, ray, current_time, si);
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
        rtTrace(top_shadower, shadowRay, current_time, prd_shadow);
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
            float em_path_length = path_length + lightDist;
            float path_length_weight = eval_modulation_weight(current_time, em_path_length);
            L *= path_length_weight;
            // L *= float(num_lights);
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
        rtTrace(top_object, ray, current_time, si);
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
