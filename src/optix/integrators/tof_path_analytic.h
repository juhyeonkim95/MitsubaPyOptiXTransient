#include "optix/integrators/common.h"

using namespace optix;

// path tracer
rtDeclareVariable(float,         tof_path_w_g_mhz, , );
rtDeclareVariable(float,         tof_path_w_f_mhz, , );
rtDeclareVariable(float,         tof_path_w_g_w_f_difference_hz, , );
rtDeclareVariable(float,         tof_path_exposure_time, , );

rtBuffer<Matrix<4,4>> transform_buffers_start;
rtBuffer<Matrix<4,4>> transform_buffers_end;

namespace tof_path_analytic{

RT_FUNCTION float eval_modulation_integration_weight(float st, float et, float path_length, float path_length_at_t, float f_value_ratio_inc) {
    float temp = (2 * M_PIf * tof_path_w_g_mhz / 300);
    float w_delta = - temp * (path_length_at_t - path_length) / (et - st);
    float phi = temp * path_length;

    float a = (2 * M_PIf * tof_path_w_g_w_f_difference_hz - w_delta);
    float b = phi;
    float c = f_value_ratio_inc / (et - st);

    bool force_constant_attenuation = false;

    float s1 = 0.5;

    if(fabsf(a) < 1){
        float b_cos = cosf(b);
        float BT = b_cos * et;
        float B0 = b_cos * st;
        if(!force_constant_attenuation){
            BT += ( 0.5 * c * et * et * b_cos);
            B0 += ( 0.5 * c * st * st * b_cos);
        }
        return s1 / 2 * (BT - B0);
    } else {
        float AT = sinf(a * et + b) / a;
        float A0 = sinf(a * st + b) / a;

        if(!force_constant_attenuation){
            AT += ( c * et * sinf(a * et + b) / a + c * cosf(a * et + b) / (a * a));
            A0 += ( c * st * sinf(a * st + b) / a + c * cosf(a * st + b) / (a * a));
        }
        return s1 / 2 * (AT - A0);
    }
}


RT_FUNCTION void path_trace(Ray& ray, unsigned int& seed, PerPathData &ppd)
{
    float emission_weight = 1.0;
    float3 throughput = make_float3(1.0);
    float3 result = make_float3(0.0);
    BSDFSample3f bs;
    float path_length = 0;
    float path_length_at_T = 0;
    float f_value_ratio = 1;

    // ---------------------- First intersection ----------------------
    SurfaceInteraction si;
    si.emission = make_float3(0);
    si.is_valid = true;

    rtTrace(top_object, ray, 0, si);
    path_length += si.t;

    SurfaceInteraction si_T;
    rtTrace(top_object, ray, tof_path_exposure_time, si_T);
    path_length_at_T += si_T.t;

    if ((path_length - path_length_at_T) > 0.1){
        //ppd.result = make_float3(5,0,0);
        // return;
    }

    SurfaceInteraction prev_si;
    SurfaceInteraction prev_si_T;

#if USE_NEXT_EVENT_ESTIMATION
    int num_lights = sysLightParameters.size();
    float num_lights_inv = 1.0 / (float) (num_lights);
    LightSample lightSample;
    PerRayData_pathtrace_shadow prd_shadow;
#endif
    int depth;
    for (depth = 1; ; depth++){
        // ---------------- Intersection with emitters ----------------
        // eval_modulation_integration_weight(0, tof_path_exposure_time, path_length, path_length_at_T, f_value_ratio - 1);
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
        rtTrace(top_shadower, shadowRay, 0, prd_shadow);
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

            float3 em_p_at_T = lightSample.p;

            float em_path_length = path_length + lightDist;
            float em_path_length_at_T = path_length_at_T + length(em_p_at_T - si_T.p);

            float dist_sqr_1 = lightDist * lightDist;
            float cos_i_1 = dot(si.normal, lightSample.wi);
            float cos_o_1 = is_light_delta ? 1 : dot(lightSample.n, -lightSample.wi);
            float f_1 = fabsf(cos_i_1) * fabsf(cos_o_1) / dist_sqr_1;

            float dist_sqr_2 = dot(em_p_at_T - si_T.p, em_p_at_T - si_T.p);
            float cos_i_2 = dot(si_T.normal, normalize(em_p_at_T - si_T.p));
            float cos_o_2 = is_light_delta ? 1 : dot(lightSample.n, -normalize(em_p_at_T - si_T.p));
            float f_2 = fabsf(cos_i_2) * fabsf(cos_o_2) / dist_sqr_2;
            float f_value_ratio_em = f_value_ratio * f_2 / f_1;
            float path_length_weight = eval_modulation_integration_weight(0, tof_path_exposure_time, em_path_length, em_path_length_at_T, (f_value_ratio_em - 1));

            float v = em_path_length_at_T - em_path_length;
            ppd.result = make_float3(cos_i_1);
            return;

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
        prev_si.p = si.p;
        prev_si.normal = si.normal;

        prev_si_T.p = si_T.p;
        prev_si_T.normal = si_T.normal;

        // clear emission & trace again
        si.emission = make_float3(0.0);
        si.seed = seed;
        rtTrace(top_object, ray, 0, si);
        seed = si.seed;

        path_length += si.t;
        // TODO : change to velocity
        si_T.p = si.p + si.velocity * tof_path_exposure_time;
        si_T.normal = si.normal;
        path_length_at_T += length(si_T.p - prev_si_T.p);

        float dist_sqr_1 = si.t * si.t;
        float cos_i_1 = dot(prev_si.normal, ray.direction);
        float cos_o_1 = dot(si.normal, -ray.direction);
        float f_1 = cos_i_1 * cos_o_1 / dist_sqr_1;

        float dist_sqr_2 = dot(si_T.p - prev_si_T.p, si_T.p - prev_si_T.p);
        float cos_i_2 = dot(prev_si_T.normal, normalize((si_T.p - prev_si_T.p)));
        float cos_o_2 = dot(si_T.normal, -normalize((si_T.p - prev_si_T.p)));
        float f_2 = cos_i_2 * cos_o_2 / dist_sqr_2;

        f_value_ratio = f_value_ratio * f_2 / f_1;


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
