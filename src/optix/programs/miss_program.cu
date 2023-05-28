#include <optixu/optixu_math_namespace.h>
#include "optix/common/prd_struct.h"
#include "optix/light/light_parameters.h"
#include "optix/common/helpers.h"

using namespace optix;

rtDeclareVariable(SurfaceInteraction, si, rtPayload, );
rtDeclareVariable(float3, bg_color, , );
rtBuffer<LightParameter> sysLightParameters;
rtDeclareVariable(Ray, ray, rtCurrentRay, );

RT_PROGRAM void miss()
{
    si.is_valid = false;
}

RT_PROGRAM void miss_environment_mapping()
{
    si.is_valid = false;
    LightParameter& light = sysLightParameters[0];

    float3 ray_direction = transform_normal(light.transformation.transpose(), ray.direction);

    float phi = atan2f(ray_direction.x, -ray_direction.z);
    float theta = acosf(-ray_direction.y);
    float u = (phi + M_PIf) * (0.5f * M_1_PIf);
    float v = theta * M_1_PIf;

    si.emission = make_float3(optix::rtTex2D<float4>(light.envmapID , u, v));
}