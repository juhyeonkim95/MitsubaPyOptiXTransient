/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optixu/optixu_math_namespace.h>
#include "optix/common/prd_struct.h"
#include "optix/common/helpers.h"

using namespace optix;

rtDeclareVariable( float3, shading_normal, attribute shading_normal, );
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );

rtDeclareVariable( float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable( float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable( float3, texcoord, attribute texcoord, );


rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(SurfaceInteraction, si, rtPayload, );

rtBuffer<MaterialParameter> sysMaterialParameters;
rtDeclareVariable(int, materialId, , );

RT_PROGRAM void closest_hit()
{
    // Transform normal from object to world coordinate
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal ) );

    // Face forwarding normal (ffnormal dot ray_direction > 0)
    float3 ff_normal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

    // Material parameter
	MaterialParameter& mat = sysMaterialParameters[materialId];
    float3 normal = mat.isTwosided? ff_normal : world_shading_normal;

    si.material_id = materialId;
    si.normal = normal;
    si.uv = texcoord;

    float3 wi = -ray.direction;
    optix::Onb onb( normal );

    si.wi = to_local(onb, wi);
    si.p = ray.origin + t_hit * ray.direction;
    si.t = t_hit;
}