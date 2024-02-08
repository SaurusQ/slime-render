#pragma once

#include "definitions.hpp"
#include <curand_kernel.h>

void kl_updateTrailMap(dim3 grid, dim3 block,
    float4* trailMapFront,
    float4* trailMapBack,
    int* relativeIdxs,
    float diffuseDeltaW,
    float evaporateDeltaW,
    unsigned int padWidth,
    unsigned int padOffset
);

__global__ void k_updateTrailMap(
    float4* trailMapFront,
    float4* trailMapBack,
    int* relativeIdxs,
    float diffuseDeltaW,
    float evaporateDeltaW,
    unsigned int padWidth,
    unsigned int padOffset
);

void kl_updateAgents(dim3 grid, dim3 block,
    curandState* randomState,
    float4* trailMap,
    Agent* agents,
    unsigned int nAgents,
    float speed,
    float turnSpeed,
    float sensorAngleSpacing,
    float sensorOffsetDst,
    unsigned int sensorSize,
    float trailDeltaW,
    unsigned int width,
    unsigned int heigth
);

__global__ void k_updateAgents(
    curandState* randomState,
    float4* trailMap,
    Agent* agents,
    unsigned int nAgents,
    float speed,
    float turnSpeed,
    float sensorAngleSpacing,
    float sensorOffsetDst,
    unsigned int sensorSize,
    float trailDeltaW,
    unsigned int width,
    unsigned int heigth
);

__device__ float sense(
    Agent a,
    float angleOffset,
    float4* trailMap,
    float sensorOffSetDst,
    int sensorSize,
    unsigned int width,
    unsigned int heigth
);

void kl_initCurand32(dim3 grid, dim3 block,
    curandState* state
);

__global__ void k_initCurand32(
    curandState* state
);