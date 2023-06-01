#pragma once

#include "definitions.hpp"
#include <curand_kernel.h>

void kl_convolution(dim3 grid, dim3 block,
    RGB* imgPtr,
    RGB* imgPadPtr,
    int* relativeIdxs,
    float* kernel,
    unsigned int kernelValues,
    unsigned int width,
    unsigned int padWidth,
    unsigned int padding,
    unsigned int padOffset
);

__global__ void k_convolution(
    RGB* imgPtr,
    RGB* imgPadPtr,
    int* relativeIdxs,
    float* kernel,
    unsigned int kernelValues,
    unsigned int width,
    unsigned int padWidth,
    unsigned int padding,
    unsigned int padOffset
);

void kl_evaporate(dim3 grid, dim3 block, RGB* imgPtr, float strength, unsigned int width);

__global__ void k_evaporate(RGB* imgPtr, float strength, unsigned int width);

void kl_updateAgents(dim3 grid, dim3 block,
    curandState* randomState,
    RGB* imgPtr,
    Agent* agents,
    unsigned int nAgents,
    float speed,
    unsigned int width,
    unsigned int heigth
);

__global__ void k_updateAgents(
    curandState* randomState,
    RGB* imgPtr,
    Agent* agents,
    unsigned int nAgents,
    float speed,
    unsigned int width,
    unsigned int heigth
);

__device__ float sense(Agent a, float angleOffset, RGB* imgPtr, unsigned int width, unsigned int heigth);

void kl_initCurand32(dim3 grid, dim3 block,
    curandState* state
);

__global__ void k_initCurand32(
    curandState* state
);