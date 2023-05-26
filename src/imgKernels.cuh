#pragma once

#include "definitions.hpp"

__global__ void k_imgToPadded(
    RGB* imgPtr,
    RGB* imgPadPtr,
    unsigned int padding,
    unsigned int width,
    unsigned int height
);

__global__ void k_convolution(
    RGB* imgPtr,
    RGB* imgPadPtr,
    int* relativeIdxs,
    float* kernel,
    unsigned int kernelValues,
    unsigned int width
);
