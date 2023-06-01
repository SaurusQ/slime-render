#pragma once

#include <cuda_runtime.h>

#include <inttypes.h>

#if 1
constexpr unsigned int W_4K = 3840;
constexpr unsigned int H_4K = 2160;
#else
constexpr unsigned int W_4K = 1000;
constexpr unsigned int H_4K = 1000;
#endif
struct RGB
{
    float r;
    float g;
    float b;
};

struct Agent
{
    float2 pos;
    float angle;
};
