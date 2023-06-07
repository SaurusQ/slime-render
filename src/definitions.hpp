#pragma once

#include <cuda_runtime.h>

#include <inttypes.h>

#if 0
constexpr unsigned int W_4K = 3840;
constexpr unsigned int H_4K = 2160;
#elif 1
constexpr unsigned int W_4K = 2560;
constexpr unsigned int H_4K = 1440;
#else
constexpr unsigned int W_4K = 320;
constexpr unsigned int H_4K = 180;
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

struct AgentConfig
{
    float speed;
    float turnSpeed;
    float sensorAngleSpacing;
    float sensorOffsetDst;
    unsigned int sensorSize;
};

