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

#define DIFFERENT_SPECIES 4

constexpr const char* agentNames[] = {"X", "Y", "Z", "W"};

struct RGBA
{
    float r;
    float g;
    float b;
    float a;
};

struct RGB
{
    float r;
    float g;
    float b;
};

typedef RGB AgentColor;

struct Agent
{
    float2 pos;
    float angle;
    int4 speciesMask;
    int speciesIdx;
};

struct AgentConfig
{
    float speed;
    float turnSpeed;
    float sensorAngleSpacing;
    float sensorOffsetDst;
    unsigned int sensorSize;
};

typedef AgentConfig AgentConfig_I;
typedef AgentConfig AgentConfig_GPU;
typedef AgentColor AgentColors_I;
typedef AgentColor AgentColors_GPU;

enum class StartFormation {
    CONFIGURED,
    RANDOM,
    MIDDLE,
    CIRCLE,
    RANDOM_CIRCLE
};

struct SimConfig
{
    AgentConfig aConfigs[DIFFERENT_SPECIES];
    AgentColor aColors[DIFFERENT_SPECIES];
    float agentShare[DIFFERENT_SPECIES];
    int numAgents;
    float evaporate;
    float diffuse;
    float trailWeight;
    bool updateAgents;
    bool clearOnSpawn;
    StartFormation startFormation;
};

struct SimUpdate
{
    bool agentSettings;
    bool spawn;
    bool population;
    bool populationShare;
    bool clearImg;
};
