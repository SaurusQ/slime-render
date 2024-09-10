#pragma once

#include <cuda_runtime.h>

#include <inttypes.h>

#if 1
constexpr unsigned int IMG_W_CONFIG = 3840;
constexpr unsigned int IMG_H_CONFIG = 2160;
#elif 0
constexpr unsigned int IMG_W_CONFIG = 2560;
constexpr unsigned int IMG_H_CONFIG = 1440;
#else
constexpr unsigned int IMG_W_CONFIG = 320;
constexpr unsigned int IMG_H_CONFIG = 180;
#endif

#define BLOCK_SIZE 32
#define BLOCK_SIZE_F static_cast<float>(BLOCK_SIZE)

#define BLOCK_SIZE_AGENT 256
#define BLOCK_SIZE_AGENT_F static_cast<float>(BLOCK_SIZE_AGENT)

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
