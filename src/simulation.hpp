#pragma once

#include "image.hpp"
#include "definitions.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glad/glad.h>

#include <vector>
#include <unordered_map>
#include <cmath>

class Simulation
{
public:
    Simulation(const Image& img);
    ~Simulation();

    void activateCuda();
    void deactivateCuda();

    void update(const Image& img);
    void readBack(const Image& img) const;
    void clearImage();
    void swapBuffers();

    GLuint getTexture() const { return texture_; }
    GLuint getPbo() const { return pbo_; }

    // GPU mods
    void trailMapToDisplay();
    void updateTrailMap(double deltaTime, float diffuseWeight, float evaporateWeight);
    void spawnAgents(unsigned int newAgents, float* agentShares, StartFormation startFormation, bool clear);
    void updatePopulationSize(unsigned int newAgents);
    void updatePopulationShare(float* newPopulationShare);
    
    void configAgentParameters(AgentConfig* aConfigs, AgentColor* aColors);
    void configAgentSpeed(int idx, float speed) { agentConfig_[idx].speed = speed; }
    void configAgentTurnSpeed(int idx, float turnSpeed) { agentConfig_[idx].turnSpeed = turnSpeed * (M_PI / 180.0); }
    void configAgentSensorAngleSpacing(int idx, float sensorAngleSpacing) { agentConfig_[idx].sensorAngleSpacing = sensorAngleSpacing * (M_PI / 180.0); }
    void configAgentSensorOffsetDst(int idx, float sensorOffsetDst) { agentConfig_[idx].sensorOffsetDst = sensorOffsetDst; }
    void configAgentSensorSize(int idx, float sensorSize) { agentConfig_[idx].sensorSize = sensorSize; }

    void updateAgentConfig();

    void updateAgents(double deltaTime, float trailWeight);
private:
    void updateSpecies();
    std::vector<unsigned int> getPopulationCount() const;
    void loadTexture();
    bool checkCudaError(cudaError_t cs, std::string msg) const;

    cudaGraphicsResource_t cudaPboResource_ = nullptr;
    RGBA* resultCudaImg_ = nullptr;
    RGBA* trailMapFront_ = nullptr;
    RGBA* trailMapBack_ = nullptr;
    GLuint texture_ = 0;
    GLuint pbo_ = 0;

    bool cudaActive_ = false;

    unsigned int width_;
    unsigned int height_;
    unsigned int bufferSize_;
    unsigned int bufferSizePadded_;
    unsigned int padding_;
    unsigned int padWidth_;
    unsigned int padOffset_;

    // Trail map
    int* relativeIdxsGPUptr_ = nullptr;

    // Agent
    Agent* agents_ = nullptr;
    AgentColors_I agentColors_[DIFFERENT_SPECIES];
    AgentConfig_I agentConfig_[DIFFERENT_SPECIES];
    AgentConfig_GPU* agentConfigGPU_ = nullptr;
    AgentColors_GPU* agentColorsGPU_ = nullptr;
    curandState* agentRandomState_ = nullptr;
    float agentShares_[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    unsigned int nAgents_ = 0;
    unsigned int nAgentsGpuSize_ = 0;

    dim3 gridI_;
    dim3 block_ = dim3(BLOCK_SIZE, BLOCK_SIZE);
};