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
    void trailMapToResult();
    void updateTrailMap(double deltaTime, float diffuseWeight, float evaporateWeight);
    void spawnAgents(unsigned int newAgents, float* agentShares, StartFormation startFormation, bool clear);
    void updatePopulationSize(unsigned int newAgents);
    void updatePopulationShare(float* newPopulationShare);
    
    void configAgentParameters(AgentConfig ac) { 
        this->configAgentSpeed(ac.speed); 
        this->configAgentTurnSpeed(ac.turnSpeed); 
        this->configAgentSensorAngleSpacing(ac.sensorAngleSpacing); 
        this->configAgentSensorOffsetDst(ac.sensorOffsetDst);
        this->configAgentSensorSize(ac.sensorSize);
    }
    void configAgentSpeed(float speed) { agentConfig_.speed = speed; }
    void configAgentTurnSpeed(float turnSpeed) { agentConfig_.turnSpeed = turnSpeed * (M_PI / 180.0); }
    void configAgentSensorAngleSpacing(float sensorAngleSpacing) { agentConfig_.sensorAngleSpacing = sensorAngleSpacing * (M_PI / 180.0); }
    void configAgentSensorOffsetDst(float sensorOffsetDst) { agentConfig_.sensorOffsetDst = sensorOffsetDst; }
    void configAgentSensorSize(float sensorSize) { agentConfig_.sensorSize = sensorSize; }

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
    AgentConfig_I agentConfig_ = {
        1.0,
        90.0 * (M_PI / 180.0),
        30.0 * (M_PI / 180.0),
        9.0,
        0
    };
    curandState* agentRandomState_ = nullptr;
    float agentShares_[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    unsigned int nAgents_ = 0;
    unsigned int nAgentsGpuSize_ = 0;
};