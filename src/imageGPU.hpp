#pragma once

#include "image.hpp"
#include "definitions.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glad/gl.h>

#include <vector>
#include <unordered_map>
#include <cmath>

class ImageGPU
{
public:
    ImageGPU(const Image& img, unsigned int padding);
    ~ImageGPU();
    void activateCuda();
    void deactivateCuda();
    void update(const Image& img);
    void readBack(const Image& img) const;
    GLuint getTexture() const { return texture_; }
    GLuint getPbo() const { return pbo_; }
    // GPU mods
    void addConvKernel(unsigned int kernelId, std::vector<float> kernel);
    void convolution(unsigned int kernelSize, unsigned int kernelId);
    void evaporate(float strength);
    void configAgents(unsigned int num);
    
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
    
    void updateAgents();
private:
    void loadTexture();
    bool checkCudaError(cudaError_t cs, std::string msg) const;
    void imgToPadded();

    cudaGraphicsResource_t cudaPboResource_ = nullptr;
    RGB* imgCudaArray_ = nullptr;
    RGB* imgPadCudaArray_ = nullptr;
    GLuint texture_ = 0;
    GLuint pbo_ = 0;

    bool cudaActive_ = false;

    unsigned int width_;
    unsigned int height_;
    unsigned int bufferSize_;
    unsigned int bufferSizePadded_;
    unsigned int padding_;
    unsigned int padWidth_;

    // Convolution
    std::unordered_map<int, int*> convRelIdxsGPUptrs_;
    std::unordered_map<int, float*> convKernelGPUptrs_;

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
    unsigned int nAgents_ = 0;
    float agentSpeed_ = 1.0;
};