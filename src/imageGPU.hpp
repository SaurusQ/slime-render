#pragma once

#include "image.hpp"
#include "definitions.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glad/glad.h>

#include <vector>
#include <unordered_map> 

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
    void configAgentParameters(float speed);
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
    curandState* agentRandomState_ = nullptr;
    unsigned int nAgents_ = 0;
    float agentSpeed_ = 1.0;
};