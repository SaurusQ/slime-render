#pragma once

#include "image.hpp"
#include "definitions.hpp"

#include <cuda_runtime.h>
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
    // Kernel starters
    void convolution(unsigned int kernelSize, const std::vector<float>& kernel);
private:
    void loadTexture();
    bool checkCudaError(cudaError_t cs, std::string msg) const;
    // Kernel starters
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
    std::unordered_map<int, std::vector<int>> convRelIdxMap;
};