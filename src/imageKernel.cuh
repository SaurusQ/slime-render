#pragma once

#include "image.hpp"
#include "definitions.hpp"

#include <cuda_runtime.h>

class ImageKernel
{
public:
    ImageKernel(const Image& img, unsigned int padding);
    ~ImageKernel();
    void update(const Image& img);
    void readBack(const Image& img) const;
    // Kernel starters
    void imgToPadded();
    void convolution(const std::vector<float>& kernel, unsigned int kernelSize);
private:
    bool checkCudaError(cudaStatus cs, std::string msg) const;
    // Kernels
    __global__ k_imgToPadded(RGB* imgPtr, RGB* imgPadPtr, unsigned int padding = padding_, unsigned int width = width_, unsigned int height = height_);
    __global__ k_convolution(RGB* imgPtr, int* relativeIdxs, float* kernel, unsigned int kernelValues, unsigned int width = width_, unsigned int height = height_);
    RGB* imageGPUptr_;
    RGB* imageGPUpaddedPtr_;
    unsigned int width_;
    unsigned int height_;
    unsigned int bufferSize_;
    unsigned int bufferSizePadded_;
    unsigned int padding_;
};