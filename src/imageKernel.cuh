#pragma once

#include "image.hpp"
#include "definitions.hpp"

#include <cuda_runtime.h>
#include <vector>

class ImageKernel
{
public:
    ImageKernel(const Image& img, unsigned int padding);
    ~ImageKernel();
    void update(const Image& img);
    void readBack(const Image& img) const;
    // Kernel starters
    void convolution(unsigned int kernelSize, const std::vector<float>& kernel);
private:
    bool checkCudaError(cudaError_t cs, std::string msg) const;
    // Kernel starters
    void imgToPadded();

    RGB* imageGPUptr_;
    RGB* imageGPUpaddedPtr_;
    unsigned int width_;
    unsigned int height_;
    unsigned int bufferSize_;
    unsigned int bufferSizePadded_;
    unsigned int padding_;
};