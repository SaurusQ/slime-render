#pragma once

#include "image.hpp"
#include "definitions.hpp"

#include <cuda_runtime.h>

class ImageKernel
{
public:
    ImageKernel(const Image& img);
    ~ImageKernel();
    void update(const Image& img);
private:
    RGB* imageGPUptr_;
    unsigned int bufferSize_;
};