#include "imageKernel.cuh"

#include <iostream>

ImageKernel::ImageKernel(const Image& img)
{
    bufferSize_ = img.getBufferSize();
    cudaError_t cudaStatus = cudaMalloc((void**)&imageGPUptr_, bufferSize_);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
    this->update(img);
}

ImageKernel::~ImageKernel()
{
    cudaFree(imageGPUptr_);
}

void ImageKernel::update(const Image& img)
{
    if(img.getBufferSize() != bufferSize_)
    {
        std::cout << "Different buffer sizes " << bufferSize_ << " : " << img.getBufferSize() << std::endl;
        return;
    }
    cudaError_t cudaStatus = cudaMemcpy(imageGPUptr_, img.getPtr(), bufferSize_, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
}