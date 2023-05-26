#include "imageKernel.cuh"

#include <iostream>
#include <stdexcept>

ImageKernel::ImageKernel(const Image& img, unsigned int padding)
{
    width_              = img.getWidth();
    height_             = img.getHeight();
    bufferSize_         = img.getBufferSize();
    bufferSizePadded_   = img.getPaddedBufferSize();

    this->checkCudaError(
        cudaMalloc((void**)&imageGPUptr_, bufferSize_),
        "cudaMalloc image buffer"
    );
    this->checkCudaError(
        cudaMalloc((void**)&imageGPUpaddedPtr_, bufferSizePadded_),
        "cudaMallow padded image buffer"
    );
    this->checkCudaError(
        cudaMemset(imageGPUpaddedPtr_, 0, bufferSizePadded_),
        "cudaMemset"
    );
    this->update(img);
}

ImageKernel::~ImageKernel()
{
    cudaFree(imageGPUptr_);
    cudaFree(imageGPUpaddedPtr_);
}

void ImageKernel::update(const Image& img)
{
    if(img.getBufferSize() != bufferSize_)
    {
        // TODO allow different buffer sizes by reallocatig memory
        std::cout << "Different buffer sizes " << bufferSize_ << " : " << img.getBufferSize() << std::endl;
        return;
    }
    this->checkCudaError(
        cudaMemcpy(imageGPUptr_, img.getPtr(), bufferSize_, cudaMemcpyHostToDevice),
        "cudaMemcpy"
    );
}

void ImageKernel::readBack(const Image& img) const
{
    this->checkCudaError(
        cudaMemcpy(img.getPtr(), imageGPUptr_, bufferSize_, cudaMemcpyDeviceToHost),
        "cudaMemcpy"
    );
}

bool ImageKernel::checkCudaError(cudaStatus cs, std::string msg) const
{
    bool failure = cs != cudaSuccess;
    if(failure)
    {
        msg = std::string("FAIL: ") + msg + std::string(" WHAT: ") + std::string(cudaGetErrorString(cudaStatus));
        throw std::runtime_error(msg);
    }
    return failure;
}

void ImageKernel::imgToPadded()
{
    <<<TODO>>>k_imgToPadded(imageGPUptr_, imageGPUpaddedPtr_);
}

__global__ ImageKernel::k_imgToPadded(RGB* imgPtr, RGB* imgPadPtr, unsigned int padding, unsigned int width, unsigned int height)
{

}

void ImageKernel::convolution(unsigned int kernelSize, const std::vector<float>& kernel)
{
    unsigned int kernelValues = (kernelSize * 2 + 1) * (kernelSize * 2 + 1);
    if(kernelSize > padding) 
    {
        std::cerr << "Too large kernel size for padding. Kernel: " << kernelSize << " Padding: " << padding << std::endl;
        return; 
    }
    if(kernel.size() < kernelValues)
    {
        std::cerr << "Not enough variables on kernel" << std::endl;
        return;
    }
    this->imgToPadded();
    std::vector<int> relativeIdxs;
    for (int y = -kernelSize; y <= kernelSize; y++)
    {
        for (int x = -kernelSize; x <= kernelSize; x++)
        {
            relativeIdxs.append(x + y * width_);
        }
    }
    int* relativeIdxsGPUptr = nullptr;
    float* kernelGPUptr = nullptr;
    this->checkCudaError(
        cudaMalloc((void**)&relativeIdxsGPUptr, kernelValues * sizeof(int)),
        "cudaMalloc relativeIdxsGPUptr"
    );
    this->checkCudaError(
        cudaMalloc((void**)&kernelGPUptr, kernelValues * sizeof(float)),
        "cudaMalloc kernelGPUptr"
    );
    <<<TODO>>>k_convolution(imageGPUptr_, imageGPUpaddedPtr_, relativeIdxsGPUptr, kernelGPUptr, kernelValues);
}

__global__ ImageKernel::k_convolution(RGB* img, RGB* imgPadded, int* relativeIdxs, float* kernel, unsigned int kernelValues, unsigned int width, unsigned int height)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    int idx = x + y * width; 

    float valueR = 0;
    float valueG = 0;
    float valueB = 0;

    for (int i = 0; i < kernelValues; i++)
    {
        valueR += imgPadded[relativeIdxs[i]].r * kernel[i];
        valueG += imgPadded[relativeIdxs[i]].g * kernel[i];
        valueB += imgPadded[relativeIdxs[i]].b * kernel[i];
    }    

    img[idx].r = valueR / kernelValues;
    img[idx].g = valueG / kernelValues;
    img[idx].b = valueb / kernelValues;
    
}