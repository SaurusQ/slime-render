#include "imageKernel.cuh"
#include "imgKernels.cuh"

#include <iostream>
#include <stdexcept>

ImageKernel::ImageKernel(const Image& img, unsigned int padding)
{
    width_              = img.getWidth();
    height_             = img.getHeight();
    bufferSize_         = img.getBufferSize();
    bufferSizePadded_   = img.getPaddedBufferSize(padding);

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
        cudaMemcpy((void*)imageGPUptr_, (void*)img.getPtr(), bufferSize_, cudaMemcpyHostToDevice),
        "cudaMemcpy"
    );
}

void ImageKernel::readBack(const Image& img) const
{
    this->checkCudaError(
        cudaMemcpy((void*)img.getPtr(), (void*)imageGPUptr_, bufferSize_, cudaMemcpyDeviceToHost),
        "cudaMemcpy"
    );
}

bool ImageKernel::checkCudaError(cudaError_t ce, std::string msg) const
{
    bool failure = ce != cudaSuccess;
    if(failure)
    {
        msg = std::string("FAIL: ") + msg + std::string(" WHAT: ") + std::string(cudaGetErrorString(ce));
        throw std::runtime_error(msg);
    }
    return failure;
}

void ImageKernel::imgToPadded()
{
    <<<TODO>>>k_imgToPadded(imageGPUptr_, imageGPUpaddedPtr_);
}

void ImageKernel::convolution(unsigned int kernelSize, const std::vector<float>& kernel)
{
    unsigned int kernelValues = (kernelSize * 2 + 1) * (kernelSize * 2 + 1);
    if(kernelSize > padding_) 
    {
        std::cerr << "Too large kernel size for padding. Kernel: " << kernelSize << " Padding: " << padding_ << std::endl;
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
            relativeIdxs.push_back(x + y * width_);
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
    <<<height_, width_>>>k_convolution(imageGPUptr_, imageGPUpaddedPtr_, relativeIdxsGPUptr, kernelGPUptr, kernelValues, width_);
    cudaFree(relativeIdxsGPUptr);
    cudaFree(kernelGPUptr);
}
