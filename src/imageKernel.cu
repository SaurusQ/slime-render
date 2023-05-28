#include "imageKernel.cuh"
#include "imgKernels.cuh"

#include <cuda_gl_interop.h>

#include <iostream>
#include <stdexcept>

ImageKernel::ImageKernel(const Image& img, unsigned int padding)
{
    width_              = img.getWidth();
    height_             = img.getHeight();
    bufferSize_         = img.getBufferSize();
    bufferSizePadded_   = img.getPaddedBufferSize(padding);

    this->loadTexture();
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
    cudaGraphicsUnregisterResource(*cudaTextureResource_);
    glDeleteTextures(1, &texture_);
}

void ImageKernel::activateCuda()
{
    if (cudaTextureResource_ != nullptr)
    {
        cudaGraphicsMapResources(1, cudaTextureResource_);
    }
}

void ImageKernel::deactivateCuda()
{
    if (cudaTextureResource_ != nullptr)
    {
        cudaGraphicsUnmapResources(1, cudaTextureResource_);
    }
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

void ImageKernel::loadTexture()
{
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Create texture data (4-component unsigned byte)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width_, height_, 0, GL_RGB, GL_FLOAT, NULL);
    // Unbind the texture
    glBindTexture(GL_TEXTURE_2D, texture_);

    // Register the texture with cuda
    this->checkCudaError(
        cudaGraphicsGLRegisterImage((cudaGraphicsResource**)&cudaTextureResource_, texture_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard),
        "cudaGraphicsGLRegisterImage"
    );
    // cuda device pointer from cuda graphics resource
    this->checkCudaError(
        cudaGraphicsMapResources(1, cudaTextureResource_),
        "cudaGraphicsMapResources"
    );

    size_t cudaSize;
    this->checkCudaError(
        cudaGraphicsResourceGetMappedPointer((void**)&imageGPUptr_, &cudaSize, *cudaTextureResource_),
        "cudaGraphicsSubResourceGetMappedArray"
    );

    std::cout << "Got cuda size: " << cudaSize << std::endl;
    std::cout << "Actual cuda size: " << bufferSize_ << std::endl;
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
    //k_imgToPadded<<<TODO>>>(imageGPUptr_, imageGPUpaddedPtr_);
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
    k_convolution<<<height_, width_>>>(imageGPUptr_, imageGPUpaddedPtr_, relativeIdxsGPUptr, kernelGPUptr, kernelValues, width_);
    cudaFree(relativeIdxsGPUptr);
    cudaFree(kernelGPUptr);
}
