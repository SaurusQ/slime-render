#include "imageGPU.hpp"
#include "cudaKernels.cuh"

#include <cuda_gl_interop.h>

#include <iostream>
#include <stdexcept>

#define REQUIRE_CUDA if(!cudaActive_) { std::cout << "cuda not active" << std::endl; return; };

ImageGPU::ImageGPU(const Image& img, unsigned int padding)
    : padding_(padding)
{
    width_              = img.getWidth();
    height_             = img.getHeigth();
    bufferSize_         = img.getBufferSize();
    bufferSizePadded_   = img.getPaddedBufferSize(padding_);
    padWidth_           = 2 * padding_ + width_;

    //const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    /*this->checkCudaError(
        cudaMallocArray(&imgPadCudaArray_, &channelDesc, width_ + padding_ * 2, height_ + padding_ * 2),
        "cudaMallocArray for imgPadCudaArray_"
    );*/
    this->checkCudaError(
        cudaMalloc((void**)&imgPadCudaArray_, bufferSizePadded_),
        "cudaMalloc image padded"
    );

    this->checkCudaError(
        cudaMemset((void*)imgPadCudaArray_, 0, bufferSizePadded_),
        "cudaMemset image padded"
    );
    this->loadTexture();
    this->activateCuda();
    this->update(img);
    this->deactivateCuda();
}

ImageGPU::~ImageGPU()
{
    cudaFree((void*)imgPadCudaArray_);
    cudaGraphicsUnregisterResource(cudaPboResource_);
    glDeleteTextures(1, &texture_);
}

void ImageGPU::activateCuda()
{
    if (cudaPboResource_ != nullptr)
    {
        cudaActive_ = true;
        this->checkCudaError(
            cudaGraphicsMapResources(1, &cudaPboResource_, 0),
            "cudaGraphicsMapResources"
        );

        size_t cudaPboSize;
        this->checkCudaError(
            //cudaGraphicsResourceGetMappedPointer((void**)&imageGPUptr_, &cudaSize, cudaTextureResource_),
            cudaGraphicsResourceGetMappedPointer((void**)&imgCudaArray_, &cudaPboSize, cudaPboResource_),
            "cudaGraphicsSubResourceGetMappedArray"
        );

        if (cudaPboSize != bufferSize_)
        {
            std::cerr << "Something wrong with buffer sizes: pbo: " << cudaPboSize << " buffer: " << bufferSize_ << std::endl;
        }
    }
}

void ImageGPU::deactivateCuda()
{
    cudaActive_ = false;
    if (cudaPboResource_ != nullptr)
    {
        cudaGraphicsUnmapResources(1, &cudaPboResource_, 0);
    }
    cudaDeviceSynchronize();
}

void ImageGPU::update(const Image& img)
{
    REQUIRE_CUDA
    if(img.getBufferSize() != bufferSize_)
    {
        // TODO allow different buffer sizes by reallocatig memory
        std::cout << "Different buffer sizes " << bufferSize_ << " : " << img.getBufferSize() << std::endl;
        return;
    }
    this->checkCudaError(
        //cudaMemcpy2DToArray(imgCudaArray_, 0, 0, (void*)img.getPtr(), width_ * sizeof(RGB), width_ * sizeof(RGB), height_, cudaMemcpyHostToDevice),
        // ((cudaArray_t)imageGPUptr_, 0, 0, (void*)img.getPtr(), bufferSize_, cudaMemcpyHostToDevice),
        cudaMemcpy((void*)imgCudaArray_, (void*)img.getPtr(), bufferSize_, cudaMemcpyHostToDevice),
        "cudaMemcpy update()"
    );
    cudaDeviceSynchronize();
}

void ImageGPU::readBack(const Image& img) const
{
    REQUIRE_CUDA
    /*this->checkCudaError(
        cudaMemcpy((void*)img.getPtr(), (void*)imageGPUptr_, bufferSize_, cudaMemcpyDeviceToHost),
        "cudaMemcpy readback()"
    );*/
}

void ImageGPU::loadTexture()
{
    // Pixel buffer object
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize_, nullptr, GL_STREAM_DRAW);

    // Texture
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


    // Create texture data (4-component unsigned byte)
    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width_, height_, 0, GL_RGB, GL_FLOAT, NULL);
    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    // Unbind the texture
    //glBindTexture(GL_TEXTURE_2D, texture_);

    // cuda device pointer from cuda graphics resource
    
    this->checkCudaError(
        cudaGraphicsGLRegisterBuffer(&cudaPboResource_, pbo_, cudaGraphicsMapFlagsWriteDiscard),
        "cudaGraphicsGLRegisterBUffer"
    );
    cudaDeviceSynchronize();
}

bool ImageGPU::checkCudaError(cudaError_t ce, std::string msg) const
{
    bool failure = ce != cudaSuccess;
    if(failure)
    {
        msg = std::string("FAIL: ") + msg + std::string(" WHAT: ") + std::string(cudaGetErrorString(ce));
        throw std::runtime_error(msg);
    }
    return failure;
}

void ImageGPU::imgToPadded()
{
    REQUIRE_CUDA
    /*
    this->checkCudaError(
        cudaMemcpy2DArrayToArray(imgPadCudaArray_, padding_, padding_, imgCudaArray_, 0, 0, width_, height_),
        "cudaMemcpy2DArrayToArray imgToPadded()"
    );*/
    this->checkCudaError(
        cudaMemcpy2D((void*)(imgPadCudaArray_ + padding_ + padWidth_ * padding_), padWidth_ * sizeof(RGB), (void*)imgCudaArray_, width_ * sizeof(RGB), width_ * sizeof(RGB), height_, cudaMemcpyDeviceToDevice),
        "cudaMemcpy imgToPadded()"
    );
}

void ImageGPU::convolution(unsigned int kernelSize, const std::vector<float>& kernel)
{
    REQUIRE_CUDA
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
    
    int* relativeIdxs;
    auto it = convRelIdxMap.find(kernelSize);
    if (it == convRelIdxMap.end())
    {
        // Generate new relative idxs
        std::vector<int> newRelIdx;
        int k = kernelSize;
        for (int y = -k; y <= k; y++)
        {
            for (int x = -k; x <= k; x++)
            {
                newRelIdx.push_back(x + y * static_cast<int>(padWidth_));
            }
        }
        convRelIdxMap[kernelSize] = newRelIdx;
        relativeIdxs = convRelIdxMap[kernelSize].data();
    }
    else
    {
        relativeIdxs = it->second.data();
    }

    int* relativeIdxsGPUptr = nullptr;
    float* kernelGPUptr = nullptr;
    this->checkCudaError(
        cudaMalloc((void**)&relativeIdxsGPUptr, kernelValues * sizeof(int)),
        "cudaMalloc relativeIdxsGPUptr"
    );
    this->checkCudaError(
        cudaMemcpy((void*)relativeIdxsGPUptr, relativeIdxs, kernelValues * sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy relativeIdxs" 
    );
    this->checkCudaError(
        cudaMalloc((void**)&kernelGPUptr, kernelValues * sizeof(float)),
        "cudaMalloc kernelGPUptr"
    );
    this->checkCudaError(
        cudaMemcpy((void*)kernelGPUptr, kernel.data(), kernelValues * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy kernel" 
    );

    dim3 dimGrid(width_ / 32, height_ / 32);
    dim3 dimBlock(32, 32);
    kl_convolution(dimGrid, dimBlock, (RGB*)imgCudaArray_, (RGB*)imgPadCudaArray_, relativeIdxsGPUptr, kernelGPUptr, kernelValues, width_, padWidth_, padding_);
    this->checkCudaError(cudaGetLastError(), "kl_convolution");

    cudaDeviceSynchronize();
    cudaFree(relativeIdxsGPUptr);
    cudaFree(kernelGPUptr);
}