#include "imageKernel.cuh"
#include "imgKernels.cuh"

#include <cuda_gl_interop.h>

#include <iostream>
#include <stdexcept>

ImageKernel::ImageKernel(const Image& img, unsigned int padding)
    : padding_(padding)
{
    width_              = img.getWidth();
    height_             = img.getHeight();
    bufferSize_         = img.getBufferSize();
    bufferSizePadded_   = img.getPaddedBufferSize(padding_);

    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
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
        "cudaMemset"
    );
    this->loadTexture();
    this->update(img);

    /*cudaChannelFormatDesc channelDesc;

    // Retrieve the channel format description from the cudaArray
    cudaGetChannelDesc(&channelDesc, imgCudaArray_);

    // Print the fields of the channel format description
    std::cout << "Channel Format Description:" << std::endl;
    std::cout << "x: " << channelDesc.x << std::endl;
    std::cout << "y: " << channelDesc.y << std::endl;
    std::cout << "z: " << channelDesc.z << std::endl;
    std::cout << "w: " << channelDesc.w << std::endl;
    std::cout << "f: " << channelDesc.f << std::endl;
*/
}

ImageKernel::~ImageKernel()
{
    cudaFree((void*)imgPadCudaArray_);
    cudaGraphicsUnregisterResource(cudaPboResource_);
    glDeleteTextures(1, &texture_);
}

void ImageKernel::activateCuda()
{
    if (cudaPboResource_ != nullptr)
    {
        cudaGraphicsMapResources(1, &cudaPboResource_);
    }
}

void ImageKernel::deactivateCuda()
{
    if (cudaPboResource_ != nullptr)
    {
        cudaGraphicsUnmapResources(1, &cudaPboResource_);
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
        //cudaMemcpy2DToArray(imgCudaArray_, 0, 0, (void*)img.getPtr(), width_ * sizeof(RGB), width_ * sizeof(RGB), height_, cudaMemcpyHostToDevice),
        // ((cudaArray_t)imageGPUptr_, 0, 0, (void*)img.getPtr(), bufferSize_, cudaMemcpyHostToDevice),
        cudaMemcpy((void*)imgCudaArray_, (void*)img.getPtr(), bufferSize_, cudaMemcpyHostToDevice),
        "cudaMemcpy update()"
    );
    cudaDeviceSynchronize();
}

void ImageKernel::readBack(const Image& img) const
{
    /*this->checkCudaError(
        cudaMemcpy((void*)img.getPtr(), (void*)imageGPUptr_, bufferSize_, cudaMemcpyDeviceToHost),
        "cudaMemcpy readback()"
    );*/
}

void ImageKernel::loadTexture()
{
    // Pixel buffer object
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize_, nullptr, GL_DYNAMIC_DRAW);

    // Texture
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


    // Create texture data (4-component unsigned byte)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width_, height_, 0, GL_RGB, GL_FLOAT, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    // Unbind the texture
    //glBindTexture(GL_TEXTURE_2D, texture_);

    // cuda device pointer from cuda graphics resource
    
    this->checkCudaError(
        cudaGraphicsGLRegisterBuffer(&cudaPboResource_, pbo_, cudaGraphicsMapFlagsWriteDiscard),
        "cudaGraphicsGLRegisterBUffer"
    );

    this->checkCudaError(
        cudaGraphicsMapResources(1, &cudaPboResource_, 0),
        "cudaGraphicsMapResources"
    );

    std::cout << "cudaGraphicsSubResourceGetMappedArray" << std::endl;
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
    cudaDeviceSynchronize();
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
    /*
    this->checkCudaError(
        cudaMemcpy2DArrayToArray(imgPadCudaArray_, padding_, padding_, imgCudaArray_, 0, 0, width_, height_),
        "cudaMemcpy2DArrayToArray imgToPadded()"
    );*/
    this->checkCudaError(
        cudaMemcpy2D((void*)imgPadCudaArray_, (width_ + 2 * padding_) * sizeof(RGB), imgCudaArray_, width_ * sizeof(RGB), width_, height_, cudaMemcpyDeviceToDevice),
        "cudaMemcpy imgToPadded()"
    );
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
    std::cout << "convolustion" << std::endl;
    k_convolution<<<height_, width_>>>((RGB*)imgCudaArray_, (RGB*)imgPadCudaArray_, relativeIdxsGPUptr, kernelGPUptr, kernelValues, width_);
    cudaDeviceSynchronize();
    cudaFree(relativeIdxsGPUptr);
    cudaFree(kernelGPUptr);
}
