#include "imageGPU.hpp"
#include "cudaKernels.cuh"

#include <cuda_gl_interop.h>

#include <iostream>
#include <stdexcept>
#include <random>

#define REQUIRE_CUDA if(!cudaActive_) { std::cout << "cuda not active" << std::endl; return; };

unsigned int roundUpToPowerOfTwo(unsigned int x) {
    if (x <= 1) {
        return 1;
    }

    // Subtract 1 to handle the case when x is already a power of two
    x--;

    // Set all bits to the right of the most significant bit
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    // Add 1 to get the next power of two
    x++;

    return x;
}

ImageGPU::ImageGPU(const Image& img, unsigned int padding)
    : padding_(padding)
{
    width_              = img.getWidth();
    height_             = img.getHeigth();
    bufferSize_         = img.getBufferSize();
    bufferSizePadded_   = img.getPaddedBufferSize(padding_);
    padWidth_           = 2 * padding_ + width_;

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

    for (const auto& pair : convRelIdxsGPUptrs_)
    {
        cudaFree(pair.second);
    }
    for (const auto& pair : convKernelGPUptrs_)
    {
        cudaFree(pair.second);
    }
    if (agents_ != nullptr) cudaFree(agents_);
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
}

void ImageGPU::update(const Image& img)
{
    REQUIRE_CUDA
    if(img.getBufferSize() != bufferSize_)
    {
        std::cout << "Different buffer sizes " << bufferSize_ << " : " << img.getBufferSize() << std::endl;
        return;
    }
    this->checkCudaError(
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
    this->checkCudaError(
        cudaMemcpy2D((void*)(imgPadCudaArray_ + padding_ + padWidth_ * padding_), padWidth_ * sizeof(RGB), (void*)imgCudaArray_, width_ * sizeof(RGB), width_ * sizeof(RGB), height_, cudaMemcpyDeviceToDevice),
        "cudaMemcpy imgToPadded()"
    );
}

void ImageGPU::addConvKernel(unsigned int kernelId, std::vector<float> kernel)
{
    float* kernelGPUptr;
    this->checkCudaError(
        cudaMalloc((void**)&kernelGPUptr, kernel.size() * sizeof(float)),
        "cudaMalloc kernelGPUptr"
    );
    this->checkCudaError(
        cudaMemcpy((void*)kernelGPUptr, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy kernel" 
    );
    convKernelGPUptrs_[kernelId] = kernelGPUptr;
}

void ImageGPU::convolution(unsigned int kernelSize, unsigned int kernelId, float convWeight)
{
    REQUIRE_CUDA
    unsigned int kernelValues = (kernelSize * 2 + 1) * (kernelSize * 2 + 1);
    if(kernelSize > padding_) 
    {
        std::cerr << "Too large kernel size for padding. Kernel: " << kernelSize << " Padding: " << padding_ << std::endl;
        return; 
    }
    this->imgToPadded();

    int* relativeIdxsGPUptr = nullptr;
    auto idxIt = convRelIdxsGPUptrs_.find(kernelSize);
    if (idxIt == convRelIdxsGPUptrs_.end())
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

        this->checkCudaError(
            cudaMalloc((void**)&relativeIdxsGPUptr, kernelValues * sizeof(int)),
            "cudaMalloc relativeIdxsGPUptr"
        );
        this->checkCudaError(
            cudaMemcpy((void*)relativeIdxsGPUptr, newRelIdx.data(), kernelValues * sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy relativeIdxs" 
        );
        convRelIdxsGPUptrs_[kernelSize] = relativeIdxsGPUptr;
    }
    else
    {
        relativeIdxsGPUptr = idxIt->second;
    }

    float* kernelGPUptr = nullptr;
    auto kernelIt = convKernelGPUptrs_.find(kernelId);
    if (kernelIt == convKernelGPUptrs_.end())
    {
        std::cout << "coudn't find kernel data call addConvKernel() to add more kernels" << std::endl;
        return;
    }
    kernelGPUptr = kernelIt->second;

    unsigned int padOffset = padding_ * padWidth_ + padding_;

    dim3 grid(width_ / 32, height_ / 32);
    dim3 block(32, 32);
    kl_convolution(grid, block, (RGB*)imgCudaArray_, (RGB*)imgPadCudaArray_, relativeIdxsGPUptr, kernelGPUptr, kernelValues, convWeight, width_, padWidth_, padding_, padOffset);
    this->checkCudaError(cudaGetLastError(), "kl_convolution");

    cudaDeviceSynchronize();
}

void ImageGPU::evaporate(float strength)
{
    REQUIRE_CUDA
    dim3 grid(width_ / 32, height_ / 32);
    dim3 block(32, 32);
    kl_evaporate(grid, block, imgCudaArray_, strength, width_);
}

void ImageGPU::setAgentStart(unsigned int num, StartFormation startFormation)
{
    nAgents_ = num;
    if (agents_ != nullptr) cudaFree(agents_);
    nAgentsGpuSize_ = roundUpToPowerOfTwo(nAgents_);
    this->checkCudaError(
        cudaMalloc((void**)&agents_, nAgentsGpuSize_ * sizeof(Agent)),
        "cudaMalloc agents"
    );
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(0.0, 2 * M_PI);
    std::uniform_real_distribution<float> randn(0.0, 1.0);

    std::unique_ptr<Agent[]> cpuAgents = std::make_unique<Agent[]>(nAgents_);
    
    for (int i = 0; i < nAgents_; i++)
    {
        Agent a;
        switch (startFormation)
        {
            case StartFormation::CIRCLE:
            {
                float r = std::sqrt(randn(rng)) * 1000; // Radius
                float x = randn(rng) * 2 * M_PI;
                a.pos = float2{width_ / 2.0 + static_cast<float>(r * std::cos(x)), height_ / 2.0 + static_cast<float>(r * std::sin(x))};
                a.angle = dist(rng);
                break;
            }
            case StartFormation::MIDDLE:
                a.pos = float2{width_ / 2.0f, height_ / 2.0f};
                a.angle = dist(rng);
                break;
            case StartFormation::RANDOM:
            default:
                break;
        }
        cpuAgents[i] = a;
    }
    
    this-checkCudaError(
        cudaMemcpy(agents_, cpuAgents.get(), nAgents_ * sizeof(Agent), cudaMemcpyHostToDevice),
        "cudaMemcpy agent from cpu to gpu"
    );

    
    if (agentRandomState_ != nullptr) cudaFree(agentRandomState_);

    this->checkCudaError(
        cudaMalloc(&agentRandomState_, 32 * sizeof(curandState)),
        "cudaMalloc agentRandomState_"
    );

    dim3 grid(1, 1);
    dim3 block(32, 1);
    kl_initCurand32(grid, block, agentRandomState_);
}

void ImageGPU::updatePopulationSize(unsigned int num)
{
    if (nAgents_ == num) return;

    if (nAgentsGpuSize_ < num || nAgentsGpuSize_ > roundUpToPowerOfTwo(num)) {
        
        std::cout << "round p2: " << num << " | " << roundUpToPowerOfTwo(num) << std::endl;
        unsigned int newArraySize = roundUpToPowerOfTwo(num);
        std::cout << "Resizing array " << nAgents_ << " -> " << newArraySize << std::endl; 
        Agent* newArray;
        this->checkCudaError(
            cudaMalloc((void**)&newArray, newArraySize * sizeof(Agent)),
            "cudaMalloc agents"
        );
        this->checkCudaError(
            cudaMemcpy(newArray, agents_, std::min(nAgents_, num) * sizeof(Agent), cudaMemcpyDeviceToDevice),
            "cudaMemcpy oldAgents to new agent array from gpu to gpu"
        );
        cudaFree(agents_);
        agents_ = newArray;
        nAgentsGpuSize_ = newArraySize;
    }
    if (nAgents_ < num) // We need to set the parameters of the new agents
    {
        std::cout << "agents old: " << nAgents_ << " agents new: " << num << std::endl;
        while (num - nAgents_ > nAgents_) // Added too many agents to make unique copies
        {
            unsigned int newAgents = num - nAgents_;
            std::cout << "adding copying new agents " << newAgents << std::endl;
            this->checkCudaError(
                cudaMemcpy(agents_ + nAgents_, agents_, nAgents_ * sizeof(Agent), cudaMemcpyDeviceToDevice),
                "cudaMemcpy copy whole array of agents to the end of itself from gpu to gpu"
            );
            nAgents_ += nAgents_;
        }
        if (nAgents_ != num)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(0, nAgents_ - 1 - (num - nAgents_));
            unsigned int idx = dist(gen);
            std::cout << "copyging random part from the gpu" << " dist: " << idx << std::endl;
            if (idx + (num - nAgents_) > nAgents_) std::cout << "FAILURE IN IDX ########" << std::endl;
            this->checkCudaError(
                cudaMemcpy(agents_ + nAgents_, agents_ + idx, (num - nAgents_) * sizeof(Agent), cudaMemcpyDeviceToDevice),
                "cudaMemcpy agent from gpu to gpu"
            );
        }
    }
    nAgents_ = num;
    std::cout << "Final count: " << nAgents_ << std::endl;
}

void ImageGPU::updateAgents()
{
    REQUIRE_CUDA
    dim3 grid(std::ceil(nAgents_ / 32.0), 1);
    dim3 block(32, 1);
    kl_updateAgents(grid, block, agentRandomState_, imgCudaArray_, agents_, nAgents_,
        agentConfig_.speed,
        agentConfig_.turnSpeed,
        agentConfig_.sensorAngleSpacing,
        agentConfig_.sensorOffsetDst,
        agentConfig_.sensorSize,
        width_, height_);
}