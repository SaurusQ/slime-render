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

    if (relativeIdxsGPUptr_ != nullptr)
    {
        cudaFree(relativeIdxsGPUptr_);
    }
    if (imgCudaArray_)
    {
        cudaFree(imgCudaArray_);
    }
    if (imgPadCudaArray_)
    {
        cudaFree(imgPadCudaArray_);
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

void ImageGPU::clearImage()
{
    if (imgCudaArray_)
    {
        cudaMemset(imgCudaArray_, 0, bufferSize_);
    }
    if (imgPadCudaArray_)
    {
        cudaMemset(imgPadCudaArray_, 0, bufferSizePadded_);
    }
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
        "cudaGraphicsGLRegisterBuffer"
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

void ImageGPU::updateTrailMap(double deltaTime, float diffuseWeight, float evaporateWeight)
{
    REQUIRE_CUDA
    //unsigned int kernelValues = (kernelSize * 2 + 1) * (kernelSize * 2 + 1);
    //
    this->imgToPadded();

    if (relativeIdxsGPUptr_ == nullptr)
    {
        if(padding_ < 1) 
        {
            std::cerr << "Insuffisient padding size Padding: " << padding_ << std::endl;
            return; 
        }
        // Generate new relative idxs
        std::vector<int> newRelIdx;
        unsigned int kernelValues = 9; // 3x3 diffuse
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                newRelIdx.push_back(x + y * static_cast<int>(padWidth_));
            }
        }

        this->checkCudaError(
            cudaMalloc((void**)&relativeIdxsGPUptr_, kernelValues * sizeof(int)),
            "cudaMalloc relativeIdxsGPUptr"
        );
        this->checkCudaError(
            cudaMemcpy((void*)relativeIdxsGPUptr_, newRelIdx.data(), kernelValues * sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy relativeIdxs" 
        );
    }

    unsigned int padOffset = padding_ * padWidth_ + padding_;

    dim3 grid(width_ / 32, height_ / 32);
    dim3 block(32, 32);
    kl_updateTrailMap(grid, block, deltaTime, (RGB*)imgCudaArray_, (RGB*)imgPadCudaArray_, relativeIdxsGPUptr_, diffuseWeight * deltaTime, evaporateWeight * deltaTime, width_, padWidth_, padding_, padOffset);
    this->checkCudaError(cudaGetLastError(), "kl_convolution");

    cudaDeviceSynchronize();
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
    std::uniform_real_distribution<float> dist2pi(0.0, 2 * M_PI);
    std::uniform_real_distribution<float> randn(0.0, 1.0);

    std::unique_ptr<Agent[]> cpuAgents = std::make_unique<Agent[]>(nAgents_);
    
    for (int i = 0; i < nAgents_; i++)
    {
        Agent ag;
        switch (startFormation)
        {
            case StartFormation::CIRCLE:
            {
                // Polar coordinates
                float r = (std::min(width_, height_) / 2.0f) * 0.9f * randn(rng);
                float a = dist2pi(rng);
                ag.pos = float2{width_ / 2.0f + static_cast<float>(r * std::cos(a)), height_ / 2.0f + static_cast<float>(r * std::sin(a))};
                ag.angle = std::atan2((height_ / 2.0f) - ag.pos.y, (width_ / 2.0f) - ag.pos.x);
                break;
            }
            case StartFormation::MIDDLE:
            {
                ag.pos = float2{width_ / 2.0f, height_ / 2.0f};
                ag.angle = dist2pi(rng);
                break;
            }
            case StartFormation::RANDOM:
            default:
            {
                ag.pos = float2{randn(rng) * width_, randn(rng) * height_};
                ag.angle = dist2pi(rng);
                break;
            }
        }
        cpuAgents[i] = ag;
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

void ImageGPU::updateAgents(double deltaTime)
{
    REQUIRE_CUDA
    dim3 grid(std::ceil(nAgents_ / 32.0), 1);
    dim3 block(32, 1);
    kl_updateAgents(grid, block, deltaTime, agentRandomState_, imgCudaArray_, agents_, nAgents_,
        agentConfig_.speed,
        agentConfig_.turnSpeed,
        agentConfig_.sensorAngleSpacing,
        agentConfig_.sensorOffsetDst,
        agentConfig_.sensorSize,
        width_, height_);
}