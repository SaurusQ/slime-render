#include "simulation.hpp"
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

Simulation::Simulation(const Image& img)
{
    width_              = img.getWidth();
    height_             = img.getHeigth();
    padding_            = 1;
    bufferSize_         = img.getBufferSize();
    bufferSizePadded_   = img.getPaddedBufferSize(padding_);
    padWidth_           = 2 * padding_ + width_;
    padOffset_ = padding_ * padWidth_ + padding_;


    this->checkCudaError(
        cudaMalloc((void**)&trailMapFront_, bufferSizePadded_),
        "cudaMalloc trailMapFront_"
    );
    this->checkCudaError(
        cudaMalloc((void**)&trailMapBack_, bufferSizePadded_),
        "cudaMalloc trailMapBack_"
    );

    this->checkCudaError(
        cudaMemset((void*)trailMapFront_, 0, bufferSizePadded_),
        "cudaMemset trailMapFront_"
    );
    this->checkCudaError(
        cudaMemset((void*)trailMapBack_, 0, bufferSizePadded_),
        "cudaMemset trailMapBack_"
    );

    this->loadTexture();
    this->activateCuda();
    this->update(img);
    this->deactivateCuda();
}

Simulation::~Simulation()
{
    cudaGraphicsUnregisterResource(cudaPboResource_);
    glDeleteTextures(1, &texture_);

    if (relativeIdxsGPUptr_ != nullptr)
    {
        cudaFree(relativeIdxsGPUptr_);
    }
    if (agentConfigGPU_ != nullptr)
    {
        cudaFree(relativeIdxsGPUptr_);
    }
    if (agentColorsGPU_ != nullptr)
    {
        cudaFree(relativeIdxsGPUptr_);
    }
    cudaFree(resultCudaImg_);
    cudaFree(trailMapFront_);
    cudaFree(trailMapBack_);
    if (agents_ != nullptr) cudaFree(agents_);
    if (agentRandomState_ != nullptr) cudaFree(agentRandomState_);
}

void Simulation::activateCuda()
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
            cudaGraphicsResourceGetMappedPointer((void**)&resultCudaImg_, &cudaPboSize, cudaPboResource_),
            "cudaGraphicsSubResourceGetMappedArray"
        );

        if (cudaPboSize != bufferSize_)
        {
            std::cerr << "Something wrong with buffer sizes: pbo: " << cudaPboSize << " buffer: " << bufferSize_ << std::endl;
        }
    }
}

void Simulation::deactivateCuda()
{
    cudaActive_ = false;
    if (cudaPboResource_ != nullptr)
    {
        cudaGraphicsUnmapResources(1, &cudaPboResource_, 0);
    }
}

void Simulation::update(const Image& img)
{
    REQUIRE_CUDA
    if(img.getBufferSize() != bufferSize_)
    {
        std::cout << "Different buffer sizes " << bufferSize_ << " : " << img.getBufferSize() << std::endl;
        return;
    }
    this->checkCudaError(
        cudaMemcpy((void*)resultCudaImg_, (void*)img.getPtr(), bufferSize_, cudaMemcpyHostToDevice),
        "cudaMemcpy update()"
    );
    cudaDeviceSynchronize();
}

void Simulation::readBack(const Image& img) const
{
    REQUIRE_CUDA
    /*this->checkCudaError(
        cudaMemcpy((void*)img.getPtr(), (void*)imageGPUptr_, bufferSize_, cudaMemcpyDeviceToHost),
        "cudaMemcpy readback()"
    );*/
}

void Simulation::clearImage()
{
    cudaMemset(resultCudaImg_, 0, bufferSize_);
    cudaMemset(trailMapFront_, 0, bufferSizePadded_);
    cudaMemset(trailMapBack_,  0, bufferSizePadded_);
}

void Simulation::swapBuffers()
{
    auto temp = trailMapFront_;
    trailMapFront_ = trailMapBack_;
    trailMapBack_ = temp;
}

void Simulation::loadTexture()
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

bool Simulation::checkCudaError(cudaError_t ce, std::string msg) const
{
    bool failure = ce != cudaSuccess;
    if(failure)
    {
        msg = std::string("FAIL: ") + msg + std::string(" WHAT: ") + std::string(cudaGetErrorString(ce));
        throw std::runtime_error(msg);
    }
    return failure;
}

void Simulation::trailMapToDisplay()
{
    REQUIRE_CUDA
    /*this->checkCudaError(
        cudaMemcpy2D((void*)resultCudaImg_, width_ * sizeof(RGBA), (void*)(trailMapFront_ + padOffset_), padWidth_ * sizeof(RGBA), width_ * sizeof(RGBA), height_, cudaMemcpyDeviceToDevice),
        "cudaMemcpy trailMapToDisplay()"
    );*/
    dim3 grid(width_ / 32, height_ / 32);
    dim3 block(32, 32);
    kl_trailMapToDisplay(grid, block,
        reinterpret_cast<float4*>(trailMapFront_),
        reinterpret_cast<float4*>(resultCudaImg_),
        reinterpret_cast<float3*>(agentColorsGPU_),
        width_,
        height_,
        padWidth_,
        padOffset_
    );
}

void Simulation::updateTrailMap(double deltaTime, float diffuseWeight, float evaporateWeight)
{
    REQUIRE_CUDA
    //unsigned int kernelValues = (kernelSize * 2 + 1) * (kernelSize * 2 + 1);
    //
    this->swapBuffers();

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

    dim3 grid(width_ / 32, height_ / 32);
    dim3 block(32, 32);
    kl_updateTrailMap(grid, block,
        reinterpret_cast<float4*>(trailMapFront_),
        reinterpret_cast<float4*>(trailMapBack_),
        relativeIdxsGPUptr_,
        diffuseWeight * deltaTime,
        evaporateWeight * deltaTime,
        padWidth_,
        padOffset_
    );
    this->checkCudaError(cudaGetLastError(), "kl_updateTrailMap");

    cudaDeviceSynchronize();
}

void Simulation::spawnAgents(unsigned int newAgents, float* agentShares, StartFormation startFormation, bool clear)
{
    nAgents_ = newAgents;

    if (agents_ != nullptr) cudaFree(agents_);
    nAgentsGpuSize_ = roundUpToPowerOfTwo(nAgents_);
    this->checkCudaError(
        cudaMalloc((void**)&agents_, nAgentsGpuSize_ * sizeof(Agent)),
        "cudaMalloc agents"
    );
    
    if (clear) this->clearImage();

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
            case StartFormation::RANDOM_CIRCLE:
            {
                float r = (std::min(width_, height_) / 2.0f) * 0.9f * randn(rng);
                float a = dist2pi(rng);
                ag.pos = float2{width_ / 2.0f + static_cast<float>(r * std::cos(a)), height_ / 2.0f + static_cast<float>(r * std::sin(a))};
                ag.angle = (randn(rng) * 2 * M_PI) - M_PI;
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
        ag.speciesIdx = -1; // Set later
        cpuAgents[i] = ag;
    }

    this-checkCudaError(
        cudaMemcpy(agents_, cpuAgents.get(), nAgents_ * sizeof(Agent), cudaMemcpyHostToDevice),
        "cudaMemcpy agent from cpu to gpu"
    );

    this->updatePopulationShare(agentShares);

    // Init random state
    if (agentRandomState_ == nullptr)
    {
        this->checkCudaError(
            cudaMalloc(&agentRandomState_, 32 * sizeof(curandState)),
            "cudaMalloc agentRandomState_"
        );
    }
    dim3 grid(1, 1);
    dim3 block(32, 1);
    kl_initCurand32(grid, block, agentRandomState_);
}

void Simulation::updatePopulationSize(unsigned int newAgents)
{
    if (nAgents_ == newAgents) return;

    if (nAgentsGpuSize_ < newAgents || nAgentsGpuSize_ > roundUpToPowerOfTwo(newAgents)) {
        
        //std::cout << "round p2: " << newAgents << " | " << roundUpToPowerOfTwo(newAgents) << std::endl;
        unsigned int newArraySize = roundUpToPowerOfTwo(newAgents);
        //std::cout << "Resizing array " << nAgents_ << " -> " << newArraySize << std::endl; 
        Agent* newArray;
        this->checkCudaError(
            cudaMalloc((void**)&newArray, newArraySize * sizeof(Agent)),
            "cudaMalloc agents"
        );
        this->checkCudaError(
            cudaMemcpy(newArray, agents_, std::min(nAgents_, newAgents) * sizeof(Agent), cudaMemcpyDeviceToDevice),
            "cudaMemcpy oldAgents to new agent array from gpu to gpu"
        );
        cudaFree(agents_);
        agents_ = newArray;
        nAgentsGpuSize_ = newArraySize;
    }

    if (nAgents_ < newAgents) // We need to set the parameters of the new agents
    {
        //std::cout << "agents old: " << nAgents_ << " agents new: " << newAgents << std::endl;
        while (newAgents - nAgents_ > nAgents_) // Added too many agents to make unique copies
        {
            unsigned int newAgents = newAgents - nAgents_;
            //std::cout << "adding copying new agents " << newAgents << std::endl;
            this->checkCudaError(
                cudaMemcpy(agents_ + nAgents_, agents_, nAgents_ * sizeof(Agent), cudaMemcpyDeviceToDevice),
                "cudaMemcpy copy whole array of agents to the end of itself from gpu to gpu"
            );
            nAgents_ += nAgents_;
        }
        if (nAgents_ != newAgents)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(0, nAgents_ - 1 - (newAgents - nAgents_));
            unsigned int idx = dist(gen);
            //std::cout << "copyging random part from the gpu" << " dist: " << idx << std::endl;
            if (idx + (newAgents - nAgents_) > nAgents_) std::cout << "FAILURE IN IDX ########" << std::endl;
            this->checkCudaError(
                cudaMemcpy(agents_ + nAgents_, agents_ + idx, (newAgents - nAgents_) * sizeof(Agent), cudaMemcpyDeviceToDevice),
                "cudaMemcpy agent from gpu to gpu"
            );
        }
    }

    nAgents_ = newAgents;
    this->updateSpecies();
    //std::cout << "Final count: " << nAgents_ << std::endl;
}

void Simulation::configAgentParameters(AgentConfig* aConfigs, AgentColor* aColors) {
    std::copy(aColors, aColors + DIFFERENT_SPECIES, agentColors_);
    for (int i = 0; i < DIFFERENT_SPECIES; i++)
    {
        this->configAgentSpeed(i, aConfigs[i].speed); 
        this->configAgentTurnSpeed(i, aConfigs[i].turnSpeed); 
        this->configAgentSensorAngleSpacing(i, aConfigs[i].sensorAngleSpacing); 
        this->configAgentSensorOffsetDst(i, aConfigs[i].sensorOffsetDst);
        this->configAgentSensorSize(i, aConfigs[i].sensorSize);
    }
}

std::vector<unsigned int> Simulation::getPopulationCount() const
{
    std::vector<unsigned int> counts(DIFFERENT_SPECIES, 0);
    unsigned int sum = 0;
    for (int i = 0; i < DIFFERENT_SPECIES; i++)
    {
        counts[i] = nAgents_ * agentShares_[i];
        sum += counts[i];
    }
    counts[0] += nAgents_ - sum; // Balance rounding errors
    return counts;
}

void Simulation::updateAgentConfig()
{
    REQUIRE_CUDA
    if (agentConfigGPU_ == nullptr)
    {
        this->checkCudaError(
            cudaMalloc((void**)&agentConfigGPU_, sizeof(AgentConfig_GPU) * DIFFERENT_SPECIES),
            "cudaMalloc for agentConfigGPU_ failed"
        );
    }
    if (agentColorsGPU_ == nullptr)
    {
        this->checkCudaError(
            cudaMalloc((void**)&agentColorsGPU_, sizeof(AgentColor) * DIFFERENT_SPECIES),
            "cudaMalloc for agentColorsGPU_ failed"
        );
    }

    this->checkCudaError(
        cudaMemcpy(agentConfigGPU_, agentConfig_, sizeof(AgentConfig_I) * DIFFERENT_SPECIES, cudaMemcpyHostToDevice),
        "cudaMemcpy agentConfigGPU_ failed" 
    );
    this->checkCudaError(
        cudaMemcpy(agentColorsGPU_, agentColors_, sizeof(AgentColor) * DIFFERENT_SPECIES, cudaMemcpyHostToDevice),
        "cudaMemcpy agentColorsGPU_ failed" 
    );
}

void Simulation::updateAgents(double deltaTime, float trailWeight)
{
    REQUIRE_CUDA
    dim3 grid(std::ceil(nAgents_ / 32.0), 1);
    dim3 block(32, 1);
    kl_updateAgents(grid, block,
        deltaTime,
        agentRandomState_,
        reinterpret_cast<float4*>(trailMapFront_),
        agents_,
        nAgents_,
        agentConfigGPU_,
        static_cast<float>(trailWeight * deltaTime),
        width_,
        height_,
        padWidth_,
        padOffset_);
}

void Simulation::updatePopulationShare(float* newPopulationShare)
{
    std::copy(newPopulationShare, newPopulationShare + DIFFERENT_SPECIES, agentShares_);
    this->updateSpecies();
}

void Simulation::updateSpecies()
{
    std::unique_ptr<Agent[]> cpuAgents = std::make_unique<Agent[]>(nAgents_);
    this->checkCudaError(
        cudaMemcpy(cpuAgents.get(), agents_, sizeof(Agent) * nAgents_, cudaMemcpyDeviceToHost),
        "Copying agents from GPU"
    );
    auto populationReserve = this->getPopulationCount();
    // Set the species of the population
    for (int idx = 0; idx < nAgents_; idx++)
    {
        Agent& ag = cpuAgents[idx];
        if (populationReserve[ag.speciesIdx] == 0)
        {
            ag.speciesIdx = (ag.speciesIdx + 1) % DIFFERENT_SPECIES;
            ag.speciesMask = {
                ag.speciesIdx == 0 ? 1 : 0,
                ag.speciesIdx == 1 ? 1 : 0,
                ag.speciesIdx == 2 ? 1 : 0,
                ag.speciesIdx == 3 ? 1 : 0
            };
            idx--;
        }
        else
        {
            populationReserve[ag.speciesIdx]--;
        }
    }
    this->checkCudaError(
        cudaMemcpy(agents_, cpuAgents.get(), sizeof(Agent) * nAgents_, cudaMemcpyHostToDevice),
        "Copying agents to GPU"
    );
}