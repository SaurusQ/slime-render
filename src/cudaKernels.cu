#include "cudaKernels.cuh"

#include <iostream>

#define PI 3.141592653589793f

// TODO move everything here to proper configs
__device__ float turnSpeed = 0.1;
__device__ float sensorOffsetDst = 9.0;
__device__ float sensorSize = 0.0;
__device__ float sensorAngleSpacing = 30.0 * (PI / 180.0);

void kl_convolution(dim3 grid, dim3 block,
    RGB* imgPtr,
    RGB* imgPadPtr,
    int* relativeIdxs,
    float* kernel,
    unsigned int kernelValues,
    unsigned int width,
    unsigned int padWidth,
    unsigned int padding,
    unsigned int padOffset
)
{
    k_convolution<<<grid, block>>>(imgPtr, imgPadPtr, relativeIdxs, kernel, kernelValues, width, padWidth, padding, padOffset);
}

__global__ void k_convolution(
    RGB* imgPtr,
    RGB* imgPadPtr,
    int* relativeIdxs,
    float* kernel,
    unsigned int kernelValues,
    unsigned int width,
    unsigned int padWidth,
    unsigned int padding,
    unsigned int padOffset
)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    int idxPad = padOffset + x + y * padWidth;
    int idx = x + y * width; 

    //imgPtr[(idx + 1) % (3840 * 2160)] = imgPadPtr[idxPad];

    //imgPtr[idx] = imgPadPtr[(padding * (2 * padding + width) + padding) + x + y * (width + 2 * padding)];

    /*
    unsigned int count = 0;
    for(int i = 0; i < (3840 + 2 * padding) * (2160 + 2 * padding); i++)
    {
        if(imgPadPtr[i].r >= 0.5) count++;
    }

    printf("count: %u", count);
    */
    
    
    float valueR = 0;
    float valueG = 0;
    float valueB = 0;

    for (int i = 0; i < kernelValues; i++)
    {
        valueR += imgPadPtr[idxPad + relativeIdxs[i]].r * kernel[i];
        valueG += imgPadPtr[idxPad + relativeIdxs[i]].g * kernel[i];
        valueB += imgPadPtr[idxPad + relativeIdxs[i]].b * kernel[i];
    }    

    imgPtr[idx].r = valueR;
    imgPtr[idx].g = valueG;
    imgPtr[idx].b = valueB;
    
}

void kl_evaporate(dim3 grid, dim3 block, RGB* imgPtr, float strength, unsigned int width)
{
    k_evaporate<<<grid, block>>>(imgPtr, strength, width);
}

__global__ void k_evaporate(RGB* imgPtr, float strength, unsigned int width)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    int idx = x + y * width;

    imgPtr[idx].r = max(0.0, imgPtr[idx].r - strength);
    imgPtr[idx].g = max(0.0, imgPtr[idx].g - strength);
    imgPtr[idx].b = max(0.0, imgPtr[idx].b - strength);
}

void kl_updateAgents(dim3 grid, dim3 block,
    curandState* randomState,
    RGB* imgPtr,
    Agent* agents,
    unsigned int nAgents,
    float speed,
    unsigned int width,
    unsigned int heigth
)
{
    k_updateAgents<<<grid, block>>>(randomState, imgPtr, agents, nAgents, speed, width, heigth);
}

__global__ void k_updateAgents(
    curandState* randomState,
    RGB* imgPtr,
    Agent* agents,
    unsigned int nAgents,
    float speed,
    unsigned int width,
    unsigned int heigth
)
{
    int agentIdx = blockIdx.x * 32 + threadIdx.x;
    if (agentIdx >= nAgents) return;
    Agent a = agents[agentIdx];

    float2 direction = make_float2(cosf(a.angle), sinf(a.angle));
    float2 newPos = make_float2(speed * direction.x + a.pos.x, speed * direction.y + a.pos.y); // TODO * deltaTime

    if (newPos.x < 0 || newPos.x >= width || newPos.y < 0 || newPos.y >= heigth)
    {
        newPos.x = min(width - 0.01, max(0.0, newPos.x));
        newPos.y = min(heigth - 0.01, max(0.0, newPos.y));
        agents[agentIdx].angle = curand_uniform(randomState + threadIdx.x) * 2 * PI;
    }
    
    agents[agentIdx].pos = newPos;

    imgPtr[__float2uint_rd(newPos.x) + __float2uint_rd(newPos.y) * width] = RGB{1.0, 1.0, 1.0};

    
    // Sense and turn
    float wf = sense(a, 0.0, imgPtr, width, heigth);
    float wl = sense(a, sensorAngleSpacing, imgPtr, width, heigth);
    float wr = sense(a, -sensorAngleSpacing, imgPtr, width, heigth);

    float randomSteer = 0.0;//= curand_uniform(randomState + threadIdx.x);

    if (wf > wl && wf > wr)
    {
        agents[agentIdx].angle += 0;
    }
    else if (wf < wl && wf < wr)
    {
        agents[agentIdx].angle += (randomSteer - 0.5) * 2 * turnSpeed; // TODO * deltaTime
    }
    else if (wr > wl) {
        agents[agentIdx].angle -= randomSteer * turnSpeed; // TODO * deltaTime
    }
    else if (wl > wr)
    {
        agents[agentIdx].angle += randomSteer * turnSpeed; // TODO * deltaTime
    }
}

__device__ float sense(Agent a, float sensorAngleOffset, RGB* imgPtr, unsigned int width, unsigned int heigth)
{
    float sensorAngle = a.angle + sensorAngleOffset;
    float2 sensorDir = make_float2(cosf(sensorAngle), sinf(sensorAngle));
    int2 sensorCentre = make_int2(a.pos.x + sensorDir.x * sensorOffsetDst, a.pos.y + sensorDir.y * sensorOffsetDst);
    
    RGB sum = RGB{0.0, 0.0, 0.0};

    for (int ox = -sensorSize; ox <= sensorSize; ox++)
    {
        for (int oy = -sensorSize; oy <= sensorSize; oy++)
        {
            int2 pos = make_int2(sensorCentre.x + ox, sensorCentre.y + oy);

            if (pos.x >= 0 && pos.x < width && pos.y >= 0 && pos.y < heigth)
            {
                int idx = pos.x + pos.y * width;
                sum.r += imgPtr[idx].r;
                sum.g += imgPtr[idx].g;
                sum.b += imgPtr[idx].b;

                //imgPtr[idx].r = 1.0; // MARK
            }
        }
    }
    return sum.r + sum.g + sum.b;
}

void kl_initCurand32(dim3 grid, dim3 block,
    curandState* state
)
{
    k_initCurand32<<<grid, block>>>(state);
}

__global__ void k_initCurand32(
    curandState* state
)
{
    int idx = threadIdx.x;
    curand_init(clock64(), idx, 0, state + idx);
}
