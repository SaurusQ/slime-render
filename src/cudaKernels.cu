#include "cudaKernels.cuh"

#include <iostream>

#define PI 3.141592653589793f

void kl_updateTrailMap(dim3 grid, dim3 block,
    double deltaTime,
    RGB* imgPtr,
    RGB* imgPadPtr,
    int* relativeIdxs,
    float diffuseDeltaW,
    float evaporateDeltaW,
    unsigned int width,
    unsigned int padWidth,
    unsigned int padding,
    unsigned int padOffset
)
{
    k_updateTrailMap<<<grid, block>>>(deltaTime, imgPtr, imgPadPtr, relativeIdxs, diffuseDeltaW, evaporateDeltaW, width, padWidth, padding, padOffset);
}

__global__ void k_updateTrailMap(
    double deltaTime,
    RGB* imgPtr,
    RGB* imgPadPtr,
    int* relativeIdxs,
    float diffuseDeltaW,
    float evaporateDeltaW,
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

    // Diffuse
    float valueR = 0;
    float valueG = 0;
    float valueB = 0;
    for (int i = 0; i < 9; i++) // 3x3 grid
    {
        valueR += imgPadPtr[idxPad + relativeIdxs[i]].r;
        valueG += imgPadPtr[idxPad + relativeIdxs[i]].g;
        valueB += imgPadPtr[idxPad + relativeIdxs[i]].b;
    }    
    float diffusedR = imgPadPtr[idxPad].r * (1 - diffuseDeltaW) + (valueR / 9.0) * (diffuseDeltaW);
    float diffusedG = imgPadPtr[idxPad].g * (1 - diffuseDeltaW) + (valueB / 9.0) * (diffuseDeltaW);
    float diffusedB = imgPadPtr[idxPad].b * (1 - diffuseDeltaW) + (valueG / 9.0) * (diffuseDeltaW);
    // Evaporate
    imgPtr[idx].r = max(0.0, diffusedR - evaporateDeltaW);
    imgPtr[idx].g = max(0.0, diffusedG - evaporateDeltaW);
    imgPtr[idx].b = max(0.0, diffusedB - evaporateDeltaW);
}

void kl_updateAgents(dim3 grid, dim3 block,
    double deltaTime,
    curandState* randomState,
    RGB* imgPtr,
    Agent* agents,
    unsigned int nAgents,
    float speed,
    float turnSpeed,
    float sensorAngleSpacing,
    float sensorOffsetDst,
    unsigned int sensorSize,
    float trailDeltaW,
    unsigned int width,
    unsigned int heigth
)
{
    k_updateAgents<<<grid, block>>>(deltaTime, randomState, imgPtr, agents, nAgents, speed, turnSpeed, sensorAngleSpacing, sensorOffsetDst, sensorSize, trailDeltaW, width, heigth);
}

__global__ void k_updateAgents(
    double deltaTime,
    curandState* randomState,
    RGB* imgPtr,
    Agent* agents,
    unsigned int nAgents,
    float speed,
    float turnSpeed,
    float sensorAngleSpacing,
    float sensorOffsetDst,
    unsigned int sensorSize,
    float trailDeltaW,
    unsigned int width,
    unsigned int heigth
)
{
    int agentIdx = blockIdx.x * 32 + threadIdx.x;
    if (agentIdx >= nAgents) return;
    Agent* agent = agents + agentIdx;
    
    // Sense and turn
    float wf = sense(*agent,                 0.0, imgPtr, sensorOffsetDst, sensorSize, width, heigth);
    float wl = sense(*agent,  sensorAngleSpacing, imgPtr, sensorOffsetDst, sensorSize, width, heigth);
    float wr = sense(*agent, -sensorAngleSpacing, imgPtr, sensorOffsetDst, sensorSize, width, heigth);
    
    float randomSteer = curand_uniform(randomState + threadIdx.x);


    if (wf > wl && wf > wr)
    {
        agent->angle += 0;
    }
    else if (wf < wl && wf < wr)
    {
        agent->angle += (randomSteer - 0.5) * 2 * turnSpeed * deltaTime;
    }
    else if (wr > wl) {
        agent->angle -= randomSteer * turnSpeed * deltaTime;
    }
    else if (wl > wr)
    {
        agent->angle += randomSteer * turnSpeed * deltaTime;
    }

    // Update position
    float2 direction = make_float2(cosf(agent->angle), sinf(agent->angle));
    float2 newPos = make_float2(deltaTime * speed * direction.x + agent->pos.x, deltaTime * speed * direction.y + agent->pos.y);

    if (newPos.x < 0 || newPos.x >= width || newPos.y < 0 || newPos.y >= heigth)
    {
        newPos.x = min(width - 0.01, max(0.0, newPos.x));
        newPos.y = min(heigth - 0.01, max(0.0, newPos.y));
        agent->angle = curand_uniform(randomState + threadIdx.x) * 2 * PI;
    }
    else
    {
        int idx = __float2uint_rd(newPos.x) + __float2uint_rd(newPos.y) * width;
        float3 value = make_float3(imgPtr[idx].r, imgPtr[idx].g, imgPtr[idx].b);
        value.x = min(1.0f, value.x + agent->speciesMask.x * trailDeltaW);
        value.y = min(1.0f, value.y + agent->speciesMask.y * trailDeltaW);
        value.z = min(1.0f, value.z + agent->speciesMask.z * trailDeltaW);
        imgPtr[idx] = RGB{value.x, value.y, value.z};
    }
    
    agent->pos = newPos;
}

__device__ float sense(Agent a, float sensorAngleOffset, RGB* imgPtr, float sensorOffsetDst, int sensorSize, unsigned int width, unsigned int heigth)
{
    float sensorAngle = a.angle + sensorAngleOffset;
    float2 sensorDir = make_float2(cosf(sensorAngle), sinf(sensorAngle));
    int2 sensorCentre = make_int2(a.pos.x + sensorDir.x * sensorOffsetDst, a.pos.y + sensorDir.y * sensorOffsetDst);
    
    float sum = 0.0f;

    int senseWeightX = a.speciesMask.x * 2 - 1;
    int senseWeightY = a.speciesMask.y * 2 - 1;
    int senseWeightZ = a.speciesMask.z * 2 - 1;

    for (int ox = -sensorSize; ox <= sensorSize; ox++)
    {
        for (int oy = -sensorSize; oy <= sensorSize; oy++)
        {
            int2 pos = make_int2(sensorCentre.x + ox, sensorCentre.y + oy);

            if (pos.x >= 0 && pos.x < width && pos.y >= 0 && pos.y < heigth)
            {
                int idx = pos.x + pos.y * width;
                sum += imgPtr[idx].r * senseWeightX
                    + imgPtr[idx].g * senseWeightY
                    + imgPtr[idx].b * senseWeightZ;
            }
        }
    }
    return sum;
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