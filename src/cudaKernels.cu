#include "cudaKernels.cuh"

#include <iostream>

#define PI 3.141592653589793f

__device__ float4 operator+(const float4 &a, const float4 &b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ float4 operator*(const float4 &a, const float4 &b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.x* b.x, a.w * b.w);
}

void kl_updateTrailMap(dim3 grid, dim3 block,
    float4* trailMapFront,
    float4* trailMapBack,
    int* relativeIdxs,
    float diffuseDT,
    float evaporateDT,
    unsigned int padWidth,
    unsigned int padOffset
)
{
    k_updateTrailMap<<<grid, block>>>(trailMapFront, trailMapBack, relativeIdxs, diffuseDT, evaporateDT, padWidth, padOffset);
}

__global__ void k_updateTrailMap(
    float4* trailMapFront,
    float4* trailMapBack,
    int* relativeIdxs,
    float diffuseDT,
    float evaporateDT,
    unsigned int padWidth,
    unsigned int padOffset
)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    int idxPad = padOffset + x + y * padWidth;

    // Diffuse
    float4 sum{0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 9; i++) // 3x3 grid
    {
        sum = sum + trailMapBack[idxPad + relativeIdxs[i]];
    }
    float diffusedX = trailMapBack[idxPad].x * (1.0f - diffuseDT) + (sum.x / 9.0f) * (diffuseDT);
    float diffusedY = trailMapBack[idxPad].y * (1.0f - diffuseDT) + (sum.y / 9.0f) * (diffuseDT);
    float diffusedZ = trailMapBack[idxPad].z * (1.0f - diffuseDT) + (sum.z / 9.0f) * (diffuseDT);
    float diffusedW = trailMapBack[idxPad].w * (1.0f - diffuseDT) + (sum.w / 9.0f) * (diffuseDT);

    // Evaporate
    trailMapFront[idxPad].x = max(0.0f, diffusedX - evaporateDT);
    trailMapFront[idxPad].y = max(0.0f, diffusedY - evaporateDT);
    trailMapFront[idxPad].z = max(0.0f, diffusedZ - evaporateDT);
    trailMapFront[idxPad].w = max(0.0f, diffusedW - evaporateDT);
}

void kl_updateAgents(dim3 grid, dim3 block,
    float deltaTime,
    curandState* randomState,
    float4* imgPtr,
    Agent* agents,
    unsigned int nAgents,
    AgentConfig* aConfigs,
    float trailWeightDT,
    unsigned int width,
    unsigned int heigth,
    unsigned int padWidth,
    unsigned int padOffset
)
{
    k_updateAgents<<<grid, block>>>(deltaTime, randomState, imgPtr, agents, nAgents, aConfigs, trailWeightDT, width, heigth, padWidth, padOffset);
}

__global__ void k_updateAgents(
    float deltaTime,
    curandState* randomState,
    float4* trailMap,
    Agent* agents,
    unsigned int nAgents,
    AgentConfig* aConfigs,
    float trailWeightDT,
    unsigned int width,
    unsigned int heigth,
    unsigned int padWidth,
    unsigned int padOffset
)
{
    int agentIdx = blockIdx.x * 32 + threadIdx.x;
    if (agentIdx >= nAgents) return;
    Agent* agent = agents + agentIdx;
    
    AgentConfig ac = aConfigs[agent->speciesIdx];

    // Sense and turn
    float wf = sense(*agent,                    0.0, trailMap, ac.sensorOffsetDst, ac.sensorSize, width, heigth, padWidth, padOffset);
    float wl = sense(*agent,  ac.sensorAngleSpacing, trailMap, ac.sensorOffsetDst, ac.sensorSize, width, heigth, padWidth, padOffset);
    float wr = sense(*agent, -ac.sensorAngleSpacing, trailMap, ac.sensorOffsetDst, ac.sensorSize, width, heigth, padWidth, padOffset);
    
    float randomSteer = curand_uniform(randomState + threadIdx.x);
    //float randomSteer = curand_normal(randomState + threadIdx.x);


    if (wf > wl && wf > wr)
    {
        agent->angle += 0;
    }
    else if (wf < wl && wf < wr)
    {
        agent->angle += (randomSteer - 0.5) * 2 * ac.turnSpeed * deltaTime;
    }
    else if (wr > wl) {
        agent->angle -= randomSteer * ac.turnSpeed * deltaTime;
    }
    else if (wl > wr)
    {
        agent->angle += randomSteer * ac.turnSpeed * deltaTime;
    }

    // Update position
    float2 direction = make_float2(cosf(agent->angle), sinf(agent->angle));
    float2 newPos = make_float2(ac.speed * deltaTime * direction.x + agent->pos.x, ac.speed * deltaTime * direction.y + agent->pos.y);

    if (newPos.x < 0 || newPos.x >= width || newPos.y < 0 || newPos.y >= heigth)
    {
        newPos.x = min(width - 0.01, max(0.0, newPos.x));
        newPos.y = min(heigth - 0.01, max(0.0, newPos.y));
        agent->angle = curand_uniform(randomState + threadIdx.x) * 2 * PI;
    }
    else
    {
        int idxPad = padOffset + __float2uint_rd(newPos.x) + __float2uint_rd(newPos.y) * padWidth;
        float4 value = trailMap[idxPad];
        value.x = min(1.0f, value.x + agent->speciesMask.x * trailWeightDT);
        value.y = min(1.0f, value.y + agent->speciesMask.y * trailWeightDT);
        value.z = min(1.0f, value.z + agent->speciesMask.z * trailWeightDT);
        value.w = min(1.0f, value.w + agent->speciesMask.w * trailWeightDT);
        trailMap[idxPad] = make_float4(value.x, value.y, value.z, value.w);
    }
    
    agent->pos = newPos;
}

__device__ float sense(
    Agent a,
    float sensorAngleOffset,
    float4* trailMap,
    float sensorOffsetDst,
    int sensorSize,
    unsigned int width,
    unsigned int heigth,
    unsigned int padWidth,
    unsigned int padOffset
)
{
    float sensorAngle = a.angle + sensorAngleOffset;
    float2 sensorDir = make_float2(cosf(sensorAngle), sinf(sensorAngle));
    int2 sensorCentre = make_int2(a.pos.x + sensorDir.x * sensorOffsetDst, a.pos.y + sensorDir.y * sensorOffsetDst);
    
    float sum = 0.0f;

    int senseWeightX = a.speciesMask.x * 2 - 1;
    int senseWeightY = a.speciesMask.y * 2 - 1;
    int senseWeightZ = a.speciesMask.z * 2 - 1;
    int senseWeightW = a.speciesMask.w * 2 - 1;

    for (int ox = -sensorSize; ox <= sensorSize; ox++)
    {
        for (int oy = -sensorSize; oy <= sensorSize; oy++)
        {
            int2 pos = make_int2(sensorCentre.x + ox, sensorCentre.y + oy);

            if (pos.x >= 0 && pos.x < width && pos.y >= 0 && pos.y < heigth)
            {
                int idxPad = padOffset + pos.x + pos.y * padWidth;
                sum += 
                      trailMap[idxPad].x * senseWeightX
                    + trailMap[idxPad].y * senseWeightY
                    + trailMap[idxPad].z * senseWeightZ
                    + trailMap[idxPad].w * senseWeightW;
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