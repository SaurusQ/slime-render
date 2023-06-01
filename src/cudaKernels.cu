#include "cudaKernels.cuh"
#include <iostream>

void kl_convolution(dim3 grid, dim3 block,
    RGB* imgPtr,
    RGB* imgPadPtr,
    int* relativeIdxs,
    float* kernel,
    unsigned int kernelValues,
    unsigned int width,
    unsigned int padding
)
{
    k_convolution<<<grid, block>>>(imgPtr, imgPadPtr, relativeIdxs, kernel, kernelValues, width, padding);
}

__global__ void k_convolution(RGB* imgPtr, RGB* imgPadPtr, int* relativeIdxs, float* kernel, unsigned int kernelValues, unsigned int width, unsigned int padding)
{
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32  + threadIdx.y;

    RGB* iPtr = imgPtr;
    RGB* iPadPtr = imgPadPtr;

    int idxPad = (padding * (2 * padding + width) + padding) + x + y * (width + 2 * padding);
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
        valueR += iPadPtr[idxPad + relativeIdxs[i]].r * kernel[i];
        valueG += iPadPtr[idxPad + relativeIdxs[i]].g * kernel[i];
        valueB += iPadPtr[idxPad + relativeIdxs[i]].b * kernel[i];
    }    

    iPtr[idx].r = valueR;
    iPtr[idx].g = valueG;
    iPtr[idx].b = valueB;
    
}