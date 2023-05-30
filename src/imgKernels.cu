#include "imgKernels.cuh"


__global__ void k_convolution(RGB* imgPtr, RGB* imgPadPtr, int* relativeIdxs, float* kernel, unsigned int kernelValues, unsigned int width)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    imgPtr[x + y * width] = RGB{1.0, 1.0, 1.0};
    
/*
    RGB* iPtr = imgPtr;
    RGB* iPadPtr = imgPadPtr;

    int idx = x + y * width; 

    float valueR = 0;
    float valueG = 0;
    float valueB = 0;

    for (int i = 0; i < kernelValues; i++)
    {
        valueR += iPadPtr[relativeIdxs[i]].r * kernel[i];
        valueG += iPadPtr[relativeIdxs[i]].g * kernel[i];
        valueB += iPadPtr[relativeIdxs[i]].b * kernel[i];
    }    

    iPtr[idx].r = valueR / kernelValues;
    iPtr[idx].g = valueG / kernelValues;
    iPtr[idx].b = valueB / kernelValues;
    */  
}