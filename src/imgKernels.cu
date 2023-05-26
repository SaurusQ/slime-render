#include "imgKernels.cuh"


__global__ void k_imgToPadded(RGB* imgPtr, RGB* imgPadPtr, unsigned int padding, unsigned int width, unsigned int height)
{

}

__global__ void k_convolution(RGB* imgPtr, RGB* imgPadPtr, int* relativeIdxs, float* kernel, unsigned int kernelValues, unsigned int width)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    int idx = x + y * width; 

    float valueR = 0;
    float valueG = 0;
    float valueB = 0;

    for (int i = 0; i < kernelValues; i++)
    {
        valueR += imgPadPtr[relativeIdxs[i]].r * kernel[i];
        valueG += imgPadPtr[relativeIdxs[i]].g * kernel[i];
        valueB += imgPadPtr[relativeIdxs[i]].b * kernel[i];
    }    

    imgPtr[idx].r = valueR / kernelValues;
    imgPtr[idx].g = valueG / kernelValues;
    imgPtr[idx].b = valueB / kernelValues;
    
}