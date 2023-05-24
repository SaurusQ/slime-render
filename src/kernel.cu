#include "kernelHeader.cuh"

#include <cuda_runtime.h>

__global__ void kernel(int x, int y, int z, float *res)
{
    res[x + y + z] = 100.0;
}

