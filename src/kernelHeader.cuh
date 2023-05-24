#pragma once

#include <cuda_runtime.h>

__global__ void kernel(int x, int y, int z, float *res);
