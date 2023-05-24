#include "kernelHeader.cuh"
#include "image.hpp"

#include <algorithm>
#include <iostream>

int main()
{
    Image img{};

    while (true)
    {
        std::cout << "Image update" << std::endl;
        img.display();
        img.randomize();
    }

    return 0;
}


/*static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

int main()
{
    cudaSetDevice(0);

    float* test = new float[2000];
    float* testGPU = nullptr;

    std::fill(test, test + 2000, 0);


    //cudaMalloc((void**)&testGPU, 10000 * sizeof(float));
    CHECK(cudaMalloc((void**)&testGPU, 2000 * sizeof(float)));

    CHECK(cudaMemcpy(testGPU, test, 2000 * sizeof(float), cudaMemcpyHostToDevice));

    kernel<<<1, 1>>>(2, 2, 2, testGPU);

    CHECK(cudaMemcpy(test, testGPU, 2000 * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(testGPU);
    delete[] test; 

    return 0;
}*/
