#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__ void add(float* x, float* y, float* sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        sum[i] = x[i] + y[i];
}
int main(void)
{
    int N = 1 << 20; // 1 million

    float* x, * y, * sum;             // CPU pointers
    float* d_x, * d_y, * d_sum;       // GPU pointers

    // Allocate CPU arrays
    x = new float[N];
    y = new float[N];
    sum = new float[N];

    // Initialize CPU arrays
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Allocate GPU memory
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_sum, N * sizeof(float));

    // Copy CPU → GPU
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define threads/blocks
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Launch kernel (THIS is the GPU launch)
    add << <blocks, threads >> > (d_x, d_y, d_sum, N);

    // Copy GPU → CPU
    cudaMemcpy(sum, d_sum, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));

    std::cout << "Max error: " << maxError << std::endl;

    return 0;
}
