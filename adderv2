#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

    // Create CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy CPU → GPU
    cudaEventRecord(start);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_h2d = 0.0f;
    cudaEventElapsedTime(&ms_h2d, start, stop);
    std::cout << "H2D transfer time: " << ms_h2d << " ms\n";


    // Define threads/blocks
    int threads = 256;
    int blocks = (N + threads - 1) / threads;//ceiling function

    cudaEventRecord(start);
    add <<<blocks, threads >>> (d_x, d_y, d_sum, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_kernel = 0.0f;
    cudaEventElapsedTime(&ms_kernel, start, stop);
    std::cout << "Kernel time: " << ms_kernel << " ms\n";


    // Copy GPU → CPU
    cudaEventRecord(start);
    cudaMemcpy(sum, d_sum, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_d2h = 0.0f;
    cudaEventElapsedTime(&ms_d2h, start, stop);
    std::cout << "D2H transfer time: " << ms_d2h << " ms\n";

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(sum[i] - 3.0f));

    std::cout << "Max error: " << maxError << std::endl;

    return 0;
}
