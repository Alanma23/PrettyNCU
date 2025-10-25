// Simple CUDA kernel with uncoalesced memory access pattern
// This kernel intentionally has poor memory access patterns for testing

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define STRIDE 128

// Kernel with uncoalesced memory access (performance issue)
__global__ void badMemoryAccess(float* input, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        // Uncoalesced access - threads access memory with large stride
        output[tid] = input[tid * STRIDE] * 2.0f;
    }
}

int main() {
    float *d_input, *d_output;
    size_t size = N * STRIDE * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, N * sizeof(float));

    // Initialize input
    float* h_input = (float*)malloc(size);
    for (int i = 0; i < N * STRIDE; i++) {
        h_input[i] = (float)i;
    }
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    badMemoryAccess<<<numBlocks, blockSize>>>(d_input, d_output, N);

    // Wait for completion
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    printf("Kernel executed successfully\n");
    return 0;
}
