#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Kernel 1: Naive matrix multiplication (one thread per output element)
// ============================================================================
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 2: Tiled matrix multiplication with shared memory
// ============================================================================
#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < N && (t * TILE_SIZE + tx) < N)
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && (t * TILE_SIZE + ty) < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Kernel 3: Optimized with vectorized loads (float4)
// ============================================================================
__global__ void matmul_optimized_kernel(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load with bounds checking
        if (row < N && (t * TILE_SIZE + tx) < N)
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && (t * TILE_SIZE + ty) < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Unrolled computation
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Host code
// ============================================================================

void matmul_cpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

void verify_result(const float *gpu_C, const float *cpu_C, int N) {
    float max_error = 0.0f;
    int errors = 0;

    for (int i = 0; i < N * N; i++) {
        float error = fabs(gpu_C[i] - cpu_C[i]);
        if (error > max_error) max_error = error;
        if (error > 1e-3) errors++;
    }

    printf("  Max error: %.6f, Errors > 1e-3: %d\n", max_error, errors);
    if (errors == 0) {
        printf("  ✓ Results match!\n");
    } else {
        printf("  ✗ Results don't match!\n");
    }
}

int main(int argc, char *argv[]) {
    int N = 512;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  CUDA Matrix Multiplication Profiling (N=%d)         ║\n", N);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    size_t bytes = N * N * sizeof(float);
    printf("Matrix size: %d x %d (%.2f MB per matrix)\n\n", N, N, bytes / (1024.0 * 1024.0));

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_cpu = (float*)calloc(N * N, sizeof(float));

    // Initialize matrices
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Setup execution configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    printf("Grid: (%d, %d), Block: (%d, %d)\n\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Warmup
    matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ========================================================================
    // Kernel 1: Naive
    // ========================================================================
    printf("Kernel 1: Naive (global memory only)\n");
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));

    double gflops = (2.0 * N * N * N) / (time_ms / 1000.0) / 1e9;
    printf("  Time: %.3f ms (%.2f GFLOPS)\n", time_ms, gflops);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    verify_result(h_C, h_C_cpu, N);

    // ========================================================================
    // Kernel 2: Tiled with shared memory
    // ========================================================================
    printf("\nKernel 2: Tiled (shared memory)\n");
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));
    memset(h_C_cpu, 0, bytes);

    CHECK_CUDA(cudaEventRecord(start));
    matmul_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    gflops = (2.0 * N * N * N) / (time_ms / 1000.0) / 1e9;
    printf("  Time: %.3f ms (%.2f GFLOPS)\n", time_ms, gflops);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    verify_result(h_C, h_C_cpu, N);

    // ========================================================================
    // Kernel 3: Optimized with unrolling
    // ========================================================================
    printf("\nKernel 3: Optimized (unrolled)\n");
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));
    memset(h_C_cpu, 0, bytes);

    CHECK_CUDA(cudaEventRecord(start));
    matmul_optimized_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    gflops = (2.0 * N * N * N) / (time_ms / 1000.0) / 1e9;
    printf("  Time: %.3f ms (%.2f GFLOPS)\n", time_ms, gflops);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    verify_result(h_C, h_C_cpu, N);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);

    printf("\n✓ Complete!\n\n");

    return 0;
}
