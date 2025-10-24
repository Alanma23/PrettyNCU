#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Naive matrix multiplication: C = A * B
void matmul_naive(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Cache-friendly version: reordered loops (ikj instead of ijk)
void matmul_reordered(double *A, double *B, double *C, int N) {
    // Initialize C to zero
    memset(C, 0, N * N * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a_ik = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// Blocked/tiled version for better cache utilization
void matmul_blocked(double *A, double *B, double *C, int N) {
    int BLOCK_SIZE = 64;  // Tune this based on cache size

    memset(C, 0, N * N * sizeof(double));

    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
                // Process block
                int i_max = (i0 + BLOCK_SIZE < N) ? i0 + BLOCK_SIZE : N;
                int j_max = (j0 + BLOCK_SIZE < N) ? j0 + BLOCK_SIZE : N;
                int k_max = (k0 + BLOCK_SIZE < N) ? k0 + BLOCK_SIZE : N;

                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        double a_ik = A[i * N + k];
                        for (int j = j0; j < j_max; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void benchmark(const char *name, void (*func)(double*, double*, double*, int),
               double *A, double *B, double *C, int N, int warmup, int iterations) {

    // Warmup
    for (int i = 0; i < warmup; i++) {
        func(A, B, C, N);
    }

    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        func(A, B, C, N);
    }
    double end = get_time_ms();

    double avg_time = (end - start) / iterations;
    double gflops = (2.0 * N * N * N) / (avg_time / 1000.0) / 1e9;

    printf("%-20s: %8.2f ms  (%6.2f GFLOPS)\n", name, avg_time, gflops);
}

int main(int argc, char *argv[]) {
    int N = 512;  // Matrix size NxN

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("Matrix multiplication benchmarks (N=%d)\n", N);
    printf("Matrix size: %d x %d (%.1f MB per matrix)\n\n",
           N, N, (N * N * sizeof(double)) / (1024.0 * 1024.0));

    // Allocate matrices
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Initialize with random values
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    // Run benchmarks
    benchmark("Naive (ijk)", matmul_naive, A, B, C, N, 1, 3);
    benchmark("Reordered (ikj)", matmul_reordered, A, B, C, N, 1, 3);
    benchmark("Blocked/Tiled", matmul_blocked, A, B, C, N, 1, 3);

    // Verify correctness (spot check)
    printf("\nSpot check (C[0,0]): %.6f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
