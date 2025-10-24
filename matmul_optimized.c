#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Version 4: Blocked + unrolled + compiler hints
void matmul_optimized(double * __restrict__ A,
                      double * __restrict__ B,
                      double * __restrict__ C,
                      int N) {
    int BLOCK_SIZE = 64;
    memset(C, 0, N * N * sizeof(double));

    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
                int i_max = (i0 + BLOCK_SIZE < N) ? i0 + BLOCK_SIZE : N;
                int j_max = (j0 + BLOCK_SIZE < N) ? j0 + BLOCK_SIZE : N;
                int k_max = (k0 + BLOCK_SIZE < N) ? k0 + BLOCK_SIZE : N;

                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        double a_ik = A[i * N + k];

                        // Unroll inner loop by 4 for better ILP
                        int j = j0;
                        for (; j < j_max - 3; j += 4) {
                            C[i * N + j + 0] += a_ik * B[k * N + j + 0];
                            C[i * N + j + 1] += a_ik * B[k * N + j + 1];
                            C[i * N + j + 2] += a_ik * B[k * N + j + 2];
                            C[i * N + j + 3] += a_ik * B[k * N + j + 3];
                        }

                        // Handle remainder
                        for (; j < j_max; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// Version 5: With OpenMP parallelization
#ifdef _OPENMP
#include <omp.h>
void matmul_parallel(double * __restrict__ A,
                     double * __restrict__ B,
                     double * __restrict__ C,
                     int N) {
    int BLOCK_SIZE = 64;
    memset(C, 0, N * N * sizeof(double));

    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
                int i_max = (i0 + BLOCK_SIZE < N) ? i0 + BLOCK_SIZE : N;
                int j_max = (j0 + BLOCK_SIZE < N) ? j0 + BLOCK_SIZE : N;
                int k_max = (k0 + BLOCK_SIZE < N) ? k0 + BLOCK_SIZE : N;

                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        double a_ik = A[i * N + k];
                        int j = j0;
                        for (; j < j_max - 3; j += 4) {
                            C[i * N + j + 0] += a_ik * B[k * N + j + 0];
                            C[i * N + j + 1] += a_ik * B[k * N + j + 1];
                            C[i * N + j + 2] += a_ik * B[k * N + j + 2];
                            C[i * N + j + 3] += a_ik * B[k * N + j + 3];
                        }
                        for (; j < j_max; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
#endif

void matmul_reordered(double *A, double *B, double *C, int N) {
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

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void benchmark(const char *name, void (*func)(double*, double*, double*, int),
               double *A, double *B, double *C, int N, int iterations) {
    // Warmup
    func(A, B, C, N);

    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        func(A, B, C, N);
    }
    double end = get_time_ms();

    double avg_time = (end - start) / iterations;
    double gflops = (2.0 * N * N * N) / (avg_time / 1000.0) / 1e9;

    printf("%-25s: %8.2f ms  (%6.2f GFLOPS)\n", name, avg_time, gflops);
}

int main(int argc, char *argv[]) {
    int N = 512;
    if (argc > 1) N = atoi(argv[1]);

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║        Advanced Matrix Multiplication (N=%d)              ║\n", N);
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));

    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    int iterations = (N <= 512) ? 3 : 1;

    benchmark("Reordered (ikj)", matmul_reordered, A, B, C, N, iterations);
    benchmark("Optimized (unrolled)", matmul_optimized, A, B, C, N, iterations);

#ifdef _OPENMP
    printf("\n[OpenMP enabled with %d threads]\n", omp_get_max_threads());
    benchmark("Parallel (OpenMP)", matmul_parallel, A, B, C, N, iterations);
#else
    printf("\n[Compile with -fopenmp to enable parallel version]\n");
#endif

    printf("\n💡 Further optimizations possible:\n");
    printf("  - SIMD intrinsics (AVX2/AVX-512)\n");
    printf("  - Prefetching\n");
    printf("  - Better blocking strategies\n");
    printf("  - Use optimized BLAS (OpenBLAS: ~100+ GFLOPS)\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
