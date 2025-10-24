#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Select which version to compile
#ifdef NAIVE
void matmul(double *A, double *B, double *C, int N) {
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
#elif defined(REORDERED)
void matmul(double *A, double *B, double *C, int N) {
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
#elif defined(BLOCKED)
void matmul(double *A, double *B, double *C, int N) {
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
                        for (int j = j0; j < j_max; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
#endif

int main(int argc, char *argv[]) {
    int N = 512;
    if (argc > 1) N = atoi(argv[1]);

    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));

    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    // Run multiple times to get stable perf measurements
    for (int iter = 0; iter < 5; iter++) {
        matmul(A, B, C, N);
    }

    printf("Result: %.6f\n", C[0]);

    free(A);
    free(B);
    free(C);
    return 0;
}
