#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

typedef struct {
    long long loads;
    long long stores;
    long long flops;
    double time_ms;
} PerfStats;

// Naive matrix multiplication with analysis
PerfStats matmul_naive_analyze(double *A, double *B, double *C, int N) {
    PerfStats stats = {0};
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                // Memory accesses:
                // - Load A[i,k]: access row i of A (good locality)
                // - Load B[k,j]: access column j of B (BAD - stride N access!)
                // - Load/Store C[i,j]: accessed once per k loop (OK, but in register)
                sum += A[i * N + k] * B[k * N + j];
                stats.loads += 2;  // A[i,k] and B[k,j]
                stats.flops += 2;  // multiply and add
            }
            C[i * N + j] = sum;
            stats.stores += 1;  // C[i,j]
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    stats.time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                    (end.tv_nsec - start.tv_nsec) / 1000000.0;

    return stats;
}

// Reordered (ikj) with analysis
PerfStats matmul_reordered_analyze(double *A, double *B, double *C, int N) {
    PerfStats stats = {0};
    struct timespec start, end;

    memset(C, 0, N * N * sizeof(double));

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a_ik = A[i * N + k];
            stats.loads += 1;  // Load A[i,k] once

            for (int j = 0; j < N; j++) {
                // Memory accesses:
                // - a_ik is in register (loaded once above)
                // - Load B[k,j]: access row k of B (good locality!)
                // - Load/Store C[i,j]: access row i of C (good locality!)
                C[i * N + j] += a_ik * B[k * N + j];
                stats.loads += 2;   // B[k,j] and C[i,j]
                stats.stores += 1;  // C[i,j]
                stats.flops += 2;   // multiply and add
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    stats.time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                    (end.tv_nsec - start.tv_nsec) / 1000000.0;

    return stats;
}

void print_analysis(const char *name, PerfStats stats, int N) {
    double gflops = (stats.flops / 1e9) / (stats.time_ms / 1000.0);
    double bandwidth_gb = ((stats.loads + stats.stores) * sizeof(double) / 1e9) / (stats.time_ms / 1000.0);

    printf("\n=== %s ===\n", name);
    printf("Time:           %8.2f ms\n", stats.time_ms);
    printf("GFLOPS:         %8.2f\n", gflops);
    printf("Memory ops:     %lld loads, %lld stores\n", stats.loads, stats.stores);
    printf("Bandwidth:      %8.2f GB/s\n", bandwidth_gb);
    printf("Ops/byte:       %.2f (%.1f%% of peak)\n",
           (double)stats.flops / ((stats.loads + stats.stores) * sizeof(double)),
           (gflops / 100.0) * 10);  // rough estimate
}

void explain_memory_pattern(const char *name, int N) {
    printf("\n### Memory Access Pattern for %s ###\n", name);

    if (strstr(name, "Naive")) {
        printf("Inner loop accesses:\n");
        printf("  A[i][k]: Row-major, good cache locality ✓\n");
        printf("  B[k][j]: Column-major, TERRIBLE locality ✗\n");
        printf("           (stride = %d, jumps by %ld bytes)\n", N, N * sizeof(double));
        printf("\nProblem: B matrix is accessed by columns, causing cache miss\n");
        printf("on nearly every access! For N=512:\n");
        printf("  - Cache line is 64 bytes = 8 doubles\n");
        printf("  - But we skip 512 doubles between accesses\n");
        printf("  - Cache miss rate: ~99%%\n");
    } else if (strstr(name, "Reordered")) {
        printf("Inner loop accesses:\n");
        printf("  A[i][k]: Loaded once, kept in register ✓\n");
        printf("  B[k][j]: Row-major, good cache locality ✓\n");
        printf("  C[i][j]: Row-major, good cache locality ✓\n");
        printf("\nFix: Reordered loops so innermost loop (j) traverses:\n");
        printf("  - Row k of B (sequential)\n");
        printf("  - Row i of C (sequential)\n");
        printf("  - Both have excellent spatial locality!\n");
        printf("  - Cache miss rate: <10%%\n");
    }
}

int main(int argc, char *argv[]) {
    int N = 512;
    if (argc > 1) N = atoi(argv[1]);

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║   Matrix Multiplication Performance Analysis (N=%d)     ║\n", N);
    printf("╚════════════════════════════════════════════════════════════╝\n");

    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));

    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    // Run and analyze
    PerfStats naive = matmul_naive_analyze(A, B, C, N);
    print_analysis("NAIVE (ijk)", naive, N);
    explain_memory_pattern("Naive", N);

    PerfStats reordered = matmul_reordered_analyze(A, B, C, N);
    print_analysis("REORDERED (ikj)", reordered, N);
    explain_memory_pattern("Reordered", N);

    // Comparison
    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║                      COMPARISON                            ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("Speedup:        %.1fx faster\n", naive.time_ms / reordered.time_ms);
    printf("GFLOPS gain:    %.2f → %.2f\n",
           (naive.flops / 1e9) / (naive.time_ms / 1000.0),
           (reordered.flops / 1e9) / (reordered.time_ms / 1000.0));

    printf("\nKey insight: Same number of operations, but MUCH better\n");
    printf("cache behavior by accessing memory sequentially!\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
