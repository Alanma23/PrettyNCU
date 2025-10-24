# Matrix Multiplication Profiling and Optimization

A hands-on journey through optimizing matrix multiplication, achieving **51× speedup** through progressive optimizations.

## 🚀 Quick Start

```bash
# Compile all versions
gcc -O2 -o matmul matmul.c
gcc -O2 -o matmul_analysis matmul_analysis.c
gcc -O3 -march=native -fopenmp -o matmul_omp matmul_optimized.c

# Run basic benchmarks
./matmul 512

# Run detailed analysis
./matmul_analysis 512

# Run optimized parallel version
./matmul_omp 512
```

## 📁 Files

| File | Description |
|------|-------------|
| **matmul.c** | All three basic versions (naive, reordered, blocked) |
| **matmul_analysis.c** | Detailed performance analysis with explanations |
| **matmul_optimized.c** | Advanced optimizations (unrolling, OpenMP) |
| **ANALYSIS.md** | Visual memory access pattern explanations |
| **JOURNEY_SUMMARY.md** | Complete optimization journey with all details |
| **README.md** | This file |

## 🎯 Results Summary (512×512 matrices)

```
Naive (ijk)              469ms   0.57 GFLOPS   (baseline)
Reordered (ikj)           54ms   5.0 GFLOPS    (9× faster!)  🔥
With -O3 -march=native    23ms  11.6 GFLOPS    (20× faster)
With OpenMP               9ms   29.3 GFLOPS    (51× faster!) 🚀
```

## 🧪 What You'll Learn

### 1. **Why Naive is Slow** (matmul.c:7-16)
The naive `ijk` loop order accesses matrix B column-wise:
```c
for (i) for (j) for (k)
    C[i][j] += A[i][k] * B[k][j];  // B[k][j] has stride N!
```
Result: ~99% cache miss rate, terrible performance

### 2. **The Power of Loop Reordering** (matmul.c:19-32)
Simple reordering to `ikj` makes B access row-wise:
```c
for (i) for (k) {
    double a = A[i][k];
    for (j)
        C[i][j] += a * B[k][j];  // Sequential access!
}
```
Result: <10% cache miss, **9× faster with no extra work!**

### 3. **Blocking for Large Matrices** (matmul.c:35-60)
Work on cache-sized chunks to maximize reuse:
```c
for (i0 = 0; i0 < N; i0 += BLOCK_SIZE)
    for (j0 = 0; j0 < N; j0 += BLOCK_SIZE)
        for (k0 = 0; k0 < N; k0 += BLOCK_SIZE)
            // Process block...
```

### 4. **Compiler Magic**
`-O3 -march=native` enables:
- Auto-vectorization (process 4-8 doubles at once)
- FMA instructions (fused multiply-add)
- Loop unrolling
Result: **2× faster from compiler alone!**

### 5. **Parallelization**
OpenMP directive distributes work across cores:
```c
#pragma omp parallel for collapse(2)
```
Result: **Another 2.5× speedup** on 160 cores

## 📊 Key Metrics

### Memory Access Patterns
| Version | B Access | Cache Misses | Bandwidth |
|---------|----------|--------------|-----------|
| Naive   | Column-wise (stride N) | ~99% | 4.6 GB/s |
| Reordered | Row-wise (sequential) | <10% | 60 GB/s |

### Cache Hierarchy
```
L1: ~32KB,  ~4 cycles   (fast!)
L2: ~256KB, ~12 cycles
L3: ~8MB,   ~40 cycles
RAM: ~GB,   ~200 cycles (avoid!)
```

## 🔬 Try It Yourself

### Test different sizes
```bash
./matmul 128   # Fits in L2 cache
./matmul 512   # Fits in L3 cache
./matmul 1024  # Exceeds L3 cache - blocking helps here!
./matmul 2048  # Watch naive version collapse!
```

### Compile with different flags
```bash
# No optimization
gcc -O0 -o matmul_O0 matmul.c
./matmul_O0 512

# Full optimization
gcc -O3 -march=native -o matmul_O3 matmul.c
./matmul_O3 512

# Compare the difference!
```

### Profile (if you have permissions)
```bash
# Cache statistics
perf stat -e cache-references,cache-misses ./matmul_naive 512
perf stat -e cache-references,cache-misses ./matmul_reordered 512

# CPU cycles and instructions
perf stat -e cycles,instructions ./matmul_naive 512
```

## 📚 Further Reading

### Next Steps
- **SIMD intrinsics**: Use AVX-512 for 8× parallelism per core
- **Prefetching**: `__builtin_prefetch()` to hint cache
- **Mixed precision**: Use FP16/BF16 for 2× more data per cache line
- **GPU**: Use CUDA/ROCm for 1000+ GFLOPS

### Production Libraries
```c
// OpenBLAS (CPU)
#include <cblas.h>
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, 1.0, A, N, B, N, 0.0, C, N);

// Achieves 100-200 GFLOPS on CPU
// Achieves 1000+ GFLOPS on GPU (cuBLAS)
```

## 🎓 Key Takeaways

1. **Memory access patterns matter MORE than raw FLOPS**
   - Cache miss = 50× slower than cache hit
   - 9× speedup from just reordering loops!

2. **Spatial locality is critical**
   - Sequential access → cache hits
   - Strided access → cache misses

3. **Enable compiler optimizations**
   - `-O3 -march=native` gives 2× for free
   - But only if code is cache-friendly first!

4. **Don't reinvent the wheel**
   - Use tuned libraries (OpenBLAS, MKL, cuBLAS)
   - They use assembly + every CPU trick imaginable

5. **Optimization hierarchy**
   - Algorithm > Data structures > Memory access > Compiler > Parallelization > SIMD

## 🤝 Contributing

This is an educational project! Feel free to:
- Add more optimization techniques
- Test on different architectures
- Compare with other algorithms (Strassen, etc.)
- Add GPU implementations

## 📄 License

Educational/Public Domain - use freely!

---

**Made with ❤️ to demonstrate the power of cache-friendly code**
