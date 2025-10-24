# Matrix Multiplication Optimization Journey

## The Problem

Matrix multiplication C = A × B computes: `C[i][j] = Σ(A[i][k] * B[k][j])`

```
     B
   ┌───┐
 A │   │ = C
   └───┘
```

For N×N matrices, we need:
- **2N³ floating-point operations** (N³ multiplies + N³ adds)
- But performance depends heavily on **memory access patterns**!

---

## Version 1: Naive Implementation (ijk order)

```c
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

### Memory Access Pattern

```
Matrix A (row-major):  [a00 a01 a02 ... a0N | a10 a11 ...]
Access A[i][k]:         >>>>>>>>>>>>>>>>>>>    (sequential ✓)

Matrix B (row-major):  [b00 b01 b02 ... b0N | b10 b11 ...]
Access B[k][j]:         ↓       ↓       ↓       (stride N ✗)
                       b0j     b1j     b2j
```

**The Problem:**
- Inner loop increments `k`, accessing B column-wise
- B[k][j] has stride N (4096 bytes for N=512!)
- Cache line is only 64 bytes (8 doubles)
- **Result: ~99% cache miss rate on B!**

**Performance: 0.57 GFLOPS (469ms for 512×512)**

---

## Version 2: Loop Reordering (ikj order)

```c
for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
        double a_ik = A[i][k];  // Load once!
        for (int j = 0; j < N; j++) {
            C[i][j] += a_ik * B[k][j];
        }
    }
}
```

### Memory Access Pattern

```
Matrix A:              [a00 a01 a02 ...]
Access A[i][k]:         >>>>                (once per k, in register!)

Matrix B:              [... | bk0 bk1 bk2 bk3 bk4 ...]
Access B[k][j]:                 >>>>>>>>>>>>>>>>>>>    (sequential ✓)

Matrix C:              [... | ci0 ci1 ci2 ci3 ci4 ...]
Access C[i][j]:                 >>>>>>>>>>>>>>>>>>>    (sequential ✓)
```

**The Fix:**
- Inner loop now increments `j`, accessing B row-wise (sequential!)
- C also accessed sequentially
- Both fit in cache, reused for multiple operations
- **Result: <10% cache miss rate**

**Performance: 4.99 GFLOPS (54ms) = 8.7× FASTER!**

---

## Version 3: Blocking/Tiling

For large matrices that don't fit in cache, we can do even better:

```c
int BLOCK_SIZE = 64;
for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
        for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
            // Process BLOCK_SIZE × BLOCK_SIZE submatrices
            for (int i = i0; i < i0+BLOCK_SIZE; i++) {
                for (int k = k0; k < k0+BLOCK_SIZE; k++) {
                    double a_ik = A[i][k];
                    for (int j = j0; j < j0+BLOCK_SIZE; j++) {
                        C[i][j] += a_ik * B[k][j];
                    }
                }
            }
        }
    }
}
```

**The Idea:**
- Work on blocks that fit in L1/L2 cache
- Reuse data within blocks before eviction
- Better for large matrices (N > 1024)

**Performance: 5.25 GFLOPS (51ms) - slightly better for N=512**

---

## Key Lessons

### 1. **Spatial Locality Matters!**
   - Sequential access → cache hits
   - Strided access → cache misses
   - **9× speedup just from loop reordering!**

### 2. **Memory Hierarchy:**
   ```
   L1 cache:  ~32KB,  ~4 cycles   (fast!)
   L2 cache:  ~256KB, ~12 cycles
   L3 cache:  ~8MB,   ~40 cycles
   RAM:       ~GB,    ~200 cycles (slow!)
   ```

### 3. **Arithmetic Intensity:**
   - Naive: 4.58 GB/s bandwidth, 0.57 GFLOPS → **memory bound**
   - Reordered: 59.91 GB/s bandwidth, 4.99 GFLOPS → **better balance**

### 4. **Optimization Hierarchy:**
   1. ✅ Algorithm choice (O(n³) vs O(n^2.81) Strassen)
   2. ✅ Cache-friendly memory access (this tutorial!)
   3. ✅ SIMD vectorization (use AVX/NEON)
   4. ✅ Parallelization (OpenMP/threads)
   5. ✅ Use tuned libraries (OpenBLAS, MKL, cuBLAS)

---

## Testing Different Sizes

The benefits become even more dramatic with larger matrices:

| N    | Naive    | Reordered | Blocked  | Speedup |
|------|----------|-----------|----------|---------|
| 128  | 7ms      | 1.2ms     | 1.1ms    | 6.4×    |
| 256  | 58ms     | 7.5ms     | 7.0ms    | 7.7×    |
| 512  | 469ms    | 54ms      | 51ms     | 8.7×    |
| 1024 | ~4000ms  | ~450ms    | ~380ms   | 10.5×   |

For N=1024, blocking starts to show significant gains over simple reordering!
