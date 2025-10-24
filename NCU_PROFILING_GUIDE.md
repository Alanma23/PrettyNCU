# NCU (NVIDIA Nsight Compute) Profiling Guide

## Complete Guide to GPU Kernel Profiling with NVIDIA B200

---

## 🎯 Executive Summary

We profiled 3 CUDA matrix multiplication kernels on **NVIDIA B200 GPU**:
- **Naive**: Direct global memory access
- **Tiled**: Shared memory optimization
- **Optimized**: Tiled + loop unrolling

### Key Results

| Kernel | Cycles | Memory Throughput | Compute Throughput | L1 Hit Rate | Occupancy |
|--------|--------|-------------------|-------------------|-------------|-----------|
| **Naive** | 109,396 | 79.75% | 53.15% | **87.36%** ✓ | 69.12% |
| **Tiled** | 76,540 | 71.82% | 56.21% | **0.29%** ✗ | 67.32% |
| **Optimized** | 76,220 | 72.07% | 56.41% | **0.29%** ✗ | 66.76% |

**Surprising Finding**: Tiled version is **30% faster** (fewer cycles) but has **terrible L1 hit rate**!

---

## 🔧 NCU vs NSYS: Key Differences

| Feature | NSYS | NCU |
|---------|------|-----|
| **Purpose** | System-level timeline | Kernel-level detailed metrics |
| **Target** | CPU + GPU | GPU kernels only |
| **Overhead** | ~3-10× | **~100-1000×** (runs kernels multiple times) |
| **Output** | Timeline, process/thread tracking | Detailed performance counters |
| **Use Case** | Find bottlenecks, CPU-GPU overlap | Optimize specific kernels |

**When to Use Each:**
- **NSYS**: Profile entire application, find hotspots
- **NCU**: Deep dive into specific GPU kernels

---

## 📊 NCU Key Metrics Explained

### 1. GPU Speed of Light (SOL)

**What it shows**: How close you are to peak hardware limits

```
Memory Throughput:        79.75%  ← Memory-bound!
Compute (SM) Throughput:  53.15%  ← Not compute-bound
```

**Interpretation:**
- **Memory > 70%**: Memory-bound kernel
  - Fix: Reduce memory traffic, improve cache hits
- **Compute > 70%**: Compute-bound kernel
  - Fix: Optimize instruction mix, reduce register pressure
- **Both high**: Well-balanced kernel

**Our Kernels**: All are memory-bound (70-80% memory throughput)

---

### 2. Memory Workload Analysis

#### L1/TEX Cache Hit Rate
- **Naive kernel**: 87.36% ✓ (Good!)
- **Tiled kernel**: 0.29% ✗ (Terrible!)

**Why the paradox?**
- Shared memory bypasses L1 cache
- L1 hit rate only counts global memory accesses
- Tiled kernel uses shared memory → fewer L1 accesses → low "hit rate"
- **This is actually correct behavior!**

#### L2 Cache Hit Rate
- **All kernels**: ~88-90% (Good)
- Data reused across SMs
- Global memory traffic reduced

#### DRAM Throughput
- **All kernels**: ~0.3-0.5% (Excellent!)
- Almost everything served from caches
- B200's massive cache hierarchy working well

---

### 3. Occupancy

```
Theoretical Occupancy:  100%
Achieved Occupancy:     69.12%
```

**What is Occupancy?**
- % of max concurrent warps actually active
- Higher = better latency hiding

**Our Results:**
- 67-69% occupancy (pretty good!)
- Limited by registers (32 per thread)
- Block size: 256 threads = 8 warps

**Occupancy Limiters:**
```
Block Limit Registers:    8 blocks  ← Limiting factor!
Block Limit Shared Mem:  21 blocks  (for tiled)
Block Limit Warps:        8 blocks
```

**To Improve:**
- Reduce register usage (currently 32/thread)
- Increase threads per block (currently 256)
- But: May hurt performance in other ways!

---

### 4. Instruction Statistics

```
Executed IPC Active:     1.40-1.65 inst/cycle
Issued IPC Active:       1.40-1.65 inst/cycle
Issue Slots Busy:        35-41%
```

**What it means:**
- B200 can issue multiple instructions per cycle
- We're issuing 1.4-1.65 (decent)
- Issue slots only 35-41% busy (room for improvement)

**Tiled/Optimized slightly better:**
- 1.65 IPC vs 1.40 IPC for naive
- Better instruction-level parallelism

---

### 5. Launch Configuration

```
Grid Size:      (32, 32, 1) = 1,024 blocks
Block Size:     (16, 16, 1) = 256 threads/block
Total Threads:  262,144 threads
Waves Per SM:   0.86 waves
```

**For 512×512 matrix:**
- Each thread computes 1 output element
- 512×512 = 262,144 outputs = 262,144 threads needed
- 262,144 / 256 = 1,024 blocks
- 1,024 blocks / 148 SMs = ~6.9 blocks/SM
- 6.9 blocks × 8 warps/block = 55 warps/SM

**B200 can handle:**
- 64 warps/SM theoretically
- We're using ~55 warps/SM (good utilization)

---

## 🎓 Key Insights from Our Profiling

### 1. **Naive Kernel is Memory-Bound**
- 79% memory throughput
- High L1 hit rate (87%) but still limited by memory
- Each thread reads entire row of A and column of B
- No data reuse between threads

### 2. **Tiled Kernel Reduces Cycles by 30%**
- 109K cycles → 76K cycles
- Shared memory enables block-level reuse
- Lower L1 hit rate is OK (using shared memory instead)
- Better instruction throughput (1.65 vs 1.40 IPC)

### 3. **Unrolling Provides Minimal Benefit**
- Only 320 cycle improvement (76,540 → 76,220)
- Compiler may already be unrolling
- ILP benefit is marginal

### 4. **All Kernels Have Good Cache Behavior**
- L2 hit rate: ~88-90%
- DRAM throughput: <0.5%
- B200's cache hierarchy is working well

---

## 🚀 How to Use NCU

### Basic Profiling

```bash
# Profile with default metrics
ncu --export profile --force-overwrite ./my_cuda_app

# Profile with full metric set (slow!)
ncu --set full --export profile_full --force-overwrite ./my_cuda_app

# Profile specific kernel
ncu --kernel-name my_kernel --export profile --force-overwrite ./my_cuda_app
```

### Export Options

```bash
# Export to CSV
ncu --import profile.ncu-rep --csv --page details > details.csv

# Export raw metrics
ncu --import profile.ncu-rep --csv --page raw > raw_metrics.csv

# Export to JSON
ncu --import profile.ncu-rep --export profile --format json
```

### Interactive Analysis

```bash
# Open in GUI (if available)
ncu-ui profile.ncu-rep

# Or transfer to local machine and open with Nsight Compute
```

---

## 📈 Profiling Best Practices

### 1. **Profile After System-Level Analysis**
```
NSYS first → Find hotspot kernels → NCU on those kernels
```

### 2. **Use Appropriate Metric Sets**
- `--set full`: Comprehensive but slow (39 passes)
- `--set detailed`: Good balance
- `--section LaunchStats`: Just launch info
- Custom: `--metrics metric1,metric2,...`

### 3. **Understand the Overhead**
```
Normal run:      0.059 ms/kernel
NCU profiling:   8,507 ms/kernel  (144,000× overhead!)
```

**Why so slow?**
- NCU runs kernel multiple times (39 passes)
- Different instrumentation each pass
- Collects hundreds of performance counters
- **Use small problem sizes for profiling!**

### 4. **Profile Multiple Kernel Versions**
```bash
# Profile all versions at once
ncu --set full --export all_kernels --force-overwrite ./matmul_cuda
```

---

## 🔍 Interpreting NCU Results

### Memory-Bound Kernel

```
Memory Throughput:  79%   ← High!
Compute Throughput: 53%   ← Lower
DRAM Throughput:    0.3%  ← Good (caches working)
L2 Hit Rate:        90%   ← Good
```

**Actions:**
1. ✅ Check cache hit rates (ours are good)
2. ✅ Use shared memory for data reuse (tiled version does this)
3. ✅ Coalesce global memory accesses
4. ❌ Don't focus on compute optimizations

### Compute-Bound Kernel

```
Memory Throughput:  40%   ← Lower
Compute Throughput: 85%   ← High!
```

**Actions:**
1. Optimize instruction mix
2. Use fused multiply-add (FMA)
3. Reduce register spills
4. Improve instruction-level parallelism

### Occupancy-Limited Kernel

```
Theoretical Occupancy: 100%
Achieved Occupancy:     30%   ← Low!
Block Limit Registers:   2    ← Bottleneck
```

**Actions:**
1. Reduce register usage
2. Decrease shared memory
3. Increase threads per block
4. Launch more blocks

---

## 💡 Optimization Strategies by Metric

| If this is high... | Then... | Actions |
|-------------------|---------|---------|
| **Memory Throughput > 70%** | Memory-bound | Use shared memory, improve coalescing, reduce traffic |
| **Compute Throughput > 70%** | Compute-bound | Optimize instructions, use FMA, reduce divergence |
| **Low L1 Hit Rate (<70%)** | Poor locality | Use shared memory, improve access patterns, tile |
| **Low L2 Hit Rate (<70%)** | Cache thrashing | Adjust block size, improve temporal locality |
| **Low Occupancy (<50%)** | Under-utilized | Reduce registers/shared mem, increase block size |
| **Low IPC (<1.0)** | Poor ILP | Unroll loops, reduce dependencies, improve scheduling |

---

## 📊 Our Results Summary

### Performance Comparison

```
Kernel          Cycles      Speedup    Memory-Bound?  Occupancy
──────────────────────────────────────────────────────────────
Naive           109,396     1.0×       Yes (80%)      69%
Tiled            76,540     1.43×      Yes (72%)      67%
Optimized        76,220     1.43×      Yes (72%)      67%
```

### Key Findings

1. **Tiling helps significantly** (30% fewer cycles)
2. **All kernels are memory-bound** (70-80% throughput)
3. **Good cache behavior** (L2 ~90%, DRAM <1%)
4. **Occupancy is reasonable** (~67-69%)
5. **Further optimization possible**:
   - Could achieve 4,500+ GFLOPS with cuBLAS
   - Our best: ~4,583 GFLOPS (naive, unprofilied)
   - cuBLAS on B200: likely 10,000+ GFLOPS

---

## 🎯 Next Steps for Further Optimization

### 1. **Use Tensor Cores**
```cuda
// Use WMMA (Warp Matrix Multiply-Accumulate)
#include <mma.h>
using namespace nvcuda::wmma;

// 16x16x16 matrix multiply on tensor cores
wmma::fragment<...> a, b, c;
wmma::load_matrix_sync(a, ...);
wmma::load_matrix_sync(b, ...);
wmma::mma_sync(c, a, b, c);
```

**Expected**: 50-100× faster than our current implementation

### 2. **Tune Block/Tile Sizes**
```bash
# Profile different configurations
for TILE in 16 32 64 128; do
  ncu --set full --export tile_$TILE ./matmul_cuda_tiled_$TILE
done
```

### 3. **Use cuBLAS**
```cuda
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// Highly optimized GEMM
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
```

**Expected**: >10,000 GFLOPS on B200

---

## 📁 Files Created

### CUDA Source
- **matmul_cuda.cu** - 3 kernel implementations

### Profiling Data
- **ncu_matmul_profile.ncu-rep** (5.4MB) - Binary NCU report
- **ncu_details.csv** (92KB) - Detailed metrics CSV
- **ncu_raw_metrics.csv** (232KB) - Raw counters CSV

### Analysis Scripts
- **ncu_comprehensive_analysis.py** - Full analysis tool
- **analyze_ncu.py** - Basic analysis

### Documentation
- **NCU_PROFILING_GUIDE.md** - This file!

---

## 🏆 Key Takeaways

1. **NCU is for kernel-level optimization**
   - Provides hundreds of performance counters
   - Massive profiling overhead (100-1000×)
   - Essential for understanding GPU bottlenecks

2. **Memory-bound vs Compute-bound matters**
   - Our kernels: All memory-bound
   - Optimization: Focus on memory access, not compute

3. **Shared memory is powerful**
   - 30% cycle reduction from tiling
   - Reduces global memory traffic
   - Enables block-level data reuse

4. **B200 has excellent cache hierarchy**
   - L2 hit rate: ~90%
   - DRAM throughput: <1%
   - Massive caches working well

5. **There's always room for improvement**
   - Current: ~4,583 GFLOPS
   - cuBLAS: likely 10,000+ GFLOPS
   - Tensor cores: potentially 50,000+ GFLOPS (FP16)

---

## 📚 Resources

- [NCU Documentation](https://docs.nvidia.com/nsight-compute/)
- [NCU CLI Reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
- [GPU Performance Analysis](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)

---

*Complete GPU profiling journey with NCU on NVIDIA B200*
