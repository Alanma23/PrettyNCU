#!/usr/bin/env python3
"""
NCU Actionable Optimizer - Code-Level Edition
Analyzes NCU profiling output and generates precise, code-level optimization tasks.

NEW: Points to exact code sections, provides detailed fix outlines with before/after examples.
Based on Simon Boehm and Pranjal Shankhdhar's proven methodologies.
"""

import csv
import sys
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class CodeFix:
    """Detailed code-level fix with before/after examples"""
    section: str  # e.g., "Main kernel loop", "Global memory load", "Thread indexing"
    location: str  # Where to look: "Lines 45-60" or "Search for pattern: for(int k=0...)"
    problem_pattern: str  # What the problematic code looks like
    fix_pattern: str  # What the fixed code should look like
    explanation: str  # Why this fixes the issue
    assembly_check: str  # What to verify in assembly (PTX/SASS)


@dataclass
class KernelMetrics:
    """Key metrics for a kernel"""
    kernel_id: str
    kernel_name: str

    # Memory metrics
    memory_throughput: float = 0.0
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    dram_throughput: float = 0.0
    global_load_efficiency: float = 0.0

    # Compute metrics
    compute_throughput: float = 0.0
    ipc_active: float = 0.0
    ipc_elapsed: float = 0.0

    # Occupancy metrics
    achieved_occupancy: float = 0.0
    theoretical_occupancy: float = 0.0
    occupancy_limiter: str = "Unknown"
    registers_per_thread: int = 0
    shared_mem_per_block: int = 0
    block_size: str = ""

    # Warp stall metrics
    stall_not_selected: float = 0.0
    stall_memory_throttle: float = 0.0
    stall_memory_dependency: float = 0.0
    stall_execution_dependency: float = 0.0
    stall_barrier: float = 0.0

    # Performance
    duration_us: float = 0.0

    # NCU insights
    ncu_insights: List[Dict] = field(default_factory=list)


@dataclass
class OptimizationTask:
    """Actionable optimization task with code-level details"""
    priority: int  # 1=critical, 2=high, 3=medium, 4=low
    category: str
    title: str
    description: str
    reasoning: str
    expected_impact: str

    # NEW: Code-level guidance
    code_fixes: List[CodeFix]
    investigation_steps: List[str]  # How to locate the exact issue
    verification_steps: List[str]  # How to verify the fix worked

    related_metrics: List[str]
    references: List[str]


class NCUActionableOptimizer:
    """Analyzes NCU data and generates code-level optimization tasks"""

    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.kernels: Dict[str, KernelMetrics] = {}
        self.tasks: List[OptimizationTask] = []

    def load_ncu_data(self):
        """Load and parse NCU CSV data"""
        print(f"Loading NCU data from {self.csv_file}...")

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                kernel_id = row.get('ID', '')
                kernel_name = row.get('Kernel Name', '')
                metric_name = row.get('Metric Name', '')
                metric_value = row.get('Metric Value', '')

                if not kernel_id:
                    continue

                if kernel_id not in self.kernels:
                    self.kernels[kernel_id] = KernelMetrics(
                        kernel_id=kernel_id,
                        kernel_name=kernel_name.split('(')[0]
                    )

                kernel = self.kernels[kernel_id]
                self._parse_metric(kernel, metric_name, metric_value)

                # Collect NCU insights
                if row.get('Rule Type') == 'OPT' and row.get('Rule Description'):
                    kernel.ncu_insights.append({
                        'rule': row.get('Rule Name', ''),
                        'description': row.get('Rule Description', ''),
                        'speedup': row.get('Estimated Speedup', 'N/A')
                    })

        print(f"Loaded {len(self.kernels)} kernel(s)")

    def _parse_metric(self, kernel: KernelMetrics, name: str, value: str):
        """Parse a metric value into the kernel metrics"""
        try:
            val = value.replace(',', '').replace('%', '')
            if not val or val == 'N/A':
                return
            val_float = float(val)

            # Memory metrics
            if 'Memory Throughput' in name or 'dram__throughput.avg.pct_of_peak' in name:
                kernel.memory_throughput = val_float
            elif 'L1/TEX Hit Rate' in name or 'l1tex__t_sector_hit_rate' in name:
                kernel.l1_hit_rate = val_float
            elif 'L2 Hit Rate' in name or 'lts__t_sector_hit_rate' in name:
                kernel.l2_hit_rate = val_float
            elif 'global_load_efficiency' in name or 'Global Load Efficiency' in name:
                kernel.global_load_efficiency = val_float

            # Compute metrics
            elif 'Compute (SM) Throughput' in name or 'sm__throughput.avg.pct_of_peak' in name:
                kernel.compute_throughput = val_float
            elif 'Executed Ipc Active' in name:
                kernel.ipc_active = val_float
            elif 'Executed Ipc Elapsed' in name:
                kernel.ipc_elapsed = val_float

            # Occupancy
            elif 'Achieved Occupancy' in name or 'sm__warps_active.avg.pct_of_peak' in name:
                kernel.achieved_occupancy = val_float
            elif 'Theoretical Occupancy' in name:
                kernel.theoretical_occupancy = val_float
            elif 'Registers Per Thread' in name:
                kernel.registers_per_thread = int(val_float)
            elif 'Shared Memory Per Block' in name or 'Static Shared Memory Per Block' in name:
                kernel.shared_mem_per_block = int(val_float)
            elif 'Block Size' in name:
                kernel.block_size = value

            # Warp stalls
            elif 'Stall Not Selected' in name:
                kernel.stall_not_selected = val_float
            elif 'Stall MIO Throttle' in name or 'stall_memory_throttle' in name:
                kernel.stall_memory_throttle = val_float
            elif 'Stall Memory Dependency' in name:
                kernel.stall_memory_dependency = val_float
            elif 'Stall Execution Dependency' in name:
                kernel.stall_execution_dependency = val_float
            elif 'Stall Barrier' in name:
                kernel.stall_barrier = val_float

            # Performance
            elif 'Duration' in name or 'gpu__time_duration' in name:
                kernel.duration_us = val_float

        except (ValueError, AttributeError):
            pass

    def analyze_kernel(self, kernel: KernelMetrics) -> List[OptimizationTask]:
        """Analyze a kernel and generate code-level optimization tasks"""
        tasks = []

        print(f"\n{'='*90}")
        print(f"Analyzing Kernel: {kernel.kernel_name}")
        print(f"{'='*90}")

        # Memory subsystem analysis
        tasks.extend(self._analyze_memory_coalescing(kernel))
        tasks.extend(self._analyze_memory_bottleneck(kernel))

        # Cache analysis
        tasks.extend(self._analyze_cache_efficiency(kernel))

        # Occupancy analysis
        tasks.extend(self._analyze_occupancy(kernel))

        # Warp stall analysis
        tasks.extend(self._analyze_warp_stalls(kernel))

        # Roofline guidance
        tasks.extend(self._analyze_roofline(kernel))

        return tasks

    def _analyze_memory_coalescing(self, k: KernelMetrics) -> List[OptimizationTask]:
        """Detect and fix memory coalescing issues with exact code guidance"""
        tasks = []

        if k.global_load_efficiency > 0 and k.global_load_efficiency < 80:
            # Bad coalescing detected
            tasks.append(OptimizationTask(
                priority=1,
                category="Memory Coalescing",
                title=f"CRITICAL: Uncoalesced Memory Access ({k.global_load_efficiency:.1f}% efficiency)",
                description=f"Global memory loads are only {k.global_load_efficiency:.1f}% efficient. "
                           f"Threads are NOT accessing consecutive memory addresses.",
                reasoning="Memory coalescing requires threads with consecutive threadIds to access "
                         "consecutive memory addresses. Current pattern causes 4-32 separate transactions "
                         "instead of 1 coalesced transaction. Simon Boehm's Kernel 2 shows 6x speedup from fixing this.",
                expected_impact="MAJOR: 3-10x speedup (Kernel 2: 309 → 1,986 GFLOPs)",

                code_fixes=[
                    CodeFix(
                        section="Thread-to-Data Mapping",
                        location="Search for: Row/column index calculation at kernel start (typically lines 1-10 of kernel)",
                        problem_pattern="""// WRONG: Row-major thread mapping
__global__ void kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // ❌ BAD
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread 0 accesses A[0*N + 0]
    // Thread 1 accesses A[0*N + 1]  ← NOT CONSECUTIVE!
    // Threads access stride of N elements
    float val = A[row * N + col];  // ❌ Strided access
}""",
                        fix_pattern="""// CORRECT: Column-major thread mapping
__global__ void kernel(float* A, float* B, float* C, int N) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // ✅ GOOD
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread 0 accesses A[0*N + 0]
    // Thread 1 accesses A[1*N + 0]  ← Consecutive in memory!
    // Threads access consecutive elements
    float val = A[row * N + col];  // ✅ Coalesced
}""",
                        explanation="Swap row/col calculation. If matrix is row-major (C[row][col] = C[row*N+col]), "
                                   "then consecutive threads should have consecutive 'row' values, not 'col' values. "
                                   "This ensures threadIdx.x increments the innermost memory dimension.",
                        assembly_check="PTX: Look for LDG.E.128 (good) vs LDG.E.32 (bad). "
                                      "Use: nvcc -ptx -o kernel.ptx kernel.cu && grep LDG kernel.ptx"
                    ),
                    CodeFix(
                        section="Array Access Pattern",
                        location="Search for: A[i*N + j] or B[row*width + col] patterns",
                        problem_pattern="""// WRONG: Inner loop iterates over strided dimension
for (int k = 0; k < K; k++) {
    // If A is row-major, this accesses A[row][k] with stride K
    float a = A[row * K + k];  // ❌ Stride-K access
    float b = B[k * N + col];
    sum += a * b;
}""",
                        fix_pattern="""// CORRECT: Ensure inner dimension is coalesced
for (int k = 0; k < K; k++) {
    // Access pattern depends on thread mapping
    // If thread row = threadIdx.x (consecutive), then:
    float a = A[row * K + k];  // ✅ A[row] consecutive
    // Or transpose A in memory/shared memory
    float b = B[k * N + col];
    sum += a * b;
}

// OR use shared memory to fix both:
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];
// Load with coalesced pattern, compute from shared memory""",
                        explanation="The innermost varying index (threadIdx.x) must match the fastest-changing "
                                   "memory dimension. For row-major C arrays, that's the column. "
                                   "Either fix thread mapping OR use shared memory to rearrange data.",
                        assembly_check="SASS: Look for LDG.E.128 instead of multiple LDG.E.32 instructions. "
                                      "Use: cuobjdump -sass kernel.cubin | grep LDG"
                    ),
                    CodeFix(
                        section="Vectorized Access (Follow-up)",
                        location="After fixing basic coalescing, look for: Load instructions",
                        problem_pattern="""// SUBOPTIMAL: Loading one float at a time
float a0 = A[row * K + k];
float a1 = A[row * K + k + 1];
float a2 = A[row * K + k + 2];
float a3 = A[row * K + k + 3];""",
                        fix_pattern="""// OPTIMAL: Vectorized load (4x floats at once)
float4 avec = reinterpret_cast<float4*>(&A[row * K + k])[0];
float a0 = avec.x;
float a1 = avec.y;
float a2 = avec.z;
float a3 = avec.w;

// Or if loading to shared memory:
reinterpret_cast<float4*>(&As[ty][tx*4])[0] =
    reinterpret_cast<float4*>(&A[...][tx*4])[0];""",
                        explanation="Once basic coalescing is fixed, vectorize with float4 (128-bit loads). "
                                   "Requires 16-byte alignment. Kernel 6 shows 14% speedup from this.",
                        assembly_check="Look for LDG.E.128 with vec4 in PTX, or LDS.128 for shared memory"
                    )
                ],

                investigation_steps=[
                    "1. FIND THREAD MAPPING:",
                    "   grep -n 'threadIdx' your_kernel.cu",
                    "   Look for row = ... threadIdx.x or threadIdx.y",
                    "",
                    "2. IDENTIFY ACCESS PATTERN:",
                    "   Find A[...] accesses in main loop",
                    "   Check if threadIdx.x varies the innermost array index",
                    "",
                    "3. CHECK MATRIX LAYOUT:",
                    "   Is it row-major C[row*N+col] or column-major?",
                    "   Consecutive threads should access consecutive memory",
                    "",
                    "4. VERIFY IN ASSEMBLY:",
                    "   nvcc -ptx -o kernel.ptx kernel.cu",
                    "   grep 'ld.global' kernel.ptx",
                    "   Should see .v4 or .128 suffixes for vectorization"
                ],

                verification_steps=[
                    "1. PROFILE AGAIN:",
                    "   ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./app",
                    "   Should see >90% (was {:.1f}%)".format(k.global_load_efficiency),
                    "",
                    "2. CHECK ASSEMBLY:",
                    "   cuobjdump -sass kernel.cubin | grep 'LDG.*128'",
                    "   Should see LDG.E.128 (128-bit loads)",
                    "",
                    "3. MEASURE SPEEDUP:",
                    "   Time kernel before and after",
                    "   Expect 3-6x speedup from this fix alone",
                    "",
                    "4. MEMORY BANDWIDTH:",
                    "   ncu --metrics dram__bytes_read.sum ./app",
                    "   Should decrease (fewer transactions)"
                ],

                related_metrics=[
                    f"Global Load Efficiency: {k.global_load_efficiency:.1f}% (target: >90%)",
                    f"Memory Throughput: {k.memory_throughput:.1f}%"
                ],

                references=[
                    "Simon Boehm Kernel 2: Memory Coalescing (309 → 1,986 GFLOPs)",
                    "CUDA C++ Programming Guide: Section 5.3.2 Device Memory Accesses",
                    "Example: https://siboehm.com/articles/22/CUDA-MMM (see Kernel 2 diff)"
                ]
            ))

        return tasks

    def _analyze_memory_bottleneck(self, k: KernelMetrics) -> List[OptimizationTask]:
        """Detect memory bottleneck and provide shared memory solution"""
        tasks = []

        if k.memory_throughput > 70 and k.compute_throughput < 50:
            tasks.append(OptimizationTask(
                priority=1,
                category="Memory Bandwidth",
                title=f"Memory-Bound Kernel: Add Shared Memory Caching",
                description=f"Memory throughput {k.memory_throughput:.1f}% >> Compute {k.compute_throughput:.1f}%. "
                           f"GPU spends most time waiting for DRAM.",
                reasoning="Memory bandwidth is the bottleneck. Repeatedly loading same data from DRAM. "
                         "Shared memory (12,080 GB/s) is 16x faster than global memory (750 GB/s). "
                         "Simon's Kernel 3 shows 1.5x speedup from caching tiles in shared memory.",
                expected_impact="MAJOR: 1.5-3x speedup from shared memory + blocktiling",

                code_fixes=[
                    CodeFix(
                        section="Shared Memory Declaration",
                        location="Add at top of kernel, after __global__ declaration",
                        problem_pattern="""__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = ...;
    int col = ...;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        // ❌ Loading from global memory every time
        sum += A[row*N + k] * B[k*N + col];
    }
    C[row*N + col] = sum;
}""",
                        fix_pattern="""// Add shared memory tile buffers
#define TILE_SIZE 32  // Tune: 16, 32, 64

__global__ void matmul(float* A, float* B, float* C, int N) {
    // ✅ Shared memory caches (16KB - 48KB per SM)
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Tile the loop
    for (int t = 0; t < N; t += TILE_SIZE) {
        // Load tile into shared memory
        As[threadIdx.y][threadIdx.x] = A[row*N + (t + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y)*N + col];
        __syncthreads();  // ⚠️ REQUIRED

        // Compute using shared memory (fast!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();  // ⚠️ REQUIRED before next tile
    }

    C[row*N + col] = sum;
}""",
                        explanation="Load matrix tiles into shared memory once, reuse TILE_SIZE times. "
                                   "Each element loaded once from DRAM, used TILE_SIZE times. "
                                   "Arithmetic intensity increases from 1 to TILE_SIZE FLOPs/byte.",
                        assembly_check="Look for STS (store to shared) and LDS (load from shared) instructions. "
                                      "Should see __syncthreads() as BAR.SYNC in assembly."
                    ),
                    CodeFix(
                        section="Synchronization Barriers",
                        location="After each shared memory load AND before reuse",
                        problem_pattern="""// ❌ WRONG: Missing synchronization
As[ty][tx] = A[...];
Bs[ty][tx] = B[...];
// Missing __syncthreads()!
for (int k = 0; k < TILE; k++) {
    sum += As[ty][k] * Bs[k][tx];  // Race condition!
}""",
                        fix_pattern="""// ✅ CORRECT: Proper synchronization
As[ty][tx] = A[...];
Bs[ty][tx] = B[...];
__syncthreads();  // ✅ Wait for all threads to finish loading

for (int k = 0; k < TILE; k++) {
    sum += As[ty][k] * Bs[k][tx];  // Safe to read
}
__syncthreads();  // ✅ Wait before loading next tile""",
                        explanation="All threads in block must finish loading before any thread reads. "
                                   "Missing __syncthreads() causes race conditions and wrong results.",
                        assembly_check="Should see BAR.SYNC instructions. Count should match __syncthreads() calls."
                    ),
                    CodeFix(
                        section="Tile Size Tuning",
                        location="Top of file, #define TILE_SIZE",
                        problem_pattern="""// ❌ Fixed tile size
#define TILE_SIZE 16  // Too small, underutilizes SMEM""",
                        fix_pattern="""// ✅ Tune for your GPU
// For Ampere/Hopper: 32-64 works well
// Check occupancy with: nsight compute --section Occupancy
#define TILE_SIZE 32  // Start here

// Advanced: Template for autotuning
template<int BM, int BN, int BK>
__global__ void matmul_tiled(...) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    ...
}""",
                        explanation="Larger tiles = better data reuse BUT more shared memory = lower occupancy. "
                                   "Profile with different sizes: 16, 32, 64. Kernel 9 does autotuning.",
                        assembly_check="Check shared memory usage: cuobjdump -sass kernel.cubin | grep 'Shared Memory'"
                    )
                ],

                investigation_steps=[
                    "1. CHECK CURRENT DATA REUSE:",
                    "   Count how many times each A[i] is loaded in inner loop",
                    "   Example: A[row*N+k] loaded N times (once per col iteration)",
                    "",
                    "2. IDENTIFY TILE BOUNDARIES:",
                    "   What data is reused across threads?",
                    "   Matmul: Each A row tile used by entire thread block row",
                    "",
                    "3. CALCULATE SHARED MEMORY NEEDED:",
                    "   Size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float)",
                    "   Example: 2 * 32 * 32 * 4 = 8,192 bytes (8KB)",
                    "   Max shared memory per block: 48KB (Ampere), 164KB (Hopper)",
                    "",
                    "4. ESTIMATE ARITHMETIC INTENSITY:",
                    "   Before: 2 FLOPs per 2 loads = 1 FLOP/load",
                    "   After: 2*TILE^2 FLOPs per 2*TILE loads = TILE FLOPs/load",
                    "   With TILE=32: 32x improvement!"
                ],

                verification_steps=[
                    "1. CORRECTNESS FIRST:",
                    "   Compare output with reference (cuBLAS or naive)",
                    "   If wrong, check __syncthreads() placement",
                    "",
                    "2. PROFILE MEMORY THROUGHPUT:",
                    "   ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./app",
                    "   Should DECREASE (less DRAM traffic)",
                    "",
                    "3. CHECK SHARED MEMORY USAGE:",
                    "   ncu --metrics shared_memory_per_block ./app",
                    "   Should match: 2 * TILE * TILE * 4 bytes",
                    "",
                    "4. MEASURE SPEEDUP:",
                    "   Expect 1.5-2x speedup (Kernel 3)",
                    "   If less: Check if occupancy dropped due to SMEM usage"
                ],

                related_metrics=[
                    f"Memory Throughput: {k.memory_throughput:.1f}% (should decrease after fix)",
                    f"Compute Throughput: {k.compute_throughput:.1f}% (should increase)",
                    f"Stall Memory Throttle: {k.stall_memory_throttle:.1f}% (should decrease)"
                ],

                references=[
                    "Simon Boehm Kernel 3: Shared Memory Caching (2,980 → 2,980 GFLOPs base)",
                    "CUDA Programming Guide: Section 3.2.3 Shared Memory",
                    "Pattern: https://github.com/siboehm/SGEMM_CUDA/blob/master/sgemm.cu#L50"
                ]
            ))

        return tasks

    def _analyze_cache_efficiency(self, k: KernelMetrics) -> List[OptimizationTask]:
        """Analyze and fix cache efficiency issues"""
        tasks = []

        if k.l1_hit_rate > 0 and k.l1_hit_rate < 70:
            tasks.append(OptimizationTask(
                priority=2,
                category="L1 Cache",
                title=f"Low L1 Cache Hit Rate: {k.l1_hit_rate:.1f}%",
                description="L1 cache is underutilized. Poor temporal/spatial locality.",
                reasoning="L1 cache thrashing or poor access patterns. Shared memory gives explicit control.",
                expected_impact="MODERATE: 1.5-2x from better locality",

                code_fixes=[
                    CodeFix(
                        section="Access Pattern - Inner Loop",
                        location="Find innermost loop where data is accessed multiple times",
                        problem_pattern="""// ❌ Large strides, poor locality
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        // Stride of M elements between i iterations
        data[i * M + j] = ...;  // Poor L1 reuse
    }
}""",
                        fix_pattern="""// ✅ Better locality - process in blocks
#define BLOCK 32
for (int ii = 0; ii < N; ii += BLOCK) {
    for (int jj = 0; jj < M; jj += BLOCK) {
        // Process BLOCK x BLOCK tile
        for (int i = ii; i < min(ii+BLOCK, N); i++) {
            for (int j = jj; j < min(jj+BLOCK, M); j++) {
                data[i * M + j] = ...;  // Better L1 locality
            }
        }
    }
}""",
                        explanation="Tiling improves L1 reuse by processing data in cache-sized chunks.",
                        assembly_check="ncu --metrics l1tex__t_sector_hit_rate.pct should increase to >70%"
                    )
                ],

                investigation_steps=[
                    "1. Profile L1 access pattern:",
                    "   ncu --section MemoryWorkloadAnalysis ./app",
                    "",
                    "2. Check stride patterns:",
                    "   Look for large strides in array access",
                    "",
                    "3. Identify reuse opportunities:",
                    "   What data is accessed multiple times?"
                ],

                verification_steps=[
                    "1. ncu --metrics l1tex__t_sector_hit_rate.pct ./app",
                    "   Should increase to >70%",
                    "",
                    "2. Check memory throughput decrease"
                ],

                related_metrics=[f"L1 Hit Rate: {k.l1_hit_rate:.1f}%"],
                references=["CUDA Best Practices: Memory Optimization"]
            ))

        return tasks

    def _analyze_occupancy(self, k: KernelMetrics) -> List[OptimizationTask]:
        """Analyze and fix occupancy issues with exact register/shared memory guidance"""
        tasks = []

        if k.achieved_occupancy > 0 and k.achieved_occupancy < 50:
            # Determine likely limiter
            limiter_hint = "Unknown"
            if k.registers_per_thread > 64:
                limiter_hint = "Likely REGISTERS (using {})".format(k.registers_per_thread)
            elif k.shared_mem_per_block > 32768:
                limiter_hint = "Likely SHARED MEMORY (using {} bytes)".format(k.shared_mem_per_block)

            tasks.append(OptimizationTask(
                priority=1,
                category="Occupancy",
                title=f"Low Occupancy: {k.achieved_occupancy:.1f}% ({limiter_hint})",
                description=f"Only {k.achieved_occupancy:.1f}% of warps active. Limits latency hiding.",
                reasoning="Low occupancy means fewer warps to hide memory/compute latency. "
                         "Critical for memory-bound kernels. Need to reduce resource usage per thread/block.",
                expected_impact="MAJOR: 2-4x speedup from increased occupancy",

                code_fixes=[
                    CodeFix(
                        section="Register Usage Reduction",
                        location="Look for: Large local arrays, many variables in registers",
                        problem_pattern="""__global__ void kernel(...) {
    // ❌ Too many registers
    float tmp[64];  // Each element uses a register!
    double accum1, accum2, ..., accum20;  // 20 registers

    // Complex expressions create many temp registers
    float x = (a*b + c*d) * (e*f + g*h) * (i*j + k*l);
}""",
                        fix_pattern="""__global__ void kernel(...) {
    // ✅ Reduce register pressure

    // Option 1: Move to shared memory
    __shared__ float tmp[BLOCK_SIZE][64];

    // Option 2: Recompute instead of store
    // If cheap to recompute, don't store in register

    // Option 3: Compiler flag
    // Compile with: nvcc -maxrregcount=64 kernel.cu

    // Option 4: Simplify expressions
    float x = a*b + c*d;
    x *= (e*f + g*h);  // Break into steps
    x *= (i*j + k*l);
}""",
                        explanation="Each thread has limited registers (64K per SM / threads). "
                                   "Reduce local arrays, intermediate variables, complex expressions.",
                        assembly_check="nvcc --ptxas-options=-v kernel.cu → shows registers per thread. "
                                      "Target: <64 registers for good occupancy."
                    ),
                    CodeFix(
                        section="Shared Memory Reduction",
                        location="Look for: __shared__ declarations, TILE_SIZE defines",
                        problem_pattern="""// ❌ Too much shared memory
#define TILE 64
__shared__ float As[TILE][TILE];  // 16KB
__shared__ float Bs[TILE][TILE];  // 16KB
__shared__ float Cs[TILE][TILE];  // 16KB
// Total: 48KB per block
// Max blocks per SM: 48KB / 48KB = 1 block → Low occupancy!""",
                        fix_pattern="""// ✅ Reduce shared memory usage
#define TILE 32  // 4KB per array
__shared__ float As[TILE][TILE];  // 4KB
__shared__ float Bs[TILE][TILE];  // 4KB
// Total: 8KB per block
// Max blocks per SM: 48KB / 8KB = 6 blocks → Better occupancy!

// OR reuse shared memory
__shared__ float shared_buf[TILE][TILE];
// Use for As, then reuse for Bs after sync""",
                        explanation="48KB shared memory per SM (Ampere). More SMEM per block = fewer blocks = lower occupancy.",
                        assembly_check="Check: cuobjdump -sass kernel.cubin | grep 'Shared Memory'"
                    ),
                    CodeFix(
                        section="Block Size Tuning",
                        location="Kernel launch configuration: kernel<<<grid, block>>>",
                        problem_pattern="""// ❌ Small block size
dim3 block(8, 8);  // Only 64 threads
kernel<<<grid, block>>>(...);
// Too few threads per block → low occupancy""",
                        fix_pattern="""// ✅ Optimal block size
dim3 block(16, 16);  // 256 threads (good)
// Or 32x32 = 1024 threads (max)
kernel<<<grid, block>>>(...);

// Rule of thumb:
// - At least 128-256 threads per block
// - Multiple of 32 (warp size)
// - Balance with register/SMEM usage""",
                        explanation="More threads per block = better occupancy (up to limits). "
                                   "But watch register/SMEM usage increase.",
                        assembly_check="Use CUDA Occupancy Calculator spreadsheet to find optimal block size"
                    ),
                    CodeFix(
                        section="Launch Bounds Directive",
                        location="Add before __global__ declaration",
                        problem_pattern="""__global__ void kernel(...) {
    // Compiler doesn't know target occupancy
}""",
                        fix_pattern="""// Tell compiler to optimize for specific occupancy
__global__
__launch_bounds__(256, 4)  // 256 threads/block, 4 blocks/SM target
void kernel(...) {
    // Compiler will optimize register usage for this config
}""",
                        explanation="__launch_bounds__ tells compiler to reduce registers to hit target occupancy.",
                        assembly_check="Check register count decreases with bounds specified"
                    )
                ],

                investigation_steps=[
                    "1. IDENTIFY LIMITER:",
                    "   ncu --section Occupancy ./app",
                    "   Look for 'Theoretical Occupancy' and limiter reason",
                    "",
                    "2. CHECK REGISTER USAGE:",
                    "   nvcc --ptxas-options=-v kernel.cu",
                    "   Output shows: 'Used X registers'",
                    f"   Current: {k.registers_per_thread} registers/thread" if k.registers_per_thread > 0 else "",
                    "",
                    "3. CHECK SHARED MEMORY:",
                    "   Look at __shared__ declarations",
                    f"   Current: {k.shared_mem_per_block} bytes/block" if k.shared_mem_per_block > 0 else "",
                    "",
                    "4. USE OCCUPANCY CALCULATOR:",
                    "   CUDA Toolkit includes spreadsheet",
                    "   Input: registers, SMEM, block size",
                    "   Output: Theoretical occupancy"
                ],

                verification_steps=[
                    "1. RECOMPILE WITH LIMITS:",
                    "   nvcc -maxrregcount=64 kernel.cu",
                    "   OR add __launch_bounds__",
                    "",
                    "2. PROFILE OCCUPANCY:",
                    "   ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./app",
                    "   Target: >66% (was {:.1f}%)".format(k.achieved_occupancy),
                    "",
                    "3. CHECK DOESN'T HURT PERFORMANCE:",
                    "   Lower registers might cause spills to local memory",
                    "   Profile: ncu --metrics local_memory_overhead ./app",
                    "",
                    "4. MEASURE SPEEDUP:",
                    "   Time kernel before/after",
                    "   Higher occupancy should improve memory-bound kernels most"
                ],

                related_metrics=[
                    f"Achieved Occupancy: {k.achieved_occupancy:.1f}% (target: >66%)",
                    f"Theoretical Occupancy: {k.theoretical_occupancy:.1f}%",
                    f"Registers/Thread: {k.registers_per_thread}" if k.registers_per_thread > 0 else "Registers/Thread: Unknown",
                    f"Shared Memory/Block: {k.shared_mem_per_block} bytes" if k.shared_mem_per_block > 0 else "Shared Memory/Block: Unknown"
                ],

                references=[
                    "CUDA Occupancy Calculator: <CUDA_PATH>/tools/CUDA_Occupancy_Calculator.xls",
                    "Programming Guide: Section 5.2.3 Occupancy",
                    "nvcc flags: -maxrregcount=N, --ptxas-options=-v"
                ]
            ))

        return tasks

    def _analyze_warp_stalls(self, k: KernelMetrics) -> List[OptimizationTask]:
        """Analyze warp stalls with context-aware interpretation"""
        tasks = []

        # Context-aware "Stall Not Selected" interpretation
        if k.stall_not_selected > 40 and k.achieved_occupancy < 50:
            tasks.append(OptimizationTask(
                priority=1,
                category="Warp Scheduling",
                title="High 'Not Selected' Stalls + Low Occupancy = CRITICAL",
                description=f"{k.stall_not_selected:.1f}% not selected, but only {k.achieved_occupancy:.1f}% occupancy",
                reasoning="IMPORTANT: 'Stall Not Selected' with LOW occupancy is BAD. "
                         "Scheduler doesn't have enough eligible warps. This is different from "
                         "high 'not selected' with high occupancy (which is OK).",
                expected_impact="CRITICAL: Fix occupancy first (see occupancy tasks above)",

                code_fixes=[
                    CodeFix(
                        section="Same as Occupancy Fixes",
                        location="See occupancy tasks above",
                        problem_pattern="Insufficient warps available for scheduler",
                        fix_pattern="Increase occupancy by reducing registers or shared memory",
                        explanation="Not enough warps → scheduler starved → appears as 'not selected' stalls",
                        assembly_check="Fix occupancy first, then re-profile"
                    )
                ],

                investigation_steps=[
                    "This is a symptom of low occupancy.",
                    "See occupancy investigation steps above."
                ],

                verification_steps=[
                    "Fix occupancy first, then re-profile warp stalls.",
                    "Expect 'not selected' stalls to remain but not be a problem."
                ],

                related_metrics=[
                    f"Stall Not Selected: {k.stall_not_selected:.1f}%",
                    f"Achieved Occupancy: {k.achieved_occupancy:.1f}%"
                ],

                references=["Simon's note: Stall interpretation depends on occupancy context"]
            ))

        # Memory throttle stalls
        if k.stall_memory_throttle > 30:
            tasks.append(OptimizationTask(
                priority=1,
                category="Memory Pipeline",
                title=f"High Memory Pipeline Stalls: {k.stall_memory_throttle:.1f}%",
                description="Memory pipeline congested (MIO Throttle). Too many outstanding memory requests.",
                reasoning="Memory subsystem saturated. Need to reduce global memory traffic via shared memory caching.",
                expected_impact="MAJOR: 2-3x from reducing memory pressure",

                code_fixes=[
                    CodeFix(
                        section="Same as Shared Memory Fixes",
                        location="See memory bottleneck tasks above",
                        problem_pattern="Too many global memory loads",
                        fix_pattern="Add shared memory caching (see detailed fix above)",
                        explanation="Shared memory reduces DRAM traffic → less pipeline congestion",
                        assembly_check="Count LDG instructions, should decrease"
                    )
                ],

                investigation_steps=["See memory bottleneck investigation above"],
                verification_steps=["ncu --metrics stall_memory_throttle should decrease to <20%"],
                related_metrics=[f"Memory Throttle Stalls: {k.stall_memory_throttle:.1f}%"],
                references=["Simon: MIO Throttle = memory pipeline congestion"]
            ))

        return tasks

    def _analyze_roofline(self, k: KernelMetrics) -> List[OptimizationTask]:
        """Roofline analysis with blocktiling guidance"""
        tasks = []

        if k.memory_throughput > 70 and k.compute_throughput < 50:
            # Memory bound - need blocktiling
            tasks.append(OptimizationTask(
                priority=2,
                category="Arithmetic Intensity",
                title="Low Arithmetic Intensity - Need Blocktiling",
                description=f"Memory-bound: Each byte does too little work (low FLOPs/byte ratio)",
                reasoning="Loading data but not computing enough with it. Blocktiling increases FLOPs per byte loaded.",
                expected_impact="MAJOR: 3-10x from 2D blocktiling (Kernels 4-5)",

                code_fixes=[
                    CodeFix(
                        section="1D Blocktiling (First Step)",
                        location="Change each thread to compute multiple outputs",
                        problem_pattern="""// ❌ 1 thread = 1 output
__global__ void matmul(...) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0;
    for (int k = 0; k < K; k++) {
        sum += A[row*K + k] * B[k*N + col];
    }
    C[row*N + col] = sum;  // 1 output
}""",
                        fix_pattern="""// ✅ 1 thread = TM outputs (1D blocktiling)
#define TM 8  // Each thread computes 8 outputs

__global__ void matmul(...) {
    int row = (blockIdx.y * TILE + threadIdx.y) * TM;  // Note *TM
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum[TM];  // TM accumulators
    for (int i = 0; i < TM; i++) sum[i] = 0;

    for (int k = 0; k < K; k++) {
        float b = B[k*N + col];  // Load once
        for (int i = 0; i < TM; i++) {
            sum[i] += A[(row+i)*K + k] * b;  // Reuse b
        }
    }

    for (int i = 0; i < TM; i++) {
        C[(row+i)*N + col] = sum[i];
    }
}""",
                        explanation="Each thread computes TM outputs instead of 1. "
                                   "B value loaded once, reused TM times. Arithmetic intensity TM× better. "
                                   "Kernel 4 shows 5.4x speedup.",
                        assembly_check="Count FMA instructions vs LDG. Ratio should increase by TM factor."
                    ),
                    CodeFix(
                        section="2D Blocktiling (Advanced)",
                        location="Further extension: TM × TN outputs per thread",
                        problem_pattern="""// 1D blocktiling: TM outputs per thread
// Still loading A and B multiple times""",
                        fix_pattern="""// ✅ 2D blocktiling: TM×TN outputs per thread
#define TM 8
#define TN 8

__global__ void matmul(...) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * TM;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * TN;

    float sum[TM][TN];  // TM×TN accumulators
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            sum[i][j] = 0;

    for (int k = 0; k < K; k++) {
        float a[TM], b[TN];
        // Load TM from A, TN from B
        for (int i = 0; i < TM; i++) a[i] = A[(row+i)*K + k];
        for (int j = 0; j < TN; j++) b[j] = B[k*N + col+j];

        // Outer product: TM×TN operations from TM+TN loads
        for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
                sum[i][j] += a[i] * b[j];
    }

    // Write TM×TN outputs
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(row+i)*N + col+j] = sum[i][j];
}""",
                        explanation="Each thread computes TM×TN outputs from TM+TN loads per iteration. "
                                   "Much better arithmetic intensity. Kernel 5 shows 2x on top of Kernel 4.",
                        assembly_check="Should see many more FFMA than LDG instructions"
                    )
                ],

                investigation_steps=[
                    "1. CALCULATE CURRENT ARITHMETIC INTENSITY:",
                    "   FLOPs per iteration: 2 (multiply + add)",
                    "   Loads per iteration: 2 (A + B)",
                    "   Intensity = 2/2 = 1 FLOP/load (very low!)",
                    "",
                    "2. CALCULATE WITH 1D BLOCKTILING (TM=8):",
                    "   FLOPs: 2 * TM = 16",
                    "   Loads: 1 + TM = 9",
                    "   Intensity = 16/9 = 1.78 FLOP/load (better)",
                    "",
                    "3. CALCULATE WITH 2D BLOCKTILING (TM=8, TN=8):",
                    "   FLOPs: 2 * TM * TN = 128",
                    "   Loads: TM + TN = 16",
                    "   Intensity = 128/16 = 8 FLOP/load (much better!)"
                ],

                verification_steps=[
                    "1. PROFILE ARITHMETIC INTENSITY:",
                    "   ncu --metrics sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_ldgsts_pred_on.sum ./app",
                    "   Ratio should increase",
                    "",
                    "2. CHECK COMPUTE THROUGHPUT:",
                    "   ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./app",
                    "   Should increase toward 50-70% (balanced)",
                    "",
                    "3. MEASURE SPEEDUP:",
                    "   1D blocktiling: Expect 3-5x",
                    "   2D blocktiling: Expect 2x more (6-10x total)"
                ],

                related_metrics=[
                    f"Memory Throughput: {k.memory_throughput:.1f}%",
                    f"Compute Throughput: {k.compute_throughput:.1f}%"
                ],

                references=[
                    "Simon Kernel 4: 1D Blocktiling (2,980 → 15,971 GFLOPs)",
                    "Simon Kernel 5: 2D Blocktiling (15,971 → 18,237 GFLOPs)"
                ]
            ))

        return tasks

    def generate_report(self):
        """Generate detailed code-level optimization report"""
        print("\n" + "="*90)
        print("NCU ACTIONABLE OPTIMIZATION REPORT - CODE-LEVEL EDITION")
        print("Precise code fixes with exact locations and before/after examples")
        print("="*90)

        all_tasks = []
        for kernel_id, kernel in sorted(self.kernels.items()):
            kernel_tasks = self.analyze_kernel(kernel)
            all_tasks.extend([(kernel, task) for task in kernel_tasks])

        # Sort by priority
        all_tasks.sort(key=lambda x: (x[1].priority, x[1].category))

        # Print tasks with code-level details
        print(f"\n{'='*90}")
        print(f"ACTIONABLE TASKS WITH CODE FIXES (Total: {len(all_tasks)})")
        print(f"{'='*90}\n")

        current_priority = None
        task_num = 1

        for kernel, task in all_tasks:
            if task.priority != current_priority:
                current_priority = task.priority
                priority_name = {1: "🔴 CRITICAL", 2: "🟠 HIGH", 3: "🟡 MEDIUM", 4: "🟢 LOW"}[current_priority]
                print(f"\n{'='*90}")
                print(f"Priority {current_priority}: {priority_name}")
                print(f"{'='*90}\n")
                task_num = 1

            print(f"TASK #{task_num}: [{task.category}] {task.title}")
            print(f"Kernel: {kernel.kernel_name}")
            print(f"\n{'─'*90}")
            print(f"DESCRIPTION:")
            print(f"  {task.description}")
            print(f"\nREASONING:")
            for line in task.reasoning.split('. '):
                if line.strip():
                    print(f"  • {line.strip()}.")
            print(f"\nEXPECTED IMPACT: {task.expected_impact}")

            # Code fixes - the main new feature
            if task.code_fixes:
                print(f"\n{'─'*90}")
                print(f"CODE-LEVEL FIXES:")
                print(f"{'─'*90}")

                for i, fix in enumerate(task.code_fixes, 1):
                    print(f"\nFIX {i}: {fix.section}")
                    print(f"{'.'*90}")
                    print(f"📍 LOCATION: {fix.location}")
                    print()
                    print(f"❌ PROBLEM CODE:")
                    print("─" * 90)
                    for line in fix.problem_pattern.split('\n'):
                        print(f"  {line}")
                    print()
                    print(f"✅ FIXED CODE:")
                    print("─" * 90)
                    for line in fix.fix_pattern.split('\n'):
                        print(f"  {line}")
                    print()
                    print(f"💡 EXPLANATION:")
                    print(f"  {fix.explanation}")
                    print()
                    print(f"🔍 VERIFY IN ASSEMBLY:")
                    print(f"  {fix.assembly_check}")
                    print()

            # Investigation steps
            if task.investigation_steps:
                print(f"\n{'─'*90}")
                print(f"INVESTIGATION STEPS:")
                print(f"{'─'*90}")
                for step in task.investigation_steps:
                    print(f"  {step}")

            # Verification steps
            if task.verification_steps:
                print(f"\n{'─'*90}")
                print(f"VERIFICATION STEPS:")
                print(f"{'─'*90}")
                for step in task.verification_steps:
                    print(f"  {step}")

            # Related metrics
            print(f"\n{'─'*90}")
            print(f"RELATED METRICS:")
            for metric in task.related_metrics:
                print(f"  📊 {metric}")

            # References
            if task.references:
                print(f"\nREFERENCES:")
                for ref in task.references:
                    print(f"  📚 {ref}")

            print(f"\n{'='*90}\n")
            task_num += 1

        # Summary
        print(f"\n{'='*90}")
        print("SUMMARY")
        print(f"{'='*90}\n")

        by_priority = defaultdict(int)
        by_category = defaultdict(int)
        for _, task in all_tasks:
            by_priority[task.priority] += 1
            by_category[task.category] += 1

        print("By Priority:")
        for p in sorted(by_priority.keys()):
            name = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}[p]
            print(f"  Priority {p} ({name}): {by_priority[p]} task(s)")

        print("\nBy Category:")
        for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count} task(s)")

        # Workflow
        print(f"\n{'='*90}")
        print("RECOMMENDED WORKFLOW")
        print(f"{'='*90}\n")
        print("1. Start with Priority 1 (CRITICAL) tasks")
        print("2. For each task:")
        print("   a. Run investigation steps to locate exact code")
        print("   b. Apply code fixes (use before/after examples)")
        print("   c. Verify with assembly checks")
        print("   d. Run verification steps to confirm improvement")
        print("3. Re-profile after each major fix")
        print("4. Move to Priority 2 tasks once P1 is done")
        print("5. Iterate until performance goals met")

        # Save report
        output_file = self.csv_file.replace('.csv', '_code_level_tasks.txt')
        self._save_detailed_report(all_tasks, output_file)
        print(f"\nFull report saved to: {output_file}\n")

    def _save_detailed_report(self, tasks, output_file):
        """Save detailed report with code examples"""
        with open(output_file, 'w') as f:
            f.write("NCU ACTIONABLE OPTIMIZATION TASKS - CODE-LEVEL EDITION\n")
            f.write("="*90 + "\n\n")

            for kernel, task in tasks:
                f.write(f"[Priority {task.priority}] [{task.category}] {task.title}\n")
                f.write(f"Kernel: {kernel.kernel_name}\n")
                f.write(f"Description: {task.description}\n")
                f.write(f"Expected Impact: {task.expected_impact}\n\n")

                if task.code_fixes:
                    f.write("CODE FIXES:\n")
                    f.write("-" * 90 + "\n")
                    for i, fix in enumerate(task.code_fixes, 1):
                        f.write(f"\nFix {i}: {fix.section}\n")
                        f.write(f"Location: {fix.location}\n\n")
                        f.write("Problem Pattern:\n")
                        f.write(fix.problem_pattern + "\n\n")
                        f.write("Fix Pattern:\n")
                        f.write(fix.fix_pattern + "\n\n")
                        f.write(f"Explanation: {fix.explanation}\n")
                        f.write(f"Verify: {fix.assembly_check}\n")
                        f.write("-" * 90 + "\n")

                f.write("\n" + "="*90 + "\n\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ncu_actionable_optimizer.py <ncu_details.csv>")
        print()
        print("Example workflow:")
        print("  1. ncu --set basic --export profile ./app")
        print("  2. ncu --import profile.ncu-rep --page details --csv > ncu_details.csv")
        print("  3. python3 ncu_actionable_optimizer.py ncu_details.csv")
        print()
        print("NEW: Get precise code-level fixes with exact locations and before/after examples!")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not Path(csv_file).exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    optimizer = NCUActionableOptimizer(csv_file)
    optimizer.load_ncu_data()
    optimizer.generate_report()


if __name__ == "__main__":
    main()
