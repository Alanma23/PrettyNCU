# NCU Diagnostic System

NCU-Claude includes an intelligent diagnostic system that helps Claude select the right NCU profiling commands based on your kernel's characteristics.

## Overview

Instead of always running comprehensive profiling (which can be 100-200x slower), the diagnostic system allows you to:

1. **Use Presets** - Quick selection of common diagnostic patterns
2. **Custom Configuration** - Fine-tune which NCU sections and metrics to profile
3. **Smart Context** - Compiled diagnostic info is passed to Claude for intelligent NCU command selection

## Quick Start

### List Available Presets

```bash
npm run dev -- diagnostics
```

Output:
```
📊 Available Diagnostic Presets:

  quick        Fast SpeedOfLight analysis (low overhead) - determines if memory or compute bound
  memory       Memory bottleneck investigation - cache analysis and coalescing
  compute      Compute workload analysis - pipeline utilization and instruction mix
  occupancy    Low occupancy investigation - register/shared memory pressure
  latency      Latency and warp stall analysis - pipeline stalls and scheduling
  tensor       Tensor Core optimization - GEMM and matrix operations
  full         Comprehensive analysis - all diagnostics (very high overhead)
```

### Use a Preset

```bash
npm run dev -- analyze kernel.cu --diagnostics memory
```

This tells Claude to focus on memory-related diagnostics.

## Available Presets

### 1. `quick` (Recommended Default)

**What it does**: Runs only `--section SpeedOfLight` to get a high-level view

**Overhead**: Low (~2-3x runtime)

**Use when**: Starting analysis on any new kernel

**NCU commands**:
- `ncu --section SpeedOfLight -o quick_check ./app`

**Key metrics**:
- Memory Throughput %
- Compute Throughput %

**Decision logic**: If Memory > Compute → use `memory` preset next. If Compute > Memory → use `compute` preset.

### 2. `memory` (Memory-Bound Kernels)

**What it does**: Deep dive into memory system - cache hierarchy, DRAM, coalescing

**Overhead**: Medium-High (~10-20x runtime)

**Use when**: Quick analysis shows Memory Throughput < 60% or you see uncoalesced access patterns

**NCU commands**:
- `ncu --section SpeedOfLight -o quick`
- `ncu --section MemoryWorkloadAnalysis --section SpeedOfLight -o mem_focused`
- `ncu --metrics lts__t_sectors_hit_rate.pct,lts__t_sectors_miss_rate.pct -o cache`

**Key metrics**:
- L2 Cache Hit Rate
- DRAM Throughput
- Memory coalescing efficiency

**Claude guidance**: Focus on shared memory tiling, stride-1 access patterns, and cache optimization

### 3. `compute` (Compute-Bound Kernels)

**What it does**: Analyzes pipeline utilization and instruction mix

**Overhead**: Medium (~10-15x runtime)

**Use when**: Quick analysis shows Compute Throughput < 60%

**NCU commands**:
- `ncu --section SpeedOfLight -o quick`
- `ncu --section ComputeWorkloadAnalysis --section InstructionStats --section SpeedOfLight -o compute`

**Key metrics**:
- FMA/FP64/FP16 pipeline utilization
- Tensor Core operations
- LSU (Load/Store) utilization

**Claude guidance**: Suggest FMA fusion, intrinsics, or instruction-level parallelism improvements

### 4. `occupancy` (Low Occupancy Issues)

**What it does**: Investigates occupancy, register pressure, and shared memory usage

**Overhead**: High (~15-25x runtime)

**Use when**: Both memory and compute throughput < 60%, suggesting latency hiding problems

**NCU commands**:
- `ncu --section SpeedOfLight -o quick`
- `ncu --section Occupancy --metrics sm__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.pct_of_peak_sustained_active -o occ`
- `ncu --section WarpStateStats --section SourceCounters --section SchedulerStats -o stalls`

**Key metrics**:
- Theoretical vs Achieved Occupancy
- Warp stall reasons
- Stall Long Scoreboard (memory latency)

**Claude guidance**: Recommend __launch_bounds__, reducing register usage, or block size tuning

### 5. `latency` (Latency/Stall Analysis)

**What it does**: Deep dive into warp stalls and pipeline bubbles

**Overhead**: High (~20-30x runtime)

**Use when**: Throughput is low but occupancy is reasonable

**NCU commands**:
- `ncu --section SpeedOfLight -o quick`
- `ncu --section WarpStateStats --section SourceCounters --section SchedulerStats -o stalls`
- `ncu --section Occupancy -o occ`

**Key metrics**:
- Warp stall breakdown percentages
- Long Scoreboard (memory dependency stalls)
- LG Throttle (memory system throttling)

**Claude guidance**: Suggest prefetching, async operations, or increasing instruction-level parallelism

### 6. `tensor` (Tensor Core / GEMM)

**What it does**: Verifies Tensor Core usage and optimizes matrix operations

**Overhead**: Low-Medium (~5-10x runtime)

**Use when**: Kernel contains matrix multiplications or conv operations

**NCU commands**:
- `ncu --section SpeedOfLight -o quick`
- `ncu --metrics sm__inst_executed_pipe_tensor_op_hmma.sum,tensor_precision_fu_utilization -o tensor`
- `ncu --section MemoryWorkloadAnalysis -o mem`

**Key metrics**:
- Tensor Core instruction count
- Tensor precision utilization

**Claude guidance**: Check wmma/mma API usage, tile sizes, and data layouts for Tensor Cores

### 7. `full` (Comprehensive Analysis)

**What it does**: Runs all available diagnostics

**Overhead**: Very High (~100-200x runtime)

**Use when**: You need every possible metric (rarely recommended - use targeted presets instead)

**NCU commands**:
- All of the above

**Warning**: Can take hours for large kernels

## Diagnostic Database

The full diagnostic configuration is stored in `diagnostics-database.json` with structure:

```json
{
  "diagnostics": [
    {
      "id": "memory_bottleneck",
      "name": "Memory Bottleneck Diagnosis",
      "description": "...",
      "priority": "high",
      "overhead": "high",
      "flags": {
        "comprehensive": "--set full -o memory_report",
        "focused": "--section MemoryWorkloadAnalysis ...",
        ...
      },
      "key_metrics": [...],
      "when_to_use": "...",
      "llm_guidance": "..."
    }
  ],
  "workflows": [...],
  "thresholds": {...}
}
```

## How It Works

1. **User selects preset** via `--diagnostics <preset>`

2. **DiagnosticsManager compiles context**:
   ```
   === NCU DIAGNOSTIC CONFIGURATION ===

   [Memory Bottleneck Diagnosis]
   Description: Comprehensive memory system analysis
   NCU Command: ncu --section MemoryWorkloadAnalysis ...
   Key Metrics:
     - lts__t_sector_hit_rate.pct - L2 Cache Hit Rate
     - ...
   LLM Guidance: Focus on memory access patterns...
   ```

3. **Context is injected into Claude's prompt**:
   - Claude reads the diagnostic configuration
   - Claude analyzes the kernel code
   - Claude chooses appropriate NCU commands from the diagnostic config
   - Claude executes profiling via Bash tool
   - Claude parses outputs and generates insights

4. **Result**: Actionable insight + hot-fix + explanation

## Performance Thresholds

The system uses these thresholds to guide recommendations:

| Metric | Threshold | Action |
|--------|-----------|--------|
| Memory Throughput | < 60% | Investigate latency issues |
| Compute Throughput | < 60% | Investigate latency issues |
| Occupancy | < 50% | Interferes with latency hiding |
| L2 Miss Rate | > 30% | Check coalescing or working set size |

## Examples

### Example 1: Unknown Issue

```bash
npm run dev -- analyze kernel.cu --diagnostics quick
```

Claude will run SpeedOfLight and determine the bottleneck type.

### Example 2: Known Memory Issue

```bash
npm run dev -- analyze kernel.cu --diagnostics memory
```

Claude will focus on memory hierarchy analysis.

### Example 3: Matrix Multiplication

```bash
npm run dev -- analyze matmul.cu --diagnostics tensor
```

Claude will check Tensor Core usage and memory patterns for GEMM.

### Example 4: Combined with Custom Context

```bash
npm run dev -- analyze kernel.cu --diagnostics occupancy --context-inline "Target A100 GPU, focus on register pressure"
```

Both the diagnostic preset AND user context are passed to Claude.

## Advanced: Custom Diagnostic Configurations

You can create custom profiles by:

1. Copying `diagnostics-database.json`
2. Modifying diagnostic entries
3. Using `DiagnosticsManager` API in your own scripts

Example:
```typescript
import { DiagnosticsManager } from './services/diagnostics-manager.js';

const manager = new DiagnosticsManager();

// Custom selection
manager.setDiagnostic('memory_bottleneck', 'focused', true);
manager.setDiagnostic('tensor_core', 'verify', true);

// Compile to context
const context = manager.compileToContext();
```

## Compilation Requirements

Some diagnostics require special compilation flags:

```bash
# For source-level correlation
nvcc -lineinfo kernel.cu -o kernel

# Then use with SourceCounters diagnostic
npm run dev -- analyze kernel.cu --diagnostics latency
```

## Best Practices

1. **Always start with `quick`** to determine bottleneck type
2. **Use targeted presets** instead of `full` to save time
3. **Combine with user context** for hardware-specific optimizations
4. **Skip warmup iterations** for multi-launch kernels (add to user context)

## Troubleshooting

**Q: Diagnostic preset not being used by Claude?**

A: Check that the diagnostic context is properly compiled. Try adding `--context-inline "Use the NCU diagnostic configuration exactly as specified"`.

**Q: Profiling overhead too high?**

A: Use a lighter preset like `quick` or `memory` instead of `full`. Or add to context: "Minimize profiling overhead - use only essential metrics".

**Q: Want to see what NCU commands will be used?**

A: The diagnostic context is printed when you use `--diagnostics`. Claude will reference this when choosing commands.

## Future Enhancements

- Interactive diagnostic selector TUI (use arrow keys to enable/disable diagnostics)
- Save/load custom diagnostic profiles
- Automatic preset selection based on code analysis
- Workflow support (multi-step diagnostic strategies)
