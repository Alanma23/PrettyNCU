# Custom NCU Commands Guide

This guide explains how to provide your own custom NCU (NVIDIA Nsight Compute) commands to the NCU-Claude analyzer.

## Why Use Custom Commands?

While NCU-Claude includes a comprehensive database of predefined diagnostics, you may want to:

- **Run specific metrics** not covered by the predefined diagnostics
- **Test experimental NCU features** or flags
- **Reproduce exact profiling setups** from your own workflows
- **Benchmark with precise configurations** for research or production

## Methods to Provide Custom Commands

### Method 1: Single Command via CLI Flag

Use `--ncu-command` to provide a single custom NCU command:

```bash
ncu-claude analyze kernel.cu --ncu-command "ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__cycles_elapsed.avg -o my_output"
```

**Example:**
```bash
ncu-claude analyze matmul.cu \
  --ncu-command "ncu --section MemoryWorkloadAnalysis --set detailed -o matmul_memory"
```

### Method 2: Multiple Commands from File

Create a text file with one NCU command per line:

**`my-ncu-commands.txt`:**
```bash
# Quick overview
ncu --section SpeedOfLight -o quick_check ./kernel

# Detailed memory analysis
ncu --section MemoryWorkloadAnalysis --section SpeedOfLight_RooflineChart -o memory_deep ./kernel

# Warp stall breakdown
ncu --section WarpStateStats --section SchedulerStats -o stalls ./kernel
```

Then run:
```bash
ncu-claude analyze kernel.cu --ncu-commands-file my-ncu-commands.txt
```

**Features:**
- Comments starting with `#` are ignored
- Empty lines are skipped
- Commands are executed in order
- Each command's output is analyzed

### Method 3: Via Context File

You can also embed custom NCU commands in your context file:

**`optimization-context.txt`:**
```
TARGET: Real-time ray tracing kernels
PRIORITY: Minimize latency
HARDWARE: RTX 4090

CUSTOM_NCU_COMMANDS:
ncu --metrics sm__inst_executed_pipe_tensor_op_hmma.sum -o tensor_check ./kernel
ncu --section SourceCounters --lineinfo -o source_level ./kernel
```

Then:
```bash
ncu-claude analyze kernel.cu --context optimization-context.txt
```

## Command Format

### Basic Structure

```
ncu [options] [executable] [executable-args]
```

### Common NCU Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--metrics <list>` | Specific metrics to collect | `--metrics sm__warps_active.avg` |
| `--section <name>` | Predefined metric sections | `--section MemoryWorkloadAnalysis` |
| `--set <level>` | Metric set level | `--set basic\|detailed\|full` |
| `-o <file>` | Output file path | `-o my_profile` |
| `--kernel-name <name>` | Profile specific kernel | `--kernel-name myKernel` |
| `-k <regex>` | Kernel regex filter | `-k "gemm.*"` |
| `--launch-count <N>` | Number of launches to profile | `--launch-count 1` |
| `-s <N>` | Skip first N launches | `-s 10` |
| `--lineinfo` | Enable source correlation | `--lineinfo` |
| `--csv` | CSV output format | `--csv --page raw` |

### Example Custom Commands

**1. Tensor Core Verification:**
```bash
ncu --metrics sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_tensor_op_dmma.sum -o tensor_check ./matmul
```

**2. Memory Hierarchy Deep Dive:**
```bash
ncu --metrics lts__t_sectors_hit_rate.pct,lts__t_sectors_miss_rate.pct,lts__t_requests_srcunit_tex.sum,dram__bytes.sum -o memory_hierarchy ./kernel
```

**3. Occupancy Analysis:**
```bash
ncu --metrics sm__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.pct_of_peak_sustained_active,sm__threads_launched.avg -o occupancy ./kernel
```

**4. Source-Level Profiling:**
```bash
ncu --section SourceCounters --lineinfo -o source_profile ./kernel
```
*(Requires compilation with `-lineinfo` flag)*

**5. Multi-Kernel Filtering:**
```bash
ncu --kernel-name "myKernel" --kernel-name "anotherKernel" -o multi_kernel ./app
```

**6. Skip Warmup Iterations:**
```bash
ncu -s 100 -c 1 --set detailed -o iteration_profile ./iterative_solver
```

## How It Works

When you provide custom NCU commands:

1. **Command Validation**: Commands are parsed and validated
2. **Context Injection**: Commands are injected into Claude's system prompt
3. **Execution Priority**: Custom commands take precedence over automated selection
4. **Exact Execution**: Claude executes the EXACT commands you provide (no modification)
5. **Output Parsing**: Results are read from the specified output files
6. **Analysis**: Claude analyzes the custom metrics to generate insights

### Example Workflow

```bash
# 1. Create custom commands file
cat > custom-profile.txt << 'EOF'
# Focus on register usage and occupancy
ncu --metrics sm__sass_thread_inst_executed_op_integer_pred_on.avg,launch__registers_per_thread -o registers ./kernel

# Memory coalescing check
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum -o coalescing ./kernel
EOF

# 2. Run analysis with custom commands
ncu-claude analyze kernel.cu --ncu-commands-file custom-profile.txt

# 3. Claude executes exactly these commands
#    Reads output from: registers.ncu-rep and coalescing.ncu-rep
#    Generates insights based on these specific metrics
```

## Combining with Other Features

### Custom Commands + User Context

```bash
ncu-claude analyze kernel.cu \
  --ncu-command "ncu --metrics sm__warps_active.avg -o warps ./kernel" \
  --context-inline "Focus on warp efficiency. Target A100 GPU."
```

### Custom Commands + Diagnostic Configuration

```bash
# Use custom commands AND select additional diagnostics
ncu-claude analyze kernel.cu \
  --ncu-commands-file my-commands.txt \
  --configure-diagnostics
```

In this case:
- Your custom commands run first
- Then Claude can choose to run additional diagnostics from the selector

## Advanced Examples

### Research: Comparing Metrics Across Configurations

**`compare-configs.txt`:**
```bash
# Configuration 1: Block size 128
ncu --metrics sm__warps_active.avg,achieved_occupancy -o config_128 ./kernel 128

# Configuration 2: Block size 256
ncu --metrics sm__warps_active.avg,achieved_occupancy -o config_256 ./kernel 256

# Configuration 3: Block size 512
ncu --metrics sm__warps_active.avg,achieved_occupancy -o config_512 ./kernel 512
```

```bash
ncu-claude analyze kernel.cu --ncu-commands-file compare-configs.txt
```

### Production: Minimal Overhead Profiling

```bash
ncu-claude analyze kernel.cu \
  --ncu-command "ncu --kernel-id :::1 --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -o quick_prod ./app"
```

### Debugging: Full Trace with Source Correlation

```bash
ncu-claude analyze kernel.cu \
  --ncu-command "ncu --section SourceCounters --section WarpStateStats --lineinfo --set full -o debug_full ./kernel"
```

## Output Files

When using `-o <filename>` in your custom commands:

- NCU creates: `<filename>.ncu-rep` (binary report)
- You can also request CSV: Add `--csv --page raw`
- Claude automatically reads these files
- Files are saved in `./ncu-llm-output/` by default (or current directory if you specify a path)

## Tips & Best Practices

1. **Start Simple**: Begin with one custom command to verify it works
2. **Use Descriptive Output Names**: `-o memory_analysis` instead of `-o output`
3. **Check NCU Documentation**: Run `ncu --help` to see all available options
4. **List Available Metrics**: Use `ncu --query-metrics` to see all metrics
5. **Test Commands First**: Run your NCU command manually before using with Claude
6. **Combine Strategically**: Use custom commands for specific deep-dives, let Claude choose for general analysis
7. **Comment Your Files**: Use `#` comments in command files to explain intent

## Troubleshooting

**Problem**: Command fails with "metric not found"
```bash
# Solution: Check available metrics
ncu --query-metrics | grep <metric_name>
```

**Problem**: Output file not found
```bash
# Solution: Use absolute paths or check cwd
ncu --metrics xyz -o /absolute/path/output ./kernel
```

**Problem**: Source correlation not working
```bash
# Solution: Compile with -lineinfo
nvcc -lineinfo kernel.cu -o kernel
```

**Problem**: Too much overhead
```bash
# Solution: Use --kernel-id to profile specific launches
ncu --kernel-id :::1 ...  # Profile only first launch
```

## Reference: Common Use Cases

| Use Case | Command Template |
|----------|------------------|
| Quick check | `ncu --section SpeedOfLight -o quick ./kernel` |
| Memory bound | `ncu --section MemoryWorkloadAnalysis --section SpeedOfLight_RooflineChart -o mem ./kernel` |
| Compute bound | `ncu --section ComputeWorkloadAnalysis --section InstructionStats -o compute ./kernel` |
| Latency issues | `ncu --section WarpStateStats --section SchedulerStats -o latency ./kernel` |
| Tensor Cores | `ncu --metrics sm__inst_executed_pipe_tensor_op_hmma.sum -o tensor ./kernel` |
| Cache analysis | `ncu --metrics lts__t_sectors_hit_rate.pct,lts__t_sectors_miss_rate.pct -o cache ./kernel` |
| Source-level | `ncu --section SourceCounters --lineinfo -o source ./kernel` |

## Example Session

```bash
# Terminal session showing custom command usage

$ cat > my-profile.txt
# Check memory throughput
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed -o memory_check ./matmul

# Check compute throughput
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -o compute_check ./matmul
^D

$ ncu-claude analyze matmul.cu --ncu-commands-file my-profile.txt
Loaded 2 custom NCU command(s) from file

🔍 Analyzing matmul.cu...

🔧 Bash
   └─ nvcc matmul.cu -o matmul

🔧 Bash
   └─ ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed -o memory_check ./matmul

🔧 Read
   └─ Reading: memory_check.ncu-rep

🔧 Bash
   └─ ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -o compute_check ./matmul

📝 Analyzing NCU output...

✅ Analysis Complete

⚡ P1 Actionable Insight:
Memory throughput at 15% while compute throughput at 85% indicates memory-bound kernel...

[...]
```

## Further Reading

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NCU Metric Reference](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)
- [NCU Command Line Options](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

## Summary

Custom NCU commands give you complete control over profiling:

✅ **Flexibility**: Run any NCU command you need
✅ **Precision**: Collect exactly the metrics you want
✅ **Reproducibility**: Use the same commands across runs
✅ **Integration**: Works with all other NCU-Claude features
✅ **No Limits**: Not restricted to predefined diagnostics
