# NCU-LLM Scripts

This directory contains the `ncu-llm` wrapper scripts used by NCU-Claude for profiling.

## Files

- **`ncu-llm`** - Main wrapper script that provides different profiling modes
- **`ncu-llm-lib.sh`** - Library functions used by the main script

## What is ncu-llm?

The `ncu-llm` wrapper is an **LLM-friendly** interface to NVIDIA Nsight Compute (NCU). It provides:

1. **Token-budget aware profiling** - All outputs stay under 50K tokens
2. **Multiple profiling modes** - quick, standard, bottleneck, memory, compute, etc.
3. **Structured output** - Grep-able text files and CSVs
4. **Actionable insights** - Focus on optimization recommendations

## Usage

The wrapper is called by Claude during analysis:

```bash
# Quick profiling (8 essential metrics, ~5K tokens)
./scripts/ncu-llm quick ./executable

# Standard profiling (core metrics + insights, ~30K tokens)
./scripts/ncu-llm standard ./executable

# Bottleneck detection (memory vs compute, ~2K tokens)
./scripts/ncu-llm bottleneck ./executable

# Memory deep-dive (~40K tokens)
./scripts/ncu-llm memory ./executable

# Compute deep-dive (~35K tokens)
./scripts/ncu-llm compute ./executable
```

## Output

All profiling outputs are saved to `./ncu-llm-output/` with timestamped filenames:

```
./ncu-llm-output/
├── quick-20251024-120000.txt       # Human-readable report
├── quick-20251024-120000.csv       # CSV data (for parsing)
├── quick-20251024-120000-raw.ncu-rep  # Binary NCU report
└── index.txt                        # Run history
```

## Editing the Scripts

### To Modify Profiling Modes

**Edit `ncu-llm-lib.sh`** - This file contains all the profiling functions.

Each function follows this pattern:

```bash
ncu_llm_<mode>() {
    init_output

    # Define which metrics to collect
    local metrics="metric1,metric2,metric3"

    # Run NCU
    ncu --metrics "$metrics" -o "$output_rep" "$@"

    # Export to CSV
    ncu --import "${output_rep}.ncu-rep" --csv > "$output_csv"

    # Generate human-readable report
    # ... (custom formatting)
}
```

### Example: Adding a New Profiling Mode

To add a new mode called "registers":

1. **Edit `ncu-llm-lib.sh`** and add:

```bash
# ============================================================================
# Command: registers - Register usage analysis
# ============================================================================
ncu_llm_registers() {
    init_output

    echo "ncu-llm: Running REGISTERS profile..."

    local output_txt=$(get_output_file "registers" ".txt")
    local output_csv=$(get_output_file "registers" ".csv")
    local output_rep=$(get_output_file "registers" "-raw")

    # Metrics for register usage
    local metrics="launch__registers_per_thread"
    metrics="$metrics,launch__registers_per_thread_allocated"
    metrics="$metrics,sm__maximum_warps_per_active_cycle_pct"

    # Profile
    ncu --metrics "$metrics" \
        --launch-count 1 \
        -o "$output_rep" \
        "$@" > /dev/null 2>&1

    # Export to CSV
    ncu --import "${output_rep}.ncu-rep" --page details --csv > "$output_csv" 2>/dev/null

    # Generate report
    cat > "$output_txt" << EOF
NCU-LLM Register Usage Report
Generated: $(date)
Command: $@

Register analysis results...
EOF

    # Add custom reporting logic here

    log_to_index "registers" "$output_txt, $output_csv, $output_rep.ncu-rep"

    echo "Results: $output_txt"
}
```

2. **Edit `ncu-llm`** and add the command to the dispatcher:

```bash
case "$command" in
    quick)
        ncu_llm_quick "$@"
        ;;
    standard)
        ncu_llm_standard "$@"
        ;;
    # ... other commands ...
    registers)    # <-- ADD THIS
        ncu_llm_registers "$@"
        ;;
    # ...
esac
```

3. **Test it:**

```bash
./scripts/ncu-llm registers ./test-kernel
```

### Example: Modifying Quick Mode

To add more metrics to the "quick" mode:

**Edit `ncu-llm-lib.sh`** around line 52:

```bash
# Add your new metric to the list
local metrics="dram__throughput.avg.pct_of_peak_sustained_elapsed"
metrics="$metrics,sm__throughput.avg.pct_of_peak_sustained_elapsed"
metrics="$metrics,gpu__time_duration.sum"
metrics="$metrics,launch__occupancy_limit_warps"
metrics="$metrics,launch__occupancy_limit_blocks"
metrics="$metrics,l1tex__t_sector_hit_rate.pct"
metrics="$metrics,lts__t_sector_hit_rate.pct"
metrics="$metrics,smsp__cycles_elapsed.avg"
metrics="$metrics,YOUR_NEW_METRIC_HERE"    # <-- ADD HERE
```

⚠️ **Warning:** Adding too many metrics increases profiling overhead and token count!

### Example: Changing Output Format

To modify the text report format, edit the section that generates the report:

**In `ncu-llm-lib.sh`**, find the `cat > "$output_txt"` section:

```bash
# Generate human-readable report
cat > "$output_txt" << EOF
NCU-LLM Quick Profile Report
Generated: $(date)
Command: $@

═══════════════════════════════════════════════════════════════

YOUR CUSTOM HEADER HERE

ESSENTIAL METRICS (8 total)
EOF
```

### To Change Output Directory

Set the environment variable:

```bash
export NCU_LLM_OUTPUT_DIR=/path/to/your/output
./scripts/ncu-llm quick ./executable
```

Or modify the default in `ncu-llm`:

```bash
export NCU_LLM_OUTPUT_DIR="${NCU_LLM_OUTPUT_DIR:-/your/custom/path}"
```

## Common Modifications

### 1. Increase Profiling Verbosity

In `ncu-llm-lib.sh`, remove the `> /dev/null 2>&1` redirections:

```bash
# Before:
ncu --metrics "$metrics" -o "$output_rep" "$@" > /dev/null 2>&1

# After (shows NCU progress):
ncu --metrics "$metrics" -o "$output_rep" "$@"
```

### 2. Profile Multiple Kernel Launches

Change `--launch-count 1` to a higher number:

```bash
ncu --metrics "$metrics" \
    --launch-count 10 \    # Profile 10 launches instead of 1
    -o "$output_rep" \
    "$@"
```

### 3. Filter Specific Kernels

Add kernel filtering:

```bash
ncu --metrics "$metrics" \
    --kernel-name "mySpecificKernel" \    # Only profile this kernel
    --launch-count 1 \
    -o "$output_rep" \
    "$@"
```

### 4. Add Source-Level Profiling

For source correlation (requires `-lineinfo` compilation):

```bash
ncu --metrics "$metrics" \
    --section SourceCounters \    # Add source-level data
    --launch-count 1 \
    -o "$output_rep" \
    "$@"
```

## Available NCU Metrics

To see all available metrics:

```bash
ncu --query-metrics
```

To see all available sections:

```bash
ncu --query-sections
```

## Testing Your Changes

After editing the scripts:

1. **Test manually:**
   ```bash
   ./scripts/ncu-llm quick ./test-kernel
   ```

2. **Check output:**
   ```bash
   cat ./ncu-llm-output/quick-*.txt
   ```

3. **Test with NCU-Claude:**
   ```bash
   npm run dev -- analyze test-kernel.cu
   ```

## Script Structure

### `ncu-llm` (Main Script)

- Command dispatcher
- Argument parsing
- Help text
- Calls functions from `ncu-llm-lib.sh`

### `ncu-llm-lib.sh` (Library)

Functions:
- `init_output()` - Initialize output directory and timestamp
- `log_to_index()` - Log runs to index file
- `get_output_file()` - Generate output filenames
- `ncu_llm_quick()` - Quick profiling mode
- `ncu_llm_standard()` - Standard profiling mode
- `ncu_llm_bottleneck()` - Bottleneck detection
- `ncu_llm_memory()` - Memory analysis (placeholder)
- `ncu_llm_compute()` - Compute analysis (placeholder)
- etc.

## Troubleshooting

**Script not executable:**
```bash
chmod +x ./scripts/ncu-llm ./scripts/ncu-llm-lib.sh
```

**NCU not found:**
```bash
which ncu
# If not found, add to PATH or use full path
```

**Permission denied on output directory:**
```bash
mkdir -p ./ncu-llm-output
chmod 755 ./ncu-llm-output
```

**Python import errors:**
The scripts use embedded Python for CSV parsing. Ensure Python 3 is available:
```bash
which python3
```

## References

- [NCU Documentation](https://docs.nvidia.com/nsight-compute/)
- [NCU Metrics Reference](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)
- [NCU CLI Options](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

## Contributing

When modifying these scripts:

1. ✅ Keep outputs under 50K tokens per file
2. ✅ Use consistent formatting (grep-able)
3. ✅ Add comments explaining your changes
4. ✅ Test with sample kernels before committing
5. ✅ Update this README if adding new features

---

**Location in Repo:** `/home/ubuntu0/KernelBench/ncu-claude/scripts/`
**Used By:** Claude Agent SDK via NCU-Claude tool
**Last Updated:** 2025-10-24
