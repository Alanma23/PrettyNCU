# NCU-LLM Scripts Integration

This document explains how the `ncu-llm` scripts were integrated into the NCU-Claude repository.

## Overview

The `ncu-llm` wrapper scripts are now **part of the NCU-Claude repository** in the `scripts/` directory. This makes them easy to:

- ✅ **Version control** - Track changes to profiling modes
- ✅ **Edit** - Modify profiling behavior without leaving the repo
- ✅ **Customize** - Add new profiling modes or metrics
- ✅ **Deploy** - Everything needed is in one place

## What Was Done

### 1. Scripts Copied to Repo

**From:** `/home/ubuntu0/ncu-llm` and `/home/ubuntu0/ncu-llm-lib.sh`

**To:** `/home/ubuntu0/KernelBench/ncu-claude/scripts/`

```
scripts/
├── ncu-llm          # Main wrapper script (chmod +x)
├── ncu-llm-lib.sh   # Library functions (chmod +x)
└── README.md        # Complete editing guide
```

### 2. Path Updates

**Updated Files:**

1. **`src/services/prompt-loader.ts`** - Changed paths in task instructions:
   ```
   OLD: /home/ubuntu0/ncu-llm
   NEW: ./scripts/ncu-llm
   ```

2. **`README.md`** - Updated architecture diagram and added customization section

3. **`QUICKSTART.md`** - Updated troubleshooting to reference new location

### 3. Documentation Created

**New File:** `scripts/README.md` (comprehensive guide)

Includes:
- ✅ What ncu-llm is and how it works
- ✅ How to add new profiling modes
- ✅ How to modify existing modes
- ✅ How to change metrics
- ✅ How to customize output formatting
- ✅ Complete examples and code templates

## How to Use

### Running Scripts Directly

```bash
# From ncu-claude directory
./scripts/ncu-llm quick ./test-kernel
./scripts/ncu-llm standard ./test-kernel
./scripts/ncu-llm bottleneck ./test-kernel
```

### Via NCU-Claude

Claude automatically uses the scripts during analysis:

```bash
npm run dev -- analyze test-kernel.cu
```

You'll see in the TUI:
```
🔧 Bash
   └─ ./scripts/ncu-llm quick ./kernel_executable
```

## Editing the Scripts

### Quick Example: Add a Metric

**Edit `scripts/ncu-llm-lib.sh`** around line 52:

```bash
# Add your metric to the quick mode
local metrics="dram__throughput.avg.pct_of_peak_sustained_elapsed"
metrics="$metrics,sm__throughput.avg.pct_of_peak_sustained_elapsed"
# ... existing metrics ...
metrics="$metrics,YOUR_NEW_METRIC_HERE"    # <-- ADD HERE
```

**No rebuild needed!** Changes take effect immediately.

### Example: Create New Mode

**1. Edit `scripts/ncu-llm-lib.sh`** - Add function:

```bash
ncu_llm_mymode() {
    init_output

    local output_txt=$(get_output_file "mymode" ".txt")
    local output_csv=$(get_output_file "mymode" ".csv")
    local output_rep=$(get_output_file "mymode" "-raw")

    local metrics="your,metrics,here"

    ncu --metrics "$metrics" -o "$output_rep" "$@" > /dev/null 2>&1
    ncu --import "${output_rep}.ncu-rep" --csv > "$output_csv" 2>/dev/null

    # Generate report...

    log_to_index "mymode" "$output_txt, $output_csv"
    echo "Results: $output_txt"
}
```

**2. Edit `scripts/ncu-llm`** - Add to dispatcher:

```bash
case "$command" in
    # ... existing commands ...
    mymode)
        ncu_llm_mymode "$@"
        ;;
esac
```

**3. Test it:**
```bash
./scripts/ncu-llm mymode ./test-kernel
```

See `scripts/README.md` for more detailed examples!

## Directory Structure

### Before Integration

```
/home/ubuntu0/
├── ncu-llm              # Standalone script
├── ncu-llm-lib.sh       # Standalone library
└── KernelBench/
    └── ncu-claude/
        └── ... (no scripts)
```

### After Integration

```
/home/ubuntu0/KernelBench/ncu-claude/
├── scripts/
│   ├── ncu-llm          # ✅ Now in repo
│   ├── ncu-llm-lib.sh   # ✅ Now in repo
│   └── README.md        # ✅ Complete guide
├── src/
├── prompts/
└── ...
```

## Benefits

### 1. Easy Editing
No need to navigate to a different directory. Everything is in one repo:

```bash
cd /home/ubuntu0/KernelBench/ncu-claude
vim scripts/ncu-llm-lib.sh    # Edit profiling behavior
vim prompts/ncu-analyzer.txt  # Edit Claude's strategy
```

### 2. Version Control
All changes to profiling modes are tracked in git:

```bash
git log scripts/ncu-llm-lib.sh
git diff scripts/ncu-llm-lib.sh
```

### 3. Deployment
Deploy everything together:

```bash
git clone your-repo
cd ncu-claude
npm install
npm run build
# Scripts are already there!
```

### 4. Customization
Users can easily customize profiling for their specific needs:

```bash
# Add GPU-specific metrics
vim scripts/ncu-llm-lib.sh

# Add custom profiling mode
vim scripts/ncu-llm-lib.sh
vim scripts/ncu-llm

# Test immediately
./scripts/ncu-llm mymode ./test
```

## Available Profiling Modes

The scripts include these modes (see `scripts/README.md` for details):

| Mode | Token Count | Use Case |
|------|------------|----------|
| `quick` | ~5K | Fast checks, CI/CD |
| `standard` | ~30K | Standard workflow |
| `bottleneck` | ~2K | Memory vs compute? |
| `memory` | ~40K | Memory deep-dive |
| `compute` | ~35K | Compute deep-dive |
| `compare` | ~20K | Side-by-side comparison |
| `insights` | ~10K | Recommendations only |
| `summary` | ~5K | Executive overview |

## Configuration

### Change Output Directory

```bash
export NCU_LLM_OUTPUT_DIR=/your/custom/path
./scripts/ncu-llm quick ./kernel
```

### Disable Timestamps

```bash
export NCU_LLM_TIMESTAMP=no
./scripts/ncu-llm quick ./kernel
```

### Verbose Output

```bash
export NCU_LLM_VERBOSE=yes
./scripts/ncu-llm quick ./kernel
```

## Integration with Claude

Claude is instructed to use these scripts via the prompts:

**From `prompts/system-prompt.txt`:**
```
Available NCU profiling modes via ncu-llm:
- quick: Ultra-minimal (8 metrics, ~5K tokens)
- bottleneck: Memory vs compute determination
- standard: Balanced analysis (core metrics + insights)
```

**From `src/services/prompt-loader.ts`:**
```
- Use the ncu-llm wrapper script located at ./scripts/ncu-llm
- Example: ./scripts/ncu-llm quick ./kernel_executable
```

Claude sees these instructions and automatically uses the scripts during analysis.

## Troubleshooting

### Script Not Executable

```bash
chmod +x ./scripts/ncu-llm ./scripts/ncu-llm-lib.sh
```

### Wrong Path in Prompts

If Claude can't find the script:

1. Check the path in prompts:
   ```bash
   grep "ncu-llm" prompts/*.txt
   grep "ncu-llm" src/services/prompt-loader.ts
   ```

2. Ensure it's `./scripts/ncu-llm` (relative to ncu-claude directory)

### Script Changes Not Taking Effect

- **Prompts**: Rebuild TypeScript (`npm run build`)
- **Scripts**: No rebuild needed, changes are immediate

### NCU Command Not Found

The script calls `ncu` - ensure it's in PATH:

```bash
which ncu
# If not found, install NCU or add to PATH
```

## Advanced: Creating Domain-Specific Modes

You can create specialized profiling modes for specific domains:

### Example: Ray Tracing Mode

**Edit `scripts/ncu-llm-lib.sh`:**

```bash
ncu_llm_raytracing() {
    init_output

    # Ray tracing specific metrics
    local metrics="sm__inst_executed_pipe_tensor_op_hmma.sum"
    metrics="$metrics,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
    metrics="$metrics,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
    metrics="$metrics,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"

    # Profile and report...
}
```

### Example: ML Training Mode

```bash
ncu_llm_mltraining() {
    # Focus on Tensor Cores, memory bandwidth, mixed precision
    local metrics="sm__inst_executed_pipe_tensor_op_hmma.sum"
    metrics="$metrics,dram__bytes.sum"
    metrics="$metrics,sm__inst_executed_pipe_fp16.sum"
    # ...
}
```

See `scripts/README.md` for complete templates!

## Summary

✅ **Scripts are now in the repo** at `scripts/`
✅ **Easy to edit** with complete documentation
✅ **No rebuild needed** for script changes
✅ **Version controlled** with the rest of the project
✅ **Fully integrated** with Claude's prompts

**Key Files:**
- `scripts/ncu-llm` - Main wrapper
- `scripts/ncu-llm-lib.sh` - Library with all profiling modes
- `scripts/README.md` - Complete editing guide

**To customize profiling:**
```bash
vim scripts/ncu-llm-lib.sh
```

No rebuild, no hassle, immediate effect! 🎉

---

**Created:** 2025-10-24
**Location:** `/home/ubuntu0/KernelBench/ncu-claude/`
