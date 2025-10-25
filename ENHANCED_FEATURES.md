# NCU-Claude Enhanced Features

This document describes the enhanced features added to the NCU-Claude tool for improved observability and diagnostics configuration.

## 🎯 New Features

### 1. Enhanced Tool Execution Observability

The tool now shows **detailed information** about every tool Claude executes, not just "Executing tool: Bash".

**What You See Now:**

```
🔧 Bash
   └─ nvcc test-kernel.cu -o test-kernel

🔧 Bash
   └─ /home/ubuntu0/ncu-llm quick ./test-kernel

🔧 Read
   └─ Reading: ./ncu-llm-output/quick-20251024-120000.txt

🔧 Grep
   └─ Searching for: Memory Throughput
```

**Implementation:**
- Enhanced `claude-agent.ts` to extract detailed tool use information from Claude's message stream
- Added `toolDetails` to `AnalysisProgress` interface with `toolName`, `command`, `description`
- Updated TUI to display tool execution details with tree structure

### 2. Side-by-Side Diff Visualization

Instead of showing a plain unified diff, the final hot-fix is now displayed as a **side-by-side comparison** with color coding.

**Example Output:**

```
═══════════════════════════════════════════════════════════════════════════════
Original                                    │ Optimized
═══════════════════════════════════════════════════════════════════════════════
  1 void myKernel(float* data) {            │   1 void myKernel(float* data) {
  2   float val = data[tid * 128];          │   2   __shared__ float tile[256];
                                             │   3   tile[tid] = data[tid];
                                             │   4   __syncthreads();
                                             │   5   float val = tile[tid];
  3 }                                        │   6 }
═══════════════════════════════════════════════════════════════════════════════
```

**Implementation:**
- Created `src/utils/side-by-side-diff.ts` with side-by-side diff rendering
- Functions: `generateSideBySideDiff()`, `formatHotFix()`, `extractCodeFromDiffString()`
- Integrated into TUI results display

### 3. Interactive Diagnostic Configuration

Select which NCU diagnostics Claude should use via an **interactive TUI selector**.

**How to Use:**

```bash
# Method 1: Use during analyze
ncu-claude analyze kernel.cu --configure-diagnostics

# Method 2: Configure separately
ncu-claude configure
```

**Selector Interface:**

```
╭─────────────────────────────────────────────────────────────────╮
│ 📊 NCU Diagnostic Configuration                                 │
│ Select diagnostics to guide Claude's NCU profiling strategy     │
╰─────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────╮
│ Controls:                                                        │
│ ↑/↓ Navigate   Space Toggle   A Select All   N Select None      │
│ Enter Confirm   Q Cancel                                         │
╰─────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────╮
│ Available Diagnostics (3 selected)                              │
│                                                                  │
│ → [✓] Quick Diagnostic (SpeedOfLight) [CRITICAL] (low overhead) │
│       Fast high-level overview - determine if memory or...       │
│   [✓] Memory Bottleneck Diagnosis [HIGH] (high overhead)        │
│   [ ] Occupancy & Roofline Analysis [HIGH] (medium overhead)    │
│   [✓] Warp Stall & Latency Analysis [HIGH] (high overhead)      │
│   [ ] Compute Workload Analysis [MEDIUM] (medium overhead)      │
│   [ ] Tensor Core & GEMM Optimization [MEDIUM] (low overhead)   │
│   [ ] Cache Control & Persistence [LOW] (medium overhead)       │
│   [ ] Kernel Filtering & Overhead Reduction [UTILITY] (varies)  │
│   [ ] Export to CSV for Post-Processing [UTILITY] (none)        │
╰─────────────────────────────────────────────────────────────────╯
```

**Features:**
- Navigate with arrow keys
- Toggle selections with Space
- Select All (A) or Select None (N)
- Color-coded priorities: RED (Critical/High), YELLOW (Medium), GRAY (Low)
- Color-coded overhead: RED (High), YELLOW (Medium), GREEN (Low)
- Shows description when hovering over an item

**Implementation:**
- `diagnostics-database.json` - Complete database of all NCU diagnostics
- `src/services/diagnostics-compiler.ts` - Compiles selections into LLM context
- `src/diagnostic-selector.tsx` - Interactive TUI selector component
- CLI integration in `src/index.ts`

## 📊 Diagnostics Database

The tool includes a comprehensive database of NCU diagnostic configurations:

### Available Diagnostics

1. **Quick Diagnostic (SpeedOfLight)** [CRITICAL, low overhead]
   - Fast determination of memory vs compute bound
   - Always recommended as first step

2. **Memory Bottleneck Diagnosis** [HIGH, high overhead]
   - Cache hierarchy analysis (L1/L2/DRAM)
   - Coalescing detection
   - Roofline charts

3. **Occupancy & Roofline Analysis** [HIGH, medium overhead]
   - Theoretical vs achieved occupancy
   - Register and shared memory pressure

4. **Warp Stall & Latency Analysis** [HIGH, high overhead]
   - Warp stall reasons breakdown
   - Scheduler statistics
   - Source-level correlation

5. **Compute Workload Analysis** [MEDIUM, medium overhead]
   - Pipeline utilization (FMA, FP64, Tensor cores)
   - Instruction mix statistics

6. **Tensor Core & GEMM Optimization** [MEDIUM, low overhead]
   - Tensor Core usage verification
   - Matrix operation optimization

7. **Cache Control & Persistence** [LOW, medium overhead]
   - Inter-kernel data reuse analysis

8. **Kernel Filtering** [UTILITY, varies]
   - Smart kernel selection to reduce profiling time

9. **CSV Export** [UTILITY, none overhead]
   - Programmatic analysis of NCU data

### Pre-defined Workflows

The database also includes recommended workflows:

1. **Initial Profiling** - Start with quick diagnostic, then deep dive based on results
2. **Memory Bottleneck Investigation** - Comprehensive memory analysis
3. **Low Occupancy Investigation** - Occupancy and stall analysis
4. **Iterative Solver Profiling** - Skip warmup iterations

### NCU Flag Combinations

Each diagnostic includes multiple flag variants:

```json
{
  "id": "memory_bottleneck",
  "flags": {
    "comprehensive": "--set full -o memory_report",
    "focused": "--section MemoryWorkloadAnalysis --section SpeedOfLight -o report",
    "cache_analysis": "--metrics lts__t_sectors_hit_rate.pct,lts__t_sectors_miss_rate.pct -o cache_report",
    "roofline": "--section SpeedOfLight_RooflineChart --set detailed -o roofline"
  }
}
```

### LLM Guidance

Each diagnostic includes specific guidance for Claude:

```
"llm_guidance": "Focus on memory access patterns, cache utilization, and coalescing.
                 Recommend shared memory tiling or access pattern changes."
```

## 🚀 Usage Examples

### Example 1: Quick Analysis with Default Settings

```bash
ncu-claude analyze kernel.cu
```

Uses default quick diagnostic workflow.

### Example 2: Interactive Diagnostic Selection

```bash
ncu-claude analyze kernel.cu --configure-diagnostics
```

1. Launches diagnostic selector
2. Choose which diagnostics to enable
3. Proceeds with analysis using selected diagnostics

### Example 3: With Custom Context + Diagnostics

```bash
ncu-claude analyze kernel.cu \
  --context my-preferences.txt \
  --configure-diagnostics
```

Combines user preferences with selected diagnostics.

### Example 4: Configure Diagnostics First

```bash
# First, preview and configure
ncu-claude configure

# Then analyze (you'll be prompted for diagnostics)
ncu-claude analyze kernel.cu --configure-diagnostics
```

## 🔧 Technical Details

### Architecture

```
User Request
    ↓
CLI (index.ts)
    ↓
Diagnostic Selector (diagnostic-selector.tsx) [if --configure-diagnostics]
    ↓
Diagnostics Compiler (diagnostics-compiler.ts)
    ↓
Prompt Loader (prompt-loader.ts)
    ↓
Claude Agent (claude-agent.ts) - Enhanced with tool observability
    ↓
TUI (tui.tsx) - Enhanced with side-by-side diff
    ↓
Results Display
```

### Data Flow

1. **Diagnostic Selection** → Set of diagnostic IDs
2. **Compilation** → LLM context string with NCU flags and guidance
3. **Prompt Building** → Full prompt = system + diagnostics + user context + code
4. **Execution** → Claude uses diagnostics to choose NCU commands
5. **Tool Observability** → Real-time display of tool execution details
6. **Result Formatting** → Side-by-side diff visualization

### Files Modified/Created

**New Files:**
- `diagnostics-database.json` - Complete diagnostic database
- `src/services/diagnostics-compiler.ts` - Diagnostic compiler service
- `src/diagnostic-selector.tsx` - Interactive TUI selector
- `src/utils/side-by-side-diff.ts` - Diff visualization utilities

**Modified Files:**
- `src/services/claude-agent.ts` - Enhanced tool observability
- `src/services/prompt-loader.ts` - Diagnostic context integration
- `src/tui.tsx` - Enhanced progress display + side-by-side diff
- `src/index.ts` - CLI commands for diagnostics

## 📝 Output Format

### Progress Display (Enhanced)

```
🔍 Analyzing test-kernel.cu...

🤔 Analyzing memory access patterns...
🔧 Bash
   └─ nvcc -lineinfo test-kernel.cu -o test-kernel
🔧 Bash
   └─ /home/ubuntu0/ncu-llm quick ./test-kernel
🔧 Read
   └─ Reading: ./ncu-llm-output/quick-20251024-120000.txt
📝 Analyzing NCU output...
✅ Analysis complete!
```

### Results Display (Enhanced)

```
╭─────────────────────────────────────────────────────────────────╮
│ ✅ Analysis Complete                                             │
╰─────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────╮
│ ⚡ P1 Actionable Insight:                                        │
│                                                                  │
│ Uncoalesced global memory access causing 85% memory stalls      │
╰─────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────╮
│ 🔧 Hot Fix (Side-by-Side Diff):                                 │
│                                                                  │
│ Original                           │ Optimized                   │
│ ═══════════════════════════════════════════════════════════════ │
│   1 float val = data[tid * 128];  │   1 __shared__ float tile[] │
│                                    │   2 tile[tid] = data[tid];  │
│                                    │   3 __syncthreads();        │
│                                    │   4 float val = tile[tid];  │
╰─────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────╮
│ 📖 Explanation:                                                  │
│                                                                  │
│ NCU shows Memory Throughput at 15% with warp stalls on memory.  │
│ Root cause: Strided access pattern (tid * 128).                 │
│                                                                  │
│ Fix: Use shared memory tiling to coalesce accesses.             │
│                                                                  │
│ References:                                                      │
│ - NVIDIA CUDA Programming Guide, Section 5.3.2                  │
│   https://docs.nvidia.com/cuda/cuda-c-programming-guide/...     │
╰─────────────────────────────────────────────────────────────────╯
```

## 🎨 Benefits

1. **Better Observability**: See exactly what NCU commands are being run
2. **Visual Clarity**: Side-by-side diff makes changes immediately clear
3. **Flexible Configuration**: Choose exactly which diagnostics to use
4. **Reduced Overhead**: Select only necessary diagnostics to minimize profiling time
5. **Guided Analysis**: Database provides structured guidance to Claude

## 🔮 Future Enhancements

Potential future additions:
- Save/load diagnostic profiles
- Custom diagnostic definitions
- Workflow templates
- Integration with visualization tools
- Multi-file analysis support
