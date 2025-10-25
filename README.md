# PrettyNCU: Intelligent CUDA Profiling with AI

An interactive CLI/TUI tool that uses PrettyNCU to intelligently profile CUDA kernels with NVIDIA Nsight Compute (NCU), automatically selecting minimal metrics and generating actionable optimization insights all in terminal.

## Features

### Core Capabilities
- 🤖 **AI-Powered Analysis**: Sonnet 4.5 chooses the right NCU profiling mode
- 🔬 **Diagnostic Presets**: 9 built-in presets for common profiling scenarios (memory, compute, occupancy, tensor, etc.)
- 📊 **Minimal Profiling Overhead**: Collects only the metrics needed to identify bottlenecks
- 🎯 **Actionable Insights**: Prioritized optimization recommendations (P1-P4)
- 🔧 **Hot Fixes**: Concrete code changes in diff format with side-by-side visualization
- 📚 **Documentation References**: Explanations cite official NVIDIA docs and reputable sources

### Enhanced User Experience
- 🎨 **Interactive TUI**: Beautiful terminal interface with real-time progress
- 👁️ **Enhanced Observability**: See exactly which tools are executed with command details
- 🔍 **Side-by-Side Diff**: Visual comparison of original vs optimized code in terminal
- ⚠️ **Mandatory NCU Execution**: Enforced profiling workflow - no static analysis without data

### Customization & Control
- ⚙️ **Customizable Context**: Provide optimization preferences and constraints via files or inline
- 🛠️ **Custom NCU Commands**: Run your own specific NCU profiling commands
- 📋 **Externalized Prompts**: All system prompts in editable .txt files
- 🔧 **Editable Profiling Scripts**: Modify NCU wrapper scripts without rebuilding

## Quick Start

```bash
# Clone and setup
cd ncu-claude
npm install
npm run build

# Make scripts executable
chmod +x ./scripts/ncu-llm ./scripts/ncu-llm-lib.sh

# Analyze a kernel
npm run dev -- analyze test-kernel.cu
```

That's it! PrettyNCU will compile your kernel, run NCU profiling, and provide optimization insights.

## Prerequisites

- Node.js 18+ and npm
- NVIDIA GPU with CUDA toolkit installed
- NCU (NVIDIA Nsight Compute) installed
- `ncu-llm` wrapper script (included in `scripts/` directory)
- MCP API access (via PrettyNCU Agent SDK)

## Installation

```bash
cd ncu-claude
npm install
npm run build
```

Optionally, install globally:
```bash
npm link
```

## Usage

### Basic Usage

Analyze a CUDA kernel with default settings:

```bash
npm run dev -- analyze test-kernel.cu
```

Or if installed globally:
```bash
ncu-claude analyze test-kernel.cu
```

### With Diagnostic Presets

Guide PrettyNCU profiling strategy with intelligent presets:

```bash
# List all available diagnostic presets
npm run dev -- diagnostics

# Use a specific preset
npm run dev -- analyze kernel.cu --diagnostics memory
npm run dev -- analyze kernel.cu --diagnostics compute
npm run dev -- analyze kernel.cu --diagnostics tensor
```

**Available Presets** (9 total):
- `quick` - Fast SpeedOfLight analysis (default)
- `memory` - Memory bottleneck investigation
- `compute` - Compute pipeline analysis
- `occupancy` - Low occupancy debugging
- `latency` - Warp stall analysis
- `tensor` - Tensor Core optimization
- `branching` - Divergent warp analysis
- `registers` - Register pressure investigation
- `full` - Comprehensive profiling (very slow)

The diagnostics system provides:
- Interactive TUI for selecting multiple diagnostics
- Pre-configured NCU flag combinations
- LLM-optimized guidance for each scenario
- Workflow recommendations for common issues

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for detailed documentation.

### With Custom Context

Provide optimization preferences via a context file:

```bash
ncu-claude analyze kernel.cu --context my-preferences.txt
```

Example context file (`my-preferences.txt`):
```
TARGET: Real-time inference kernels
PRIORITY: Minimize latency over throughput
HARDWARE: RTX 4090
CONSTRAINTS: Must maintain FP32 precision
STYLE: Prefer simple optimizations
DOCS: Include code examples in explanations
```

### With Inline Context

```bash
ncu-claude analyze kernel.cu --context-inline "Focus on L1 cache optimization. Target A100 GPU."
```

### With Custom NCU Commands

Provide your own specific NCU commands to execute:

**Single command:**
```bash
ncu-claude analyze kernel.cu --ncu-command "ncu --metrics sm__warps_active.avg -o warps"
```

**Multiple commands from file:**

Create `my-commands.txt`:
```
# Memory analysis
ncu --section MemoryWorkloadAnalysis -o memory ./kernel

# Compute analysis
ncu --section ComputeWorkloadAnalysis -o compute ./kernel
```

Then run:
```bash
ncu-claude analyze kernel.cu --ncu-commands-file my-commands.txt
```

See [CUSTOM_NCU_COMMANDS.md](CUSTOM_NCU_COMMANDS.md) for complete documentation and examples.

### Combined: Diagnostics + Context

Combine diagnostic presets with custom context:

```bash
ncu-claude analyze kernel.cu --diagnostics memory --context-inline "Target RTX 4090, focus on coalescing"
```

### Interactive Mode

Prompts you to enter context interactively:

```bash
ncu-claude analyze kernel.cu --interactive
```

## How It Works

1. **Upload**: Provide a `.cu` CUDA kernel file
2. **Context**: Add custom optimization preferences (or use sensible defaults)
3. **AI Analysis**: PrettyNCU analyzes your code and intelligently selects NCU profiling mode
4. **Profiling**: Executes NCU commands automatically via the Agent SDK
5. **Insights**: Parses NCU output and identifies the highest-priority bottleneck
6. **Hot Fix**: Generates concrete code changes with explanations
7. **Output**: Returns structured JSON with `actionable_insight`, `hot_fix`, and `explanation`

### Enhanced Observability

The TUI provides detailed visibility into the profiling process:

**Tool Execution Details**: See exactly what PrettyNCU is running:
```
🔧 Bash
   └─ nvcc test-kernel.cu -o kernel_executable

🔧 Bash
   └─ ./scripts/ncu-llm quick ./kernel_executable

📄 Read
   └─ Reading: ./ncu-llm-output/quick-20251024-120000.txt
```

**Side-by-Side Diff Visualization**: Hot fixes are displayed as visual comparisons:
```
┌─ Original Code ─────────────────┐  ┌─ Optimized Code ────────────────┐
│ output[tid] = input[tid*STRIDE] │  │ __shared__ float tile[256];     │
│                                 │  │ tile[tid] = input[tid];         │
│                                 │  │ __syncthreads();                │
│                                 │  │ output[tid] = tile[tid] * 2.0f; │
└─────────────────────────────────┘  └─────────────────────────────────┘
```

**NCU Metadata**: Full transparency into profiling commands and outputs:
- Exact NCU commands that were executed
- Paths to all generated output files (txt, csv, ncu-rep)
- Preview of raw NCU data for manual inspection

## Architecture

```
ncu-claude/
├── src/
│   ├── index.ts                      # CLI entry point (commander.js)
│   ├── tui.tsx                       # Terminal UI with enhanced observability
│   ├── diagnostic-selector.tsx       # Interactive diagnostics selector TUI
│   ├── services/
│   │   ├── claude-agent.ts           # Claude Agent SDK integration
│   │   ├── prompt-loader.ts          # Loads and combines prompt templates
│   │   └── diagnostics-compiler.ts   # Diagnostics configuration compiler
│   └── utils/
│       ├── code-diff.ts              # Diff formatting utilities
│       └── side-by-side-diff.ts      # Side-by-side diff visualization
├── prompts/
│   ├── system-prompt.txt             # Main Claude instructions (with NCU enforcement)
│   ├── ncu-analyzer.txt              # NCU metric selection strategy
│   ├── code-optimizer.txt            # Hot-fix generation guidelines
│   └── default-context.txt           # Default optimization preferences
├── scripts/
│   ├── ncu-llm                       # NCU wrapper script (8 profiling modes)
│   ├── ncu-llm-lib.sh                # Library functions for ncu-llm
│   └── README.md                     # Complete editing guide for scripts
├── diagnostics-database.json         # 9 diagnostic presets with NCU flags
├── test-kernel.cu                    # Sample kernel for testing
└── *.md                              # Comprehensive documentation files
```

**Key Components:**

- **CLI/TUI Layer**: Commander.js for argument parsing, Ink for interactive terminal UI
- **Claude Integration**: Agent SDK with bypass permissions for automatic NCU execution
- **Diagnostics System**: JSON database with 9 presets, interactive selector, context compiler
- **Profiling Scripts**: Bash-based NCU wrappers with token-optimized output
- **Prompt System**: Externalized prompts with mandatory NCU execution enforcement
- **Observability**: Detailed tool execution tracking and side-by-side diff rendering

## Customization

### Prompt System

All Claude prompts are externalized to `.txt` files in the `prompts/` directory:

- **`system-prompt.txt`**: Core instructions for CUDA optimization
- **`ncu-analyzer.txt`**: Strategy for choosing NCU profiling modes
- **`code-optimizer.txt`**: Guidelines for generating hot-fixes
- **`default-context.txt`**: Default user preferences

This makes it easy to customize Claude's behavior without changing code.

### NCU Profiling Scripts

The `scripts/` directory contains the NCU wrapper scripts that Claude uses for profiling:

- **`ncu-llm`**: Main wrapper providing different profiling modes (quick, standard, bottleneck, etc.)
- **`ncu-llm-lib.sh`**: Library functions with metric definitions and reporting logic

**To modify profiling behavior:**
1. Edit `scripts/ncu-llm-lib.sh` to change metrics or add new modes
2. See `scripts/README.md` for detailed examples and instructions
3. Changes take effect immediately (no rebuild needed)

**Common modifications:**
- Add/remove metrics from existing modes
- Create new profiling modes
- Change output formatting
- Adjust profiling overhead settings

See [scripts/README.md](scripts/README.md) for complete documentation on editing the profiling scripts.

### Mandatory NCU Execution

NCU-Claude **enforces actual profiling execution** - Claude cannot provide optimization insights based solely on static code analysis. This is enforced at three levels:

1. **System Prompt**: Core instructions emphasize mandatory NCU execution
2. **Task Instructions**: Step-by-step workflow that requires compilation → profiling → analysis
3. **Analyzer Prompt**: NCU selection strategy explicitly requires execution

**Why this matters:**
- ❌ No guessing about performance issues
- ✅ All insights are based on actual profiling data
- ✅ Recommendations are evidence-based and actionable
- ✅ You get the exact NCU commands that were executed for reproducibility

See [MANDATORY_NCU_EXECUTION.md](MANDATORY_NCU_EXECUTION.md) for implementation details.

## Output Format

The tool returns a JSON object with these fields:

```json
{
  "actionable_insight": "Uncoalesced global memory access causing 85% memory stalls",
  "hot_fix": "@@ kernel.cu @@\n- float val = data[tid * STRIDE];\n+ __shared__ float tile[256];\n+ tile[tid] = data[tid];\n+ __syncthreads();\n+ float val = tile[tid];",
  "explanation": "NCU shows Memory Throughput at 15% with warp stalls on memory dependencies. Root cause: strided access pattern. Fix uses shared memory tiling to coalesce accesses.\n\nReferences:\n- NVIDIA CUDA Programming Guide, Section 5.3.2\n  https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses",
  "ncu_metadata": {
    "commands_executed": [
      "ncu --section SpeedOfLight -o quick_check ./a.out",
      "ncu --section MemoryWorkloadAnalysis -o memory_analysis ./a.out"
    ],
    "output_files": [
      "./ncu-llm-output/quick_check.txt",
      "./ncu-llm-output/quick_check.csv",
      "./ncu-llm-output/memory_analysis.txt",
      "./ncu-llm-output/memory_analysis.csv"
    ],
    "raw_data_snippet": ">>> Memory Throughput: 15.2%\n>>> Compute Throughput: 82.1%\n..."
  }
}
```

**New**: The `ncu_metadata` field provides transparency into the profiling process:
- **commands_executed**: Exact NCU commands PrettyNCU chose to run
- **output_files**: Paths to all NCU output files for manual inspection
- **raw_data_snippet**: Preview of raw NCU data (first 500 chars)

## NCU Profiling Modes

PrettyNCU intelligently chooses between:

- **quick**: 8 essential metrics (~5K tokens) - for simple kernels
- **bottleneck**: Memory vs compute check (~2K tokens) - when root cause unclear
- **standard**: Comprehensive analysis (~30K tokens) - for deep dives

## Complete Example Workflow

Here's what you'll see when analyzing a kernel:

```bash
$ npm run dev -- analyze test-kernel.cu --diagnostics memory

┌─ Analyzing test-kernel.cu... ─────────────────┐
│ 🤔 Analyzing memory access patterns...        │
│                                                │
│ 🔧 Bash                                        │
│    └─ nvcc test-kernel.cu -o kernel_executable│
│                                                │
│ 🔧 Bash                                        │
│    └─ ./scripts/ncu-llm quick ./kernel_exec...│
│                                                │
│ 📄 Read                                        │
│    └─ Reading: ./ncu-llm-output/quick-*.txt   │
│                                                │
│ 📊 Running NCU profiling...                   │
│ 📝 Parsing NCU output...                      │
│ ✅ Analysis complete!                         │
└────────────────────────────────────────────────┘

┌─ P1 Actionable Insight ────────────────────────┐
│ Uncoalesced global memory access pattern      │
│ causing 85% of warps to stall on memory       │
│ dependencies. Memory throughput at 15%.        │
└────────────────────────────────────────────────┘

┌─ Hot Fix (Side-by-Side) ───────────────────────┐
│                                                │
│ ORIGINAL CODE:                                 │
│   output[tid] = input[tid * STRIDE] * 2.0f;   │
│                                                │
│ OPTIMIZED CODE:                                │
│   __shared__ float tile[256];                 │
│   tile[tid] = input[tid];                     │
│   __syncthreads();                            │
│   output[tid] = tile[tid] * 2.0f;             │
│                                                │
└────────────────────────────────────────────────┘

┌─ Explanation ──────────────────────────────────┐
│ NCU profiling shows Memory Throughput at 15%  │
│ with high warp stalls on memory dependencies. │
│                                                │
│ Root cause: Strided memory access pattern     │
│ (tid * STRIDE) causes uncoalesced loads.      │
│                                                │
│ Fix: Use shared memory to coalesce accesses:  │
│ 1. Load data cooperatively into shared memory │
│ 2. Synchronize threads                        │
│ 3. Access from shared memory (coalesced)      │
│                                                │
│ Expected improvement: 5-8x memory throughput  │
│                                                │
│ References:                                    │
│ - NVIDIA CUDA Programming Guide, Section 5.3.2│
│   https://docs.nvidia.com/cuda/cuda-c-progr...│
└────────────────────────────────────────────────┘

┌─ NCU Profiling Information ────────────────────┐
│ Commands Executed:                             │
│   1. ./scripts/ncu-llm quick ./kernel_exec     │
│   2. ./scripts/ncu-llm memory ./kernel_exec    │
│                                                │
│ Output Files:                                  │
│   • ./ncu-llm-output/quick-20251024-*.txt     │
│   • ./ncu-llm-output/quick-20251024-*.csv     │
│   • ./ncu-llm-output/memory-20251024-*.txt    │
│   • ./ncu-llm-output/memory-20251024-*.csv    │
│                                                │
│ Raw NCU Data (first 500 chars):                │
│   Memory Throughput: 15.2%                    │
│   Compute Throughput: 82.1%                   │
│   L2 Cache Hit Rate: 42.3%                    │
│   Warp Stalls (Memory): 85.7%                 │
└────────────────────────────────────────────────┘
```

**What just happened:**

1. ✅ PrettyNCU compiled your kernel with nvcc
2. ✅ Executed NCU profiling using the memory diagnostic preset
3. ✅ Parsed the profiling data and identified the bottleneck
4. ✅ Generated a concrete code fix with explanation
5. ✅ Provided full transparency into the profiling process

All insights are based on **actual profiling data**, not static code analysis.

## Development

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Run in dev mode
npm run dev -- analyze test-kernel.cu

# Run built version
npm start -- analyze test-kernel.cu
```

## Configuration

The tool uses the Claude Agent SDK with these settings:

- **Model**: `claude-sonnet-4-5-20250929`
- **Permission Mode**: `bypassPermissions` (auto-executes NCU commands)
- **Max Thinking Tokens**: 10,000

## Additional Documentation

The project includes comprehensive documentation for all features:

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide for new users
- **[DIAGNOSTICS.md](DIAGNOSTICS.md)** - Complete guide to the diagnostics system and all 9 presets
- **[CUSTOM_NCU_COMMANDS.md](CUSTOM_NCU_COMMANDS.md)** - How to use custom NCU commands with examples
- **[MANDATORY_NCU_EXECUTION.md](MANDATORY_NCU_EXECUTION.md)** - Documentation of NCU execution enforcement
- **[SCRIPTS_INTEGRATION.md](SCRIPTS_INTEGRATION.md)** - How ncu-llm scripts are integrated and used
- **[scripts/README.md](scripts/README.md)** - Complete guide to editing profiling scripts
- **[ENHANCED_FEATURES.md](ENHANCED_FEATURES.md)** - Enhanced observability and side-by-side diff features

Each document provides detailed examples and usage patterns for its respective feature.

## Troubleshooting

**Error: ncu-llm not found**
- Ensure `./scripts/ncu-llm` exists and is executable: `chmod +x ./scripts/ncu-llm ./scripts/ncu-llm-lib.sh`
- Check that NCU is installed: `which ncu`
- Verify you're running from the ncu-claude directory

**Error: CUDA compilation failed**
- Verify CUDA toolkit is installed: `nvcc --version`
- Check that your `.cu` file has valid CUDA syntax
- Ensure NVIDIA GPU drivers are properly installed

**Error: PrettyNCU API timeout**
- Large kernels may take time to analyze
- Try using `--context-inline "Use quick profiling"` to force quick mode
- Use `--diagnostics quick` for faster profiling

**Error: Permission denied on output directory**
- Create output directory: `mkdir -p ./ncu-llm-output`
- Set permissions: `chmod 755 ./ncu-llm-output`

**NCU profiling not executing**
- The tool enforces mandatory NCU execution
- If PrettyNCU tries to analyze without profiling, this is a bug - please report it
- Check that the Bash tool has proper permissions in your environment

## License

ISC

## Contributing

This is a research tool for the KernelBench project. For issues or improvements, please file a GitHub issue.
