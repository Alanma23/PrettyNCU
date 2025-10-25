# Quick Start Guide

## Installation

```bash
cd /home/ubuntu0/KernelBench/ncu-claude
npm install
npm run build
```

## Basic Usage

**Option 1: Using npm run dev**
```bash
npm run dev -- analyze test-kernel.cu
```

**Option 2: Using node directly**
```bash
node dist/index.js analyze test-kernel.cu
```

**Option 3: Install globally**
```bash
npm link
ncu-claude analyze test-kernel.cu
```

## Using Diagnostic Presets (Recommended!)

List available presets:
```bash
npm run dev -- diagnostics
```

Use a preset to guide NCU profiling:
```bash
# Memory-bound kernel
npm run dev -- analyze kernel.cu --diagnostics memory

# Compute-bound kernel
npm run dev -- analyze kernel.cu --diagnostics compute

# Tensor Core / Matrix multiply
npm run dev -- analyze matmul.cu --diagnostics tensor

# Unknown issue - start here
npm run dev -- analyze kernel.cu --diagnostics quick
```

**Available Presets**:
- `quick` - Fast analysis (default, ~2-3x overhead)
- `memory` - Memory bottleneck (~10-20x overhead)
- `compute` - Compute pipeline (~10-15x overhead)
- `occupancy` - Low occupancy (~15-25x overhead)
- `latency` - Warp stalls (~20-30x overhead)
- `tensor` - Tensor Cores (~5-10x overhead)
- `full` - Everything (~100-200x overhead, not recommended)

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for details.

## Example with Custom Context

Create a context file `my-context.txt`:
```
TARGET: Matrix multiplication kernels
PRIORITY: Maximize compute throughput
HARDWARE: A100 GPU
FOCUS: Shared memory usage and register pressure
```

Run analysis:
```bash
ncu-claude analyze kernel.cu --context my-context.txt
```

## Example with Inline Context

```bash
ncu-claude analyze kernel.cu --context-inline "Focus on memory coalescing. Target RTX 4090."
```

## Combined: Diagnostics + Custom Context

You can combine diagnostic presets with custom context for maximum control:

```bash
# Memory diagnostic with hardware-specific context
npm run dev -- analyze kernel.cu --diagnostics memory --context-inline "Target A100 GPU, focus on coalescing"

# Tensor optimization with performance goals
npm run dev -- analyze matmul.cu --diagnostics tensor --context-inline "Minimize latency over throughput"
```

## Interactive Mode

```bash
ncu-claude analyze kernel.cu --interactive
# You'll be prompted to enter custom context
```

## Testing with Sample Kernel

A test kernel with intentional performance issues is provided:

```bash
npm run dev -- analyze test-kernel.cu
```

This kernel has uncoalesced memory access that NCU should detect.

## Expected Output

The tool will display:

1. **Analysis Progress**: Real-time updates as Claude analyzes and profiles
2. **P1 Actionable Insight**: The highest priority issue found
3. **Hot Fix**: Code diff showing the optimization
4. **Explanation**: Detailed explanation with NVIDIA doc references

## Customizing Prompts

Edit the files in `prompts/` directory:
- `system-prompt.txt` - Main instructions for Claude
- `ncu-analyzer.txt` - NCU metric selection strategy
- `code-optimizer.txt` - Hot-fix generation guidelines
- `default-context.txt` - Default user preferences

Changes take effect immediately (no rebuild needed).

## Troubleshooting

**Claude Agent SDK not found**
```bash
npm install @anthropic-ai/claude-agent-sdk
```

**NCU not found**
- Ensure `/home/ubuntu0/ncu-llm` exists
- Check NCU is installed: `which ncu`

**TypeScript errors**
```bash
npm run build
```

## Advanced Usage

**Override default NCU path**

Edit `prompts/system-prompt.txt` or provide in context:
```
NCU_PATH: /custom/path/to/ncu-llm
```

**Change Claude model**

Edit `src/services/claude-agent.ts:40`:
```typescript
model: 'claude-sonnet-4-5-20250929',  // Change this
```

**Adjust permission mode**

Edit `src/services/claude-agent.ts:41`:
```typescript
permissionMode: 'bypassPermissions',  // or 'default', 'acceptEdits', 'plan'
```
