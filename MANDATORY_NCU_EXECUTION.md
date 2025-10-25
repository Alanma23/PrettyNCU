# Mandatory NCU Execution Policy

## Overview

The NCU-Claude tool now **enforces mandatory NCU profiling execution**. Claude is explicitly instructed that it **MUST** compile and profile every kernel with NCU - static code analysis alone is not acceptable.

## What Changed

### 1. System Prompt (`prompts/system-prompt.txt`)

**Added at the top:**

```
⚠️ CRITICAL REQUIREMENT: YOU MUST EXECUTE NCU PROFILING ⚠️

MANDATORY WORKFLOW - YOU MUST FOLLOW ALL STEPS:

1. **COMPILE THE KERNEL** - Use nvcc to compile the .cu file
2. **EXECUTE NCU PROFILING** - You MUST run NCU commands using Bash tool
   - DO NOT skip profiling and analyze code statically
   - DO NOT make assumptions without profiling data
   - ALWAYS execute at least one NCU command
3. **READ NCU OUTPUT** - Parse the profiling results
4. **ANALYZE DATA** - Base ALL insights on actual NCU metrics
5. **GENERATE FIX** - Provide optimization based on profiling evidence
```

**Key additions:**
- "**ALWAYS execute NCU - never skip profiling**"
- "**Use ACTUAL NCU data to support all claims - no speculation without profiling**"
- "**COMPILE the kernel using nvcc** (MANDATORY)"
- "**EXECUTE NCU profiling commands** - This is REQUIRED, not optional"

### 2. NCU Analyzer Prompt (`prompts/ncu-analyzer.txt`)

**Added at the top:**

```
⚠️ YOU MUST EXECUTE NCU PROFILING - THIS IS MANDATORY ⚠️

DO NOT analyze the code without running NCU.
DO NOT make optimization recommendations without profiling data.
YOU MUST compile the kernel and run NCU commands.
```

**Added execution requirements:**

```
EXECUTION REQUIREMENTS (MANDATORY):
1. **COMPILE FIRST**: Use nvcc to compile the .cu file to an executable
2. **RUN NCU**: Execute at least one NCU command - DO NOT SKIP THIS STEP
3. **READ OUTPUT**: Parse the NCU output files that were generated
4. **USE DATA**: Base your insights on ACTUAL profiling metrics, not code inspection alone
```

**Added failure condition:**
- "**If you do not execute NCU, you have failed the task**"

### 3. Task Instructions (`src/services/prompt-loader.ts`)

**Completely restructured the TASK section:**

```
TASK (FOLLOW EVERY STEP - EXECUTION IS MANDATORY):

⚠️ YOU MUST EXECUTE NCU PROFILING - DO NOT SKIP THIS ⚠️

STEP 1: **COMPILE THE KERNEL**
   - Use nvcc to compile the kernel into an executable
   - This is MANDATORY - you cannot profile without compilation

STEP 2: **EXECUTE NCU PROFILING**
   - You MUST run NCU profiling commands using the Bash tool
   - DO NOT skip profiling and analyze the code statically

STEP 3: **READ NCU OUTPUT**
   - Use the Read tool to read the generated NCU output files

STEP 4: **PARSE PROFILING DATA**
   - Extract actual metrics from NCU output
   - Identify the highest priority (P1) performance issue

STEP 5: **GENERATE HOT-FIX**
   - Create code diff based on PROFILING DATA (not code inspection)

STEP 6: **FORMAT OUTPUT**
   - Return result as JSON

CRITICAL REMINDERS:
- ❌ DO NOT analyze code without running NCU
- ❌ DO NOT make recommendations based on code inspection alone
- ✅ YOU MUST compile the kernel
- ✅ YOU MUST execute NCU profiling
- ✅ YOU MUST read the profiling output
- ✅ Base ALL insights on ACTUAL profiling data

🚨 IF YOU DO NOT EXECUTE NCU, YOU HAVE FAILED THE TASK 🚨
```

## Why This Matters

### Problem We're Solving

Without these explicit instructions, Claude might:
1. ❌ Analyze code statically and make educated guesses
2. ❌ Suggest optimizations based on common patterns (not actual data)
3. ❌ Skip profiling to save time
4. ❌ Provide generic advice without evidence

### What We Enforce Now

With the updated prompts, Claude will:
1. ✅ **Always compile** the kernel using nvcc
2. ✅ **Always execute** at least one NCU profiling command
3. ✅ **Always read** the NCU output files
4. ✅ **Always base** recommendations on actual profiling metrics
5. ✅ **Always provide** evidence from NCU data

## Enforcement Mechanisms

### 1. Repetition
The requirement is stated **multiple times** in **multiple places**:
- System prompt (top-level instructions)
- NCU analyzer prompt (strategy selection)
- Task instructions (step-by-step workflow)

### 2. Strong Language
We use:
- ⚠️ Warning symbols
- **Bold text**
- CAPS for emphasis
- 🚨 Failure warnings
- Explicit "DO NOT" and "YOU MUST" statements

### 3. Step-by-Step Workflow
Breaking down the task into explicit steps makes it harder to skip profiling:
- STEP 1: Compile
- STEP 2: Profile (with NCU)
- STEP 3: Read output
- STEP 4: Parse data
- STEP 5: Generate fix
- STEP 6: Format output

### 4. Explicit Failure Conditions
- "If you do not execute NCU, you have failed the task"
- "🚨 IF YOU DO NOT EXECUTE NCU, YOU HAVE FAILED THE TASK 🚨"

## Expected Behavior

### Before These Changes

```
User: Analyze kernel.cu

Claude:
Looking at the code, I can see uncoalesced memory access at line 15.
This is likely causing poor memory throughput.

Recommendation: Use shared memory tiling...
```
❌ **No compilation, no profiling, no data**

### After These Changes

```
User: Analyze kernel.cu

Claude:
I will now compile and profile this kernel.

🔧 Bash: nvcc kernel.cu -o kernel
🔧 Bash: /home/ubuntu0/ncu-llm quick ./kernel
🔧 Read: ./ncu-llm-output/quick-20251024-120000.txt

NCU profiling shows:
- Memory Throughput: 15% of peak
- Compute Throughput: 85% of peak
- Warp stalls: 82% on memory dependencies

This confirms the kernel is memory-bound...
```
✅ **Compiled, profiled, data-driven**

## Verification

### How to Verify Claude is Profiling

When you run the tool, you should see in the TUI:

```
🔧 Bash
   └─ nvcc kernel.cu -o executable

🔧 Bash
   └─ /home/ubuntu0/ncu-llm quick ./executable

🔧 Read
   └─ Reading: ./ncu-llm-output/quick-*.txt
```

If you **don't** see these tool executions, Claude is not following the instructions.

### What to Look For

In the final output, the explanation should reference **actual metrics**:

✅ **Good** (data-driven):
```
NCU profiling shows Memory Throughput at 15% (from quick-*.txt)
with 85% of warps stalled on Long Scoreboard (memory latency).

The kernel executed 1,245,678 global memory transactions with
only 23% coalescing efficiency (from lts__t_sectors metric).
```

❌ **Bad** (speculation):
```
Looking at the code, this appears to have uncoalesced memory access.
The strided access pattern likely causes poor performance.
You should use shared memory.
```

## Benefits

1. **Accuracy**: Recommendations based on real profiling data, not guesses
2. **Reproducibility**: Same kernel always gets profiled the same way
3. **Trust**: Users can verify claims against actual NCU metrics
4. **Learning**: Users see the actual profiling commands that were run
5. **Debugging**: If optimization doesn't work, we have data to understand why

## Edge Cases Handled

### Custom Commands
When user provides `--ncu-command`, Claude is instructed:
```
Execute the custom NCU commands EXACTLY as provided by the user
```

### Diagnostic Configuration
When diagnostics are selected via `--configure-diagnostics`:
```
Use the diagnostic configuration to guide your NCU command selection
```

### Multiple Runs
If NCU was already run on the same kernel, Claude can:
- Check for existing output files first
- Re-run if the code has changed
- Use cached results if available (and note this)

## Testing

### Test Case 1: Basic Analysis
```bash
ncu-claude analyze test-kernel.cu
```

**Expected**: Compile → Quick profile → Read output → Data-driven insights

### Test Case 2: With Diagnostics
```bash
ncu-claude analyze test-kernel.cu --configure-diagnostics
# Select: Memory Bottleneck + Warp Stalls
```

**Expected**: Compile → Memory analysis → Warp stall analysis → Combined insights

### Test Case 3: Custom Commands
```bash
ncu-claude analyze test-kernel.cu --ncu-command "ncu --metrics xyz -o output"
```

**Expected**: Compile → Execute custom command → Read output → Analyze xyz metric

## Troubleshooting

### If Claude Still Doesn't Profile

**Check:**
1. Is nvcc in PATH? (`which nvcc`)
2. Is ncu-llm executable? (`ls -la /home/ubuntu0/ncu-llm`)
3. Does the kernel compile? (`nvcc kernel.cu -o test`)
4. Can you run NCU manually? (`ncu ./test`)

**If issues persist:**
- Check Claude's actual output in the TUI
- Look for error messages in tool execution
- Verify the Bash tool has correct permissions

## Summary

The NCU-Claude tool now **guarantees** that every analysis is based on real profiling data:

✅ **Compilation is mandatory**
✅ **NCU profiling is mandatory**
✅ **Reading output is mandatory**
✅ **Data-driven insights are mandatory**

No more guessing. No more static analysis. Only real metrics.

---

**Updated Files:**
- `prompts/system-prompt.txt` - Added mandatory execution policy
- `prompts/ncu-analyzer.txt` - Added execution requirements
- `src/services/prompt-loader.ts` - Restructured task with explicit steps

**Last Updated:** 2025-10-24
