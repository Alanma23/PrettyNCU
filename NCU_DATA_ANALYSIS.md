# NCU Data Analysis: What's Inside the CSV Export?

## Your Questions Answered

### 1. **Does it just have raw information?**
**No!** The NCU CSV contains BOTH raw metrics AND actionable insights.

### 2. **Does it give you summaries or actionable insights, like from the GUI?**
**Yes!** The CSV export includes the same analysis and recommendations as the GUI.

---

## Data Structure Breakdown

The `ncu_details.csv` has **373 rows** across **4 kernel invocations** (IDs 0-3):
- **ID 0**: `matmul_naive_kernel` (first run)
- **ID 1**: `matmul_naive_kernel` (second run)
- **ID 2**: `matmul_tiled_kernel`
- **ID 3**: `matmul_optimized_kernel`

---

## Two Types of Data in the CSV

### Type 1: Raw Metrics (Performance Counters)

These are the measured values from hardware performance counters:

```
Line 5:  "Memory Throughput"    = "79.41" %
Line 27: "L1/TEX Hit Rate"      = "87.36" %
Line 56: "Registers Per Thread" = "32" register/thread
Line 74: "Block Limit Registers" = "8" block  ← Limiting factor!
```

**What you can do with this:**
- Extract specific metrics programmatically
- Build custom dashboards
- Compare across kernel versions
- Track performance over time

---

### Type 2: Actionable Insights (Analysis Rules)

NCU automatically analyzes the metrics and provides recommendations!

#### Example 1: Memory Bottleneck (Line 12)
```csv
"Rule Name": "SOLBottleneck"
"Rule Type": "OPT"  ← Optimization opportunity
"Rule Description": "Memory is more heavily utilized than Compute:
                     Look at the Memory Workload Analysis section to
                     identify the L1 bottleneck. Check memory replay
                     (coalescing) metrics to make sure you're efficiently
                     utilizing the bytes transferred. Also consider
                     whether it is possible to do more work per memory
                     access (kernel fusion) or whether there are values
                     you can (re)compute."
```

**Actionable takeaway:** Your kernel is memory-bound, not compute-bound.
Focus on reducing memory traffic, not optimizing compute instructions.

---

#### Example 2: Uncoalesced Memory Access (Line 34)
```csv
"Rule Name": "MemoryCacheAccessPattern"
"Rule Type": "OPT"
"Rule Description": "The memory access pattern for global loads from
                     L1TEX might not be optimal. On average, only 18.0
                     of the 32 bytes transmitted per sector are utilized
                     by each thread. This could possibly be caused by a
                     stride between threads. Check the Source Counters
                     section for uncoalesced global loads."
"Estimated Speedup Type": "global"
"Estimated Speedup": "34.74"  ← Potential 34.74% speedup!
```

**Actionable takeaway:** Fix memory coalescing → potential 34.74% speedup

**How to fix:**
- Threads should access consecutive memory addresses
- Currently wasting 44% of memory bandwidth (only using 18/32 bytes)
- Reorder data layout or access pattern

---

#### Example 3: Low Issue Slot Utilization (Line 40)
```csv
"Rule Name": "IssueSlotUtilization"
"Rule Type": "OPT"
"Rule Description": "Every scheduler is capable of issuing one instruction
                     per cycle, but for this workload each scheduler only
                     issues an instruction every 2.9 cycles. This might
                     leave hardware resources underutilized and may lead
                     to less optimal performance. Out of the maximum of 16
                     warps per scheduler, this workload allocates an average
                     of 11.04 active warps per scheduler, but only an average
                     of 1.34 warps were eligible per cycle. Eligible warps
                     are the subset of active warps that are ready to issue
                     their next instruction. Every cycle with no eligible
                     warp results in no instruction being issued and the
                     issue slot remains unused. To increase the number of
                     eligible warps, reduce the time the active warps are
                     stalled by inspecting the top stall reasons on the
                     Warp State Statistics and Source Counters sections."
"Estimated Speedup Type": "local"
"Estimated Speedup": "20.59"  ← Potential 20.59% speedup!
```

**Actionable takeaway:** Reduce warp stalls → 20.59% potential speedup

**Why stalls happen:**
- Only 1.34 out of 11.04 warps eligible each cycle
- Most warps are waiting on L1TEX memory operations

---

#### Example 4: L1TEX Stalls (Line 45)
```csv
"Rule Name": "CPIStall"
"Rule Type": "OPT"
"Rule Description": "On average, each warp of this workload spends 15.9
                     cycles being stalled waiting for a scoreboard dependency
                     on a L1TEX (local, global, surface, texture) operation.
                     Find the instruction producing the data being waited
                     upon to identify the culprit. To reduce the number of
                     cycles waiting on L1TEX data accesses verify the memory
                     access patterns are optimal for the target architecture,
                     attempt to increase cache hit rates by increasing data
                     locality (coalescing), or by changing the cache
                     configuration. Consider moving frequently used data
                     to shared memory. This stall type represents about
                     50.2% of the total average of 31.6 cycles between
                     issuing two instructions."
"Estimated Speedup Type": "global"
"Estimated Speedup": "20.59"
```

**Actionable takeaway:** 50.2% of execution time is stalled on L1TEX!

**Solution:** Use shared memory (which is exactly what the tiled version does!)

---

## Rule Types Explained

### OPT (Optimization)
- Actionable recommendation to improve performance
- Usually includes estimated speedup potential
- Example: "Fix memory coalescing → 34.74% faster"

### INF (Information)
- Contextual information, no action required
- Example: "FMA pipeline is well-utilized, not a bottleneck"

### WARN (Warning)
- Potential issue that may affect correctness or performance
- Less common in this dataset

---

## Comparison: GUI vs CSV

| Feature | GUI | CSV Export |
|---------|-----|------------|
| **Raw Metrics** | ✓ Visual charts | ✓ All values in table |
| **Analysis Rules** | ✓ Highlighted sections | ✓ In "Rule Name" column |
| **Recommendations** | ✓ Clickable descriptions | ✓ In "Rule Description" |
| **Estimated Speedup** | ✓ Shown inline | ✓ In "Estimated Speedup" |
| **Source Code Mapping** | ✓ Interactive | ✗ Not in CSV |
| **Roofline Charts** | ✓ Visual | ✗ Only data points |
| **Programmability** | ✗ Manual analysis | ✓ Easy to automate |

---

## Key Insights from This Dataset

### Naive Kernel Analysis

**Bottleneck:** Memory-bound (79.75% memory throughput)

**Top 3 Optimization Opportunities:**

1. **Fix Memory Coalescing** → 34.74% speedup potential
   - Currently only using 18/32 bytes per transaction
   - Threads accessing non-contiguous addresses

2. **Reduce L1TEX Stalls** → 20.59% speedup potential
   - 50.2% of time spent waiting on memory
   - Solution: Use shared memory (tiled version)

3. **Improve Warp Scheduling** → 20.59% speedup potential
   - Only 1.34/11.04 warps eligible per cycle
   - Caused by memory stalls

**Impact of Tiled Version:**
- Reduced cycles: 109,396 → 76,540 (30% improvement!)
- Proves that shared memory solves the L1TEX stall issue

---

## How to Use This Information Programmatically

### Extract Optimization Rules

```python
import csv

with open('ncu_details.csv', 'r') as f:
    reader = csv.DictReader(f)

    for row in reader:
        # Filter for optimization rules
        if row['Rule Type'] == 'OPT' and row['Estimated Speedup']:
            print(f"Issue: {row['Rule Name']}")
            print(f"Speedup: {row['Estimated Speedup']}%")
            print(f"Fix: {row['Rule Description'][:100]}...")
            print()
```

### Compare Kernels

```python
import pandas as pd

df = pd.read_csv('ncu_details.csv')

# Get all optimization opportunities
opts = df[df['Rule Type'] == 'OPT'][
    ['ID', 'Kernel Name', 'Rule Name', 'Estimated Speedup']
]

# Group by kernel
for kid, group in opts.groupby('ID'):
    kernel = group['Kernel Name'].iloc[0].split('(')[0]
    total_speedup = group['Estimated Speedup'].astype(float).sum()
    print(f"Kernel {kid} ({kernel}): {total_speedup:.1f}% total speedup potential")
```

---

## Summary

**The NCU CSV export is NOT just raw data!**

It contains:
1. ✓ All performance metrics (memory throughput, occupancy, IPC, etc.)
2. ✓ Automatic bottleneck detection (memory-bound vs compute-bound)
3. ✓ Specific optimization recommendations (fix coalescing, use shared memory)
4. ✓ Estimated speedup potential (34.74%, 20.59%, etc.)
5. ✓ Root cause analysis (50.2% of time in L1TEX stalls)

**This is the SAME analysis as the GUI, just in CSV format!**

The advantage of CSV:
- Programmable analysis
- Custom dashboards
- Regression tracking
- CI/CD integration
- Bulk comparison across many kernels

---

## Next Steps

Based on the insights from this data:

1. **Immediate Win:** The tiled version already achieves 30% speedup
   - Validates the NCU recommendation to use shared memory

2. **Further Optimization:**
   - Fix remaining memory coalescing issues
   - Tune block/tile sizes
   - Consider using Tensor Cores for 50-100× more speedup

3. **Validation:**
   - Re-profile after each change
   - Track if "Estimated Speedup" matches actual improvement
   - Use NCU's guidance to prioritize optimizations

The NCU CSV gives you a complete performance optimization roadmap!
