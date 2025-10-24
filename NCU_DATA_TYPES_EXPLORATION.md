# What Kind of Information Can You Get from NCU?

## Complete Data Export Capabilities

NCU (NVIDIA Nsight Compute) can export a wide variety of data types beyond just CSV files. Here's everything you can extract:

---

## 1. Export Formats Overview

### A. Binary Report File (.ncu-rep)
```bash
ncu --export myreport ./myprogram

# Creates: myreport.ncu-rep (5.4MB in our case)
```

**What's inside:**
- Complete profiling session data
- All collected metrics and counters
- Source code correlation (if available)
- Multiple kernel invocations
- Device information
- Analysis rules and recommendations

**How to use:**
- Open in ncu-ui (GUI) for visual analysis
- Export to various formats using `--import` flag
- Query specific sections and metrics

---

### B. CSV Export - Multiple "Pages"

NCU has different "pages" you can export as CSV:

#### **1. Details Page** (default)
```bash
ncu --import report.ncu-rep --csv > details.csv
ncu --import report.ncu-rep --page details --csv > details.csv
```

**Contains:**
- Section metrics (Memory Throughput, Occupancy, etc.)
- Analysis rules and recommendations ✅
- Estimated speedup potential ✅
- Root cause analysis ✅

**What we got:** ncu_details.csv (92KB, 373 rows, 20 columns)

---

#### **2. Raw Metrics Page**
```bash
ncu --import report.ncu-rep --page raw --csv > raw.csv
```

**Contains:**
- ALL raw hardware counters (2,362 metrics!)
- Low-level performance counters:
  - `dram__read_throughput.avg.pct_of_peak_sustained_elapsed`
  - `lts__t_sector_hit_rate_srcunit_tex_realtime.pct`
  - `sm__warps_active.avg.per_cycle_active`
  - And 2,359 more...

**What we got:** ncu_raw_metrics.csv (232KB, 6 rows, 2,362 columns)

---

#### **3. Source Code Page**
```bash
ncu --import report.ncu-rep --page source --print-source sass --csv
ncu --import report.ncu-rep --page source --print-source cuda --csv
ncu --import report.ncu-rep --page source --print-source ptx --csv
```

**Contains:**
- SASS (GPU assembly) instructions
- PTX (intermediate representation)
- CUDA source code (if compiled with -lineinfo)
- **Per-instruction metrics!**
  - Warp stall reasons per instruction
  - Memory access patterns per load/store
  - Branch divergence per branch
  - Occupancy per instruction

**Example output:**
```
Address            Source                           Warp Stalls  Memory Ops  L1 Hit Rate
0x70c2ef594d50     LDG.E R28, desc[UR6][R26.64]     6 samples    Load Global 87.36%
0x70c2ef594d90     LDG.E R16, desc[UR6][R12.64]     6 samples    Load Global 87.36%
```

**Columns (60+ metrics per instruction):**
- Warp stall breakdown (L1TEX, barrier, branch, etc.)
- Memory operation details
- Thread execution statistics
- Cache hit rates per instruction

---

#### **4. Session Info Page**
```bash
ncu --import report.ncu-rep --page session --csv
```

**Contains:**
- Launch settings (command line used)
- Session metadata (date, hostname, OS)
- Device attributes (GPU name, compute capability, memory)
- CUDA version
- Process information

**Example:**
```csv
Device Attribute          Device 0
display_name              NVIDIA B200
compute_capability_major  10
compute_capability_minor  0
total_memory              191513886720
multiprocessor_count      148
```

---

## 2. Section-Based Collection

NCU has different "section sets" you can collect:

### Available Section Sets:

```bash
ncu --list-sets
```

| Set | Sections | Metrics | Use Case |
|-----|----------|---------|----------|
| **basic** | 4 sections | 190 | Quick profiling, basic bottleneck detection |
| **detailed** | 9 sections | 557 | Standard optimization workflow |
| **full** | 24 sections | 5,895 | Complete analysis, deep dive |
| **roofline** | 7 sections | 5,260 | Roofline analysis (compute vs memory bound) |
| **nvlink** | 3 sections | 52 | Multi-GPU communication analysis |
| **pmsampling** | 2 sections | 186 | Warp-level sampling profiling |

### Collect Specific Section Set:
```bash
# Collect detailed set
ncu --set detailed --export detailed_profile ./myprogram

# Collect full set (WARNING: high overhead!)
ncu --set full --export full_profile ./myprogram

# Collect roofline data for roofline charts
ncu --set roofline --export roofline_profile ./myprogram
```

---

## 3. Individual Sections Available

```bash
ncu --list-sections  # Shows all 24 sections
```

### Key Sections and What They Provide:

#### **Speed of Light Analysis**
- Memory throughput vs peak
- Compute throughput vs peak
- Identifies if memory-bound or compute-bound
- **Roofline chart data** ✅

Sections:
- `SpeedOfLight` - Basic throughput metrics
- `SpeedOfLight_RooflineChart` - Data for roofline visualization
- `SpeedOfLight_HierarchicalSingleRooflineChart` - FP32 roofline
- `SpeedOfLight_HierarchicalDoubleRooflineChart` - FP64 roofline
- `SpeedOfLight_HierarchicalTensorRooflineChart` - Tensor Core roofline

#### **Memory Workload Analysis**
- L1/L2 cache hit rates
- Memory bandwidth utilization
- Coalescing efficiency
- **Memory hierarchy charts** ✅

Sections:
- `MemoryWorkloadAnalysis` - Memory metrics
- `MemoryWorkloadAnalysis_Chart` - Chart data
- `MemoryWorkloadAnalysis_Tables` - Detailed breakdowns

#### **Compute Workload Analysis**
- Pipeline utilization
- Instruction throughput
- Warp efficiency

Section:
- `ComputeWorkloadAnalysis`

#### **Warp State Statistics**
- Warp stall reasons (50+ categories!)
- Cycles spent in each stall state
- Active vs eligible warps

Sections:
- `WarpStateStats` - Summary statistics
- `PmSampling_WarpStates` - Detailed sampling data

#### **Source Correlation**
- Instruction-level metrics
- Hotspot identification
- Line-by-line performance

Section:
- `SourceCounters` - Source code performance data

#### **Occupancy Analysis**
- Theoretical vs achieved occupancy
- Limiting factors (registers, shared memory, etc.)

Section:
- `Occupancy`

#### **Scheduler Statistics**
- Issue slot utilization
- Eligible warps per scheduler

Section:
- `SchedulerStats`

#### **Launch Statistics**
- Grid/block dimensions
- Resource usage (registers, shared memory)
- Thread count

Section:
- `LaunchStats`

---

## 4. What the GUI (ncu-ui) Can Generate

The NCU GUI can open .ncu-rep files and provides:

### **Visual Charts and Graphs:**

1. **Roofline Charts** 📊
   - FP32/FP64/Tensor Core roofline plots
   - Shows compute intensity vs throughput
   - Identifies optimization ceiling
   - **Can be exported as images** (PNG, SVG)

2. **Memory Hierarchy Charts** 📊
   - Visual representation of L1/L2/DRAM traffic
   - Bandwidth utilization bars
   - Cache hit rate visualization

3. **Speed of Light Gauge** 📊
   - Circular gauges showing memory/compute utilization
   - Color-coded (green/yellow/red)
   - Visual bottleneck identification

4. **Warp State Distribution** 📊
   - Pie chart of stall reasons
   - Percentage breakdown
   - Visual priority ranking

5. **Occupancy Visualization** 📊
   - Bar charts of theoretical vs achieved
   - Resource limit indicators
   - Block configuration analyzer

6. **Source View with Heatmaps** 📊
   - Color-coded source lines by metric
   - Instruction-level hotspot highlighting
   - SASS/PTX/CUDA side-by-side view

### **Interactive Features:**

- Click on any metric to drill down
- Hover for detailed tooltips
- Filter by kernel/section
- Compare multiple reports side-by-side
- Export any view to:
  - **PNG images** ✅
  - **SVG vector graphics** ✅
  - **CSV data**
  - **PDF reports** ✅

### **GUI Export Capabilities:**

From the GUI, you can:
1. File → Export → Image (PNG, SVG)
2. File → Export → Report (PDF with all charts)
3. File → Export → CSV (specific sections)
4. Right-click any chart → Save as Image

---

## 5. Advanced Query Modes

### Metric Querying:
```bash
# List all metrics for selected sections
ncu --list-metrics --section SpeedOfLight

# Query available metrics on the system
ncu --query-metrics

# Query with suffixes (avg, min, max, sum)
ncu --query-metrics-mode suffix
```

### Custom Metric Collection:
```bash
# Collect specific metrics only
ncu --metrics "dram__throughput,l1tex__throughput" \
    --export custom_profile ./myprogram

# Collect metric groups
ncu --metrics "group:memory,group:compute" \
    --export groups_profile ./myprogram

# Use regex to select metrics
ncu --metrics "regex:.*throughput.*" \
    --export throughput_profile ./myprogram
```

---

## 6. What We Actually Got from Our Profiling

### Summary of Extracted Data:

| File | Size | Type | Contents |
|------|------|------|----------|
| ncu_matmul_profile.ncu-rep | 5.4 MB | Binary | Complete report (GUI-compatible) |
| ncu_details.csv | 92 KB | CSV | Metrics + actionable insights |
| ncu_raw_metrics.csv | 232 KB | CSV | 2,362 raw hardware counters |

### What We Can Still Extract:

✅ **Already Have:**
- Actionable insights with speedup estimates
- Raw performance counters
- Session information

❌ **Haven't Extracted Yet:**
- Source code with per-instruction metrics
- Roofline chart data
- Memory hierarchy visualization data
- Warp state distribution details
- Occupancy analysis details
- Visual charts (requires GUI or export from GUI)

---

## 7. Practical Examples - What You Can Get

### Example 1: Get Source Code with Hotspots
```bash
ncu --import ncu_matmul_profile.ncu-rep \
    --page source \
    --print-source sass \
    --csv > source_hotspots.csv
```

**Output:** Every SASS instruction with 60+ metrics showing:
- Which instructions cause the most warp stalls
- Memory access patterns per load/store
- Cache behavior per instruction
- Instruction execution counts

### Example 2: Get Roofline Data
```bash
# Collect with roofline sections
ncu --set roofline --export roofline_profile ./matmul_cuda 512

# Export roofline data
ncu --import roofline_profile.ncu-rep \
    --section SpeedOfLight_RooflineChart \
    --csv > roofline_data.csv
```

**Output:** Data points for roofline chart:
- Arithmetic intensity (FLOP/byte)
- Achieved performance (GFLOPS)
- Peak performance limits
- Memory bandwidth limits

### Example 3: Get Detailed Warp Stalls
```bash
# Collect with PM sampling
ncu --set pmsampling --export warp_stalls ./matmul_cuda 512

# Export warp state data
ncu --import warp_stalls.ncu-rep \
    --section PmSampling_WarpStates \
    --csv > warp_state_breakdown.csv
```

**Output:** Detailed warp stall breakdown:
- Percentage of time in each stall state
- Top stall locations in source code
- Sampling data per instruction

### Example 4: Get Complete Session Info
```bash
ncu --import ncu_matmul_profile.ncu-rep \
    --page session \
    --csv > session_info.csv
```

**Output:** Complete environment details:
- GPU specifications (B200, 148 SMs, 191 GB memory)
- CUDA version (12.8)
- Launch configuration
- Timestamp and hostname

---

## 8. Can You Get Images from NCU?

### ❌ **Not Directly from Command Line**

NCU CLI does not generate images. It exports:
- CSV data
- Text output
- Binary .ncu-rep files

### ✅ **Yes, from the GUI**

The ncu-ui (GUI) can:
1. Open .ncu-rep files
2. Generate all visualizations:
   - Roofline charts
   - Memory hierarchy diagrams
   - Speed of Light gauges
   - Warp state pie charts
   - Occupancy bars
   - Source code heatmaps
3. Export these as:
   - PNG images
   - SVG vector graphics
   - PDF reports with all charts

**Workflow:**
```bash
# 1. Profile on server (no GUI needed)
ncu --set full --export full_report ./myprogram

# 2. Transfer .ncu-rep file to machine with GUI
scp full_report.ncu-rep user@desktop:~/

# 3. Open in GUI (on desktop)
ncu-ui full_report.ncu-rep

# 4. Export images
# File → Export → Image (PNG/SVG)
# File → Export → Report (PDF)
```

### ⚠️ **Alternative: Generate Charts from CSV**

You can use the CSV data to generate your own charts:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read NCU data
df = pd.read_csv('ncu_details.csv')

# Extract memory throughput
mem_throughput = df[df['Metric Name'] == 'Memory Throughput']

# Plot
plt.figure(figsize=(10, 6))
plt.bar(mem_throughput['Kernel Name'],
        mem_throughput['Metric Value'].astype(float))
plt.xlabel('Kernel')
plt.ylabel('Memory Throughput (%)')
plt.title('Memory Throughput Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('memory_throughput.png', dpi=300)
```

---

## 9. Summary: What Can You Get?

| Data Type | Format | Source | Contains Images? |
|-----------|--------|--------|------------------|
| **Actionable Insights** | CSV | CLI | No |
| **Raw Metrics** | CSV | CLI | No |
| **Source Code + Metrics** | CSV | CLI | No |
| **Session Info** | CSV | CLI | No |
| **Binary Report** | .ncu-rep | CLI | No (but GUI can open) |
| **Visual Charts** | PNG/SVG | GUI | **Yes!** ✅ |
| **PDF Report** | PDF | GUI | **Yes!** ✅ |
| **Roofline Plots** | PNG/SVG | GUI | **Yes!** ✅ |
| **Heatmaps** | PNG/SVG | GUI | **Yes!** ✅ |

---

## 10. What's Most Useful?

**For Automation/CI/CD:**
- CSV exports (details + raw)
- Programmatic analysis
- Regression tracking

**For Presentations:**
- GUI-generated images (roofline, gauges, etc.)
- PDF reports
- Custom charts from CSV data

**For Deep Debugging:**
- Source page with instruction-level metrics
- Warp state sampling
- Full section set

**For Quick Insights:**
- Details CSV (what we have!)
- Basic section set
- Actionable recommendations

---

## 11. Next Steps - What You Could Extract

Want to see more? Here's what we can do:

### 1. Extract Source Code with Hotspots
```bash
ncu --import ncu_matmul_profile.ncu-rep \
    --page source --print-source sass \
    --csv > sass_with_metrics.csv
```

### 2. Re-profile with Roofline Data
```bash
ncu --set roofline --export roofline_report ./matmul_cuda 512
```

### 3. Get Detailed Warp State Breakdown
```bash
ncu --set pmsampling --export warp_report ./matmul_cuda 512
```

### 4. Create Custom Visualizations
```python
# Use the CSV data to create any chart you want
# Matplotlib, Plotly, Seaborn, etc.
```

---

## Conclusion

**NCU provides a wealth of data beyond simple CSV exports:**

✅ **What you CAN get from CLI:**
- Actionable insights with speedup estimates
- 2,362 raw performance counters
- Instruction-level metrics
- Source code correlation
- Session/device information
- Roofline data points
- Warp state details

❌ **What you CANNOT get from CLI:**
- Pre-rendered images/charts
- Visual diagrams
- Interactive visualizations

✅ **What you CAN get from GUI:**
- All of the above PLUS:
- Beautiful charts (roofline, memory hierarchy, etc.)
- Export to PNG/SVG/PDF
- Interactive exploration
- Heatmaps and visual hotspots

**Best Workflow:**
1. Profile on server with NCU CLI → .ncu-rep file
2. Export CSV for automated analysis
3. Transfer .ncu-rep to desktop for visual analysis
4. Generate images/reports in GUI for presentations

**You have everything you need in CSV format!**
The data is all there - just needs visualization if you want images.
