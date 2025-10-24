# Complete File Index

## Matrix Multiplication Profiling Journey with NSYS & SQLite

All files created during this comprehensive profiling and optimization journey.

---

## 📂 Source Code (C)

### Basic Implementations
- **matmul.c** (3.8K)
  - All three versions: naive, reordered, blocked
  - Benchmarking harness
  - Performance comparison

- **matmul_single.c** (2.3K)
  - Compile-time version selection
  - Used for individual profiling

- **matmul_analysis.c** (6.5K)
  - Detailed performance analysis
  - Memory access pattern explanations
  - Cache behavior insights

- **matmul_optimized.c** (5.8K)
  - Advanced optimizations
  - Loop unrolling
  - OpenMP parallelization

---

## 🔬 Compiled Binaries

- **matmul** (17K) - All versions benchmark
- **matmul_naive** (16K) - Naive implementation only
- **matmul_reordered** (16K) - Reordered implementation
- **matmul_blocked** (16K) - Blocked/tiled version
- **matmul_opt** (17K) - Optimized (no OpenMP)
- **matmul_omp** (21K) - Optimized with OpenMP
- **matmul_analysis** (17K) - Analysis version

---

## 🗄️ Profiling Data

### NSYS Reports
- **matmul_naive_profile.nsys-rep**
  - Binary profiling report
  - Open with nsys-ui GUI

- **matmul_reordered_profile.nsys-rep**
  - Binary profiling report
  - Open with nsys-ui GUI

### SQLite Databases
- **matmul_naive_profile.sqlite** (668K)
  - 77 tables of profiling data
  - Process, thread, timing info
  - System environment
  - GPU hardware details

- **matmul_reordered_profile.sqlite** (668K)
  - Same structure as above
  - For comparison analysis

---

## 🐍 Python Analysis Scripts

### Database Exploration
- **explore_nsys_db.py**
  - Initial database exploration
  - Schema inspection
  - Table enumeration

- **analyze_nsys_simple.py**
  - Focused analysis script
  - Key metrics extraction
  - String mapping resolution

- **comprehensive_query.py**
  - Deep database queries
  - Detailed profiling analysis
  - Multi-table joins

- **final_analysis.py**
  - Complete comparative analysis
  - Side-by-side comparison
  - Summary statistics
  - **← Recommended for quick analysis**

---

## 📊 SQL Query Files

- **query_nsys.sql**
  - Collection of useful SQL queries
  - Timing, processes, threads
  - GPU info, diagnostics
  - Ready to run with sqlite3 CLI

---

## 📚 Documentation

### Main Guides
- **README.md** (5.1K)
  - Quick start guide
  - Overview of project
  - Basic usage
  - Results summary

- **ANALYSIS.md** (4.3K)
  - Memory access pattern diagrams
  - Cache behavior visualization
  - Loop order comparison
  - Key lessons

- **JOURNEY_SUMMARY.md** (7.7K)
  - Complete optimization journey
  - All 5 stages detailed
  - Performance breakdown
  - Educational takeaways
  - **← Best for understanding the journey**

- **NSYS_PROFILING_GUIDE.md** (~10K)
  - Comprehensive NSYS profiling guide
  - SQLite database structure
  - SQL query examples
  - Advanced usage
  - **← Best for NSYS/SQLite specifics**

- **SQLITE_QUICK_REFERENCE.md** (~8K)
  - Quick reference card
  - Common queries
  - Python templates
  - SQL templates
  - **← Best for quick lookups**

- **FILES_INDEX.md** (this file)
  - Complete file listing
  - Descriptions and sizes
  - Usage recommendations

---

## 📊 File Organization Summary

```
Matrix Multiplication Project Structure
├── Source Code
│   ├── matmul.c                    (all versions)
│   ├── matmul_single.c             (individual compile)
│   ├── matmul_analysis.c           (performance analysis)
│   └── matmul_optimized.c          (advanced opts)
│
├── Binaries
│   ├── matmul                      (main benchmark)
│   ├── matmul_naive/reordered/blocked
│   └── matmul_opt/omp              (optimized versions)
│
├── Profiling Data
│   ├── matmul_naive_profile.sqlite
│   ├── matmul_reordered_profile.sqlite
│   └── *.nsys-rep                  (GUI reports)
│
├── Analysis Scripts
│   ├── explore_nsys_db.py
│   ├── analyze_nsys_simple.py
│   ├── comprehensive_query.py
│   └── final_analysis.py           ⭐ Run this!
│
└── Documentation
    ├── README.md                   (quick start)
    ├── ANALYSIS.md                 (memory patterns)
    ├── JOURNEY_SUMMARY.md          (full journey)
    ├── NSYS_PROFILING_GUIDE.md     ⭐ Read this!
    ├── SQLITE_QUICK_REFERENCE.md   (cheat sheet)
    └── FILES_INDEX.md              (this file)
```

---

## 🎯 Recommended Reading Order

### For Quick Start
1. README.md
2. Run: `./matmul 512`
3. Run: `python3 final_analysis.py`

### For Deep Understanding
1. JOURNEY_SUMMARY.md (understand the optimizations)
2. ANALYSIS.md (memory access patterns)
3. NSYS_PROFILING_GUIDE.md (profiling methodology)

### For Reference
- SQLITE_QUICK_REFERENCE.md (query examples)
- FILES_INDEX.md (this file)

---

## 💻 Quick Commands

### Compile Everything
```bash
# Basic versions
gcc -O2 -o matmul matmul.c

# Analysis version
gcc -O2 -o matmul_analysis matmul_analysis.c

# Optimized versions
gcc -O3 -march=native -o matmul_opt matmul_optimized.c
gcc -O3 -march=native -fopenmp -o matmul_omp matmul_optimized.c
```

### Run Benchmarks
```bash
# Compare all versions
./matmul 512

# Run detailed analysis
./matmul_analysis 512

# Test optimized parallel version
./matmul_omp 512
```

### Profile with NSYS
```bash
# Profile a version
nsys profile --stats=true --export=sqlite --output=my_profile ./matmul_naive 512

# Analyze the database
python3 final_analysis.py
```

### Query Database
```python
# Quick query
python3 << 'EOF'
import sqlite3
conn = sqlite3.connect('matmul_naive_profile.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT duration / 1e9 FROM ANALYSIS_DETAILS")
print(f"Duration: {cursor.fetchone()[0]:.6f} seconds")
conn.close()
