#!/usr/bin/env python3
"""
Show the structure of NCU profiling data
"""

import csv

def show_csv_structure(csv_file, max_rows=10):
    """Show the structure of an NCU CSV file"""

    print(f"\n{'='*90}")
    print(f"File: {csv_file}")
    print(f"{'='*90}\n")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        # Get columns
        columns = reader.fieldnames
        print(f"📊 Columns ({len(columns)} total):")
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col}")

        # Show sample rows
        print(f"\n📝 Sample Data (first {max_rows} rows):\n")

        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)

        if not rows:
            print("  (No data)")
            return

        # Display in table format
        # Key columns to show
        key_cols = ['ID', 'Kernel Name', 'Section Name', 'Metric Name', 'Metric Value', 'Metric Unit']

        # Print header
        for col in key_cols:
            if col in columns:
                print(f"{col[:20]:<22}", end='')
        print()
        print("─" * (22 * len(key_cols)))

        # Print rows
        for row in rows:
            for col in key_cols:
                if col in row:
                    val = row[col]
                    # Shorten kernel name
                    if col == 'Kernel Name':
                        val = val.split('(')[0][:20]
                    print(f"{str(val)[:20]:<22}", end='')
            print()

def show_data_organization():
    """Show how NCU data is organized"""

    print(f"\n{'='*90}")
    print(f"NCU DATA ORGANIZATION")
    print(f"{'='*90}\n")

    print("📋 HIERARCHICAL STRUCTURE:")
    print("""
    NCU Report
    │
    ├── Kernel Invocation #0 (ID=0)
    │   ├── Kernel Name: matmul_naive_kernel(...)
    │   ├── Launch Config: Grid(32,32,1), Block(16,16,1)
    │   │
    │   ├── Section: GPU Speed Of Light
    │   │   ├── Metric: Memory Throughput = 79.41%
    │   │   ├── Metric: Compute Throughput = 52.93%
    │   │   ├── Metric: Elapsed Cycles = 109,067
    │   │   └── ...
    │   │
    │   ├── Section: Memory Workload Analysis
    │   │   ├── Metric: L1/TEX Hit Rate = 87.36%
    │   │   ├── Metric: L2 Hit Rate = 90.78%
    │   │   └── ...
    │   │
    │   ├── Section: Compute Workload Analysis
    │   │   ├── Metric: Executed IPC = 1.40
    │   │   └── ...
    │   │
    │   ├── Section: Occupancy
    │   │   ├── Metric: Achieved Occupancy = 69.27%
    │   │   └── ...
    │   │
    │   └── [Many more sections...]
    │
    ├── Kernel Invocation #1 (ID=1)
    │   └── [Same structure as #0]
    │
    └── ...
    """)

def show_metric_categories():
    """Show the different categories of metrics"""

    print(f"\n{'='*90}")
    print(f"METRIC CATEGORIES")
    print(f"{'='*90}\n")

    categories = {
        'GPU Speed Of Light': [
            'Memory Throughput (%)',
            'Compute (SM) Throughput (%)',
            'Elapsed Cycles',
            'SM Frequency (Ghz)',
            'DRAM Frequency (Ghz)',
        ],
        'Memory Workload': [
            'L1/TEX Hit Rate (%)',
            'L2 Hit Rate (%)',
            'DRAM Throughput (%)',
            'Memory Throughput (%)',
            'Mem Busy (%)',
        ],
        'Compute Workload': [
            'Executed IPC (inst/cycle)',
            'Issued IPC (inst/cycle)',
            'Warp Cycles Per Instruction',
            'Avg Active Threads Per Warp',
        ],
        'Occupancy': [
            'Theoretical Occupancy (%)',
            'Achieved Occupancy (%)',
            'Block Limit Registers',
            'Block Limit Shared Mem',
            'Theoretical Active Warps per SM',
        ],
        'Launch Statistics': [
            'Grid Size',
            'Block Size',
            'Registers Per Thread',
            'Shared Memory Per Block (bytes)',
            'Threads',
            'Waves Per SM',
        ],
        'Instruction Mix': [
            'Integer Instructions (%)',
            'Float Instructions (%)',
            'Load/Store Instructions (%)',
            'Control Flow Instructions (%)',
        ],
        'Warp State': [
            'Warp Cycles Per Issued Instruction',
            'Warp Cycles Per Executed Instruction',
            'Warp Stall Distribution',
        ],
    }

    for category, metrics in categories.items():
        print(f"📊 {category}:")
        for metric in metrics:
            print(f"   • {metric}")
        print()

def show_example_queries():
    """Show example data queries"""

    print(f"\n{'='*90}")
    print(f"EXAMPLE DATA QUERIES")
    print(f"{'='*90}\n")

    print("🔍 Python CSV Parsing:")
    print("""
import csv

# Read NCU details CSV
with open('ncu_details.csv', 'r') as f:
    reader = csv.DictReader(f)

    for row in reader:
        kernel_id = row['ID']
        kernel_name = row['Kernel Name']
        section = row['Section Name']
        metric_name = row['Metric Name']
        metric_value = row['Metric Value']
        metric_unit = row['Metric Unit']

        # Find specific metric
        if metric_name == 'Memory Throughput':
            print(f"Kernel {kernel_id}: Memory Throughput = {metric_value}%")
    """)

    print("\n🔍 Pandas DataFrame:")
    print("""
import pandas as pd

# Load as DataFrame
df = pd.read_csv('ncu_details.csv')

# Filter for specific kernel
naive_kernel = df[df['Kernel Name'].str.contains('naive')]

# Get memory throughput metrics
mem_metrics = df[df['Metric Name'] == 'Memory Throughput']
print(mem_metrics[['ID', 'Kernel Name', 'Metric Value']])

# Group by kernel and section
grouped = df.groupby(['ID', 'Section Name']).size()
print(grouped)
    """)

def main():
    """Main function"""

    print("\n" + "="*90)
    print("  NCU DATA STRUCTURE VISUALIZATION")
    print("="*90)

    # Show CSV structure
    try:
        show_csv_structure('ncu_details.csv', max_rows=15)
    except FileNotFoundError:
        print("\nncu_details.csv not found")

    # Show organization
    show_data_organization()

    # Show metric categories
    show_metric_categories()

    # Show example queries
    show_example_queries()

    print("\n" + "="*90)
    print("  Complete!")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
