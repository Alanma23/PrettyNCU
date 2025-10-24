#!/usr/bin/env python3
"""
Analyze NCU (Nsight Compute) CSV profiling data
"""

import csv
import sys

def analyze_ncu_details(csv_file):
    """Analyze NCU details CSV file"""

    print(f"\n{'='*80}")
    print(f"NCU DETAILS ANALYSIS: {csv_file}")
    print(f"{'='*80}\n")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No data found!")
        return

    # Get available columns
    print(f"📊 Available Metrics: {len(rows[0])} columns\n")

    # Group by kernel
    kernels = {}
    for row in rows:
        kernel_name = row.get('Kernel Name', 'Unknown')
        if kernel_name not in kernels:
            kernels[kernel_name] = []
        kernels[kernel_name].append(row)

    print(f"🔍 Found {len(kernels)} kernel(s):\n")

    for kernel_name, kernel_data in kernels.items():
        print(f"\n{'─'*80}")
        print(f"Kernel: {kernel_name}")
        print(f"{'─'*80}")

        # Show first row data (most metrics are the same across rows for same kernel)
        row = kernel_data[0]

        # Key metrics to display
        important_metrics = [
            'Duration',
            'Grid Size',
            'Block Size',
            'Registers Per Thread',
            'Shared Memory Configuration Size',
            'Shared Memory Per Block',
            'Achieved Occupancy',
            'Theoretical Occupancy',
            'Compute (SM) Throughput',
            'Memory Throughput',
            'DRAM Throughput',
            'L1/TEX Hit Rate',
            'L2 Hit Rate',
            'Memory Workload',
            'Compute Workload',
        ]

        print()
        for metric in important_metrics:
            if metric in row:
                value = row[metric]
                print(f"  {metric:<40}: {value}")

def analyze_ncu_raw_metrics(csv_file):
    """Analyze NCU raw metrics CSV file"""

    print(f"\n{'='*80}")
    print(f"NCU RAW METRICS ANALYSIS: {csv_file}")
    print(f"{'='*80}\n")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No data found!")
        return

    print(f"📈 Total metric rows: {len(rows)}\n")

    # Get columns
    columns = list(rows[0].keys())
    print(f"Columns: {', '.join(columns[:10])}...\n")

    # Group by ID (kernel invocation)
    kernels_by_id = {}
    for row in rows:
        kid = row.get('ID', '0')
        kernel_name = row.get('Kernel Name', 'Unknown')

        if kid not in kernels_by_id:
            kernels_by_id[kid] = {
                'name': kernel_name,
                'metrics': []
            }
        kernels_by_id[kid]['metrics'].append(row)

    print(f"🎯 Kernel Invocations: {len(kernels_by_id)}\n")

    for kid, data in sorted(kernels_by_id.items(), key=lambda x: int(x[0])):
        print(f"\n{'─'*80}")
        print(f"ID {kid}: {data['name']}")
        print(f"{'─'*80}")
        print(f"  Metrics collected: {len(data['metrics'])}")

        # Show some key metrics
        interesting_metrics = [
            'smsp__sass_thread_inst_executed_op_dadd_pred_on.sum',
            'smsp__sass_thread_inst_executed_op_dmul_pred_on.sum',
            'smsp__sass_thread_inst_executed_op_dfma_pred_on.sum',
            'dram__bytes_read.sum',
            'dram__bytes_write.sum',
            'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum',
            'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum',
        ]

        print(f"\n  Key Counters:")
        for row in data['metrics']:
            metric_name = row.get('Metric Name', '')
            if metric_name in interesting_metrics:
                value = row.get('Metric Value', '0')
                print(f"    {metric_name:<60}: {value}")

def compare_kernels(csv_file):
    """Compare different kernel versions"""

    print(f"\n{'='*80}")
    print(f"KERNEL COMPARISON")
    print(f"{'='*80}\n")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by kernel
    kernels = {}
    for row in rows:
        kernel_name = row.get('Kernel Name', 'Unknown')
        if kernel_name not in kernels:
            kernels[kernel_name] = row

    # Extract key metrics for comparison
    metrics_to_compare = [
        'Duration',
        'Achieved Occupancy',
        'Compute (SM) Throughput',
        'Memory Throughput',
        'DRAM Throughput',
    ]

    print(f"{'Kernel':<30} ", end='')
    for metric in metrics_to_compare:
        print(f"{metric:<25} ", end='')
    print()
    print('─' * 160)

    for kernel_name, data in kernels.items():
        # Shorten kernel name
        short_name = kernel_name.split('(')[0][-30:]
        print(f"{short_name:<30} ", end='')

        for metric in metrics_to_compare:
            value = data.get(metric, 'N/A')
            print(f"{str(value):<25} ", end='')
        print()

def main():
    print("\n" + "="*80)
    print("  NCU (NVIDIA Nsight Compute) Profile Analysis")
    print("="*80)

    # Analyze details
    try:
        analyze_ncu_details('ncu_details.csv')
    except FileNotFoundError:
        print("ncu_details.csv not found")
    except Exception as e:
        print(f"Error analyzing details: {e}")

    # Analyze raw metrics
    try:
        analyze_ncu_raw_metrics('ncu_raw_metrics.csv')
    except FileNotFoundError:
        print("ncu_raw_metrics.csv not found")
    except Exception as e:
        print(f"Error analyzing raw metrics: {e}")

    # Compare kernels
    try:
        compare_kernels('ncu_details.csv')
    except FileNotFoundError:
        print("ncu_details.csv not found")
    except Exception as e:
        print(f"Error comparing kernels: {e}")

    print("\n" + "="*80)
    print("  Analysis Complete")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
