#!/usr/bin/env python3
"""
Create Visualizations from NCU CSV Data
Demonstrates what you can do with the exported data
"""

import pandas as pd
import csv
from collections import defaultdict

def create_comparison_chart():
    """Create ASCII chart comparing kernel performance"""

    print("\n" + "="*90)
    print("  KERNEL PERFORMANCE COMPARISON - Visual Chart from NCU Data")
    print("="*90 + "\n")

    # Read NCU details
    df = pd.read_csv('ncu_details.csv')

    # Extract key metrics for each kernel
    kernels = {}
    for kid in ['0', '1', '2', '3']:
        kernel_data = df[df['ID'] == kid]

        # Get kernel name
        kernel_name = kernel_data['Kernel Name'].iloc[0] if len(kernel_data) > 0 else f"Kernel {kid}"
        kernel_short = kernel_name.split('(')[0]

        # Extract metrics
        metrics = {}
        for _, row in kernel_data.iterrows():
            metric_name = row['Metric Name']
            metric_value = row['Metric Value']

            if metric_name == 'Memory Throughput':
                try:
                    metrics['memory_throughput'] = float(str(metric_value).replace(',', ''))
                except:
                    pass
            elif metric_name == 'Compute (SM) Throughput':
                try:
                    metrics['compute_throughput'] = float(str(metric_value).replace(',', ''))
                except:
                    pass
            elif metric_name == 'Elapsed Cycles':
                try:
                    metrics['cycles'] = int(str(metric_value).replace(',', ''))
                except:
                    pass
            elif metric_name == 'Achieved Occupancy':
                try:
                    metrics['occupancy'] = float(str(metric_value).replace(',', ''))
                except:
                    pass
            elif metric_name == 'L1/TEX Hit Rate':
                try:
                    metrics['l1_hit_rate'] = float(str(metric_value).replace(',', ''))
                except:
                    pass

        if metrics:
            kernels[kid] = {
                'name': kernel_short,
                'metrics': metrics
            }

    # Create visual comparison
    if kernels:
        # Cycles comparison
        print("📊 EXECUTION CYCLES COMPARISON")
        print("─" * 90)

        max_cycles = max(k['metrics'].get('cycles', 0) for k in kernels.values())
        baseline_cycles = kernels['0']['metrics'].get('cycles', 1)

        for kid, data in sorted(kernels.items()):
            cycles = data['metrics'].get('cycles', 0)
            if not cycles:
                continue

            speedup = baseline_cycles / cycles if cycles > 0 else 1.0
            bar_length = 40
            filled = int((cycles / max_cycles) * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)

            print(f"{data['name']:25} [{bar}] {cycles:8,} ({speedup:.2f}×)")

        print()

        # Memory vs Compute
        print("📊 MEMORY VS COMPUTE THROUGHPUT")
        print("─" * 90)

        for kid, data in sorted(kernels.items()):
            mem = data['metrics'].get('memory_throughput', 0)
            comp = data['metrics'].get('compute_throughput', 0)

            if not mem or not comp:
                continue

            mem_bar = int(mem / 2.5)  # Scale to 40 chars max
            comp_bar = int(comp / 2.5)

            print(f"\n{data['name']}:")
            print(f"  Memory:  [{'█' * mem_bar}{'░' * (40 - mem_bar)}] {mem:.1f}%")
            print(f"  Compute: [{'█' * comp_bar}{'░' * (40 - comp_bar)}] {comp:.1f}%")

            if mem > comp + 10:
                print(f"           → MEMORY-BOUND (memory {mem - comp:.1f}% higher)")
            elif comp > mem + 10:
                print(f"           → COMPUTE-BOUND (compute {comp - mem:.1f}% higher)")
            else:
                print(f"           → BALANCED")

        print()

        # Occupancy comparison
        print("📊 OCCUPANCY COMPARISON")
        print("─" * 90)

        for kid, data in sorted(kernels.items()):
            occ = data['metrics'].get('occupancy', 0)
            if not occ:
                continue

            occ_bar = int(occ / 2.5)
            print(f"{data['name']:25} [{'█' * occ_bar}{'░' * (40 - occ_bar)}] {occ:.1f}%")

        print()

        # L1 hit rate
        print("📊 L1 CACHE HIT RATE")
        print("─" * 90)

        for kid, data in sorted(kernels.items()):
            l1 = data['metrics'].get('l1_hit_rate', 0)
            if l1 is None:
                continue

            l1_bar = int(l1 / 2.5) if l1 >= 0 else 0
            print(f"{data['name']:25} [{'█' * l1_bar}{'░' * (40 - l1_bar)}] {l1:.2f}%")

            if l1 < 1 and kid in ['2', '3']:
                print(f"{'':27} ⚠️  Low L1 hit rate - using shared memory instead!")

        print()

def create_speedup_waterfall():
    """Create waterfall chart showing speedup opportunities"""

    print("\n" + "="*90)
    print("  OPTIMIZATION WATERFALL - Speedup Opportunities")
    print("="*90 + "\n")

    # Parse insights
    with open('ncu_details.csv', 'r') as f:
        reader = csv.DictReader(f)

        opportunities = []
        seen = set()

        for row in reader:
            if row['ID'] == '0' and row['Rule Type'] == 'OPT':
                key = row['Rule Name']
                if key in seen:
                    continue
                seen.add(key)

                try:
                    speedup = float(row.get('Estimated Speedup', 0) or 0)
                except:
                    speedup = 0

                if speedup > 0:
                    opportunities.append({
                        'name': row['Rule Name'],
                        'speedup': speedup,
                        'desc': row['Rule Description'][:60] + "..."
                    })

        # Sort by speedup
        opportunities.sort(key=lambda x: x['speedup'], reverse=True)

        # Display waterfall
        baseline = 100.0
        current = baseline

        print(f"Starting Performance: {baseline:.1f}%")
        print()

        for i, opp in enumerate(opportunities, 1):
            # Calculate new performance
            gain = opp['speedup']
            new_perf = current * (1 + gain / 100)

            # Visual arrow
            arrow_len = int(gain / 2) if gain > 0 else 1
            arrow = "─" * arrow_len + ">"

            print(f"{i}. {opp['name']}")
            print(f"   {current:.1f}% {arrow} {new_perf:.1f}% (+{gain:.1f}%)")
            print(f"   💡 {opp['desc']}")
            print()

            current = new_perf

        total_gain = ((current - baseline) / baseline) * 100
        print("─" * 90)
        print(f"📈 Total Potential: {baseline:.1f}% → {current:.1f}% ({total_gain:+.1f}%)")
        print()

def create_roofline_data():
    """Extract data needed for roofline chart"""

    print("\n" + "="*90)
    print("  ROOFLINE CHART DATA (requires manual plotting)")
    print("="*90 + "\n")

    df = pd.read_csv('ncu_details.csv')

    print("To create a roofline chart, you need:")
    print("1. Peak Performance (GFLOPS)")
    print("2. Peak Memory Bandwidth (GB/s)")
    print("3. Arithmetic Intensity (FLOP/byte)")
    print("4. Achieved Performance (GFLOPS)")
    print()

    print("From our data:")
    print()

    for kid in ['0', '2', '3']:
        kernel_data = df[df['ID'] == kid]
        kernel_name = kernel_data['Kernel Name'].iloc[0].split('(')[0] if len(kernel_data) > 0 else f"Kernel {kid}"

        # Extract duration and compute throughput
        duration = None
        compute_pct = None
        memory_bw = None

        for _, row in kernel_data.iterrows():
            if row['Metric Name'] == 'Duration':
                try:
                    duration = float(str(row['Metric Value']).replace(',', ''))  # microseconds
                except:
                    pass
            elif row['Metric Name'] == 'Compute (SM) Throughput':
                try:
                    compute_pct = float(str(row['Metric Value']).replace(',', ''))
                except:
                    pass
            elif row['Metric Name'] == 'Memory Throughput':
                try:
                    memory_bw = float(str(row['Metric Value']).replace(',', ''))
                except:
                    pass

        if duration and compute_pct:
            # B200 peak FP32: ~67 TFLOPS (estimated)
            peak_gflops = 67000  # GFLOPS
            achieved_gflops = peak_gflops * (compute_pct / 100)

            print(f"{kernel_name}:")
            print(f"  Duration: {duration:.2f} μs")
            print(f"  Compute Throughput: {compute_pct:.1f}% of peak")
            print(f"  Achieved Performance: ~{achieved_gflops:.0f} GFLOPS")
            print(f"  Memory Throughput: {memory_bw:.1f}% of peak")
            print()

    print("For a complete roofline chart, use:")
    print("  ncu --set roofline --export roofline_report ./matmul_cuda 512")
    print()

def main():
    """Generate all visualizations"""

    print("\n" + "╔" + "="*88 + "╗")
    print("║" + " "*20 + "NCU DATA VISUALIZATION - ASCII Charts from CSV" + " "*21 + "║")
    print("╚" + "="*88 + "╝")

    try:
        create_comparison_chart()
        create_speedup_waterfall()
        create_roofline_data()

        print("\n" + "="*90)
        print("  💡 NEXT STEPS")
        print("="*90)
        print("""
For production-quality charts:

1. Use matplotlib/plotly to create publication-ready charts from this CSV data
2. Export the .ncu-rep file to a machine with GUI and use ncu-ui for images
3. Re-profile with --set roofline for complete roofline analysis data

All the data is here - just needs visualization! 📊
        """)

    except FileNotFoundError:
        print("\n❌ Error: ncu_details.csv not found")
        print("   Make sure you're in the directory with NCU profiling data\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
