#!/usr/bin/env python3
"""
Comprehensive NCU (NVIDIA Nsight Compute) Analysis
Analyzes GPU kernel profiling data from NCU CSV exports
"""

import csv
from collections import defaultdict
import sys

def parse_ncu_csv(csv_file):
    """Parse NCU CSV and organize by kernel"""

    kernels = defaultdict(lambda: defaultdict(list))

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel_id = row['ID']
            kernel_name = row['Kernel Name']
            section = row['Section Name']
            metric_name = row['Metric Name']
            metric_value = row['Metric Value']
            metric_unit = row['Metric Unit']

            # Store metric
            kernels[kernel_id][kernel_name].append({
                'section': section,
                'name': metric_name,
                'value': metric_value,
                'unit': metric_unit
            })

    return kernels

def get_metric_value(metrics, metric_name):
    """Find a specific metric by name"""
    for m in metrics:
        if m['name'] == metric_name:
            return m['value'], m['unit']
    return None, None

def analyze_kernel(kernel_name, metrics):
    """Analyze a single kernel's metrics"""

    print(f"\n{'='*90}")
    print(f"Kernel: {kernel_name.split('(')[0]}")
    print(f"{'='*90}\n")

    # ========================================================================
    # 1. GPU Speed of Light (SOL) - Overall throughput analysis
    # ========================================================================
    print("🚀 GPU SPEED OF LIGHT")
    print("-" * 90)

    sol_metrics = [
        ('Memory Throughput', '%'),
        ('Compute (SM) Throughput', '%'),
        ('Elapsed Cycles', 'cycle'),
        ('SM Frequency', 'Ghz'),
        ('DRAM Frequency', 'Ghz'),
    ]

    for metric, expected_unit in sol_metrics:
        val, unit = get_metric_value(metrics, metric)
        if val:
            print(f"  {metric:<35}: {val:>15} {unit}")

    # ========================================================================
    # 2. Memory Workload Analysis
    # ========================================================================
    print(f"\n💾 MEMORY WORKLOAD")
    print("-" * 90)

    memory_metrics = [
        ('Memory Throughput', '%'),
        ('L1/TEX Hit Rate', '%'),
        ('L2 Hit Rate', '%'),
        ('Mem Busy', '%'),
        ('Max Bandwidth', '%'),
        ('L1/TEX Cache Throughput', '%'),
        ('L2 Cache Throughput', '%'),
        ('DRAM Throughput', '%'),
    ]

    for metric, expected_unit in memory_metrics:
        val, unit = get_metric_value(metrics, metric)
        if val:
            print(f"  {metric:<35}: {val:>15} {unit}")

    # ========================================================================
    # 3. Compute Workload Analysis
    # ========================================================================
    print(f"\n⚡ COMPUTE WORKLOAD")
    print("-" * 90)

    compute_metrics = [
        ('Compute (SM) Throughput', '%'),
        ('Warp Cycles Per Issued Instruction', 'inst'),
        ('Warp Cycles Per Executed Instruction', 'inst'),
        ('Avg. Active Threads Per Warp', 'thread'),
        ('Avg. Not Predicated Off Threads Per Warp', 'thread'),
    ]

    for metric, expected_unit in compute_metrics:
        val, unit = get_metric_value(metrics, metric)
        if val:
            print(f"  {metric:<35}: {val:>15} {unit}")

    # ========================================================================
    # 4. Occupancy Analysis
    # ========================================================================
    print(f"\n📊 OCCUPANCY")
    print("-" * 90)

    occupancy_metrics = [
        ('Block Limit SM', 'block'),
        ('Block Limit Registers', 'block'),
        ('Block Limit Shared Mem', 'block'),
        ('Block Limit Warps', 'block'),
        ('Theoretical Active Warps per SM', 'warp'),
        ('Theoretical Occupancy', '%'),
        ('Achieved Occupancy', '%'),
        ('Achieved Active Warps Per SM', 'warp'),
    ]

    for metric, expected_unit in occupancy_metrics:
        val, unit = get_metric_value(metrics, metric)
        if val:
            print(f"  {metric:<35}: {val:>15} {unit}")

    # ========================================================================
    # 5. Launch Statistics
    # ========================================================================
    print(f"\n🎯 LAUNCH CONFIGURATION")
    print("-" * 90)

    launch_metrics = [
        ('Grid Size', None),
        ('Block Size', None),
        ('Registers Per Thread', 'register'),
        ('Static Shared Memory Per Block', 'byte'),
        ('Dynamic Shared Memory Per Block', 'byte'),
        ('Shared Memory Configuration Size', 'Kbyte'),
        ('Threads', 'thread'),
        ('Waves Per SM', None),
    ]

    for metric, expected_unit in launch_metrics:
        val, unit = get_metric_value(metrics, metric)
        if val:
            unit_str = unit if unit else ''
            print(f"  {metric:<35}: {val:>15} {unit_str}")

    # ========================================================================
    # 6. Instruction Statistics
    # ========================================================================
    print(f"\n📝 INSTRUCTION STATISTICS")
    print("-" * 90)

    inst_metrics = [
        ('Executed Ipc Active', 'inst/cycle'),
        ('Issued Ipc Active', 'inst/cycle'),
        ('Executed Ipc Elapsed', 'inst/cycle'),
        ('Issue Slots Busy', '%'),
    ]

    for metric, expected_unit in inst_metrics:
        val, unit = get_metric_value(metrics, metric)
        if val:
            print(f"  {metric:<35}: {val:>15} {unit}")

def compare_kernels_side_by_side(kernels_data):
    """Compare all kernels side by side"""

    print(f"\n{'='*90}")
    print(f"KERNEL COMPARISON - Key Metrics")
    print(f"{'='*90}\n")

    # Get kernel names
    kernel_names = []
    kernel_metrics = {}

    for kid, kdata in sorted(kernels_data.items(), key=lambda x: int(x[0] if x[0] else '0')):
        for kname, metrics in kdata.items():
            short_name = kname.split('(')[0].replace('matmul_', '').replace('_kernel', '')
            kernel_names.append(short_name)
            kernel_metrics[short_name] = metrics

    # Key metrics to compare
    comparison_metrics = [
        ('Memory Throughput', '%'),
        ('Compute (SM) Throughput', '%'),
        ('Achieved Occupancy', '%'),
        ('L1/TEX Hit Rate', '%'),
        ('L2 Hit Rate', '%'),
        ('Elapsed Cycles', 'cycle'),
    ]

    # Print table header
    print(f"{'Metric':<30} ", end='')
    for kname in kernel_names:
        print(f"{kname[:20]:>22} ", end='')
    print()
    print('─' * (30 + 24 * len(kernel_names)))

    # Print each metric
    for metric, unit in comparison_metrics:
        print(f"{metric:<30} ", end='')
        for kname in kernel_names:
            val, _ = get_metric_value(kernel_metrics[kname], metric)
            if val:
                print(f"{val:>20} {unit} ", end='')
            else:
                print(f"{'N/A':>22} ", end='')
        print()

def create_optimization_guide(kernels_data):
    """Generate optimization recommendations based on profiling data"""

    print(f"\n{'='*90}")
    print(f"OPTIMIZATION INSIGHTS & RECOMMENDATIONS")
    print(f"{'='*90}\n")

    for kid, kdata in sorted(kernels_data.items(), key=lambda x: int(x[0] if x[0] else '0')):
        for kname, metrics in kdata.items():
            short_name = kname.split('(')[0]

            print(f"📌 {short_name}")
            print("-" * 90)

            # Get key metrics
            mem_throughput, _ = get_metric_value(metrics, 'Memory Throughput')
            compute_throughput, _ = get_metric_value(metrics, 'Compute (SM) Throughput')
            occupancy, _ = get_metric_value(metrics, 'Achieved Occupancy')
            l1_hit_rate, _ = get_metric_value(metrics, 'L1/TEX Hit Rate')
            l2_hit_rate, _ = get_metric_value(metrics, 'L2 Hit Rate')

            recommendations = []

            # Memory-bound analysis
            if mem_throughput and compute_throughput:
                try:
                    mem_val = float(mem_throughput.replace(',', ''))
                    comp_val = float(compute_throughput.replace(',', ''))

                    if mem_val > 70:
                        recommendations.append("⚠️  MEMORY BOUND: Memory throughput is very high")
                        recommendations.append("   → Consider using shared memory for data reuse")
                        recommendations.append("   → Check for uncoalesced global memory accesses")
                        recommendations.append("   → Look into memory access patterns")

                    if comp_val > 70:
                        recommendations.append("⚠️  COMPUTE BOUND: SM throughput is very high")
                        recommendations.append("   → Good utilization of compute resources")
                        recommendations.append("   → Consider optimizing instruction mix")
                except:
                    pass

            # Occupancy analysis
            if occupancy:
                try:
                    occ_val = float(occupancy.replace(',', ''))
                    if occ_val < 50:
                        recommendations.append(f"⚠️  LOW OCCUPANCY: {occupancy}%")
                        recommendations.append("   → Reduce register usage per thread")
                        recommendations.append("   → Reduce shared memory usage")
                        recommendations.append("   → Increase threads per block")
                    elif occ_val > 75:
                        recommendations.append(f"✓ GOOD OCCUPANCY: {occupancy}%")
                except:
                    pass

            # Cache hit rate analysis
            if l1_hit_rate:
                try:
                    l1_val = float(l1_hit_rate.replace(',', ''))
                    if l1_val < 70:
                        recommendations.append(f"⚠️  LOW L1 HIT RATE: {l1_hit_rate}%")
                        recommendations.append("   → Improve data locality")
                        recommendations.append("   → Use shared memory for frequently accessed data")
                except:
                    pass

            if l2_hit_rate:
                try:
                    l2_val = float(l2_hit_rate.replace(',', ''))
                    if l2_val < 70:
                        recommendations.append(f"⚠️  LOW L2 HIT RATE: {l2_hit_rate}%")
                        recommendations.append("   → Consider tiling/blocking to improve locality")
                except:
                    pass

            if recommendations:
                for rec in recommendations:
                    print(f"  {rec}")
            else:
                print("  ✓ No major issues detected")

            print()

def main():
    csv_file = 'ncu_details.csv'

    print("\n" + "="*90)
    print("  NCU (NVIDIA Nsight Compute) Comprehensive GPU Kernel Analysis")
    print("  NVIDIA B200 GPU Profiling")
    print("="*90)

    try:
        # Parse CSV
        print(f"\n📂 Loading: {csv_file}")
        kernels_data = parse_ncu_csv(csv_file)
        print(f"✓ Found {len(kernels_data)} kernel invocation(s)\n")

        # Analyze each kernel
        for kid, kdata in sorted(kernels_data.items(), key=lambda x: int(x[0] if x[0] else '0')):
            for kname, metrics in kdata.items():
                analyze_kernel(kname, metrics)

        # Side-by-side comparison
        if len(kernels_data) > 1:
            compare_kernels_side_by_side(kernels_data)

        # Optimization guide
        create_optimization_guide(kernels_data)

        print("\n" + "="*90)
        print("  Analysis Complete!")
        print("  For interactive exploration, open ncu_matmul_profile.ncu-rep in Nsight Compute UI")
        print("="*90 + "\n")

    except FileNotFoundError:
        print(f"❌ Error: {csv_file} not found")
        print("   Please ensure NCU profiling data has been exported to CSV")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
