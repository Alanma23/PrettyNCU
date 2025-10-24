#!/usr/bin/env python3
"""
Extract Actionable Insights from NCU CSV
Shows that NCU exports contain optimization recommendations, not just raw data
"""

import csv
from collections import defaultdict

def extract_insights(csv_file):
    """Extract optimization rules and recommendations from NCU CSV"""

    print("\n" + "="*90)
    print("  ACTIONABLE INSIGHTS FROM NCU PROFILING DATA")
    print("="*90 + "\n")

    insights_by_kernel = defaultdict(list)

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Look for optimization opportunities
            if row['Rule Type'] == 'OPT' and row['Rule Description']:
                kernel_id = row['ID']
                kernel_name = row['Kernel Name'].split('(')[0]

                insights_by_kernel[kernel_id].append({
                    'name': kernel_name,
                    'rule': row['Rule Name'],
                    'description': row['Rule Description'],
                    'speedup': row.get('Estimated Speedup', 'N/A'),
                    'scope': row.get('Estimated Speedup Type', 'N/A')
                })

    # Display insights for each kernel
    for kid, insights in sorted(insights_by_kernel.items()):
        if not insights:
            continue

        kernel_name = insights[0]['name']
        print(f"\n{'─'*90}")
        print(f"Kernel ID {kid}: {kernel_name}")
        print(f"{'─'*90}\n")

        # Remove duplicates (same insight appears multiple times)
        seen = set()
        unique_insights = []
        for insight in insights:
            key = (insight['rule'], insight['description'])
            if key not in seen:
                seen.add(key)
                unique_insights.append(insight)

        print(f"Found {len(unique_insights)} optimization opportunities:\n")

        for i, insight in enumerate(unique_insights, 1):
            print(f"{i}. Issue: {insight['rule']}")

            if insight['speedup'] != 'N/A' and insight['speedup']:
                print(f"   Potential Speedup: {insight['speedup']}% ({insight['scope']})")

            # Print description with word wrap
            desc = insight['description']
            print(f"\n   Recommendation:")

            # Simple word wrapping
            words = desc.split()
            line = "   "
            for word in words:
                if len(line) + len(word) + 1 > 88:
                    print(line)
                    line = "   " + word
                else:
                    line += " " + word if line != "   " else word
            if line.strip():
                print(line)

            print()

def summarize_speedup_potential(csv_file):
    """Calculate total speedup potential per kernel"""

    print("\n" + "="*90)
    print("  TOTAL OPTIMIZATION POTENTIAL")
    print("="*90 + "\n")

    speedups = defaultdict(lambda: {'name': '', 'total': 0, 'count': 0})

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row['Rule Type'] == 'OPT' and row.get('Estimated Speedup'):
                try:
                    speedup = float(row['Estimated Speedup'])
                    kernel_id = row['ID']
                    kernel_name = row['Kernel Name'].split('(')[0]

                    speedups[kernel_id]['name'] = kernel_name
                    speedups[kernel_id]['total'] += speedup
                    speedups[kernel_id]['count'] += 1
                except ValueError:
                    pass

    print(f"{'Kernel ID':<12} {'Kernel Name':<30} {'Opportunities':<15} {'Total Speedup'}")
    print("─" * 90)

    for kid, data in sorted(speedups.items()):
        print(f"{kid:<12} {data['name']:<30} {data['count']:<15} {data['total']:.1f}%")

def main():
    csv_file = 'ncu_details.csv'

    try:
        extract_insights(csv_file)
        summarize_speedup_potential(csv_file)

        print("\n" + "="*90)
        print("  KEY TAKEAWAY")
        print("="*90)
        print("""
NCU CSV exports contain:
  ✓ Raw performance metrics (memory throughput, occupancy, IPC, etc.)
  ✓ Automatic bottleneck analysis (memory-bound, compute-bound, etc.)
  ✓ Specific optimization recommendations with actionable steps
  ✓ Estimated speedup potential for each optimization
  ✓ Root cause analysis (e.g., "50% of time in L1TEX stalls")

This is the SAME intelligent analysis as the GUI, just in CSV format!
Perfect for programmatic analysis, CI/CD, and tracking optimization progress.
""")
        print("="*90 + "\n")

    except FileNotFoundError:
        print(f"\n❌ Error: {csv_file} not found")
        print("   Make sure you're in the directory with NCU profiling data\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
