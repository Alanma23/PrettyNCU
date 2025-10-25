#!/bin/bash
#
# ncu-llm-lib.sh - Library functions for NCU-LLM suite
#

# Initialize output directory and timestamp
init_output() {
    mkdir -p "$NCU_LLM_OUTPUT_DIR"

    if [ "${NCU_LLM_TIMESTAMP:-yes}" = "yes" ]; then
        TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    else
        TIMESTAMP=""
    fi
}

# Log to index file
log_to_index() {
    local command="$1"
    local files="$2"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $command" >> "$NCU_LLM_OUTPUT_DIR/index.txt"
    echo "  Files: $files" >> "$NCU_LLM_OUTPUT_DIR/index.txt"
    echo "" >> "$NCU_LLM_OUTPUT_DIR/index.txt"
}

# Generate output filename
get_output_file() {
    local prefix="$1"
    local suffix="$2"

    if [ -n "$TIMESTAMP" ]; then
        echo "$NCU_LLM_OUTPUT_DIR/${prefix}-${TIMESTAMP}${suffix}"
    else
        echo "$NCU_LLM_OUTPUT_DIR/${prefix}${suffix}"
    fi
}

# ============================================================================
# Command: quick - Ultra-minimal profiling
# ============================================================================
ncu_llm_quick() {
    init_output

    echo "ncu-llm: Running QUICK profile (ultra-minimal, ~5K tokens)..."

    local output_txt=$(get_output_file "quick" ".txt")
    local output_csv=$(get_output_file "quick" ".csv")
    local output_rep=$(get_output_file "quick" "-raw")

    # Essential metrics only (8 metrics)
    local metrics="dram__throughput.avg.pct_of_peak_sustained_elapsed"
    metrics="$metrics,sm__throughput.avg.pct_of_peak_sustained_elapsed"
    metrics="$metrics,gpu__time_duration.sum"
    metrics="$metrics,launch__occupancy_limit_warps"
    metrics="$metrics,launch__occupancy_limit_blocks"
    metrics="$metrics,l1tex__t_sector_hit_rate.pct"
    metrics="$metrics,lts__t_sector_hit_rate.pct"
    metrics="$metrics,smsp__cycles_elapsed.avg"

    # Profile
    ncu --metrics "$metrics" \
        --disable-extra-suffixes \
        --launch-count 1 \
        -o "$output_rep" \
        "$@" > /dev/null 2>&1

    # Export to CSV
    ncu --import "${output_rep}.ncu-rep" --page details --csv > "$output_csv" 2>/dev/null

    # Generate human-readable report
    cat > "$output_txt" << EOF
NCU-LLM Quick Profile Report
Generated: $(date)
Command: $@

═══════════════════════════════════════════════════════════════

ESSENTIAL METRICS (8 total)
EOF

    # Extract key metrics
    python3 << PYTHON >> "$output_txt"
import csv
import sys

try:
    with open('$output_csv', 'r') as f:
        reader = csv.DictReader(f)

        metrics = {}
        for row in reader:
            kid = row['ID']
            kname = row['Kernel Name'].split('(')[0] if row['Kernel Name'] else 'Unknown'
            metric = row['Metric Name']
            value = row['Metric Value']

            if kid not in metrics:
                metrics[kid] = {'name': kname, 'data': {}}
            metrics[kid]['data'][metric] = value

        for kid in sorted(metrics.keys()):
            kname = metrics[kid]['name']
            data = metrics[kid]['data']

            print(f"\nKernel {kid}: {kname}")
            print("─" * 60)

            # Memory throughput
            mem = data.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 'N/A')
            print(f"Memory Throughput:     {mem}%")

            # Compute throughput
            comp = data.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 'N/A')
            print(f"Compute Throughput:    {comp}%")

            # Bottleneck determination
            try:
                mem_val = float(str(mem).replace('%', '').replace(',', ''))
                comp_val = float(str(comp).replace('%', '').replace(',', ''))
                if mem_val > comp_val + 10:
                    print(f">>> Bottleneck:        MEMORY-BOUND (mem {mem_val:.1f}% > compute {comp_val:.1f}%)")
                elif comp_val > mem_val + 10:
                    print(f">>> Bottleneck:        COMPUTE-BOUND (compute {comp_val:.1f}% > mem {mem_val:.1f}%)")
                else:
                    print(f">>> Bottleneck:        BALANCED")
            except:
                print(f">>> Bottleneck:        Unable to determine")

            # L1 hit rate
            l1 = data.get('l1tex__t_sector_hit_rate.pct', 'N/A')
            print(f"L1 Cache Hit Rate:     {l1}%")

            # L2 hit rate
            l2 = data.get('lts__t_sector_hit_rate.pct', 'N/A')
            print(f"L2 Cache Hit Rate:     {l2}%")

            # Duration
            duration = data.get('gpu__time_duration.sum', 'N/A')
            unit = 'us' if duration != 'N/A' else ''
            print(f"Duration:              {duration} {unit}")

            # Occupancy limits
            occ_warps = data.get('launch__occupancy_limit_warps', 'N/A')
            occ_blocks = data.get('launch__occupancy_limit_blocks', 'N/A')
            print(f"Occupancy Limit:       Warps={occ_warps}, Blocks={occ_blocks}")

            # Cycles
            cycles = data.get('smsp__cycles_elapsed.avg', 'N/A')
            print(f"Average Cycles:        {cycles}")

except Exception as e:
    print(f"\nError processing metrics: {e}", file=sys.stderr)
PYTHON

    echo "" >> "$output_txt"
    echo "═══════════════════════════════════════════════════════════════" >> "$output_txt"
    echo "Files created:" >> "$output_txt"
    echo "  - $output_txt (this file)" >> "$output_txt"
    echo "  - $output_csv (CSV data)" >> "$output_txt"
    echo "" >> "$output_txt"
    echo "Size: $(wc -c < "$output_txt") bytes (~$(($(wc -c < "$output_txt") / 3)) tokens)" >> "$output_txt"

    log_to_index "quick $*" "$output_txt, $output_csv"

    echo "✅ Quick profile complete!"
    echo "   Report: $output_txt"
    echo "   CSV:    $output_csv"
    echo ""
    echo "Quick view:"
    grep ">>>" "$output_txt" || true
}

# ============================================================================
# Command: standard - Balanced profiling
# ============================================================================
ncu_llm_standard() {
    init_output

    echo "ncu-llm: Running STANDARD profile (balanced, ~30K tokens)..."

    local output_txt=$(get_output_file "standard" ".txt")
    local output_csv=$(get_output_file "standard" ".csv")
    local output_insights=$(get_output_file "standard" "-insights.txt")
    local output_rep=$(get_output_file "standard" "-raw")

    # Use basic set with disabled suffixes
    ncu --set basic \
        --disable-extra-suffixes \
        -o "$output_rep" \
        "$@" > /dev/null 2>&1

    # Export details
    ncu --import "${output_rep}.ncu-rep" --page details --csv > "$output_csv" 2>/dev/null

    # Generate main report
    cat > "$output_txt" << EOF
NCU-LLM Standard Profile Report
Generated: $(date)
Command: $@

═══════════════════════════════════════════════════════════════
SUMMARY
═══════════════════════════════════════════════════════════════
EOF

    # Extract insights
    grep "Rule Type" "$output_csv" | grep "OPT" | cut -d',' -f18 | sort -u > "$output_insights" 2>/dev/null || true

    python3 << PYTHON >> "$output_txt"
import csv

with open('$output_csv', 'r') as f:
    reader = csv.DictReader(f)

    kernels = {}
    for row in reader:
        kid = row['ID']
        if kid not in kernels:
            kernels[kid] = {
                'name': row['Kernel Name'].split('(')[0],
                'metrics': {},
                'insights': []
            }

        metric = row['Metric Name']
        value = row['Metric Value']
        if metric and value:
            kernels[kid]['metrics'][metric] = value

        if row.get('Rule Type') == 'OPT':
            kernels[kid]['insights'].append({
                'rule': row.get('Rule Name'),
                'desc': row.get('Rule Description'),
                'speedup': row.get('Estimated Speedup')
            })

    for kid in sorted(kernels.keys()):
        k = kernels[kid]
        print(f"\nKernel {kid}: {k['name']}")
        print("─" * 70)

        # Key metrics
        metrics_to_show = [
            'Memory Throughput',
            'Compute (SM) Throughput',
            'Achieved Occupancy',
            'L1/TEX Hit Rate',
            'Duration',
            'Elapsed Cycles'
        ]

        for m in metrics_to_show:
            val = k['metrics'].get(m, 'N/A')
            print(f"  {m:30} {val}")

        if k['insights']:
            print(f"\n  Optimization Opportunities ({len(k['insights'])} found):")
            for ins in k['insights'][:3]:  # Top 3
                speedup = ins['speedup'] if ins['speedup'] else 'N/A'
                print(f"    >>> {ins['rule']:30} +{speedup}% potential")
PYTHON

    echo "" >> "$output_txt"
    echo "═══════════════════════════════════════════════════════════════" >> "$output_txt"
    echo "For detailed insights, see: $output_insights" >> "$output_txt"
    echo "For CSV data, see: $output_csv" >> "$output_txt"

    log_to_index "standard $*" "$output_txt, $output_csv, $output_insights"

    echo "✅ Standard profile complete!"
    echo "   Report:   $output_txt"
    echo "   Insights: $output_insights"
    echo "   CSV:      $output_csv"
}

# ============================================================================
# Command: bottleneck - Quick memory vs compute check
# ============================================================================
ncu_llm_bottleneck() {
    init_output

    echo "ncu-llm: Running BOTTLENECK check (~2K tokens)..."

    local output_txt=$(get_output_file "bottleneck" ".txt")
    local temp_rep=$(mktemp)
    local temp_csv=$(mktemp)

    # Just speed of light
    ncu --section SpeedOfLight \
        --disable-extra-suffixes \
        --launch-count 1 \
        -o "$temp_rep" \
        "$@" > /dev/null 2>&1

    # Export to CSV
    ncu --import "${temp_rep}.ncu-rep" --page details --csv > "$temp_csv" 2>/dev/null

    cat > "$output_txt" << EOF
NCU-LLM Bottleneck Analysis
Generated: $(date)
Command: $@

═══════════════════════════════════════════════════════════════
BOTTLENECK DETERMINATION
═══════════════════════════════════════════════════════════════
EOF

    python3 << PYTHON >> "$output_txt"
import csv
import sys

with open('$temp_csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Metric Name'] == 'Memory Throughput':
            mem_throughput = row['Metric Value']
            kernel_name = row['Kernel Name'].split('(')[0]
        elif row['Metric Name'] == 'Compute (SM) Throughput':
            comp_throughput = row['Metric Value']

            try:
                mem = float(mem_throughput.replace('%', '').replace(',', ''))
                comp = float(comp_throughput.replace('%', '').replace(',', ''))

                print(f"\nKernel: {kernel_name}")
                print(f"  Memory Throughput:  {mem:.1f}%")
                print(f"  Compute Throughput: {comp:.1f}%")
                print(f"  Difference:         {abs(mem - comp):.1f}%")
                print()

                if mem > comp + 15:
                    print(f"  >>> VERDICT: MEMORY-BOUND")
                    print(f"      Memory is {mem - comp:.1f}% more utilized than compute")
                    print(f"      Focus on: Memory optimization, cache behavior, bandwidth")
                elif comp > mem + 15:
                    print(f"  >>> VERDICT: COMPUTE-BOUND")
                    print(f"      Compute is {comp - mem:.1f}% more utilized than memory")
                    print(f"      Focus on: IPC, instruction mix, pipeline utilization")
                else:
                    print(f"  >>> VERDICT: BALANCED")
                    print(f"      Memory and compute are similarly utilized")
                    print(f"      Profile deeper to find specific bottlenecks")
                print()
            except Exception as e:
                print(f"Error: {e}")
PYTHON

    rm -f "${temp_rep}.ncu-rep" "$temp_csv"

    log_to_index "bottleneck $*" "$output_txt"

    echo "✅ Bottleneck check complete!"
    echo "   Report: $output_txt"
    echo ""
    cat "$output_txt" | grep ">>>" || true
}

# ============================================================================
# Command: insights - Extract only actionable insights
# ============================================================================
ncu_llm_insights() {
    init_output

    if [ $# -lt 1 ]; then
        echo "Error: insights command requires a .ncu-rep file"
        echo "Usage: ncu-llm insights <report.ncu-rep>"
        exit 1
    fi

    local input_rep="$1"

    if [ ! -f "$input_rep" ]; then
        echo "Error: File not found: $input_rep"
        exit 1
    fi

    echo "ncu-llm: Extracting insights from $input_rep..."

    local output_txt=$(get_output_file "insights" ".txt")

    cat > "$output_txt" << EOF
NCU-LLM Actionable Insights
Generated: $(date)
Source: $input_rep

═══════════════════════════════════════════════════════════════
OPTIMIZATION OPPORTUNITIES
═══════════════════════════════════════════════════════════════
EOF

    ncu --import "$input_rep" --page details --csv 2>/dev/null | python3 << 'PYTHON' >> "$output_txt"
import csv
import sys

reader = csv.DictReader(sys.stdin)

insights_by_kernel = {}
for row in reader:
    if row.get('Rule Type') == 'OPT':
        kid = row['ID']
        kname = row['Kernel Name'].split('(')[0]

        if kid not in insights_by_kernel:
            insights_by_kernel[kid] = {'name': kname, 'insights': []}

        insights_by_kernel[kid]['insights'].append({
            'rule': row.get('Rule Name', ''),
            'desc': row.get('Rule Description', ''),
            'speedup': row.get('Estimated Speedup', '')
        })

for kid in sorted(insights_by_kernel.keys()):
    k = insights_by_kernel[kid]
    print(f"\nKernel {kid}: {k['name']}")
    print("─" * 70)

    # Sort by speedup
    sorted_insights = sorted(k['insights'],
                            key=lambda x: float(x['speedup']) if x['speedup'] else 0,
                            reverse=True)

    for i, ins in enumerate(sorted_insights, 1):
        speedup = f"+{ins['speedup']}%" if ins['speedup'] else "N/A"
        print(f"\n{i}. >>> {ins['rule']}")
        print(f"   Potential speedup: {speedup}")
        print(f"   Recommendation:")

        # Word wrap description
        words = ins['desc'].split()
        line = "      "
        for word in words[:100]:  # Limit length
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = "      " + word
            else:
                line += " " + word if line != "      " else word
        if line.strip():
            print(line)

    print()
PYTHON

    log_to_index "insights $input_rep" "$output_txt"

    echo "✅ Insights extracted!"
    echo "   Report: $output_txt"
    echo ""
    echo "Top recommendations:"
    grep "^[0-9]\\. >>>" "$output_txt" | head -5 || true
}

# Export remaining functions placeholder
ncu_llm_memory() {
    echo "ncu-llm memory: Not yet implemented"
    echo "Coming soon: Memory subsystem deep-dive"
}

ncu_llm_compute() {
    echo "ncu-llm compute: Not yet implemented"
    echo "Coming soon: Compute subsystem deep-dive"
}

ncu_llm_compare() {
    echo "ncu-llm compare: Not yet implemented"
    echo "Coming soon: Side-by-side kernel comparison"
}

ncu_llm_summary() {
    echo "ncu-llm summary: Not yet implemented"
    echo "Coming soon: One-page executive summary"
}

ncu_llm_export() {
    echo "ncu-llm export: Not yet implemented"
    echo "Coming soon: Export to various LLM-friendly formats"
}
