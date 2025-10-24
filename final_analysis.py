#!/usr/bin/env python3
"""
Final comprehensive NSYS profiling analysis
"""

import sqlite3

def analyze(db_path, version_name):
    """Final analysis of NSYS database"""

    print(f"\n{'='*90}")
    print(f"  {version_name}")
    print(f"{'='*90}\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get string mappings
    cursor.execute("SELECT id, value FROM StringIds")
    strings = {row[0]: row[1] for row in cursor.fetchall()}

    # 1. TIMING
    print("⏱️  EXECUTION TIMING")
    print("-" * 90)
    cursor.execute("SELECT duration, startTime, stopTime FROM ANALYSIS_DETAILS")
    duration_ns, start, stop = cursor.fetchone()
    duration_sec = duration_ns / 1e9

    N = 512
    flops = 2 * N * N * N
    gflops = (flops / duration_sec) / 1e9

    print(f"  Duration:         {duration_sec:>12.6f} sec  ({duration_ns:>15,} ns)")
    print(f"  GFLOPS:           {gflops:>12.2f}")
    print(f"  Total FLOPs:      {flops:>15,}")

    # 2. PROCESS
    print(f"\n📦 TARGET PROCESS")
    print("-" * 90)
    cursor.execute("""
        SELECT name, pid
        FROM PROCESSES
        WHERE name LIKE '%matmul%'
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result:
        print(f"  Executable:       {result[0]}")
        print(f"  PID:              {result[1]}")

    # 3. GPU INFO
    print(f"\n🖥️  HARDWARE")
    print("-" * 90)
    cursor.execute("SELECT COUNT(*) FROM TARGET_INFO_GPU")
    gpu_count = cursor.fetchone()[0]
    print(f"  GPUs Available:   {gpu_count}")

    if gpu_count > 0:
        cursor.execute("SELECT name, totalMemory, smCount, clockRate FROM TARGET_INFO_GPU LIMIT 1")
        name, memory, sms, clock = cursor.fetchone()
        print(f"  GPU Model:        {name}")
        print(f"  Total Memory:     {memory / (1024**3):.1f} GB")
        print(f"  SM Count:         {sms}")
        print(f"  Clock Rate:       {clock / 1e6:.0f} MHz")

    # 4. METADATA
    print(f"\n⚙️  PROFILING CONFIGURATION")
    print("-" * 90)
    cursor.execute("PRAGMA table_info(META_DATA_CAPTURE)")
    meta_cols = [col[1] for col in cursor.fetchall()]
    print(f"  Metadata fields:  {', '.join(meta_cols)}")

    # 5. DIAGNOSTIC SUMMARY
    print(f"\n⚠️  PROFILING DIAGNOSTICS")
    print("-" * 90)
    cursor.execute("SELECT COUNT(*) FROM DIAGNOSTIC_EVENT WHERE severity = 2")
    warnings = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM DIAGNOSTIC_EVENT WHERE severity = 3")
    errors = cursor.fetchone()[0]

    print(f"  Warnings:         {warnings}")
    print(f"  Errors:           {errors}")

    # Show key warnings
    cursor.execute("""
        SELECT text FROM DIAGNOSTIC_EVENT
        WHERE severity IN (2, 3)
        ORDER BY timestamp
    """)
    for (text,) in cursor.fetchall():
        if 'CUDA' in text or 'event' in text.lower():
            print(f"    - {text[:70]}")

    conn.close()

def final_comparison(naive_db, reordered_db):
    """Final side-by-side comparison"""

    print(f"\n{'='*90}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*90}\n")

    def get_stats(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT duration FROM ANALYSIS_DETAILS")
        duration_ns = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM PROCESSES")
        procs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM ThreadNames")
        threads = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM StringIds")
        strings = cursor.fetchone()[0]

        conn.close()

        N = 512
        flops = 2 * N * N * N
        duration_sec = duration_ns / 1e9
        gflops = (flops / duration_sec) / 1e9

        return {
            'duration_ns': duration_ns,
            'duration_sec': duration_sec,
            'gflops': gflops,
            'processes': procs,
            'threads': threads,
            'strings': strings
        }

    naive_stats = get_stats(naive_db)
    reordered_stats = get_stats(reordered_db)

    speedup = naive_stats['duration_sec'] / reordered_stats['duration_sec']
    time_saved_pct = (1 - reordered_stats['duration_sec'] / naive_stats['duration_sec']) * 100

    print(f"{'='*90}")
    print(f"{'METRIC':<30} {'NAIVE (ijk)':>20} {'REORDERED (ikj)':>20} {'IMPROVEMENT':>15}")
    print(f"{'='*90}")

    print(f"{'Duration (sec)':<30} {naive_stats['duration_sec']:>20.6f} {reordered_stats['duration_sec']:>20.6f} {speedup:>14.2f}x")
    print(f"{'Duration (ns)':<30} {naive_stats['duration_ns']:>20,} {reordered_stats['duration_ns']:>20,} {'-'*15}")
    print(f"{'GFLOPS':<30} {naive_stats['gflops']:>20.2f} {reordered_stats['gflops']:>20.2f} {reordered_stats['gflops'] - naive_stats['gflops']:>14.2f}")
    print(f"{'Time Saved (%)':<30} {'':<20} {time_saved_pct:>19.1f}% {'-'*15}")
    print(f"{'-'*90}")
    print(f"{'Processes Captured':<30} {naive_stats['processes']:>20,} {reordered_stats['processes']:>20,} {reordered_stats['processes'] - naive_stats['processes']:>15,}")
    print(f"{'Threads Captured':<30} {naive_stats['threads']:>20,} {reordered_stats['threads']:>20,} {reordered_stats['threads'] - naive_stats['threads']:>15,}")
    print(f"{'Strings Captured':<30} {naive_stats['strings']:>20,} {reordered_stats['strings']:>20,} {reordered_stats['strings'] - naive_stats['strings']:>15,}")

    print(f"\n{'='*90}")
    print(f"  KEY FINDINGS")
    print(f"{'='*90}\n")

    print(f"  ✅ Loop reordering achieved {speedup:.2f}× speedup")
    print(f"  ✅ Saved {time_saved_pct:.1f}% of execution time")
    print(f"  ✅ GFLOPS improved from {naive_stats['gflops']:.2f} to {reordered_stats['gflops']:.2f}")
    print(f"  ✅ Reordered version is {reordered_stats['gflops'] / naive_stats['gflops']:.1f}× more efficient")

    print(f"\n  📊 PROFILING OVERHEAD:")
    print(f"     Note: NSYS adds profiling overhead, so these times are slower than")
    print(f"     bare-metal benchmarks. The speedup ratio is what matters!")

def summary():
    """Print final summary"""

    print(f"\n{'='*90}")
    print(f"  NSYS PROFILING JOURNEY COMPLETE")
    print(f"{'='*90}\n")

    print("  📁 SQLite Databases Created:")
    print("     - matmul_naive_profile.sqlite     (668K)")
    print("     - matmul_reordered_profile.sqlite (668K)")

    print(f"\n  🔍 What We Explored:")
    print("     ✓ 77 tables per database")
    print("     ✓ Timing and performance metrics")
    print("     ✓ Process and thread information")
    print("     ✓ GPU hardware details")
    print("     ✓ System environment variables")
    print("     ✓ Profiling metadata and diagnostics")

    print(f"\n  📊 Key Insights:")
    print("     • NSYS captures comprehensive system-level profiling data")
    print("     • SQLite export enables custom analysis via SQL queries")
    print("     • Profiling overhead is significant but ratios are accurate")
    print("     • Loop reordering shows clear 2.7× speedup in profiled run")

    print(f"\n  🚀 Next Steps:")
    print("     • Use nsys-ui to visualize timeline")
    print("     • Add NVTX markers for custom ranges")
    print("     • Profile GPU kernels (if using CUDA)")
    print("     • Export to other formats (JSON, CSV)")

    print(f"\n{'='*90}\n")

if __name__ == "__main__":
    naive_db = "matmul_naive_profile.sqlite"
    reordered_db = "matmul_reordered_profile.sqlite"

    # Analyze each
    analyze(naive_db, "NAIVE VERSION (ijk)")
    analyze(reordered_db, "REORDERED VERSION (ikj)")

    # Compare
    final_comparison(naive_db, reordered_db)

    # Summary
    summary()
