#!/usr/bin/env python3
"""
Comprehensive querying of NSYS SQLite databases
"""

import sqlite3
import sys

def run_queries(db_path, version_name):
    """Run comprehensive queries on the database"""

    print(f"\n{'='*90}")
    print(f"  {version_name}")
    print(f"{'='*90}\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get string mappings
    cursor.execute("SELECT id, value FROM StringIds")
    strings = {row[0]: row[1] for row in cursor.fetchall()}

    # 1. TIMING ANALYSIS
    print("⏱️  TIMING SUMMARY")
    print("-" * 90)
    cursor.execute("SELECT duration, startTime, stopTime FROM ANALYSIS_DETAILS")
    duration_ns, start, stop = cursor.fetchone()
    duration_sec = duration_ns / 1e9

    print(f"  Duration:     {duration_sec:>12.6f} sec  ({duration_ns:>15,} ns)")
    print(f"  Start Time:   {start:>15,} ns")
    print(f"  Stop Time:    {stop:>15,} ns")

    # Calculate performance metrics
    N = 512
    flops = 2 * N * N * N
    gflops = (flops / duration_sec) / 1e9
    print(f"\n  Performance:  {gflops:>12.2f} GFLOPS")
    print(f"  Operations:   {flops:>15,} FLOPs")

    # 2. PROCESS STATS
    print(f"\n📦 PROCESS STATISTICS")
    print("-" * 90)
    cursor.execute("SELECT COUNT(DISTINCT globalPid), COUNT(DISTINCT pid) FROM PROCESSES")
    global_procs, unique_pids = cursor.fetchone()
    print(f"  Total Processes:  {global_procs:>6}")
    print(f"  Unique PIDs:      {unique_pids:>6}")

    # Find our matmul process
    cursor.execute("""
        SELECT name, pid
        FROM PROCESSES
        WHERE name LIKE '%matmul%'
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result:
        print(f"  Matmul Process:   {result[0]} (PID: {result[1]})")

    # 3. THREAD STATS
    print(f"\n🧵 THREAD STATISTICS")
    print("-" * 90)
    cursor.execute("SELECT COUNT(DISTINCT globalTid) FROM ThreadNames")
    thread_count = cursor.fetchone()[0]
    print(f"  Total Threads:    {thread_count:>6}")

    # Show interesting threads (not system threads)
    cursor.execute("""
        SELECT DISTINCT nameId
        FROM ThreadNames
        LIMIT 50
    """)
    thread_names = set()
    for (name_id,) in cursor.fetchall():
        name = strings.get(name_id, "")
        if name and not any(kw in name for kw in ['kworker', 'ksoftirqd', 'migration', 'rcu_']):
            thread_names.add(name)

    if thread_names:
        print(f"\n  Application Threads:")
        for name in sorted(thread_names)[:10]:
            print(f"    - {name}")

    # 4. SYSTEM INFO
    print(f"\n💻 SYSTEM INFORMATION")
    print("-" * 90)
    try:
        cursor.execute("SELECT nameId, valueId FROM TARGET_INFO_SYSTEM_ENV")
        cpu_info = {}
        for name_id, value_id in cursor.fetchall():
            name = strings.get(name_id, "")
            value = strings.get(value_id, "")
            keywords = ['CPU', 'PROCESSOR', 'CORE', 'THREAD']
            if any(kw in name.upper() for kw in keywords):
                if 'NAME' in name or 'MODEL' in name or 'COUNT' in name:
                    cpu_info[name] = value

        if cpu_info:
            for key, value in sorted(cpu_info.items())[:5]:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"  System info not available: {e}")

    # 5. PROFILER OVERHEAD
    print(f"\n⚙️  PROFILER OVERHEAD")
    print("-" * 90)
    try:
        cursor.execute("SELECT * FROM PROFILER_OVERHEAD")
        overhead_data = cursor.fetchall()
        if overhead_data:
            print(f"  Overhead records: {len(overhead_data)}")
            # Show first record as example
            if len(overhead_data) > 0:
                print(f"  Sample: {overhead_data[0]}")
        else:
            print("  No overhead data recorded")
    except Exception as e:
        print(f"  Overhead data not available")

    # 6. GPU INFO
    print(f"\n🖥️  GPU INFORMATION")
    print("-" * 90)
    try:
        cursor.execute("SELECT COUNT(*) FROM TARGET_INFO_GPU")
        gpu_count = cursor.fetchone()[0]
        print(f"  GPUs detected:    {gpu_count:>6}")

        if gpu_count > 0:
            cursor.execute("SELECT * FROM TARGET_INFO_GPU LIMIT 1")
            cols = [desc[0] for desc in cursor.description]
            gpu_data = cursor.fetchone()
            if gpu_data:
                print(f"\n  GPU Details:")
                for i, col in enumerate(cols):
                    val = gpu_data[i]
                    if col.endswith('Id') and val in strings:
                        val = f"{val} -> {strings[val]}"
                    print(f"    {col}: {val}")
    except Exception as e:
        print(f"  GPU info not available")

    # 7. STRING ANALYSIS
    print(f"\n📝 STRING DATA ANALYSIS")
    print("-" * 90)
    print(f"  Total strings:    {len(strings):>6}")

    # Find relevant strings
    matmul_strings = []
    for sid, value in strings.items():
        if 'matmul' in value.lower():
            matmul_strings.append((sid, value))

    if matmul_strings:
        print(f"\n  Matmul-related strings:")
        for sid, value in matmul_strings[:10]:
            print(f"    [{sid:4d}] {value[:70]}")

    # 8. OS RUNTIME API
    print(f"\n🔌 OS RUNTIME APIs")
    print("-" * 90)
    try:
        cursor.execute("SELECT id, nameId FROM OSRT_API")
        apis = cursor.fetchall()
        print(f"  API count:        {len(apis):>6}")
        if apis:
            print(f"\n  Available APIs:")
            for api_id, name_id in apis:
                name = strings.get(name_id, f"Unknown_{name_id}")
                print(f"    [{api_id}] {name}")
    except Exception as e:
        print(f"  API data not available")

    # 9. DIAGNOSTIC EVENTS
    print(f"\n⚠️  DIAGNOSTIC EVENTS")
    print("-" * 90)
    cursor.execute("SELECT COUNT(*) FROM DIAGNOSTIC_EVENT")
    event_count = cursor.fetchone()[0]
    print(f"  Event count:      {event_count:>6}")

    if event_count > 0:
        cursor.execute("""
            SELECT timestamp, severity, text
            FROM DIAGNOSTIC_EVENT
            ORDER BY timestamp
            LIMIT 10
        """)
        print(f"\n  Recent events:")
        for ts, severity, text in cursor.fetchall():
            severity_label = ['Unknown', 'Info', 'Warning', 'Error', 'Verbose'][severity] if severity < 5 else 'Unknown'
            print(f"    [{severity_label:7s}] {text[:70]}")

    # 10. METADATA
    print(f"\n⚙️  CAPTURE METADATA")
    print("-" * 90)
    cursor.execute("SELECT COUNT(*) FROM META_DATA_CAPTURE")
    metadata_count = cursor.fetchone()[0]
    print(f"  Metadata entries: {metadata_count:>6}")

    if metadata_count > 0:
        cursor.execute("SELECT keyId, valueId FROM META_DATA_CAPTURE LIMIT 20")
        print(f"\n  Key settings:")
        for key_id, value_id in cursor.fetchall()[:10]:
            key = strings.get(key_id, f"Key_{key_id}")
            value = strings.get(value_id, f"Value_{value_id}")
            if any(kw in key for kw in ['sample', 'trace', 'enable', 'cpu', 'api']):
                print(f"    {key}: {value}")

    conn.close()

def compare_detailed(naive_db, reordered_db):
    """Detailed comparison with SQL queries"""

    print(f"\n{'='*90}")
    print(f"  DETAILED COMPARATIVE ANALYSIS")
    print(f"{'='*90}\n")

    conn_naive = sqlite3.connect(naive_db)
    conn_reordered = sqlite3.connect(reordered_db)

    cursor_naive = conn_naive.cursor()
    cursor_reordered = conn_reordered.cursor()

    # Compare key metrics
    cursor_naive.execute("SELECT duration FROM ANALYSIS_DETAILS")
    naive_duration = cursor_naive.fetchone()[0]

    cursor_reordered.execute("SELECT duration FROM ANALYSIS_DETAILS")
    reordered_duration = cursor_reordered.fetchone()[0]

    # Compare processes
    cursor_naive.execute("SELECT COUNT(*) FROM PROCESSES")
    naive_procs = cursor_naive.fetchone()[0]

    cursor_reordered.execute("SELECT COUNT(*) FROM PROCESSES")
    reordered_procs = cursor_reordered.fetchone()[0]

    # Compare threads
    cursor_naive.execute("SELECT COUNT(*) FROM ThreadNames")
    naive_threads = cursor_naive.fetchone()[0]

    cursor_reordered.execute("SELECT COUNT(*) FROM ThreadNames")
    reordered_threads = cursor_reordered.fetchone()[0]

    print("📊 METRIC COMPARISON")
    print("-" * 90)
    print(f"{'Metric':<30} {'Naive':>15} {'Reordered':>15} {'Difference':>15}")
    print("-" * 90)

    print(f"{'Duration (ns)':<30} {naive_duration:>15,} {reordered_duration:>15,} {naive_duration - reordered_duration:>15,}")
    print(f"{'Duration (sec)':<30} {naive_duration/1e9:>15.6f} {reordered_duration/1e9:>15.6f} {(naive_duration - reordered_duration)/1e9:>15.6f}")

    # Performance
    N = 512
    flops = 2 * N * N * N
    naive_gflops = (flops / (naive_duration / 1e9)) / 1e9
    reordered_gflops = (flops / (reordered_duration / 1e9)) / 1e9

    print(f"{'GFLOPS':<30} {naive_gflops:>15.2f} {reordered_gflops:>15.2f} {reordered_gflops - naive_gflops:>15.2f}")
    print(f"{'Speedup':<30} {'':<15} {naive_duration/reordered_duration:>15.2f}x {'':<15}")
    print(f"{'Time saved (%)':<30} {'':<15} {(1 - reordered_duration/naive_duration)*100:>14.1f}% {'':<15}")

    print(f"\n{'Processes':<30} {naive_procs:>15} {reordered_procs:>15} {reordered_procs - naive_procs:>15}")
    print(f"{'Threads':<30} {naive_threads:>15} {reordered_threads:>15} {reordered_threads - naive_threads:>15}")

    conn_naive.close()
    conn_reordered.close()

if __name__ == "__main__":
    naive_db = "matmul_naive_profile.sqlite"
    reordered_db = "matmul_reordered_profile.sqlite"

    # Query each database
    run_queries(naive_db, "NAIVE VERSION (ijk)")
    run_queries(reordered_db, "REORDERED VERSION (ikj)")

    # Detailed comparison
    compare_detailed(naive_db, reordered_db)

    print(f"\n{'='*90}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*90}\n")
