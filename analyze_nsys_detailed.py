#!/usr/bin/env python3
"""
Detailed analysis of NSYS profiling data focusing on key metrics
"""

import sqlite3
import sys

def query_profiling_details(db_path, version_name):
    """Extract key profiling metrics from NSYS database"""

    print(f"\n{'='*80}")
    print(f"Analyzing: {version_name}")
    print(f"Database: {db_path}")
    print(f"{'='*80}\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Get process information
    print("📦 PROCESS INFORMATION")
    print("-" * 80)
    cursor.execute("""
        SELECT
            globalPid,
            pid,
            processName,
            startTime,
            endTime,
            (endTime - startTime) / 1e9 as duration_sec
        FROM PROCESSES
    """)
    for row in cursor.fetchall():
        print(f"  Global PID: {row[0]}")
        print(f"  PID: {row[1]}")
        print(f"  Process: {row[2]}")
        print(f"  Start Time: {row[3]:,} ns")
        print(f"  End Time: {row[4]:,} ns")
        print(f"  Duration: {row[5]:.3f} seconds")

    # 2. Get string IDs to resolve names
    cursor.execute("SELECT id, value FROM StringIds")
    string_map = {row[0]: row[1] for row in cursor.fetchall()}

    # 3. Check what tables contain data
    print(f"\n📊 DATA AVAILABILITY")
    print("-" * 80)

    tables_to_check = [
        'OSRT_API_TRACE',
        'OSRT_API',
        'ThreadNames',
        'TARGET_INFO_SYSTEM_ENV',
        'PROFILER_OVERHEAD',
        'ANALYSIS_DETAILS'
    ]

    for table in tables_to_check:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table:<30}: {count:>10,} rows")
        except:
            print(f"  {table:<30}: Not available")

    # 4. Get thread names
    print(f"\n🧵 THREADS")
    print("-" * 80)
    try:
        cursor.execute("""
            SELECT
                globalTid,
                nameId,
                copyNameId
            FROM ThreadNames
            LIMIT 10
        """)
        for row in cursor.fetchall():
            name = string_map.get(row[1], f"Unknown ({row[1]})")
            print(f"  Thread {row[0]}: {name}")
    except Exception as e:
        print(f"  Thread data not available: {e}")

    # 5. Get OS Runtime API information
    print(f"\n🔍 OS RUNTIME API CALLS")
    print("-" * 80)

    # First get the API names
    try:
        cursor.execute("SELECT id, nameId FROM OSRT_API LIMIT 20")
        api_map = {}
        for api_id, name_id in cursor.fetchall():
            api_name = string_map.get(name_id, f"API_{name_id}")
            api_map[api_id] = api_name

        # Then get trace data - note there's no OSRT_API_TRACE table, we need to find the right one
        # Let's check what exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE '%TRACE%' OR name LIKE '%API%'
            ORDER BY name
        """)
        trace_tables = cursor.fetchall()
        print(f"\n  Available trace tables:")
        for t in trace_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {t[0]}")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"    - {t[0]}: {count:,} rows")

    except Exception as e:
        print(f"  API trace data error: {e}")

    # 6. System environment
    print(f"\n💻 SYSTEM ENVIRONMENT")
    print("-" * 80)
    try:
        cursor.execute("SELECT nameId, valueId FROM TARGET_INFO_SYSTEM_ENV")
        for name_id, value_id in cursor.fetchall():
            name = string_map.get(name_id, f"Unknown_{name_id}")
            value = string_map.get(value_id, f"Unknown_{value_id}")
            if 'CPU' in name or 'CORE' in name or 'MEM' in name or 'CACHE' in name:
                print(f"  {name}: {value}")
    except Exception as e:
        print(f"  System env not available: {e}")

    # 7. Profiler overhead
    print(f"\n⏱️  PROFILER OVERHEAD")
    print("-" * 80)
    try:
        cursor.execute("SELECT * FROM PROFILER_OVERHEAD")
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                print(f"  {row}")
        else:
            print("  No overhead data")
    except Exception as e:
        print(f"  Overhead data not available: {e}")

    # 8. Analysis details
    print(f"\n📈 ANALYSIS SUMMARY")
    print("-" * 80)
    try:
        cursor.execute("""
            SELECT
                duration / 1e9 as duration_sec,
                startTime,
                stopTime
            FROM ANALYSIS_DETAILS
        """)
        for row in cursor.fetchall():
            print(f"  Total Duration: {row[0]:.6f} seconds")
            print(f"  Start: {row[1]:,} ns")
            print(f"  Stop: {row[2]:,} ns")
    except Exception as e:
        print(f"  Analysis details error: {e}")

    conn.close()

def compare_databases(naive_db, reordered_db):
    """Compare timing between naive and reordered implementations"""

    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")

    def get_duration(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT duration FROM ANALYSIS_DETAILS")
            duration_ns = cursor.fetchone()[0]
            conn.close()
            return duration_ns / 1e9
        except:
            conn.close()
            return None

    naive_duration = get_duration(naive_db)
    reordered_duration = get_duration(reordered_db)

    if naive_duration and reordered_duration:
        print(f"⏱️  EXECUTION TIME COMPARISON")
        print("-" * 80)
        print(f"  Naive version:     {naive_duration:>10.6f} seconds")
        print(f"  Reordered version: {reordered_duration:>10.6f} seconds")
        print(f"  Speedup:           {naive_duration / reordered_duration:>10.2f}x")
        print(f"  Time saved:        {(naive_duration - reordered_duration):>10.6f} seconds ({(1 - reordered_duration/naive_duration)*100:.1f}%)")

def explore_interesting_tables(db_path):
    """Deep dive into interesting tables"""

    print(f"\n{'='*80}")
    print("DEEP DIVE: INTERESTING TABLES")
    print(f"{'='*80}\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find tables with actual data
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """)

    tables_with_data = []
    for (table_name,) in cursor.fetchall():
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        if count > 0 and not table_name.startswith('ENUM_'):
            tables_with_data.append((table_name, count))

    print("📊 Tables with data (non-enum):\n")
    for table_name, count in sorted(tables_with_data, key=lambda x: x[1], reverse=True):
        print(f"  {table_name:<40} {count:>10,} rows")

        # For small tables, show sample
        if count <= 5 and count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            rows = cursor.fetchall()
            if rows:
                print(f"    Sample: {rows[0]}")

    conn.close()

if __name__ == "__main__":
    naive_db = "matmul_naive_profile.sqlite"
    reordered_db = "matmul_reordered_profile.sqlite"

    # Detailed analysis of each
    query_profiling_details(naive_db, "NAIVE (ijk loop order)")
    query_profiling_details(reordered_db, "REORDERED (ikj loop order)")

    # Compare
    compare_databases(naive_db, reordered_db)

    # Deep dive
    print(f"\n\nDeep dive into naive database...")
    explore_interesting_tables(naive_db)
