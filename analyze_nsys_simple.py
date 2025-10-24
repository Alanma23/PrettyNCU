#!/usr/bin/env python3
"""
Simple focused analysis of NSYS profiling data
"""

import sqlite3

def analyze_db(db_path, version_name):
    """Analyze NSYS database"""

    print(f"\n{'='*80}")
    print(f"{version_name}")
    print(f"{'='*80}\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get string mappings
    cursor.execute("SELECT id, value FROM StringIds")
    strings = {row[0]: row[1] for row in cursor.fetchall()}

    # 1. Check PROCESSES table structure
    print("📦 PROCESS INFO")
    print("-" * 80)
    cursor.execute("PRAGMA table_info(PROCESSES)")
    process_cols = cursor.fetchall()
    col_names = [col[1] for col in process_cols]
    print(f"  Available columns: {', '.join(col_names)}\n")

    # Query with actual column names
    cursor.execute("SELECT * FROM PROCESSES LIMIT 1")
    process_data = cursor.fetchone()
    if process_data:
        for i, col in enumerate(process_cols):
            col_name = col[1]
            value = process_data[i]
            if col_name in ['nameId']:
                resolved = strings.get(value, f"ID_{value}")
                print(f"  {col_name}: {value} -> {resolved}")
            elif isinstance(value, int) and value > 1e9:
                print(f"  {col_name}: {value:,} ns")
            else:
                print(f"  {col_name}: {value}")

    # 2. Analysis details - this has the total duration
    print(f"\n⏱️  TIMING")
    print("-" * 80)
    cursor.execute("SELECT * FROM ANALYSIS_DETAILS")
    analysis = cursor.fetchone()
    if analysis:
        duration_ns = analysis[1]
        start_time = analysis[2]
        stop_time = analysis[3]
        duration_sec = duration_ns / 1e9
        print(f"  Duration: {duration_sec:.6f} seconds ({duration_ns:,} ns)")
        print(f"  Start:    {start_time:,} ns")
        print(f"  Stop:     {stop_time:,} ns")

    # 3. System info
    print(f"\n💻 SYSTEM INFO")
    print("-" * 80)
    try:
        cursor.execute("SELECT nameId, valueId FROM TARGET_INFO_SYSTEM_ENV")
        for name_id, value_id in cursor.fetchall():
            name = strings.get(name_id, f"ID_{name_id}")
            value = strings.get(value_id, f"ID_{value_id}")
            # Show only interesting fields
            keywords = ['CPU', 'CORE', 'PROCESSOR', 'THREAD', 'MEMORY', 'CACHE']
            if any(kw in name.upper() for kw in keywords):
                print(f"  {name}: {value}")
    except Exception as e:
        print(f"  Not available: {e}")

    # 4. Thread info
    print(f"\n🧵 THREADS")
    print("-" * 80)
    cursor.execute("SELECT globalTid, nameId FROM ThreadNames LIMIT 10")
    for gtid, name_id in cursor.fetchall():
        name = strings.get(name_id, f"ID_{name_id}")
        print(f"  Thread {gtid}: {name}")

    # 5. Check what profiling data we have
    print(f"\n📊 AVAILABLE DATA TABLES")
    print("-" * 80)

    # Get all non-empty tables
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        AND name NOT LIKE 'ENUM_%'
        ORDER BY name
    """)

    data_tables = []
    for (table_name,) in cursor.fetchall():
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        if count > 0:
            data_tables.append((table_name, count))

    for table_name, count in sorted(data_tables, key=lambda x: x[1], reverse=True):
        print(f"  {table_name:<40} {count:>10,} rows")

    conn.close()

def compare(naive_db, reordered_db):
    """Compare both databases"""

    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}\n")

    def get_stats(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT duration, startTime, stopTime FROM ANALYSIS_DETAILS")
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                'duration_ns': row[0],
                'duration_sec': row[0] / 1e9,
                'start': row[1],
                'stop': row[2]
            }
        return None

    naive_stats = get_stats(naive_db)
    reordered_stats = get_stats(reordered_db)

    if naive_stats and reordered_stats:
        print("⏱️  EXECUTION TIME")
        print("-" * 80)
        print(f"  Naive (ijk):      {naive_stats['duration_sec']:>10.6f} sec  ({naive_stats['duration_ns']:>15,} ns)")
        print(f"  Reordered (ikj):  {reordered_stats['duration_sec']:>10.6f} sec  ({reordered_stats['duration_ns']:>15,} ns)")
        print()
        speedup = naive_stats['duration_sec'] / reordered_stats['duration_sec']
        time_saved = naive_stats['duration_sec'] - reordered_stats['duration_sec']
        pct_saved = (1 - reordered_stats['duration_sec'] / naive_stats['duration_sec']) * 100

        print(f"  🚀 Speedup:        {speedup:>10.2f}x")
        print(f"  💾 Time saved:     {time_saved:>10.6f} sec ({pct_saved:.1f}%)")

        # Calculate GFLOPS
        N = 512
        flops = 2 * N * N * N  # matrix multiply operations

        naive_gflops = (flops / naive_stats['duration_sec']) / 1e9
        reordered_gflops = (flops / reordered_stats['duration_sec']) / 1e9

        print(f"\n  💪 Performance:")
        print(f"     Naive:         {naive_gflops:>10.2f} GFLOPS")
        print(f"     Reordered:     {reordered_gflops:>10.2f} GFLOPS")

def deep_dive_stringids(db_path):
    """Look at what's in the string IDs"""

    print(f"\n{'='*80}")
    print("STRING IDS ANALYSIS")
    print(f"{'='*80}\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, value FROM StringIds ORDER BY id")
    strings = cursor.fetchall()

    print(f"📝 Found {len(strings)} strings\n")

    # Group by patterns
    categories = {
        'system': [],
        'functions': [],
        'libs': [],
        'paths': [],
        'other': []
    }

    for sid, value in strings:
        if value.startswith('/'):
            categories['paths'].append((sid, value))
        elif '.so' in value or '.a' in value:
            categories['libs'].append((sid, value))
        elif '(' in value or 'matmul' in value.lower():
            categories['functions'].append((sid, value))
        elif value.upper() == value or '_' in value:
            categories['system'].append((sid, value))
        else:
            categories['other'].append((sid, value))

    for cat, items in categories.items():
        if items:
            print(f"\n{cat.upper()} ({len(items)}):")
            for sid, value in items[:20]:  # Show first 20
                print(f"  [{sid:3d}] {value}")
            if len(items) > 20:
                print(f"  ... and {len(items) - 20} more")

    conn.close()

if __name__ == "__main__":
    naive_db = "matmul_naive_profile.sqlite"
    reordered_db = "matmul_reordered_profile.sqlite"

    # Analyze each
    analyze_db(naive_db, "NAIVE VERSION (ijk)")
    analyze_db(reordered_db, "REORDERED VERSION (ikj)")

    # Compare
    compare(naive_db, reordered_db)

    # Look at strings
    print(f"\n\nExamining string mappings in naive database...")
    deep_dive_stringids(naive_db)
