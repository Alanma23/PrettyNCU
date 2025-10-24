#!/usr/bin/env python3
"""
Explore and analyze NSYS SQLite profiling databases
"""

import sqlite3
import sys

def explore_database(db_path):
    """Explore the structure and contents of an NSYS SQLite database"""

    print(f"\n{'='*80}")
    print(f"Exploring: {db_path}")
    print(f"{'='*80}\n")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = cursor.fetchall()

    print(f"📊 Found {len(tables)} tables:\n")
    for i, (table_name,) in enumerate(tables, 1):
        print(f"  {i:2d}. {table_name}")

    print("\n" + "="*80)
    print("TABLE SCHEMAS AND SAMPLE DATA")
    print("="*80)

    # For each table, show schema and sample data
    for (table_name,) in tables:
        print(f"\n{'─'*80}")
        print(f"📋 Table: {table_name}")
        print(f"{'─'*80}")

        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        print("\n🔧 Schema:")
        print(f"  {'Column':<30} {'Type':<15} {'NotNull':<8} {'Default':<15} {'PK'}")
        print(f"  {'-'*30} {'-'*15} {'-'*8} {'-'*15} {'-'*3}")
        for col_id, name, col_type, notnull, default_val, pk in columns:
            default_str = str(default_val) if default_val else ""
            print(f"  {name:<30} {col_type:<15} {notnull:<8} {default_str:<15} {pk}")

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        print(f"\n📈 Row count: {row_count:,}")

        # Show sample data (first 5 rows)
        if row_count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            sample_rows = cursor.fetchall()

            if sample_rows:
                print("\n📝 Sample data (first 5 rows):")
                # Get column names
                col_names = [desc[0] for desc in cursor.description]

                # Print header
                print("\n  " + " | ".join(f"{name[:20]:<20}" for name in col_names))
                print("  " + "-" * (len(col_names) * 23))

                # Print data
                for row in sample_rows:
                    formatted_row = []
                    for val in row:
                        if isinstance(val, float):
                            formatted_row.append(f"{val:<20.6f}")
                        elif isinstance(val, int):
                            formatted_row.append(f"{val:<20}")
                        elif val is None:
                            formatted_row.append(f"{'NULL':<20}")
                        else:
                            formatted_row.append(f"{str(val)[:20]:<20}")
                    print("  " + " | ".join(formatted_row))

    conn.close()

def analyze_profiling_data(naive_db, reordered_db):
    """Analyze and compare profiling data from both databases"""

    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80 + "\n")

    def get_timing_stats(db_path, version_name):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if we have the right tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='OSRT_API_TRACE';")
        if cursor.fetchone():
            cursor.execute("""
                SELECT
                    nameId,
                    COUNT(*) as call_count,
                    SUM(end - start) as total_duration_ns,
                    AVG(end - start) as avg_duration_ns,
                    MIN(end - start) as min_duration_ns,
                    MAX(end - start) as max_duration_ns
                FROM OSRT_API_TRACE
                GROUP BY nameId
                ORDER BY total_duration_ns DESC
                LIMIT 10;
            """)
            results = cursor.fetchall()
            conn.close()
            return results

        conn.close()
        return None

    # Get timing stats for both versions
    print("🔍 Analyzing timing data...\n")

    naive_stats = get_timing_stats(naive_db, "Naive")
    reordered_stats = get_timing_stats(reordered_db, "Reordered")

    if naive_stats:
        print("📊 Top API calls by duration (Naive):")
        print(f"  {'NameID':<10} {'Calls':<10} {'Total (ms)':<15} {'Avg (μs)':<15} {'Min (μs)':<15} {'Max (μs)':<15}")
        print("  " + "-"*80)
        for name_id, calls, total_ns, avg_ns, min_ns, max_ns in naive_stats[:10]:
            print(f"  {name_id:<10} {calls:<10} {total_ns/1e6:<15.3f} {avg_ns/1e3:<15.3f} {min_ns/1e3:<15.3f} {max_ns/1e3:<15.3f}")

    if reordered_stats:
        print("\n📊 Top API calls by duration (Reordered):")
        print(f"  {'NameID':<10} {'Calls':<10} {'Total (ms)':<15} {'Avg (μs)':<15} {'Min (μs)':<15} {'Max (μs)':<15}")
        print("  " + "-"*80)
        for name_id, calls, total_ns, avg_ns, min_ns, max_ns in reordered_stats[:10]:
            print(f"  {name_id:<10} {calls:<10} {total_ns/1e6:<15.3f} {avg_ns/1e3:<15.3f} {min_ns/1e3:<15.3f} {max_ns/1e3:<15.3f}")

if __name__ == "__main__":
    naive_db = "matmul_naive_profile.sqlite"
    reordered_db = "matmul_reordered_profile.sqlite"

    # Explore both databases
    explore_database(naive_db)
    explore_database(reordered_db)

    # Comparative analysis
    analyze_profiling_data(naive_db, reordered_db)
