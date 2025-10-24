# NSYS SQLite Database Quick Reference

Quick cheat sheet for querying NSYS SQLite profiling databases.

## 📊 Most Useful Queries

### 1. Get Total Execution Time
```python
import sqlite3
conn = sqlite3.connect('matmul_profile.sqlite')
cursor = conn.cursor()

cursor.execute("SELECT duration / 1e9 as sec FROM ANALYSIS_DETAILS")
print(f"Duration: {cursor.fetchone()[0]:.6f} seconds")
```

### 2. Calculate GFLOPS
```python
N = 512
flops = 2 * N * N * N  # 268,435,456 for N=512

cursor.execute("SELECT duration FROM ANALYSIS_DETAILS")
duration_ns = cursor.fetchone()[0]
gflops = (flops / (duration_ns / 1e9)) / 1e9

print(f"Performance: {gflops:.2f} GFLOPS")
```

### 3. Find Your Process
```python
cursor.execute("""
    SELECT name, pid FROM PROCESSES
    WHERE name LIKE '%your_program%'
    LIMIT 1
""")
print(cursor.fetchone())
```

### 4. Get String Mappings
```python
cursor.execute("SELECT id, value FROM StringIds")
strings = {row[0]: row[1] for row in cursor.fetchall()}

# Now resolve any nameId
name_id = 123
name = strings.get(name_id, "Unknown")
```

### 5. Count Threads
```python
cursor.execute("SELECT COUNT(DISTINCT globalTid) FROM ThreadNames")
print(f"Total threads: {cursor.fetchone()[0]}")
```

### 6. GPU Information
```python
cursor.execute("""
    SELECT name, totalMemory, smCount, clockRate
    FROM TARGET_INFO_GPU
    LIMIT 1
""")
name, mem, sms, clock = cursor.fetchone()
print(f"GPU: {name}, Memory: {mem/(1024**3):.1f}GB, SMs: {sms}")
```

### 7. Compare Two Runs
```python
def compare(db1, db2):
    conn1 = sqlite3.connect(db1)
    conn2 = sqlite3.connect(db2)

    c1 = conn1.cursor()
    c2 = conn2.cursor()

    c1.execute("SELECT duration FROM ANALYSIS_DETAILS")
    c2.execute("SELECT duration FROM ANALYSIS_DETAILS")

    dur1 = c1.fetchone()[0]
    dur2 = c2.fetchone()[0]

    speedup = dur1 / dur2
    print(f"Speedup: {speedup:.2f}x")

    conn1.close()
    conn2.close()

compare('naive.sqlite', 'optimized.sqlite')
```

## 🗂️ Key Tables Reference

| Table | Purpose | Typical Rows |
|-------|---------|--------------|
| `ANALYSIS_DETAILS` | Overall timing | 1 |
| `PROCESSES` | Process info | 1000-2000 |
| `ThreadNames` | Thread tracking | 3000-4000 |
| `StringIds` | String lookup | 1500-2000 |
| `TARGET_INFO_GPU` | GPU hardware | 1-8 |
| `TARGET_INFO_SYSTEM_ENV` | Environment | 50-100 |
| `DIAGNOSTIC_EVENT` | Warnings/errors | 5-15 |
| `PROFILER_OVERHEAD` | Overhead stats | 5-10 |
| `META_DATA_CAPTURE` | Config | 100-150 |

## 🔍 Common Fields

### ANALYSIS_DETAILS
- `duration` (INTEGER) - Total duration in nanoseconds
- `startTime` (INTEGER) - Start timestamp (ns)
- `stopTime` (INTEGER) - Stop timestamp (ns)

### PROCESSES
- `globalPid` (INTEGER) - Global process ID
- `pid` (INTEGER) - System PID
- `name` (TEXT) - Process name

### ThreadNames
- `globalTid` (INTEGER) - Global thread ID
- `nameId` (INTEGER) - Name reference → StringIds

### StringIds
- `id` (INTEGER) - Unique ID
- `value` (TEXT) - Actual string

## 🐍 Python Template

```python
#!/usr/bin/env python3
import sqlite3

def analyze_nsys_db(db_path):
    """Analyze NSYS SQLite database"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get string mappings
    cursor.execute("SELECT id, value FROM StringIds")
    strings = {row[0]: row[1] for row in cursor.fetchall()}

    # Get timing
    cursor.execute("SELECT duration FROM ANALYSIS_DETAILS")
    duration_ns = cursor.fetchone()[0]
    duration_sec = duration_ns / 1e9

    print(f"Duration: {duration_sec:.6f} sec")

    # Calculate performance
    N = 512  # Your matrix size
    flops = 2 * N * N * N
    gflops = (flops / duration_sec) / 1e9
    print(f"GFLOPS: {gflops:.2f}")

    # Find your process
    cursor.execute("""
        SELECT name, pid FROM PROCESSES
        WHERE name LIKE '%your_program%'
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result:
        print(f"Process: {result[0]} (PID: {result[1]})")

    # GPU info
    cursor.execute("SELECT name FROM TARGET_INFO_GPU LIMIT 1")
    result = cursor.fetchone()
    if result:
        print(f"GPU: {result[0]}")

    conn.close()

if __name__ == "__main__":
    analyze_nsys_db("your_profile.sqlite")
```

## 📝 SQL Query Template

```sql
-- timing.sql
.mode column
.headers on

-- Overall timing
SELECT
    'Duration (sec)' as Metric,
    CAST(duration / 1e9 AS REAL) as Value
FROM ANALYSIS_DETAILS;

-- GFLOPS calculation (for N=512 matmul)
SELECT
    'GFLOPS' as Metric,
    CAST((268435456.0 / (duration / 1e9)) / 1e9 AS REAL) as Value
FROM ANALYSIS_DETAILS;

-- Process info
SELECT
    name as Process,
    pid as PID
FROM PROCESSES
WHERE name LIKE '%matmul%'
LIMIT 1;

-- Thread count
SELECT
    'Thread Count' as Metric,
    COUNT(DISTINCT globalTid) as Value
FROM ThreadNames;
```

Run with: `sqlite3 your_profile.sqlite < timing.sql`

## 🔧 Advanced Queries

### Join with String Resolution
```sql
SELECT
    TN.globalTid as ThreadID,
    S.value as ThreadName
FROM ThreadNames TN
JOIN StringIds S ON TN.nameId = S.id
WHERE S.value NOT LIKE 'kworker%'
LIMIT 10;
```

### Environment Variables
```sql
SELECT
    SN.value as Variable,
    SV.value as Value
FROM TARGET_INFO_SYSTEM_ENV TE
JOIN StringIds SN ON TE.nameId = SN.id
JOIN StringIds SV ON TE.valueId = SV.id
WHERE SN.value LIKE '%CPU%'
   OR SN.value LIKE '%THREAD%';
```

### Diagnostic Messages
```sql
SELECT
    CASE severity
        WHEN 1 THEN 'Info'
        WHEN 2 THEN 'Warning'
        WHEN 3 THEN 'Error'
        ELSE 'Unknown'
    END as Level,
    text as Message
FROM DIAGNOSTIC_EVENT
ORDER BY timestamp;
```

## 🎯 Common Tasks

### Task: Compare Before/After Optimization
```python
import sqlite3

def compare_optimization(before_db, after_db):
    def get_time(db):
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("SELECT duration FROM ANALYSIS_DETAILS")
        result = cursor.fetchone()[0]
        conn.close()
        return result

    before = get_time(before_db) / 1e9
    after = get_time(after_db) / 1e9

    speedup = before / after
    saved_pct = (1 - after/before) * 100

    print(f"Before:     {before:.6f} sec")
    print(f"After:      {after:.6f} sec")
    print(f"Speedup:    {speedup:.2f}×")
    print(f"Time saved: {saved_pct:.1f}%")

compare_optimization('naive.sqlite', 'optimized.sqlite')
```

### Task: Extract System Info
```python
def get_system_info(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get strings
    cursor.execute("SELECT id, value FROM StringIds")
    strings = dict(cursor.fetchall())

    # Get environment
    cursor.execute("SELECT nameId, valueId FROM TARGET_INFO_SYSTEM_ENV")
    info = {}
    for name_id, value_id in cursor.fetchall():
        name = strings.get(name_id, "")
        value = strings.get(value_id, "")
        if 'CPU' in name or 'CORE' in name:
            info[name] = value

    conn.close()
    return info

info = get_system_info('profile.sqlite')
for k, v in info.items():
    print(f"{k}: {v}")
```

## 📚 Remember

1. **Always close connections**: `conn.close()`
2. **Resolve IDs via StringIds**: Most names are referenced by ID
3. **Timing is in nanoseconds**: Divide by 1e9 for seconds
4. **GFLOPS = operations / time / 1e9**
5. **Profiling adds overhead**: Compare ratios, not absolute times

## 🚀 Quick Start

```bash
# Profile your app
nsys profile --stats=true --export=sqlite --output=my_profile ./my_app

# Query it
python3 << 'EOF'
import sqlite3
conn = sqlite3.connect('my_profile.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT duration / 1e9 FROM ANALYSIS_DETAILS")
print(f"Time: {cursor.fetchone()[0]:.6f} sec")
conn.close()
EOF
```

---

*Part of NSYS Profiling Guide*
