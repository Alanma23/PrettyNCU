-- Comprehensive NSYS SQLite Profiling Queries
-- This file contains SQL queries to extract detailed profiling information

-- ============================================================================
-- 1. TIMING ANALYSIS
-- ============================================================================

.mode column
.headers on
.width 30 15 20

SELECT '=== TIMING SUMMARY ===' as Section;
SELECT
    'Duration (sec)' as Metric,
    CAST(duration / 1e9 AS REAL) as Value,
    'Total profile duration' as Description
FROM ANALYSIS_DETAILS;

SELECT
    'Start Time (ns)' as Metric,
    startTime as Value,
    'Profile start timestamp' as Description
FROM ANALYSIS_DETAILS;

SELECT
    'Stop Time (ns)' as Metric,
    stopTime as Value,
    'Profile stop timestamp' as Description
FROM ANALYSIS_DETAILS;

-- ============================================================================
-- 2. PROCESS INFORMATION
-- ============================================================================

SELECT '';
SELECT '=== PROCESS INFORMATION ===' as Section;

SELECT
    COUNT(DISTINCT globalPid) as 'Total Processes',
    COUNT(DISTINCT pid) as 'Unique PIDs'
FROM PROCESSES;

SELECT
    'Process Name' as Field,
    name as Value
FROM PROCESSES
WHERE globalPid = (SELECT MIN(globalPid) FROM PROCESSES WHERE name != 'kworker/16:1-events')
LIMIT 1;

-- ============================================================================
-- 3. THREAD ANALYSIS
-- ============================================================================

SELECT '';
SELECT '=== THREAD STATISTICS ===' as Section;

SELECT
    COUNT(DISTINCT globalTid) as 'Total Threads'
FROM ThreadNames;

-- Top threads by name (resolved via StringIds)
SELECT '';
SELECT 'Top Thread Names:' as Info;
SELECT
    TN.globalTid as 'Thread ID',
    SI.value as 'Thread Name'
FROM ThreadNames TN
JOIN StringIds SI ON TN.nameId = SI.id
WHERE SI.value NOT LIKE 'kworker%'
AND SI.value NOT LIKE 'migration%'
AND SI.value NOT LIKE 'ksoftirqd%'
LIMIT 10;

-- ============================================================================
-- 4. SYSTEM ENVIRONMENT
-- ============================================================================

SELECT '';
SELECT '=== SYSTEM ENVIRONMENT ===' as Section;

SELECT
    SN.value as 'Environment Variable',
    SV.value as 'Value'
FROM TARGET_INFO_SYSTEM_ENV TE
JOIN StringIds SN ON TE.nameId = SN.id
JOIN StringIds SV ON TE.valueId = SV.id
WHERE SN.value LIKE '%CPU%'
   OR SN.value LIKE '%THREAD%'
   OR SN.value LIKE '%CORE%'
   OR SN.value LIKE '%PROCESSOR%'
   OR SN.value LIKE '%CACHE%'
   OR SN.value LIKE '%MEMORY%'
ORDER BY SN.value;

-- ============================================================================
-- 5. PROFILER OVERHEAD
-- ============================================================================

SELECT '';
SELECT '=== PROFILER OVERHEAD ===' as Section;

SELECT * FROM PROFILER_OVERHEAD;

-- ============================================================================
-- 6. GPU INFORMATION (if available)
-- ============================================================================

SELECT '';
SELECT '=== GPU INFORMATION ===' as Section;

SELECT
    COUNT(*) as 'GPU Count'
FROM TARGET_INFO_GPU;

SELECT * FROM TARGET_INFO_GPU LIMIT 5;

-- ============================================================================
-- 7. STRING IDS ANALYSIS
-- ============================================================================

SELECT '';
SELECT '=== STRING IDS SUMMARY ===' as Section;

SELECT
    COUNT(*) as 'Total Strings'
FROM StringIds;

-- Find matmul-related strings
SELECT '';
SELECT 'Matmul-related strings:' as Info;
SELECT
    id,
    SUBSTR(value, 1, 80) as 'String Value'
FROM StringIds
WHERE value LIKE '%matmul%'
   OR value LIKE '%malloc%'
   OR value LIKE '%free%'
ORDER BY id;

-- ============================================================================
-- 8. OS RUNTIME API CALLS
-- ============================================================================

SELECT '';
SELECT '=== OS RUNTIME API ===' as Section;

SELECT
    O.id as 'API ID',
    S.value as 'API Name'
FROM OSRT_API O
JOIN StringIds S ON O.nameId = S.id;

-- ============================================================================
-- 9. METADATA
-- ============================================================================

SELECT '';
SELECT '=== CAPTURE METADATA ===' as Section;

SELECT
    COUNT(*) as 'Metadata Entries'
FROM META_DATA_CAPTURE;

-- Show capture settings
SELECT
    KS.value as 'Setting',
    VS.value as 'Value'
FROM META_DATA_CAPTURE MC
JOIN StringIds KS ON MC.keyId = KS.id
JOIN StringIds VS ON MC.valueId = VS.id
WHERE KS.value LIKE '%sample%'
   OR KS.value LIKE '%trace%'
   OR KS.value LIKE '%enable%'
ORDER BY KS.value
LIMIT 20;

-- ============================================================================
-- 10. DIAGNOSTIC EVENTS
-- ============================================================================

SELECT '';
SELECT '=== DIAGNOSTIC EVENTS ===' as Section;

SELECT
    timestamp,
    timestampType,
    source,
    severity,
    SUBSTR(text, 1, 100) as 'Message'
FROM DIAGNOSTIC_EVENT
ORDER BY timestamp
LIMIT 15;
