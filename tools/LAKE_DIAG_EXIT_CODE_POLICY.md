# Lake Diagnostic Regression — Exit Code Policy

This document defines the **exit code contract** for the lake-adjacency diagnostic regression workflow.

Target scripts:
- `tools/run_lake_diag_regression.ps1`
- `tools/analyze_diagnostic_lake_adjacency.py compare`

The goal is to enable:
- CI-friendly PASS/FAIL judgment
- Repeatable local verification
- Machine-readable failure classification

---

## Definitions

**Baseline**
- A previously recorded diagnostic snapshot used as the reference.
- Default: `artifacts/diag_baseline.log`

**Snapshot**
- A newly generated diagnostic snapshot produced by a fresh `--diagnostic` run.
- Stored as `artifacts/diag_YYYYMMDD_HHMMSS.log`

**Delta**
- A numeric difference between baseline and snapshot metrics (e.g., `adjacent`, `edges`, etc.).

**Strict pass**
- No metric deltas AND no anomaly delta AND baseline freshness is within threshold.

---

## Exit Codes

### 0 — STRICT_PASS
Meaning:
- Regression fully passed.
Conditions:
- No metric deltas vs baseline (`diff == 0` for all tracked metrics)
- `anomalies_diff == 0`
- Baseline freshness check passed (if enabled)
Typical output includes:
- `[STRICT PASS] No deltas detected vs baseline.`

---

### 10 — FAIL_METRIC_DELTA
Meaning:
- One or more tracked metric values differ vs baseline.
Examples:
- `adjacent.diff != 0`
- `edges.diff != 0`
- any other numeric metric delta detected

Expected behavior:
- Print a compare summary indicating which metric(s) changed
- Exit non-zero to fail CI / automation

---

### 11 — FAIL_ANOMALY_DELTA
Meaning:
- Anomaly count differs vs baseline (`anomalies_diff != 0`)
Notes:
- This includes cases where baseline has 0 anomalies but snapshot has anomalies, or vice versa.

---

### 12 — FAIL_BASELINE_STALE
Meaning:
- Baseline file is older than the allowed freshness threshold.
Typical checks:
- baseline mtime exceeds `threshold_days`
Expected output includes:
- `[BASELINE] ... age_days=... threshold_days=...`
and a clear FAIL line.

---

### 20 — FAIL_INPUT_MISSING
Meaning:
- Required input files are missing.
Examples:
- baseline log not found
- builder script not found
- diagnostic output log not created

---

### 21 — FAIL_PARSE_ERROR
Meaning:
- Log parsing failed in a way that prevents trustworthy comparison.
Examples:
- compare command crashes
- malformed/empty log where metrics cannot be extracted

---

### 22 — FAIL_TOOL_ERROR
Meaning:
- The tool executed but failed unexpectedly.
Examples:
- uncaught exception
- subprocess failures
- permission errors

---

### 99 — FAIL_UNKNOWN
Meaning:
- Any unclassified failure.
Policy:
- Prefer mapping to a specific code above.
- Use 99 only as a last resort.

---

## Recommended Automation Contract

For scripts intended to be used by CI:

- Exit code MUST reflect PASS/FAIL.
- Human-readable summary MUST be printed to stdout.
- The contract MUST be stable over time.

Minimum required output on completion:
- PASS case: `[STRICT PASS] ...`
- FAIL case: `[STRICT FAIL] <reason> exit_code=<code>`

---

## Notes for Implementation

This document defines policy only.
It does not require any changes to:
- Master CSV output
- Any map image processing logic
- Any non-diagnostic execution behavior

The exit code contract applies only to diagnostic tooling.

