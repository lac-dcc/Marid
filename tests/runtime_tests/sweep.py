#!/usr/bin/env python3

import subprocess
import re
import csv
import sys

# ---------------------------------------------------------------------------
# Usage:
#   python3 sweep.py [X] [output.csv]
#
# Arguments:
#   X           Number of benchmark runs per N (default: 5)
#   output.csv  Output CSV file (default: results.csv)
# ---------------------------------------------------------------------------

X          = int(sys.argv[1])  if len(sys.argv) > 1 else 5
OUTPUT_CSV = sys.argv[2]       if len(sys.argv) > 2 else "results.csv"
BENCH_SCRIPT = "log_perf_counters.py"

N_VALUES = [
    1000, 5000, 10000, 20000, 30000, 40000, 50000,
    60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 150000
]

# --- Parse average times from benchmark output ------------------------------
def parse_avg_times(output):
    dcache_avg = None
    icache_avg = None
    for line in output.splitlines():
        m = re.search(r"dcache average time:\s+([\d.]+)", line)
        if m:
            dcache_avg = float(m.group(1))
        m = re.search(r"icache average time:\s+([\d.]+)", line)
        if m:
            icache_avg = float(m.group(1))
    return dcache_avg, icache_avg

# --- Sweep ------------------------------------------------------------------
rows = []

for N in N_VALUES:
    print(f"Running benchmark for N={N}...", flush=True)
    result = subprocess.run(
        ["python3", BENCH_SCRIPT, str(N), str(X)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  ERROR for N={N}:")
        print(result.stderr)
        rows.append({"N": N, "dcache_time": "ERROR", "icache_time": "ERROR"})
        continue

    dcache_avg, icache_avg = parse_avg_times(result.stdout)

    if dcache_avg is None or icache_avg is None:
        print(f"  WARNING: could not parse output for N={N}")
        print(result.stdout)
        rows.append({"N": N, "dcache_time": "PARSE_ERROR", "icache_time": "PARSE_ERROR"})
        continue

    print(f"  dcache_avg={dcache_avg:.6f} s  icache_avg={icache_avg:.6f} s")
    rows.append({"N": N, "dcache_time": dcache_avg, "icache_time": icache_avg})

# --- Write CSV --------------------------------------------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["N", "dcache_time", "icache_time"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nResults written to {OUTPUT_CSV}")
