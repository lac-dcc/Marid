#!/usr/bin/env python3

import sys
import subprocess
import time
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Usage:
#   python3 gen_and_bench.py [N] [X]
# ---------------------------------------------------------------------------

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
X = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# ---------------------------------------------------------------------------
# Perf events (adjust depending on your CPU support)
# ---------------------------------------------------------------------------
PERF_EVENTS = [
    "instructions",
    "cycles",
#    "cache-references",
#   "cache-misses",
    "L1-icache-loads",
    "L1-icache-load-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
]

# --- Generate dcache.c ------------------------------------------------------
print(f"Generating dcache.c (N={N})...")

dcache_code = f"""// Auto-generated (N={N})
#define N {N}
int main() {{
    int a[N];
    int sum = 0;
    for (int i = 0; i < N; ++i) {{
        sum += a[i];
    }}
    return sum;
}}
"""

Path("dcache.c").write_text(dcache_code)

# --- Generate icache.c ------------------------------------------------------
print(f"Generating icache.c (N={N})...")

with open("icache.c", "w") as f:
    f.write(f"// Auto-generated (N={N})\n\n")
    f.write("int main() {\n")
    f.write("    volatile int x = 0;\n")
    f.write("    volatile int sum = 0;\n")
    for _ in range(N):
        f.write("    sum += x;\n")
    f.write("    return sum;\n")
    f.write("}\n")

# --- Compile ----------------------------------------------------------------
def compile_program(src, out):
    print(f"Compiling {src}...")
    subprocess.run(["clang", "-O0", "-o", out, src], check=True)

compile_program("dcache.c", "dcache")
compile_program("icache.c", "icache")

# --- Run perf ---------------------------------------------------------------
def run_perf(exe):
    cmd = [
        "perf", "stat",
        "-e", ",".join(PERF_EVENTS),
        f"./{exe}"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    return result.stderr

# --- Parse perf output ------------------------------------------------------
def parse_perf(output):
    counters = {}

    for line in output.splitlines():
        # matches: "  1,234,567  instructions"
        m = re.match(r"\s*([\d,]+)\s+([A-Za-z0-9\-\_]+)", line)
        if m:
            value = int(m.group(1).replace(",", ""))
            name = m.group(2)
            counters[name] = value

    return counters

# --- Benchmark --------------------------------------------------------------
def run_bench(name, exe):
    print(f"\nWarming up ./{exe}...")
    subprocess.run([f"./{exe}"], stdout=subprocess.DEVNULL)

    print(f"\nRunning ./{exe} {X} times...")

    total_time = 0.0
    total_counters = {}

    for i in range(1, X + 1):
        t0 = time.perf_counter()

        perf_out = run_perf(exe)

        t1 = time.perf_counter()
        elapsed = t1 - t0
        total_time += elapsed

        counters = parse_perf(perf_out)

        for k, v in counters.items():
            total_counters[k] = total_counters.get(k, 0) + v

        print(f"  run {i:3d}/{X}:  {elapsed:.6f} s")

    avg_time = total_time / X

    print(f"\n  {name} total time:    {total_time:.6f} s")
    print(f"  {name} average time:  {avg_time:.6f} s")

    print(f"\n  {name} counters (average):")
    for k in sorted(total_counters.keys()):
        print(f"    {k:25s}: {total_counters[k] // X}")

    return total_time, avg_time, total_counters

# --- Run benchmarks ---------------------------------------------------------
dc_total, dc_avg, dc_cnt = run_bench("dcache", "dcache")
ic_total, ic_avg, ic_cnt = run_bench("icache", "icache")

# --- Summary ----------------------------------------------------------------
print("\n========================================")
print(f"  Benchmark Summary (N={N}, X={X})")
print("========================================")
print(f"  {'Program':<10}  {'Total(s)':>10}  {'Avg(s)':>10}")
print(f"  {'-------':<10}  {'--------':>10}  {'------':>10}")
print(f"  {'dcache':<10}  {dc_total:10.6f}  {dc_avg:10.6f}")
print(f"  {'icache':<10}  {ic_total:10.6f}  {ic_avg:10.6f}")
print("========================================")
