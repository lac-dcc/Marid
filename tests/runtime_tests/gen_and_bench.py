#!/usr/bin/env python3

import sys
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Usage:
#   python3 gen_and_bench.py [N] [X]
# ---------------------------------------------------------------------------

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
X = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# --- Generate dcache.c ------------------------------------------------------
print(f"Generating dcache.c (array size N={N})...")

dcache_code = f"""// Auto-generated (N={N})
// Stresses the data cache: large stack array, sequential sum loop.
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
print(f"Generating icache.c (unrolled additions N={N})...")

with open("icache.c", "w") as f:
    f.write(f"// Auto-generated (N={N})\n")
    f.write("// Stresses the instruction cache: single stack slot, no loop.\n\n")
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
    subprocess.run(["clang", "-O0", "-o", out, src])

compile_program("dcache.c", "dcache")
compile_program("icache.c", "icache")

# --- Benchmark helper -------------------------------------------------------
def run_bench(name, exe):
    print(f"\nRunning ./{exe} {X} times...")
    total = 0.0

    for i in range(1, X + 1):
        t0 = time.perf_counter()
        subprocess.run([f"./{exe}"], stdout=subprocess.DEVNULL)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total += elapsed

        print(f"  run {i:3d}/{X}:  {elapsed:.6f} s")

    avg = total / X

    print(f"\n  {name} total:    {total:.6f} s")
    print(f"  {name} average:  {avg:.6f} s")

    return total, avg

# --- Run benchmarks ---------------------------------------------------------
dc_total, dc_avg = run_bench("dcache", "dcache")
ic_total, ic_avg = run_bench("icache", "icache")

# --- Summary ----------------------------------------------------------------
print("\n========================================")
print(f"  Benchmark Summary  (N={N}, X={X} runs)")
print("========================================")
print(f"  {'Program':<10}  {'Total (s)':>12}  {'Average (s)':>12}")
print(f"  {'-------':<10}  {'---------':>12}  {'-----------':>12}")
print(f"  {'dcache':<10}  {dc_total:12.6f}  {dc_avg:12.6f}")
print(f"  {'icache':<10}  {ic_total:12.6f}  {ic_avg:12.6f}")
print("========================================")
