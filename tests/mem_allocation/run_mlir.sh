#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run_mlir.sh — Compile and run an MLIR benchmark
#
# Usage:
#   ./run_mlir.sh <benchmark>          # e.g. nested_if  (no extension needed)
#   ./run_mlir.sh <benchmark> --keep   # also keep intermediate .ll file
# ---------------------------------------------------------------------------

usage() {
  echo "Usage: $0 <benchmark_name> [--keep]"
  echo "  <benchmark_name>  Path to the .mlir file (extension optional)"
  echo "  --keep            Keep intermediate LLVM IR (.ll) file"
  exit 1
}

# --- Parse arguments --------------------------------------------------------
KEEP=false
BENCHMARK=""

for arg in "$@"; do
  case "$arg" in
    --keep) KEEP=true ;;
    --help|-h) usage ;;
    -*) echo "Unknown option: $arg"; usage ;;
    *)
      if [[ -n "$BENCHMARK" ]]; then
        echo "Error: unexpected argument '$arg'"
        usage
      fi
      BENCHMARK="$arg"
      ;;
  esac
done

[[ -z "$BENCHMARK" ]] && { echo "Error: no benchmark specified."; usage; }

# --- Resolve paths ----------------------------------------------------------
# Strip .mlir extension if provided, then re-add it for the source path
BASENAME="${BENCHMARK%.mlir}"
MLIR_FILE="${BASENAME}.mlir"
LL_FILE="${BASENAME}.ll"
EXE_FILE="${BASENAME}"

[[ -f "$MLIR_FILE" ]] || { echo "Error: '$MLIR_FILE' not found."; exit 1; }

# --- Compile ----------------------------------------------------------------
echo "Generating executable file ${EXE_FILE}..."

mlir-opt "$MLIR_FILE" \
  --convert-scf-to-cf \
  --finalize-memref-to-llvm \
  --convert-arith-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
| mlir-translate --mlir-to-llvmir -o "$LL_FILE"

clang "$LL_FILE" -o "$EXE_FILE"

# --- Optionally clean up the .ll file --------------------------------------
if [[ "$KEEP" == false ]]; then
  rm -f "$LL_FILE"
else
  echo "Keeping intermediate file: ${LL_FILE}"
fi

# --- Run --------------------------------------------------------------------
echo ""
echo "Execution output: ${EXE_FILE}"
"./${EXE_FILE}"
