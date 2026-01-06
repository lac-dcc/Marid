# Marid

Marid is an MLIR-based static analysis and pass framework for reasoning about
memory allocation and boundedness properties of programs.

## Features

- Constant-boundedness analysis for SCF-based MLIR programs
- MLIR Analysis Framework integration
- Checker pass with diagnostic support
- Standalone driver tool (`marid-opt`)

## Building

```bash
mkdir build
cd build
cmake .. \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
make
