/**
 * @file marid-opt.cpp
 * @brief Standalone driver for Marid passes.
 */

#include "marid/Passes/CheckConstantBoundednessPass.h"
#include "marid/Analysis/ConstantBoundednessAnalysis.h"
#include "marid/Passes/LoopExpansionPass.h"
#include "marid/Passes/TreeificationPass.h"
#include "marid/Passes/MemoryAllocationPass.h"
#include "marid/Passes/TreeScanMemoryAllocationPass.h"
#include "marid/Passes/TreeScanDefragMemoryAllocationPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace marid;

/// Print an error message and exit.
[[noreturn]] static void die(llvm::StringRef msg) {
  llvm::errs() << msg << "\n";
  std::exit(1);
}

/// Input file.
static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input mlir file>"),
    llvm::cl::init("-"));

/// Memory allocation strategy.
static llvm::cl::opt<std::string> memAllocStrategy(
    "mem-alloc",
    llvm::cl::desc("Memory allocation strategy"),
    llvm::cl::value_desc("baseline|tree-scan|tree-scan-defrag"),
    llvm::cl::init("baseline"));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Marid: constant-boundedness analysis and memory allocation\n");

  // Dialect registration
  DialectRegistry registry;
  registry.insert<
      func::FuncDialect,
      arith::ArithDialect,
      scf::SCFDialect,
      cf::ControlFlowDialect,
      memref::MemRefDialect
  >();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Parse input
  llvm::SourceMgr sourceMgr;
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!buffer)
    die("Error: could not open input file");

  sourceMgr.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    die("Error: failed to parse MLIR module");

  // Build pass pipeline
  PassManager pm(&context);

  pm.addPass(createCheckConstantBoundednessPass());
  pm.addPass(createLoopExpansionPass());
  pm.addPass(createTreeificationPass());

  // Select memory allocator
  if (memAllocStrategy == "baseline") {
    pm.addPass(createMemoryAllocationPass());
  } else if (memAllocStrategy == "tree-scan") {
    pm.addNestedPass<func::FuncOp>(
        createTreeScanMemoryAllocationPass());
  } else if (memAllocStrategy == "tree-scan-defrag") {
    pm.addNestedPass<func::FuncOp>(
        createTreeScanDefragMemoryAllocationPass());
  } else {
    die("Unknown memory allocation strategy: " + memAllocStrategy);
  }

  // Run passes
  if (failed(pm.run(*module)))
    return 1;

  // Print final module
  llvm::outs() << "\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  return 0;
}
