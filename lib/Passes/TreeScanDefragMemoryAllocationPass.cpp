#include "marid/Passes/TreeScanDefragMemoryAllocationPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace marid {
namespace {

class TreeScanDefragMemoryAllocationPass
    : public PassWrapper<TreeScanDefragMemoryAllocationPass,
                          OperationPass<func::FuncOp>> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TreeScanDefragMemoryAllocationPass)

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::outs()
        << "[TreeScanDefragMemAlloc] Visiting function @"
        << func.getName() << "\n";

    // Future steps:
    // 1. Bottom-up last-use discovery
    // 2. Top-down greedy allocation
    // 3. Detect fragmentation
    // 4. Perform defragmentation via swaps
  }

  StringRef getArgument() const final {
    return "marid-tree-scan-defrag-mem-alloc";
  }

  StringRef getDescription() const final {
    return "Tree-scan memory allocation with defragmentation";
  }
};

} // namespace

std::unique_ptr<Pass> createTreeScanDefragMemoryAllocationPass() {
  return std::make_unique<TreeScanDefragMemoryAllocationPass>();
}

} // namespace marid
