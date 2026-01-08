#include "marid/Passes/TreeificationPass.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace marid {

namespace {

struct TreeificationPass
    : public PassWrapper<TreeificationPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Skeleton: do nothing for now.
    // Later, this is where we will:
    //  - traverse scf.if ops (post-order)
    //  - detect join points
    //  - duplicate continuations
    //  - rewrite SSA uses
    (void)module;
  }

  StringRef getArgument() const override {
    return "marid-treeify";
  }

  StringRef getDescription() const override {
    return "Expands structured control flow into a tree";
  }
};

} // namespace

std::unique_ptr<Pass> createTreeificationPass() {
  return std::make_unique<TreeificationPass>();
}

} // namespace marid
