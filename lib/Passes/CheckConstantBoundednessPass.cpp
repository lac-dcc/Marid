#include "marid/Passes/CheckConstantBoundednessPass.h"
#include "marid/Analysis/ConstantBoundednessAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace marid {

namespace {

struct CheckConstantBoundednessPass
    : public PassWrapper<CheckConstantBoundednessPass,
                          OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Request the analysis (computed lazily, cached by MLIR)
    auto &analysis = getAnalysis<ConstantBoundednessAnalysis>();

    if (!analysis.isConstantBounded()) {
      module.emitError()
          << "program is not constant-bounded";
      signalPassFailure();
      return;
    }

    llvm::outs() << "Program is constant-bounded.\n";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCheckConstantBoundednessPass() {
  return std::make_unique<CheckConstantBoundednessPass>();
}

} // namespace marid
