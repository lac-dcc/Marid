#include "marid/Analysis/ConstantBoundednessAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;
using namespace marid;

ConstantBoundednessAnalysis::ConstantBoundednessAnalysis(Operation *op)
    : constantBounded(true) {

  // Walk all operations in the IR rooted at `op`
  op->walk([&](Operation *nestedOp) {
    if (!constantBounded)
      return WalkResult::interrupt();

    // Reject while-loops outright
    if (isa<scf::WhileOp>(nestedOp)) {
      constantBounded = false;
      return WalkResult::interrupt();
    }

    // Check for-loops
    if (auto forOp = dyn_cast<scf::ForOp>(nestedOp)) {
      Value lb = forOp.getLowerBound();
      Value ub = forOp.getUpperBound();
      Value step = forOp.getStep();

      auto isConstant = [](Value v) {
        return v.getDefiningOp<arith::ConstantOp>() != nullptr;
      };

      if (!isConstant(lb) || !isConstant(ub) || !isConstant(step)) {
        constantBounded = false;
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
}

bool ConstantBoundednessAnalysis::invalidate(
    mlir::Operation *,
    const mlir::AnalysisManager::PreservedAnalyses &pa) {
  return !pa.isPreserved<ConstantBoundednessAnalysis>();
}
