#include "marid/Passes/LoopExpansionPass.h"
#include "marid/Analysis/ConstantBoundednessAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

static std::optional<int64_t> getConstantIndex(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  return std::nullopt;
}

static LogicalResult expandForLoop(scf::ForOp forOp) {
  auto lbOpt = getConstantIndex(forOp.getLowerBound());
  auto ubOpt = getConstantIndex(forOp.getUpperBound());
  auto stepOpt = getConstantIndex(forOp.getStep());

  if (!lbOpt || !ubOpt || !stepOpt)
    return failure();

  int64_t lb = *lbOpt;
  int64_t ub = *ubOpt;
  int64_t step = *stepOpt;

  if (step <= 0)
    return failure();

  int64_t tripCount = (ub - lb + step - 1) / step;
  if (tripCount <= 0)
    return success(); // empty loop

  OpBuilder builder(forOp);

  Block *parentBlock = forOp->getBlock();
  auto insertionPoint = Block::iterator(forOp);

  Value iv = forOp.getInductionVar();

  // Insert before the loop
  builder.setInsertionPoint(parentBlock, insertionPoint);

  for (int64_t i = 0; i < tripCount; ++i) {
    IRMapping mapping;

    // Create constant IV value
    Value ivValue =
      arith::ConstantIndexOp::create(builder, forOp.getLoc(), lb + i * step);

    mapping.map(iv, ivValue);

    // Clone body operations except scf.yield
    for (Operation &op : forOp.getBody()->without_terminator()) {
      builder.clone(op, mapping);
    }
  }

  forOp.erase();
  return success();
}

namespace marid {

namespace {

struct LoopExpansionPass
    : public PassWrapper<LoopExpansionPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 1. Query ConstantBoundednessAnalysis
    auto &analysis = getAnalysis<ConstantBoundednessAnalysis>();
    if (!analysis.isConstantBounded()) {
      module.emitError()
          << "loop expansion requires constant-bounded program";
      signalPassFailure();
      return;
    }

    // 2. Expand loops innermost-first
    SmallVector<scf::ForOp, 8> loops;
    module.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
      loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      if (failed(expandForLoop(forOp))) {
        forOp.emitError()
            << "failed to expand scf.for (expected constant bounds)";
        signalPassFailure();
        return;
      }
    }
  }

  StringRef getArgument() const override {
    return "marid-expand-loops";
  }

  StringRef getDescription() const override {
    return "Expand constant-bounded scf.for loops by unrolling them";
  }
};

} // namespace

std::unique_ptr<Pass> createLoopExpansionPass() {
  return std::make_unique<LoopExpansionPass>();
}

} // namespace marid

