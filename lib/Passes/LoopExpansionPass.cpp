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

static LogicalResult expandForLoop(scf::ForOp forOp, RewriterBase &rewriter) {
  auto lbOpt = getConstantIndex(forOp.getLowerBound());
  auto ubOpt = getConstantIndex(forOp.getUpperBound());
  auto stepOpt = getConstantIndex(forOp.getStep());

  if (!lbOpt || !ubOpt || !stepOpt || *stepOpt <= 0)
    return failure();

  int64_t lb = *lbOpt;
  int64_t ub = *ubOpt;
  int64_t step = *stepOpt;
  int64_t tripCount = (ub - lb + step - 1) / step;

  Location loc = forOp.getLoc();
  IRMapping mapping;

  // 1. Initialize mapping for iter_args
  // The initial values of the loop's block arguments come from the init_args
  for (auto [arg, initVal] :
      llvm::zip(forOp.getRegionIterArgs(), forOp.getInits())) {
    mapping.map(arg, initVal);
  }

  rewriter.setInsertionPoint(forOp);

  for (int64_t i = 0; i < tripCount; ++i) {
    // Update the Induction Variable for this iteration
    Value ivValue =
      arith::ConstantIndexOp::create(rewriter, loc, lb + i * step);
    mapping.map(forOp.getInductionVar(), ivValue);

    // Clone the body
    for (Operation &op : forOp.getBody()->without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // 2. Prepare for next iteration:
    // Map the iter_args to the values yielded at the end of THIS iteration.
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (auto [arg, yieldVal] :
        llvm::zip(forOp.getRegionIterArgs(), yieldOp.getOperands())) {
      // We look up the yielded value in the mapper because it might have
      // been newly created/cloned in this iteration.
      mapping.map(arg, mapping.lookupOrDefault(yieldVal));
    }
  }

  // 3. Replace loop results with the final mapped values of the iter_args
  SmallVector<Value> finalValues;
  for (Value arg : forOp.getRegionIterArgs()) {
    finalValues.push_back(mapping.lookup(arg));
  }

  // This step prevents the "destroyed but still has uses" crash
  rewriter.replaceOp(forOp, finalValues);

  return success();
}

namespace marid {

namespace {

  static LogicalResult expandForLoop(scf::ForOp forOp, RewriterBase &rewriter) {
    auto lbOpt = getConstantIndex(forOp.getLowerBound());
    auto ubOpt = getConstantIndex(forOp.getUpperBound());
    auto stepOpt = getConstantIndex(forOp.getStep());

    if (!lbOpt || !ubOpt || !stepOpt || *stepOpt <= 0)
      return failure();

    int64_t lb = *lbOpt;
    int64_t ub = *ubOpt;
    int64_t step = *stepOpt;
    int64_t tripCount = (ub - lb + step - 1) / step;

    Location loc = forOp.getLoc();
    IRMapping mapping;

    // 1. Initialize mapping for iter_args
    // The initial values of the loop's block arguments come from the init_args
    for (auto [arg, initVal] :
        llvm::zip(forOp.getRegionIterArgs(), forOp.getInits())) {
      mapping.map(arg, initVal);
    }

    rewriter.setInsertionPoint(forOp);

    for (int64_t i = 0; i < tripCount; ++i) {
      // Update the Induction Variable for this iteration
      Value ivValue =
        arith::ConstantIndexOp::create(rewriter, loc, lb + i * step);
      mapping.map(forOp.getInductionVar(), ivValue);

      // Clone the body
      for (Operation &op : forOp.getBody()->without_terminator()) {
        rewriter.clone(op, mapping);
      }

      // 2. Prepare for next iteration: 
      // Map the iter_args to the values yielded at the end of THIS iteration.
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (auto [arg, yieldVal] :
          llvm::zip(forOp.getRegionIterArgs(), yieldOp.getOperands())) {
        // We look up the yielded value in the mapper because it might have 
        // been newly created/cloned in this iteration.
        mapping.map(arg, mapping.lookupOrDefault(yieldVal));
      }
    }

    // 3. Replace loop results with the final mapped values of the iter_args
    SmallVector<Value> finalValues;
    for (Value arg : forOp.getRegionIterArgs()) {
      finalValues.push_back(mapping.lookup(arg));
    }

    // This is the critical step that prevents the "destroyed but still has
    // uses" crash
    rewriter.replaceOp(forOp, finalValues);

    return success();
  }

struct LoopExpansionPass
    : public PassWrapper<LoopExpansionPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    // 1. Query ConstantBoundednessAnalysis
    auto &analysis = getAnalysis<ConstantBoundednessAnalysis>();
    if (!analysis.isConstantBounded()) {
      module.emitError()
          << "loop expansion requires constant-bounded program";
      signalPassFailure();
      return;
    }

    // 2. Expand loops innermost-first
    SmallVector<scf::ForOp> loops;
    module.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
        loops.push_back(forOp);
        });

    for (scf::ForOp forOp : loops) {
      if (failed(expandForLoop(forOp, rewriter))) { // Pass rewriter here
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

