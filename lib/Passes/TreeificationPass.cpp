#include "marid/Passes/TreeificationPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace marid {

/// Treeifies a single scf.if operation.
/// We duplicate the 'continuation' (all logic following the scf.if in its 
/// current block) into both the then and else paths to eliminate join points.
static LogicalResult treeifyIf(scf::IfOp ifOp, RewriterBase &rewriter) {
  Block *currentBlock = ifOp->getBlock();
  Location loc = ifOp.getLoc();

  // 1. Isolate the continuation logic. 
  // We split the block immediately after the scf.if. The new continuationBlock
  // contains all the original users of the scf.if results.
  Block *continuationBlock = rewriter.splitBlock(currentBlock, ++Block::iterator(ifOp));

  // 2. Create target blocks in the function region.
  // This logic assumes we are treeifying from the top-down, so the parent
  // of the current block is the function's region.
  Region *parentRegion = currentBlock->getParent();
  Block *thenBlock = rewriter.createBlock(parentRegion);
  Block *elseBlock = rewriter.createBlock(parentRegion);

  // Helper to build a leaf of the tree.
  auto buildPath = [&](Region &srcRegion, Block *destBlock) {
    rewriter.setInsertionPointToStart(destBlock);
    IRMapping mapper;

    // A. Clone the operations from the scf.if branch.
    // scf.if regions are guaranteed by the dialect to have a single block.
    for (auto &op : srcRegion.front()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        // Map scf.if results to the specific values produced in this branch.
        for (auto it : llvm::zip(ifOp.getResults(), yieldOp.getOperands())) {
          mapper.map(std::get<0>(it), mapper.lookupOrDefault(std::get<1>(it)));
        }
        continue; 
      }
      rewriter.clone(op, mapper);
    }

    // B. Clone the continuation logic into this branch.
    // This creates the "Tree" shape by duplicating the tail logic.
    for (auto &op : *continuationBlock) {
      rewriter.clone(op, mapper);
    }
  };

  // 3. Populate both paths.
  buildPath(ifOp.getThenRegion(), thenBlock);
  buildPath(ifOp.getElseRegion(), elseBlock);

  // 4. Transform the original block into a conditional branch.
  rewriter.setInsertionPoint(ifOp);
  cf::CondBranchOp::create(rewriter, loc, ifOp.getCondition(), 
                           thenBlock, ValueRange{}, 
                           elseBlock, ValueRange{});

  // 5. Cleanup.
  // CRITICAL: We erase the continuation block BEFORE the scf.if.
  // This removes the original users of the scf.if results, allowing the 
  // scf.if to be erased without triggering "op has uses" assertions.
  rewriter.eraseBlock(continuationBlock);
  rewriter.eraseOp(ifOp);

  return success();
}

namespace {

struct TreeificationPass
    : public PassWrapper<TreeificationPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    module.walk([&](func::FuncOp func) {
      bool changed = true;
      while (changed) {
        changed = false;
        
        // We use a restarted walk to avoid invalid/stale pointers.
        // We only process scf.if ops that are direct children of the function
        // (not nested inside another scf.if). This ensures top-down processing.
        func.walk<WalkOrder::PreOrder>([&](scf::IfOp ifOp) {
          if (isa<func::FuncOp>(ifOp->getParentOp())) {
            if (succeeded(treeifyIf(ifOp, rewriter))) {
              changed = true;
              return WalkResult::interrupt(); // Restart search for new top-level ops
            }
          }
          return WalkResult::advance();
        });
      }
    });
  }

  StringRef getArgument() const override { return "marid-treeify"; }
  StringRef getDescription() const override {
    return "Transforms structured control flow into a tree-shaped CFG via iterative duplication.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect, scf::SCFDialect, 
                    arith::ArithDialect, func::FuncDialect>();
  }
};

} // namespace

std::unique_ptr<Pass> createTreeificationPass() {
  return std::make_unique<TreeificationPass>();
}

} // namespace marid
