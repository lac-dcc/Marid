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

/// Clones a specific branch region and the shared continuation logic into a
/// new block. This is the core "duplication" step that converts the DAG
/// structure into a Tree.
static void clonePath(scf::IfOp ifOp, Region &branchRegion, 
                      Block *continuationBlock, Block *destBlock, 
                      RewriterBase &rewriter) {
  rewriter.setInsertionPointToStart(destBlock);
  IRMapping mapper;

  // 1. Clone branch-specific logic (then { ... } or else { ... })
  for (auto &op : branchRegion.front()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      // Map the scf.if results to the values produced in this specific path.
      for (auto [ifRes, yieldVal] :
          llvm::zip(ifOp.getResults(), yieldOp.getOperands())) {
        mapper.map(ifRes, mapper.lookupOrDefault(yieldVal));
      }
      // scf.yield is not cloned into the CFG
      continue;
    }
    rewriter.clone(op, mapper);
  }

  // 2. Clone the continuation logic (everything that was after the scf.if)
  for (auto &op : *continuationBlock) {
    rewriter.clone(op, mapper);
  }
}

/// Orchestrates the treeification of a single scf.if operation.
static LogicalResult treeifyIf(scf::IfOp ifOp, RewriterBase &rewriter) {
  Block *currentBlock = ifOp->getBlock();
  Location loc = ifOp.getLoc();

  // 1. Isolate the "tail" of the current block.
  Block *continuationBlock =
    rewriter.splitBlock(currentBlock, ++Block::iterator(ifOp));

  // 2. Prepare new blocks for the divergent paths.
  Region *parentRegion = currentBlock->getParent();
  Block *thenBlock = rewriter.createBlock(parentRegion);
  Block *elseBlock = rewriter.createBlock(parentRegion);

  // 3. Duplicate logic into each path.
  clonePath(ifOp, ifOp.getThenRegion(), continuationBlock, thenBlock, rewriter);
  clonePath(ifOp, ifOp.getElseRegion(), continuationBlock, elseBlock, rewriter);

  // 4. Replace the scf.if with a conditional branch in the original block.
  rewriter.setInsertionPoint(ifOp);
  cf::CondBranchOp::create(rewriter, loc, ifOp.getCondition(), 
                           thenBlock, ValueRange{}, 
                           elseBlock, ValueRange{});

  // 5. Cleanup: Remove the shared tail and the original operation.
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
        // Search for top-level scf.if operations to process them root-to-leaf.
        func.walk<WalkOrder::PreOrder>([&](scf::IfOp ifOp) {
          if (isa<func::FuncOp>(ifOp->getParentOp())) {
            if (succeeded(treeifyIf(ifOp, rewriter))) {
              changed = true;
              // Restart to find newly promoted inner ifs
              return WalkResult::interrupt(); 
            }
          }
          return WalkResult::advance();
        });
      }
    });
  }

  StringRef getArgument() const override { return "marid-treeify"; }
  StringRef getDescription() const override {
    return "Transforms a DAG of structured control flow into a Tree-shaped CFG.";
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
