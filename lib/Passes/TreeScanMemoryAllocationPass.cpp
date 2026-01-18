#include "marid/Passes/TreeScanMemoryAllocationPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace marid {
namespace {

/// Auxiliary function to print just the SSA name
static void printValueName(Value v, llvm::raw_ostream &os) {
  v.printAsOperand(os, OpPrintingFlags().useLocalScope());
}

/// Set of values seen along a CFG path
using SeenSet = llvm::SmallPtrSet<Value, 8>;

/// Memory interval
struct Interval {
  size_t offset;
  size_t size;
};

/// Active allocator state along a path
struct AllocatorState {
  size_t nextOffset = 0;
  llvm::SmallVector<Interval, 8> freeList;
  llvm::DenseMap<Value, Interval> active;
};

class TreeScanMemoryAllocationPass
    : public PassWrapper<TreeScanMemoryAllocationPass,
                         OperationPass<func::FuncOp>> {

private:
  /// For each value, operations where it is last used
  using LastUseSet = llvm::SmallPtrSet<Operation *, 4>;
  llvm::DenseMap<Value, LastUseSet> lastUses;

  /// Final allocations (for reporting)
  llvm::DenseMap<Value, Interval> finalAllocations;

  /// Bottom-up last-use discovery
  SeenSet collectLastUsesFromBlock(Block *block);

  /// Top-down allocation
  void allocateFromBlock(Block *block, AllocatorState state);

  /// Helpers
  bool isLastUse(Value v, Operation *op) const;
  size_t getValueSize(Value v) const;
  Interval allocateInterval(AllocatorState &state, size_t size);
  void freeInterval(AllocatorState &state, const Interval &interval);

  void printLastUses(func::FuncOp func);
  void printAllocations(func::FuncOp func);

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TreeScanMemoryAllocationPass)

  void runOnOperation() override;

  StringRef getArgument() const final {
    return "marid-tree-scan-mem-alloc";
  }

  StringRef getDescription() const final {
    return "Tree-scan memory allocation pass";
  }
};

/* ========================= LAST USE DISCOVERY ========================= */

SeenSet
TreeScanMemoryAllocationPass::collectLastUsesFromBlock(Block *block) {
  SeenSet seen;

  for (Block *succ : block->getSuccessors()) {
    SeenSet childSeen = collectLastUsesFromBlock(succ);
    seen.insert(childSeen.begin(), childSeen.end());
  }

  for (auto it = block->rbegin(); it != block->rend(); ++it) {
    Operation &op = *it;

    for (Value v : op.getOperands()) {
      if (!seen.contains(v)) {
        lastUses[v].insert(&op);
        seen.insert(v);
      }
    }
  }

  return seen;
}

/* ========================= ALLOCATION HELPERS ========================= */

bool TreeScanMemoryAllocationPass::isLastUse(Value v,
                                             Operation *op) const {
  auto it = lastUses.find(v);
  if (it == lastUses.end())
    return false;
  return it->second.contains(op);
}

size_t TreeScanMemoryAllocationPass::getValueSize(Value v) const {
  if (auto intTy = dyn_cast<IntegerType>(v.getType()))
    return intTy.getWidth() / 8;

  if (auto memrefTy = dyn_cast<MemRefType>(v.getType()))
    return memrefTy.getNumElements() *
           memrefTy.getElementTypeBitWidth() / 8;

  return 0;
}

Interval
TreeScanMemoryAllocationPass::allocateInterval(AllocatorState &state,
                                               size_t size) {
  for (auto it = state.freeList.begin(); it != state.freeList.end(); ++it) {
    if (it->size >= size) {
      Interval result{it->offset, size};
      if (it->size > size) {
        it->offset += size;
        it->size -= size;
      } else {
        state.freeList.erase(it);
      }
      return result;
    }
  }

  Interval result{state.nextOffset, size};
  state.nextOffset += size;
  return result;
}

void
TreeScanMemoryAllocationPass::freeInterval(AllocatorState &state,
                                           const Interval &interval) {
  state.freeList.push_back(interval);
}

/* ========================= TOP-DOWN ALLOCATION ========================= */

void
TreeScanMemoryAllocationPass::allocateFromBlock(Block *block,
                                                AllocatorState state) {
  // Allocate block arguments
  for (BlockArgument arg : block->getArguments()) {
    size_t size = getValueSize(arg);
    if (size == 0)
      continue;

    Interval interval = allocateInterval(state, size);
    state.active[arg] = interval;
    finalAllocations.try_emplace(arg, interval);
  }

  for (Operation &op : *block) {
    // Free last uses
    for (Value v : op.getOperands()) {
      if (isLastUse(v, &op)) {
        auto it = state.active.find(v);
        if (it != state.active.end()) {
          freeInterval(state, it->second);
          state.active.erase(it);
        }
      }
    }

    // Allocate results
    for (Value res : op.getResults()) {
      size_t size = getValueSize(res);
      if (size == 0)
        continue;

      Interval interval = allocateInterval(state, size);
      state.active[res] = interval;
      finalAllocations.try_emplace(res, interval);

      // Immediately free values that have no uses
      if (res.use_empty()) {
        freeInterval(state, i);
        state.active.erase(res);
      }
    }
  }

  // Recurse into successors (tree-shaped CFG)
  for (Block *succ : block->getSuccessors()) {
    allocateFromBlock(succ, state);
  }
}

/* ========================= REPORTING ========================= */

void TreeScanMemoryAllocationPass::printLastUses(func::FuncOp func) {
  llvm::outs() << "\n[TreeScanMemAlloc] Last-use analysis for function @"
               << func.getName() << ":\n";

  for (auto &entry : lastUses) {
    llvm::outs() << "Value: ";
    printValueName(entry.first, llvm::outs());
    llvm::outs() << "\n";
    for (Operation *op : entry.second) {
      llvm::outs() << "  last used at: ";
      op->print(llvm::outs());
      llvm::outs() << "\n";
    }
  }
}

void TreeScanMemoryAllocationPass::printAllocations(func::FuncOp func) {
  size_t maxOffset = 0;

  llvm::outs() << "\nAllocation:\n";
  for (auto &entry : finalAllocations) {
    auto &interval = entry.second;
    maxOffset = std::max(maxOffset, interval.offset + interval.size);

    llvm::outs() << "["
                 << interval.offset << ", "
                 << (interval.offset + interval.size - 1) << "] ";
    printValueName(entry.first, llvm::outs());
    llvm::outs() << "\n";
  }

  llvm::outs() << "Stack size: " << maxOffset << " Bytes\n";
}

/* ========================= PASS ENTRY ========================= */

void TreeScanMemoryAllocationPass::runOnOperation() {
  func::FuncOp func = getOperation();

  lastUses.clear();
  finalAllocations.clear();

  Block &entry = func.getBody().front();

  collectLastUsesFromBlock(&entry);

  AllocatorState initialState;
  allocateFromBlock(&entry, initialState);

  printLastUses(func);
  printAllocations(func);
}

} // namespace

std::unique_ptr<Pass> createTreeScanMemoryAllocationPass() {
  return std::make_unique<TreeScanMemoryAllocationPass>();
}

} // namespace marid

