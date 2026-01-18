#include "marid/Passes/TreeScanDefragMemoryAllocationPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace marid {
namespace {

/// Print only SSA name
static void printValueName(Value v, llvm::raw_ostream &os) {
  v.printAsOperand(os, OpPrintingFlags().useLocalScope());
}

/// Seen-set used during last-use discovery
using SeenSet = llvm::SmallPtrSet<Value, 8>;

/// Maximum size of the stack seen across all paths:
size_t globalMaxStackSize = 0;

/// Memory interval
struct Interval {
  size_t offset;
  size_t size;
};

/// Allocator state along one CFG path
struct AllocatorState {
  size_t nextOffset = 0;      // current live footprint
  llvm::SmallVector<Interval, 8> freeList;
  llvm::DenseMap<Value, Interval> active;
};

/// Defragmentation statistics
struct DefragStats {
  unsigned numDefrags = 0;
  unsigned numMoves = 0;
  uint64_t bytesCopied = 0;
};

class TreeScanDefragMemoryAllocationPass
    : public PassWrapper<TreeScanDefragMemoryAllocationPass,
                         OperationPass<func::FuncOp>> {

private:
  /// Value → set of last-use operations
  using LastUseSet = llvm::SmallPtrSet<Operation *, 4>;
  llvm::DenseMap<Value, LastUseSet> lastUses;

  /// Final allocation map (for reporting)
  llvm::DenseMap<Value, Interval> finalAllocations;

  /// Global defragmentation statistics
  DefragStats stats;

  /* ================= LAST USE DISCOVERY ================= */

  SeenSet collectLastUsesFromBlock(Block *block);

  /* ================= ALLOCATION HELPERS ================= */

  bool isLastUse(Value v, Operation *op) const;
  size_t getValueSize(Value v) const;

  size_t totalFreeMemory(const AllocatorState &state) const;
  bool hasContiguousFree(const AllocatorState &state, size_t size) const;

  Interval allocateInterval(AllocatorState &state, size_t size);
  Interval allocateFromFreeList(AllocatorState &state, size_t size);
  Interval allocateFromTop(AllocatorState &state, size_t size);

  void freeInterval(AllocatorState &state, const Interval &interval);

  void defragment(AllocatorState &state);

  /* ================= TOP-DOWN ALLOCATION ================= */

  void allocateFromBlock(Block *block, AllocatorState state);

  /* ================= REPORTING ================= */

  void printAllocations(func::FuncOp func);
  void printDefragStats(func::FuncOp func);

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TreeScanDefragMemoryAllocationPass)

  void runOnOperation() override;

  StringRef getArgument() const final {
    return "marid-tree-scan-defrag-mem-alloc";
  }

  StringRef getDescription() const final {
    return "Tree-scan memory allocation with defragmentation";
  }
};

/* ========================= LAST USE DISCOVERY ========================= */

SeenSet
TreeScanDefragMemoryAllocationPass::collectLastUsesFromBlock(Block *block) {
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

/* ========================= HELPERS ========================= */

bool
TreeScanDefragMemoryAllocationPass::isLastUse(Value v,
                                              Operation *op) const {
  auto it = lastUses.find(v);
  if (it == lastUses.end())
    return false;
  return it->second.contains(op);
}

size_t
TreeScanDefragMemoryAllocationPass::getValueSize(Value v) const {
  if (auto intTy = dyn_cast<IntegerType>(v.getType()))
    return intTy.getWidth() / 8;

  if (auto memrefTy = dyn_cast<MemRefType>(v.getType()))
    return memrefTy.getNumElements() *
           memrefTy.getElementTypeBitWidth() / 8;

  return 0;
}

size_t
TreeScanDefragMemoryAllocationPass::totalFreeMemory(
    const AllocatorState &state) const {
  size_t sum = 0;
  for (const auto &i : state.freeList)
    sum += i.size;
  return sum;
}

bool
TreeScanDefragMemoryAllocationPass::hasContiguousFree(
    const AllocatorState &state, size_t size) const {
  for (const auto &i : state.freeList)
    if (i.size >= size)
      return true;
  return false;
}

/* ========================= ALLOCATION ========================= */

Interval
TreeScanDefragMemoryAllocationPass::allocateFromFreeList(
    AllocatorState &state, size_t size) {
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
  llvm_unreachable("No suitable free interval");
}

Interval
TreeScanDefragMemoryAllocationPass::allocateFromTop(
    AllocatorState &state, size_t size) {
  Interval result{state.nextOffset, size};
  state.nextOffset += size;
  globalMaxStackSize =
      std::max(globalMaxStackSize, state.nextOffset);
  return result;
}

Interval
TreeScanDefragMemoryAllocationPass::allocateInterval(
    AllocatorState &state, size_t size) {

  // Case 1: fast path
  if (hasContiguousFree(state, size)) {
    return allocateFromFreeList(state, size);
  }

  // Case 2: stack must grow — should we defragment?
  if (totalFreeMemory(state) > 0 &&
    state.nextOffset + size > globalMaxStackSize) {
    llvm::outs() << "DEBUG: Invoking defrag with global max = " << globalMaxStackSize << " vs " << (state.nextOffset + size) << "\n";
    defragment(state);
    // After defrag, we can allocate from top!
  }

  // Case 3: unavoidable growth
  return allocateFromTop(state, size);
}


void
TreeScanDefragMemoryAllocationPass::freeInterval(
    AllocatorState &state, const Interval &interval) {
  state.freeList.push_back(interval);
}

/* ========================= DEFRAGMENTATION ========================= */

void
TreeScanDefragMemoryAllocationPass::defragment(AllocatorState &state) {
  stats.numDefrags++;

  llvm::SmallVector<std::pair<Value, Interval>, 8> active;
  for (auto &kv : state.active)
    active.push_back(kv);

  llvm::sort(active, [](auto &a, auto &b) {
    return a.second.offset < b.second.offset;
  });

  size_t newOffset = 0;
  for (auto &entry : active) {
    Interval &interval = state.active[entry.first];
    if (interval.offset != newOffset) {
      stats.numMoves++;
      stats.bytesCopied += interval.size;
      interval.offset = newOffset;
    }
    newOffset += interval.size;
  }

  state.nextOffset = newOffset;
  state.freeList.clear();
}

/* ========================= TOP-DOWN ALLOCATION ========================= */

void
TreeScanDefragMemoryAllocationPass::allocateFromBlock(
    Block *block, AllocatorState state) {

  for (BlockArgument arg : block->getArguments()) {
    size_t size = getValueSize(arg);
    if (size == 0)
      continue;
    Interval i = allocateInterval(state, size);
    state.active[arg] = i;
    finalAllocations.try_emplace(arg, i);
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

      Interval i = allocateInterval(state, size);
      state.active[res] = i;
      finalAllocations.try_emplace(res, i);

      // Immediately free values that have no uses
      if (res.use_empty()) {
        freeInterval(state, i);
        state.active.erase(res);
      }
    } 

  }

  for (Block *succ : block->getSuccessors())
    allocateFromBlock(succ, state);
}

/* ========================= REPORTING ========================= */

void
TreeScanDefragMemoryAllocationPass::printAllocations(func::FuncOp func) {
  size_t maxOffset = 0;

  llvm::outs() << "\nAllocation:\n";
  for (auto &kv : finalAllocations) {
    const Interval &i = kv.second;
    maxOffset = std::max(maxOffset, i.offset + i.size);

    llvm::outs() << "["
                 << i.offset << ", "
                 << (i.offset + i.size - 1) << "] ";
    printValueName(kv.first, llvm::outs());
    llvm::outs() << "\n";
  }

  llvm::outs() << "Stack size: " << maxOffset << " Bytes\n";
  assert(globalMaxStackSize == maxOffset);
}

void
TreeScanDefragMemoryAllocationPass::printDefragStats(func::FuncOp func) {
  llvm::outs() << "\nDefragmentation statistics:\n";
  llvm::outs() << "  Defragmentations: " << stats.numDefrags << "\n";
  llvm::outs() << "  Logical moves:    " << stats.numMoves << "\n";
  llvm::outs() << "  Bytes copied:     " << stats.bytesCopied << "\n";
}

/* ========================= PASS ENTRY ========================= */

void
TreeScanDefragMemoryAllocationPass::runOnOperation() {
  func::FuncOp func = getOperation();

  lastUses.clear();
  finalAllocations.clear();
  stats = DefragStats();

  Block &entry = func.getBody().front();

  collectLastUsesFromBlock(&entry);

  AllocatorState initial;
  allocateFromBlock(&entry, initial);

  printAllocations(func);
  printDefragStats(func);
}

} // namespace

std::unique_ptr<Pass>
createTreeScanDefragMemoryAllocationPass() {
  return std::make_unique<TreeScanDefragMemoryAllocationPass>();
}

} // namespace marid
