#include "marid/Passes/MemoryAllocationPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace marid {

namespace {

struct Allocation {
  uint64_t offset;
  uint64_t size;
};

struct MemoryAllocationPass
    : public PassWrapper<MemoryAllocationPass, OperationPass<ModuleOp>> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    uint64_t currentOffset = 0;
    llvm::DenseMap<Value, Allocation> allocations;

    auto allocate = [&](Value v, uint64_t size) {
      allocations[v] = Allocation{currentOffset, size};
      currentOffset += size;
    };

    module.walk([&](func::FuncOp func) {
      // --- Function arguments ---
      for (Value arg : func.getArguments()) {
        Type ty = arg.getType();
        uint64_t size = getTypeSizeInBytes(ty);
        allocate(arg, size);
      }

      // --- Operations ---
      func.walk([&](Operation *op) {
        // Scalars (arith ops, etc.)
        for (Value result : op->getResults()) {
          if (isa<MemRefType>(result.getType()))
            continue; // handled below

          uint64_t size = getTypeSizeInBytes(result.getType());
          allocate(result, size);
        }

        // Buffers
        if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
          MemRefType memrefTy = alloc.getType();
          if (!memrefTy.hasStaticShape()) {
            alloc.emitError("dynamic memref in constant-bounded program");
            signalPassFailure();
            return;
          }

          uint64_t elemSize =
              memrefTy.getElementTypeBitWidth() / 8;
          uint64_t numElems = 1;
          for (int64_t dim : memrefTy.getShape())
            numElems *= dim;

          allocate(alloc.getResult(), elemSize * numElems);
        }
      });
    });

    // --- Report ---
    llvm::outs() << "Stack size: " << currentOffset << " Bytes\n";
    llvm::outs() << "----\nAllocation:\n";

    for (auto &[value, alloc] : allocations) {
      llvm::outs()
          << value << " -> ["
          << alloc.offset << ", "
          << (alloc.offset + alloc.size - 1)
          << "]\n";
    }
  }

  static uint64_t getTypeSizeInBytes(Type ty) {
    if (auto intTy = dyn_cast<IntegerType>(ty))
      return std::max<uint64_t>(1, intTy.getWidth() / 8);

    if (auto floatTy = dyn_cast<FloatType>(ty))
      return floatTy.getWidth() / 8;

    llvm::report_fatal_error("unsupported type in allocator");
  }

  StringRef getArgument() const override {
    return "marid-alloc";
  }

  StringRef getDescription() const override {
    return "Assigns stack memory ranges to variables and buffers "
           "in constant-bounded programs";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
                    memref::MemRefDialect,
                    func::FuncDialect>();
  }
};

} // namespace

std::unique_ptr<Pass> createMemoryAllocationPass() {
  return std::make_unique<MemoryAllocationPass>();
}

} // namespace marid
