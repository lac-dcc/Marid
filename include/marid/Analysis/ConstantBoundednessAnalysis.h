#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace marid {

class ConstantBoundednessAnalysis {
public:
  explicit ConstantBoundednessAnalysis(mlir::Operation *op);

  bool isConstantBounded() const { return constantBounded; }

  bool invalidate(mlir::Operation *op,
                  const mlir::AnalysisManager::PreservedAnalyses &pa);

private:
  bool constantBounded;
};

} // namespace marid
