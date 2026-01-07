#pragma once

#include "mlir/Pass/Pass.h"

namespace marid {

std::unique_ptr<mlir::Pass> createLoopExpansionPass();

} // namespace marid
