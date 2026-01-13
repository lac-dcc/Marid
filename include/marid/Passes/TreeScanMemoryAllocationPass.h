#ifndef MARID_PASSES_TREESCANMEMORYALLOCATIONPASS_H
#define MARID_PASSES_TREESCANMEMORYALLOCATIONPASS_H

#include "mlir/Pass/Pass.h"

namespace marid {

/// Creates the TreeScan-based memory allocation pass.
std::unique_ptr<mlir::Pass> createTreeScanMemoryAllocationPass();

} // namespace marid

#endif // MARID_PASSES_TREESCANMEMORYALLOCATIONPASS_H
