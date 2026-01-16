#ifndef MARID_PASSES_TREESCANDEFRAGMEMORYALLOCATIONPASS_H
#define MARID_PASSES_TREESCANDEFRAGMEMORYALLOCATIONPASS_H

#include "mlir/Pass/Pass.h"

namespace marid {

/// Creates the TreeScan memory allocation pass with defragmentation.
std::unique_ptr<mlir::Pass> createTreeScanDefragMemoryAllocationPass();

} // namespace marid

#endif // MARID_PASSES_TREESCANDEFRAGMEMORYALLOCATIONPASS_H
