#ifndef MARID_PASSES_TREEIFICATION_PASS_H
#define MARID_PASSES_TREEIFICATION_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace marid {

/// Creates a pass that transforms structured control flow into a tree
/// by eliminating join points through duplication.
std::unique_ptr<mlir::Pass> createTreeificationPass();

} // namespace marid

#endif // MARID_PASSES_TREEIFICATION_PASS_H
