#ifndef MARID_PASSES_TREEIFICATION_PASS_H
#define MARID_PASSES_TREEIFICATION_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace marid {

/// Creates a pass that transforms structured control flow into a tree
/// by eliminating join points through duplication.
///
/// Invariant:
///
/// After treeification, no SSA value defined before the original scf.if
/// is used by operations shared across divergent control-flow paths.
/// All continuation logic is duplicated per path, producing a tree-shaped CFG.
std::unique_ptr<mlir::Pass> createTreeificationPass();

} // namespace marid

#endif // MARID_PASSES_TREEIFICATION_PASS_H
