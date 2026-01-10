// tests/treeification/sequential_if.mlir
func.func @sequential(%c1: i1, %c2: i1, %arg: i32) -> i32 {
  // First branch
  %r1 = scf.if %c1 -> (i32) {
    %a = arith.addi %arg, %arg : i32
    scf.yield %a : i32
  } else {
    %b = arith.subi %arg, %arg : i32
    scf.yield %b : i32
  }

  // Second branch (in the continuation of the first)
  %r2 = scf.if %c2 -> (i32) {
    %c = arith.muli %r1, %arg : i32
    scf.yield %c : i32
  } else {
    %d = arith.divsi %r1, %arg : i32
    scf.yield %d : i32
  }

  return %r2 : i32
}
