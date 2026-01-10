// tests/treeification/nested_if.mlir
func.func @nested(%cond1: i1, %cond2: i1, %val: i32) -> i32 {
  %r = scf.if %cond1 -> (i32) {
    %inner = scf.if %cond2 -> (i32) {
      %v1 = arith.addi %val, %val : i32
      scf.yield %v1 : i32
    } else {
      %v2 = arith.muli %val, %val : i32
      scf.yield %v2 : i32
    }
    scf.yield %inner : i32
  } else {
    %v3 = arith.subi %val, %val : i32
    scf.yield %v3 : i32
  }
  %res = arith.addi %r, %val : i32
  return %res : i32
}
