// tests/treeification/nested_if.mlir
func.func @nested(%cond1: i1, %cond2: i1, %val: i32) -> i32 {
  // Allocation before any control flow
  %b0 = memref.alloc() : memref<32xi8>

  %r = scf.if %cond1 -> (i32) {
    // Allocation in outer then-branch
    %b1 = memref.alloc() : memref<16xi8>

    %inner = scf.if %cond2 -> (i32) {
      // Allocation in inner then-branch
      %b2 = memref.alloc() : memref<8xi8>

      %v1 = arith.addi %val, %val : i32
      scf.yield %v1 : i32
    } else {
      // Allocation in inner else-branch
      %b3 = memref.alloc() : memref<8xi8>

      %v2 = arith.muli %val, %val : i32
      scf.yield %v2 : i32
    }

    scf.yield %inner : i32
  } else {
    // Allocation in outer else-branch
    %b4 = memref.alloc() : memref<24xi8>

    %v3 = arith.subi %val, %val : i32
    scf.yield %v3 : i32
  }

  // Allocation after control flow (continuation)
  %b5 = memref.alloc() : memref<4xi8>

  %res = arith.addi %r, %val : i32
  return %res : i32
}
