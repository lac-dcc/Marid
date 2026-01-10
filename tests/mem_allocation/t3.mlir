// tests/mem_allocation/nested_for_if_alloc.mlir
module {
  func.func @nested() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true

    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %x = arith.constant 7 : i32

        scf.if %true {
          // Allocation in THEN branch
          %b_then = memref.alloc() : memref<8xi8>
        } else {
          // Allocation in ELSE branch
          %b_else = memref.alloc() : memref<16xi8>
        }
      }
    }

    return
  }
}

