// tests/mem_allocation/two_outer_loops_if_alloc.mlir
module {
  func.func @nested() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true

    // ===== First outer loop =====
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %x = arith.constant 7 : i32

        scf.if %true {
          %b_then_0 = memref.alloc() : memref<8xi8>
        } else {
          %b_else_0 = memref.alloc() : memref<16xi8>
        }
      }
    }

    // ===== Second outer loop =====
    scf.for %k = %c0 to %c2 step %c1 {
      scf.for %l = %c0 to %c2 step %c1 {
        %y = arith.constant 13 : i32

        scf.if %true {
          %b_then_1 = memref.alloc() : memref<4xi8>
        } else {
          %b_else_1 = memref.alloc() : memref<32xi8>
        }
      }
    }

    return
  }
}

