module {
  func.func @nested() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true

    %c42 = arith.constant 42 : i8

    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %x = arith.constant 7 : i32

        scf.if %true {
          // Allocation in THEN branch
          %b_then = memref.alloc() : memref<8xi8>

          // Use the buffer
          memref.store %c42, %b_then[%c0] : memref<8xi8>
          %v_then = memref.load %b_then[%c0] : memref<8xi8>

        } else {
          // Allocation in ELSE branch
          %b_else = memref.alloc() : memref<16xi8>

          // Use the buffer
          memref.store %c42, %b_else[%c0] : memref<16xi8>
          %v_else = memref.load %b_else[%c0] : memref<16xi8>
        }
      }
    }

    return
  }
}
