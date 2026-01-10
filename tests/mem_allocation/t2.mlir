module {
  func.func @nested() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %x = arith.constant 7 : i32
        %b1 = memref.alloc(): memref<8xi8>
      }
    }

    return
  }
}
