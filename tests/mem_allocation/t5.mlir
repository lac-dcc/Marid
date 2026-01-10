func.func @f(%x: i32) {
  %y = arith.addi %x, %x : i32

  %buf = memref.alloc() : memref<32xi8>

  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i8

  // Store into the buffer
  memref.store %c42, %buf[%c0] : memref<32xi8>

  // Load from the buffer
  %v = memref.load %buf[%c0] : memref<32xi8>

  return
}
