func.func @f(%x: i32) {
  %y = arith.addi %x, %x : i32
  %buf = memref.alloc() : memref<32xi8>
  return
}
