func.func @f(%x: i32) {
  %y = arith.addi %x, %x : i32
  %z = arith.addi %y, %y : i32
  return
}
