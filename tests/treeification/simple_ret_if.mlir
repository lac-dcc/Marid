func.func @simple(%x: i1, %a: i32) -> i32 {
  %r = scf.if %x -> (i32) {
    %v = arith.addi %a, %a : i32
    scf.yield %v : i32
  } else {
    %v = arith.subi %a, %a : i32
    scf.yield %v : i32
  }
  return %r : i32
}
