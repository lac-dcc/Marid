func.func @last_use_branches(%cond: i1) {
  // def x
  %x = arith.constant 1 : i32

  // def y
  %y = arith.constant 2 : i32

  // def z
  %z = arith.constant 3 : i32

  scf.if %cond {
    // use(x)  <-- last usage of x in THEN
    %u1 = arith.addi %x, %x : i32

    // use(y)  <-- last usage of y
    %u2 = arith.addi %y, %y : i32
  } else {
    // use(z)
    %u3 = arith.addi %z, %z : i32

    // use(x)  <-- last usage of x in ELSE
    %u4 = arith.addi %x, %x : i32
  }

  // use(z)  <-- last usage of z (after join)
  %u5 = arith.addi %z, %z : i32

  return
}
