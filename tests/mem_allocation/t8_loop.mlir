module {
  func.func @last_use_branches(%cond: i1) {
    %x = arith.constant 1 : i32
    %y = arith.constant 2 : i32
    %z = arith.constant 3 : i32

    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c2 step %c1 {
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
    }

    // use(z)  <-- last usage of z (after loop join)
    %u5 = arith.addi %z, %z : i32

    return
  }
}
