module {
  func.func @join_escapes_loop(%cond: i1) {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32

    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index

    // Loop with a join inside
    %w = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %c1_i32) -> i32 {
      %v = scf.if %cond -> i32 {
        %t1 = arith.addi %acc, %c2_i32 : i32
        scf.yield %t1 : i32
      } else {
        %t2 = arith.addi %acc, %c3_i32 : i32
        scf.yield %t2 : i32
      }

      // Carry the joined value to the next iteration
      scf.yield %v : i32
    }

    // Value escapes the loop and is used here
    %u = arith.addi %w, %c3_i32 : i32
    return
  }
}
