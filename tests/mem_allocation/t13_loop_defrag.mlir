module {
  func.func @foo(%cond: i1) -> i16 {
    %a = arith.constant 1 : i16
    %b = arith.constant 2 : i16
    %c = arith.constant 3 : i16
    %D = arith.extsi %b : i16 to i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0 : i16
    %v1 = scf.for %i = %c0 to %c2 step %c1
             iter_args(%acc = %init) -> (i16) {
      %v0 = scf.if %cond -> (i16) {
        %v2 = arith.addi %a, %c : i16
        %v3 = arith.trunci %D : i32 to i16
        %v4 = arith.addi %v2, %v3 : i16
        scf.yield %v4 : i16
      } else {
        %u3 = arith.addi %c, %c : i16
        scf.yield %u3 : i16
      }
      scf.yield %v0 : i16
    }

    return %v1 : i16
  }
}
