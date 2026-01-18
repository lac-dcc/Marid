module {
  func.func @defrag_required() {
    // BIG: 10 bytes (i801), no uses. That's to check if it will prevent
    // defragmentation from happening:
    %big = arith.constant 0 : i80

    // a : 2 bytes
    %a = arith.constant 1 : i16

    // b : 2 bytes
    %b = arith.constant 2 : i16

    // c : 2 bytes
    %c = arith.constant 3 : i16

    // last use of b
    %ub = arith.addi %b, %b : i16

    // D : 4 bytes
    %d = arith.constant 4 : i32

    // keep a and c live
    %ua = arith.addi %a, %a : i16
    %uc = arith.addi %c, %c : i16

    // use D
    %ud = arith.addi %d, %d : i32

    return
  }
}
