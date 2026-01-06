module {
  func.func @test() {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c10 step %c1 {
      // empty loop body
    }

    return
  }
}
