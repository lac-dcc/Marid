module {
  func.func @test(%n : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %n step %c1 {
    }

    return
  }
}
