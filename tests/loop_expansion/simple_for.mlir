// RUN: marid-opt -marid-expand-loops %s | FileCheck %s

module {
  func.func @simple() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c4 step %c1 {
      %x = arith.constant 42 : i32
    }

    return
  }
}

// CHECK-NOT: scf.for
// CHECK: arith.constant 42 : i32
// CHECK: arith.constant 42 : i32
// CHECK: arith.constant 42 : i32
// CHECK: arith.constant 42 : i32
