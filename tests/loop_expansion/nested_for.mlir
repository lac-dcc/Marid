// RUN: marid-opt -marid-expand-loops %s | FileCheck %s

module {
  func.func @nested() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %x = arith.constant 7 : i32
      }
    }

    return
  }
}

// CHECK-NOT: scf.for
// CHECK: arith.constant 7 : i32
// CHECK: arith.constant 7 : i32
// CHECK: arith.constant 7 : i32
// CHECK: arith.constant 7 : i32
