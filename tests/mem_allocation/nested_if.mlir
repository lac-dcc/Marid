// tests/treeification/nested_if.mlir

// Declare printf from libc
llvm.func @printf(!llvm.ptr, ...) -> i32

// Helper: print an i32 followed by newline
func.func @print_i32(%val: i32) {
  %fmt = llvm.mlir.addressof @fmt_str : !llvm.ptr
  %ext = arith.extsi %val : i32 to i64
  llvm.call @printf(%fmt, %ext) vararg(!llvm.func<i32 (!llvm.ptr, ...)>) : (!llvm.ptr, i64) -> i32
  return
}

llvm.mlir.global internal constant @fmt_str("%d\0A\00") : !llvm.array<4 x i8>

func.func @nested(%cond1: i1, %cond2: i1, %val: i32) -> i32 {
  %b0 = memref.alloc() : memref<32xi8>
  %r = scf.if %cond1 -> (i32) {
    %b1 = memref.alloc() : memref<16xi8>
    %inner = scf.if %cond2 -> (i32) {
      %b2 = memref.alloc() : memref<8xi8>
      %v1 = arith.addi %val, %val : i32
      scf.yield %v1 : i32
    } else {
      %b3 = memref.alloc() : memref<8xi8>
      %v2 = arith.muli %val, %val : i32
      scf.yield %v2 : i32
    }
    scf.yield %inner : i32
  } else {
    %b4 = memref.alloc() : memref<24xi8>
    %v3 = arith.subi %val, %val : i32
    scf.yield %v3 : i32
  }
  %b5 = memref.alloc() : memref<4xi8>
  %res = arith.addi %r, %val : i32
  return %res : i32
}

func.func @main() -> i32 {
  %seven = arith.constant 7 : i32
  %t = arith.constant 1 : i1
  %f = arith.constant 0 : i1

  // cond1=1, cond2=1: addi(7,7)=14, +7 => 21
  %r0 = func.call @nested(%t, %t, %seven) : (i1, i1, i32) -> i32
  func.call @print_i32(%r0) : (i32) -> ()

  // cond1=1, cond2=0: muli(7,7)=49, +7 => 56
  %r1 = func.call @nested(%t, %f, %seven) : (i1, i1, i32) -> i32
  func.call @print_i32(%r1) : (i32) -> ()

  // cond1=0, cond2=*: subi(7,7)=0,  +7 => 7
  %r2 = func.call @nested(%f, %f, %seven) : (i1, i1, i32) -> i32
  func.call @print_i32(%r2) : (i32) -> ()

  %zero = arith.constant 0 : i32
  return %zero : i32
}
