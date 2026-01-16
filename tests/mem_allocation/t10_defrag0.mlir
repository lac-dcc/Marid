module {
  func.func @defrag_required(%cond: i1) {
    // Three buffers with distinct sizes
    %a = memref.alloc() : memref<2xi32>   // a = 8 bytes
    %b = memref.alloc() : memref<2xi32>   // b = 8 bytes
    %c = memref.alloc() : memref<2xi32>   // c = 8 bytes

    // Artificial use of all to keep them live
    %i0 = arith.constant 0 : index
    %v0 = arith.constant 0 : i32
    // memref.store %v0, %a[%i0] : memref<2xi32>
    // memref.store %v0, %b[%i0] : memref<2xi32>
    // memref.store %v0, %c[%i0] : memref<2xi32>

    // Last use of b:
    memref.store %v0, %b[%i0] : memref<2xi32>

    // At this point:
    //   a and c are still live
    //   b is dead → creates a hole between a and c

    // Now we need D with a size that is twice b's:
    %d = memref.alloc() : memref<4xi32>   // D = 16 bytes

    // Use a and c again so they cannot be freed
    memref.store %v0, %a[%i0] : memref<2xi32>
    memref.store %v0, %c[%i0] : memref<2xi32>

    // Use D
    memref.store %v0, %d[%i0] : memref<4xi32>

    return
  }
}
