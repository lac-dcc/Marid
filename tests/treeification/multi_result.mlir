// tests/treeification/multi_result.mlir
func.func @multi_result(%x: i1, %a: i32, %b: i32) -> i32 {
  // scf.if returning two values
  %res:2 = scf.if %x -> (i32, i32) {
    %sum = arith.addi %a, %b : i32
    %diff = arith.subi %a, %b : i32
    scf.yield %sum, %diff : i32, i32
  } else {
    %mul = arith.muli %a, %b : i32
    %div = arith.divsi %a, %b : i32
    scf.yield %mul, %div : i32, i32
  }
  
  // Uses both results in the continuation
  %final = arith.addi %res#0, %res#1 : i32
  return %final : i32
}
