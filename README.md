<p align="center">
  <img alt="Project Banner" src="assets/images/Banner.png" width="95%" height="auto"/></br>
</p>

Marid is an MLIR-based static analysis and transformation framework for reasoning
about **boundedness**, **control flow structure**, and **memory allocation**
in programs with statically predictable behavior.

## Features

- Constant-boundedness analysis for SCF-based MLIR programs
- Loop expansion for statically bounded iteration spaces
- Treeification of structured control flow into a tree-shaped CFG
- Simple stack-based memory allocation for constant-bounded programs
- MLIR Analysis Framework integration
- Checker pass with diagnostic support
- Standalone driver tool (`marid-opt`)

## Building

```bash
mkdir build
cd build
cmake ..
make
```

---

## Usage

```bash
./bin/marid-opt program.mlir
```

The tool prints:

* diagnostics from the boundedness checker,
* the memory allocation report,
* followed by the (unchanged) transformed MLIR module.

## Pipeline Overview

Marid implements an MLIR-based transformation pipeline that converts structured
programs into a **Tree-shaped Control Flow Graph (CFG)** and then performs
a **simple, deterministic memory allocation** over the resulting program.

By expanding all loops and duplicating control-flow continuations,
Marid ensures that every possible execution path is represented by its own
linear sequence of basic blocks ending in a unique `return` statement.

The transformation follows a four-stage process:

1. **Analysis**: Verify that the program has finite, predictable complexity.
2. **Expansion**: Flatten all loops into sequences of straight-line code.
3. **Treeification**: Eliminate join points by duplicating control flow.
4. **Memory Allocation**: Assign fixed stack locations to all values and buffers.

---

## 1. Constant Boundedness Analysis

Before any transformation, Marid verifies that the program is **constant-bounded**.
Intuitively, this means that both execution time and the number of execution paths
are statically known.

### Rules

* No `scf.while` loops (termination depends on runtime data).
* `scf.for` loops must have:

  * constant lower bounds,
  * constant upper bounds,
  * constant step sizes.

### Pseudo-code

```python
def check_constant_boundedness(module):
    for op in module:
        if op is scf.while:
            return False  # Non-deterministic termination
        
        if op is scf.for:
            if not (is_constant(op.lb) and 
                    is_constant(op.ub) and 
                    is_constant(op.step)):
                return False  # Trip count must be known at compile time
    return True
```

---

## 2. Loop Expansion

Once a program is proven constant-bounded, Marid expands all `scf.for` loops.
Because loop bounds are constant, each loop can be fully unrolled at compile time.

The loop body is cloned once per iteration, with the induction variable replaced
by the corresponding constant value.

### Pseudo-code

```python
def expand_loops(module):
    # Process innermost loops first to handle nesting
    for for_op in module.walk_post_order(scf.for):
        trip_count = (for_op.ub - for_op.lb) / for_op.step
        
        for i in range(trip_count):
            iv_value = for_op.lb + (i * for_op.step)
            mapping = {for_op.induction_var: iv_value}
            clone_body(for_op.body, mapping)
        
        for_op.erase()
```

---

## 3. Treeification

After loop expansion, the program may still contain structured conditionals
(`scf.if`) whose control flow *joins* after the branch.
This results in a **Directed Acyclic Graph (DAG)**.

Treeification transforms this DAG into a **strict tree** by eliminating join points.
It does so by **duplicating the continuation logic** (the code following a branch)
into each branch path.

After this transformation:

* The CFG has no join points.
* Every basic block (except the entry block) has exactly one predecessor.
* Every `return` corresponds to a unique execution path.

### Pseudo-code

```python
def treeify_module(func):
    while func.contains(scf.if):
        if_op = func.get_top_level_if()
        
        # 1. Isolate the continuation logic
        tail_block = split_block_after(if_op)
        
        # 2. Create fresh CFG blocks
        then_path = func.add_block()
        else_path = func.add_block()
        
        # 3. Duplicate logic into each path
        for branch in [if_op.then_region, if_op.else_region]:
            dest = then_path if branch is then_region else else_path
            clone_ops(branch, dest)
            clone_ops(tail_block, dest)
        
        # 4. Replace structured control flow
        replace_cond_br(if_op, then_path, else_path)
        
        # 5. Cleanup
        tail_block.erase()
        if_op.erase()
```

---

## 4. Simple Memory Allocation

Once the CFG has been fully treeified, the program consists solely of
straight-line execution paths with no loops and no join points.
This makes memory allocation straightforward.

Marid implements a **simple stack-based memory allocator** with the following
properties:

* Each SSA value and memory buffer is assigned a **unique, fixed stack range**.
* Allocation sizes are derived from statically known types.
* No reuse or lifetime analysis is performed.
* The allocator is **deterministic** and **purely compile-time**.

This pass does **not** modify the IR. Instead, it computes a memory layout
and prints a human-readable report.

### Example Output

```text
Stack size: 40 Bytes
----
Allocation:
[0, 3]   <block argument> of type 'i32' at index: 0
[4, 7]   %0 = arith.addi %arg0, %arg0 : i32
[8, 39]  %alloc = memref.alloc() : memref<32xi8>
----
```

This allocator serves as a foundation for future work, such as:

* lifetime-aware stack reuse,
* lowering `memref.alloc` to explicit stack offsets,
* backend-specific code generation,
* or formal reasoning about memory safety.

---

## CFG Properties After the Pipeline

After running the full Marid pipeline, the resulting MLIR module satisfies:

1. **No Loops**: The CFG is acyclic.
2. **No Join Points**: Each block has at most one predecessor.
3. **Path Isolation**: Each `return` corresponds to one execution trace.
4. **Static Memory Layout**: All values have fixed, statically known locations.
