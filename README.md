<p align="center">
  <img alt="Project Banner" src="assets/images/Banner.png" width="95%" height="auto"/></br>
</p>


Marid is an MLIR-based static analysis and pass framework for reasoning about
memory allocation and boundedness properties of programs.

## Features

- Constant-boundedness analysis for SCF-based MLIR programs
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

## Pipeline Overview

Marid is an MLIR-based transformation pipeline that converts structured programs into a **Tree-shaped Control Flow Graph (CFG)**. By expanding all loops and duplication-based treeification of conditionals, Marid ensures that every unique execution path in a function is represented by its own linear sequence of basic blocks ending in a unique return statement.

The transformation follows a three-stage process:

1. **Analysis**: Verify that the program has finite, predictable complexity.
2. **Expansion**: Flatten all loops into sequences of conditional logic.
3. **Treeification**: Duplicates code following branches to eliminate join points in the CFG.

---

## 1. Constant Boundedness Analysis

Before transformation, Marid ensures the program is "Constant-Bounded." This means the execution time and path count are statically determinable.

**Rules:**

* No `scf.while` loops (as termination depends on runtime data).
* `scf.for` loops must have `constant` lower bounds, upper bounds, and steps.

**Pseudo-code:**

```python
def check_constant_boundedness(module):
    for op in module:
        if op is scf.while:
            return False # Non-deterministic termination
        
        if op is scf.for:
            if not (is_constant(op.lb) and 
                    is_constant(op.ub) and 
                    is_constant(op.step)):
                return False # Trip count must be known at compile time
    return True

```

---

## 2. Loop Expansion

Once verified, Marid unrolls all `scf.for` loops. Because the bounds are constants, the loop is completely replaced by  copies of its body, where  is the trip count.

**Pseudo-code:**

```python
def expand_loops(module):
    # Process innermost loops first to handle nested expansion
    for for_op in module.walk_post_order(scf.for):
        trip_count = (for_op.ub - for_op.lb) / for_op.step
        
        for i in range(trip_count):
            iv_value = for_op.lb + (i * for_op.step)
            # Map the loop induction variable to the constant value
            mapping = {for_op.induction_var: iv_value}
            # Clone the entire body into the parent block
            clone_body(for_op.body, mapping)
        
        for_op.erase()

```

---

## 3. Treeification

The final stage converts a Directed Acyclic Graph (DAG) of conditionals into a strict Tree. It eliminates **Join Points** (blocks where control flow merges) by duplicating the "continuation" logic (the code following the branch) into both the `then` and `else` paths.

**Pseudo-code:**

```python
def treeify_module(func):
    # Repeat until no structured conditionals remain
    while func.contains(scf.if):
        # Always pick an 'if' that is a direct child of the function (Top-Down)
        # This ensures we expand the tree from the root to the leaves.
        if_op = func.get_top_level_if()
        
        # 1. Isolate the 'tail' (continuation) logic
        # Everything after if_op in the current block is moved to a temp block
        tail_block = split_block_after(if_op)
        
        # 2. Create fresh CFG blocks for the 'then' and 'else' paths
        then_path = func.add_block()
        else_path = func.add_block()
        
        # 3. Populate each path (The Duplication Step)
        for branch in [if_op.then_region, if_op.else_region]:
            dest = then_path if branch is then_region else else_path
            
            # A. Clone branch-specific logic
            clone_ops(branch, dest)
            
            # B. Clone the tail (continuation)
            # If the tail contained nested 'if' ops, they are now cloned here.
            # They will be processed in the next iteration of the 'while' loop.
            clone_ops(tail_block, dest)
            
        # 4. Replace the structured 'if' with a primitive conditional branch
        replace_cond_br(if_op, target_true=then_path, target_false=else_path)
        
        # 5. Cleanup
        tail_block.erase()
        if_op.erase()
```

---

## CFG Properties

After running the Marid pipeline, the resulting MLIR module satisfies the following properties:

1. **No Loops**: The CFG is a Directed Acyclic Graph.
2. **No Join Points**: Every block (except the entry block) has exactly one predecessor.
3. **Path Isolation**: Every `return` statement in the function represents a unique, independent execution trace.

## Usage

```bash
./bin/marid-opt program.mlir

```
