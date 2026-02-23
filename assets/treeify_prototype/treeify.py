from dataclasses import dataclass
from typing import Optional

import itertools

_id_gen = itertools.count()


class Instruction:
    def __init__(self, name):
        self.name = name
        self.id = next(_id_gen)


class SimpleInst(Instruction):
    def __init__(self, name, next_inst=None):
        super().__init__(name)
        self.next_inst = next_inst


class BranchInst(Instruction):
    def __init__(self, name, then_inst, else_inst):
        super().__init__(name)
        self.then_inst = then_inst
        self.else_inst = else_inst

def print_cfg(inst, indent=0, visited=None):
    """
    Pretty-print a CFG.
    `visited` is used only to avoid infinite printing in the presence of joins.
    """
    if inst is None:
        return

    if visited is None:
        visited = set()

    prefix = "  " * indent
    print(f"{prefix}{inst.name}#{inst.id}")

    # Detect joins (same node reached twice)
    if inst in visited:
        print(f"{prefix}  (join)")
        return

    visited.add(inst)

    if isinstance(inst, BranchInst):
        print(f"{prefix}  then:")
        print_cfg(inst.then_inst, indent + 2, visited)
        print(f"{prefix}  else:")
        print_cfg(inst.else_inst, indent + 2, visited)

    elif isinstance(inst, SimpleInst):
        print_cfg(inst.next_inst, indent + 1, visited)


def duplicate(inst: Optional[Instruction]) -> Optional[Instruction]:
    """
    Recursively clones the CFG rooted at `inst`, duplicating all paths.
    The result is a tree-shaped CFG with no join nodes.
    """
    if inst is None:
        return None

    if isinstance(inst, BranchInst):
        return BranchInst(
            name=inst.name,
            then_inst=duplicate(inst.then_inst),
            else_inst=duplicate(inst.else_inst),
        )

    if isinstance(inst, SimpleInst):
        return SimpleInst(
            name=inst.name,
            next_inst=duplicate(inst.next_inst),
        )

    raise TypeError(f"Unknown instruction type: {type(inst)}")

def treeify(cfg_entry: Instruction) -> Instruction:
    """
    Takes the entry instruction of a fully unrolled CFG and
    returns the root of a tree-shaped CFG with no joins.
    """
    return duplicate(cfg_entry)

# Exit
exit_inst = SimpleInst("exit")

# Join node
join_inst = SimpleInst("join", exit_inst)

# Then / else paths
then_inst = SimpleInst("then", join_inst)
else_inst = SimpleInst("else", join_inst)

# Branch
branch_inst = BranchInst("if", then_inst, else_inst)

# Entry
entry = SimpleInst("entry", branch_inst)

print("Original CFG:")
print_cfg(entry)

tree = treeify(entry)

print("\nTreeified CFG:")
print_cfg(tree)
