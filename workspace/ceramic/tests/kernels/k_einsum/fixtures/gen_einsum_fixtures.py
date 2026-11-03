#!/usr/bin/env python3
"""Einsum fixture generator: python + numpy, raw binary, one directory per case.

Each fixture directory under this script's directory contains:

- meta.json: recipe, the einsum statement, per-operand shape/strides,
  output shape/strides, xorhash checksums of every file
- one raw little-endian float32 .bin per operand, the storage-order elements
  of the logical array over a C-order or Fortran-order host layout
- o.bin, the expected output in raw float32, C order

The reference runs in float64 end to end: the operand bins contain
the float64 values cast to float32. np.einsum computes the expected output from the same
float64 arrays, so the kernel replays f32 inputs against the f64 truth.

Run:  uv run gen_einsum_fixtures.py [name ...]   (default: all fixtures)
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def xorhash_checksum(arr):
    """Xor over the raw bit patterns, order-independent, exact."""
    return int(np.bitwise_xor.reduce(arr.ravel().view(np.uint32)))


def write_bin(path, arr):
    arr.tofile(path)


def logical_array(rng, shape):
    return rng.standard_normal(shape)


def store_array(a, order):
    """Float32 storage of the logical array `a` in the given order.

    - rm: C order, strides are the C-order prefix products
    - cm: Fortran order, stored as the contiguous transpose, read back
      with the transposed strides

    The cm storage exercises the einsum relabel path. Permutations are built
    by copying (np.ascontiguousarray), never through strided views
    combined with in-place writes.
    """
    if order == "rm":
        return np.ascontiguousarray(a).astype(np.float32)
    assert order == "cm"
    return np.ascontiguousarray(a.T).astype(np.float32)


def strides_for(shape, order):
    """Element strides of a contiguous C-order ("rm") or Fortran-order ("cm")
    layout over the float64 logical shape."""
    layout = np.empty(shape, dtype=np.float64,
                      order={"rm": "C", "cm": "F"}[order])
    return [s // layout.itemsize for s in layout.strides]


# Recipe spec keys:
# - sub: the np.einsum subscript
# - stmt: the einsum statement the kernel replays
# - operands: (file, logical shape) pairs
# - out: the output logical shape
RECIPES = {
    "transpose": dict(
        stmt="D[j,i] = A[i,j]", sub="ij->ji",
        operands=[("a.bin", (3, 4))], out=(4, 3)),
    "sum_total": dict(
        stmt="S = A[i,j]", sub="ij->",
        operands=[("a.bin", (3, 4))], out=(1, 1)),
    "sum_row": dict(
        stmt="D[i] = A[i,j]", sub="ij->i",
        operands=[("a.bin", (3, 4))], out=(3,)),
    "sum_col": dict(
        stmt="D[j] = A[i,j]", sub="ij->j",
        operands=[("a.bin", (3, 4))], out=(4,)),
    "matvec": dict(
        stmt="D[i] = A[i,j] * v[j]", sub="ij,j->i",
        operands=[("a.bin", (3, 4)), ("v.bin", (4,))], out=(3,)),
    "mm": dict(
        stmt="D[i,k] = A[i,j] * B[j,k]", sub="ij,jk->ik",
        operands=[("a.bin", (3, 4)), ("b.bin", (4, 5))], out=(3, 5)),
    "dot": dict(
        stmt="d = u[i] * v[i]", sub="i,i->",
        operands=[("u.bin", (5,)), ("v.bin", (5,))], out=(1, 1)),
    "hadamard": dict(
        stmt="D[i,j] = A[i,j] * B[i,j]", sub="ij,ij->ij",
        operands=[("a.bin", (3, 4)), ("b.bin", (3, 4))], out=(3, 4)),
    "outer": dict(
        stmt="D[i,j] = u[i] * v[j]", sub="i,j->ij",
        operands=[("u.bin", (5,)), ("v.bin", (7,))], out=(5, 7)),
    "contraction_abc": dict(
        stmt="D[a,c] = A[a,b,c] * B[c,b]", sub="abc,cb->ac",
        operands=[("a.bin", (2, 3, 4)), ("b.bin", (4, 3))], out=(2, 4)),
    "contraction_2bound": dict(
        stmt="D[i,l] = A[i,j,k] * B[k,j,l]", sub="ijk,kjl->il",
        operands=[("a.bin", (2, 3, 4)), ("b.bin", (4, 3, 5))], out=(2, 5)),
    "contraction_3op": dict(
        stmt="D[a,b] = A[a,b,c] * B[c] * C[b,c]", sub="abc,c,bc->ab",
        operands=[("a.bin", (2, 3, 4)), ("b.bin", (4,)), ("c.bin", (3, 4))],
        out=(2, 3)),
    "contraction_batched": dict(
        stmt="D[b,i,k] = A[b,i,j] * B[b,j,k]", sub="bij,bjk->bik",
        operands=[("a.bin", (2, 3, 4)), ("b.bin", (2, 4, 5))], out=(2, 3, 5)),
}

# one entry per fixture: (name, recipe, storage order per operand)
CASES = [
    ("transpose_rm", "transpose", ["rm"]),
    ("transpose_cm", "transpose", ["cm"]),
    ("sum_total_rm", "sum_total", ["rm"]),
    ("sum_row_rm", "sum_row", ["rm"]),
    ("sum_row_cm", "sum_row", ["cm"]),
    ("sum_col_rm", "sum_col", ["rm"]),
    ("sum_col_cm", "sum_col", ["cm"]),
    ("matvec_cm", "matvec", ["cm", "rm"]),
    ("mm_rm", "mm", ["rm", "rm"]),
    ("mm_cm", "mm", ["cm", "rm"]),
    ("dot_rm", "dot", ["rm", "rm"]),
    ("hadamard_cm", "hadamard", ["rm", "cm"]),
    ("outer_rm", "outer", ["rm", "rm"]),
    ("contraction_abc_cm", "contraction_abc", ["rm", "cm"]),
    ("contraction_2bound_cm", "contraction_2bound", ["rm", "cm"]),
    ("contraction_3op_cm", "contraction_3op", ["cm", "rm", "rm"]),
    ("contraction_batched_rm", "contraction_batched", ["rm", "rm"]),
]


def make_fixture(name, recipe, orders, seed):
    spec = RECIPES[recipe]
    rng = np.random.default_rng(seed)
    arrays = [logical_array(rng, shape) for _, shape in spec["operands"]]

    # Reference input: the float64 values that the script stores in the bins as float32
    expected = np.einsum(spec["sub"], *arrays).reshape(spec["out"])

    out = os.path.join(HERE, name)
    os.makedirs(out, exist_ok=True)
    meta = {
        "name": name,
        "recipe": recipe,
        "dtype": "float32",
        "stmt": spec["stmt"],
        "operands": [],
        "out": {
            "shape": list(spec["out"]),
            "strides": strides_for(list(spec["out"]), "rm"),
        },
        "checksum": xorhash_checksum(expected.astype(np.float32)),
    }
    xorhashes = {}
    for (file, shape), a, order in zip(spec["operands"], arrays, orders):
        stored = store_array(a, order)
        write_bin(os.path.join(out, file), stored)
        entry = {
            "file": file,
            "shape": list(shape),
            "strides": strides_for(list(shape), order),
        }
        meta["operands"].append(entry)
        xorhashes[file] = xorhash_checksum(stored)
    write_bin(os.path.join(out, "o.bin"), expected.astype(np.float32))
    xorhashes["o.bin"] = xorhash_checksum(expected.astype(np.float32))
    meta["xorhash_checksums"] = xorhashes
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name}: recipe={recipe} stmt='{spec['stmt']}' "
          f"orders={orders} out_shape={spec['out']}")


def main():
    names = sys.argv[1:]
    wanted = [c for c in CASES if not names or c[0] in names]
    for i, (name, recipe, orders) in enumerate(wanted):
        make_fixture(name, recipe, orders, seed=100 + i)
    print(f"{len(wanted)} fixtures written")


if __name__ == "__main__":
    main()
