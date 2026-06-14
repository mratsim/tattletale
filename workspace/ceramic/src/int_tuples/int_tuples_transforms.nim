# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros, std/typetraits
import ./int_tuples_datatypes

# ═══════════════════════════════════════════════════════════════
#  flatten — recursively collect leaf elements of a tuple
# ═══════════════════════════════════════════════════════════════

macro flattenImpl(t: IntOrIntTuple): untyped =
  let tNode = t
  let ttype = tNode.getTypeImpl()

  proc isLeaf(t: NimNode): bool =
    (t.kind == nnkSym and $t == "int") or
    (t.kind == nnkBracketExpr and $t[0] == "Int")

  proc collect(acc: var NimNode; e: NimNode; t: NimNode) =
    if t.kind == nnkTupleConstr:
      for idx in 0 ..< t.len:
        let fd = t[idx]
        let fa = newTree(nnkBracketExpr, e, newLit(idx))
        if isLeaf(fd):
          acc.add fa
        else:
          collect(acc, fa, fd)
    else:
      acc.add e

  # nnkPar: single-item result stays scalar (avoids explicit `if result.len == 1`).
  # Multi-item: construct a tuple like nnkTupleConstr.
  result = newNimNode(nnkPar)
  collect(result, tNode, ttype)

proc flatten*(t: IntOrIntTuple): auto {.inline, noInit.}=
  ## Recursively collect leaf fields of a (possibly nested) tuple.
  ## Scalars and Int[N] are leaves; tuples are expanded.
  ##
  ## CuTe: `flatten(t)` → flat tuple of leaf elements.
  ##
  ## Examples:
  ##   flatten(5)          → 5
  ##   flatten((1,2,3))    → (1,2,3)
  ##   flatten((1,(2,3)))  → (1,2,3)
  flattenImpl(t)

proc flatten*(t: static IntOrIntTuple): static auto {.inline, noInit.} =
  ## Recursively collect leaf fields of a (possibly nested) tuple.
  ## Scalars and Int[N] are leaves; tuples are expanded.
  ##
  ## CuTe: `flatten(t)` → flat tuple of leaf elements.
  ##
  ## Examples:
  ##   flatten(5)          → 5
  ##   flatten((1,2,3))    → (1,2,3)
  ##   flatten((1,(2,3)))  → (1,2,3)
  ##
  ## Compile-time overload for full-compile-time input
  flattenImpl(t)

# ═══════════════════════════════════════════════════════════════
#  concat — concatenate two tuples or a scalar and a tuple
# ═══════════════════════════════════════════════════════════════
#
#  CuTe: `append(t, x)` / `prepend(t, x)` — concat is the unified version.
#  Replaces both append and prepend via overloads.
#
#  Overloads:
#    concat(a: int;        b: tuple)     — int + tuple
#    concat(a: static int; b: tuple)     — static int + tuple  (→ Int[N] concatenation)
#    concat(a: Int[N];     b: tuple)     — Int[N] + tuple
#    concat(a: tuple;      b: int)       — tuple + int
#    concat(a: tuple;      b: static int) — tuple + static int
#    concat(a: tuple;      b: Int[N])    — tuple + Int[N]
#    concat(a, b: tuple)                  — tuple + tuple
#    concat(a, b: int)                    — int + int  (→ tuple)
#    concat(a: int; b: static int)        — int + static int
#    concat(a: static int; b: int)        — static int + int
#    concat(a, b: static int)             — static int + static int
#    concat(a: Int[N]; b: Int[N])         — Int[N] + Int[N]
# ═══════════════════════════════════════════════════════════════

proc concat*(a: int; b: tuple): auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let bType = bNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    for idx in 0 ..< bType.len:
      result.add newTree(nnkBracketExpr, bNode, newLit(idx))
  concatImpl()

proc concat*(a: static int; b: tuple): auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let bType = bNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", aNode))
    for idx in 0 ..< bType.len:
      result.add newTree(nnkBracketExpr, bNode, newLit(idx))
  concatImpl()

proc concat*[V: static int](a: Int[V]; b: tuple): auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let bType = bNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    for idx in 0 ..< bType.len:
      result.add newTree(nnkBracketExpr, bNode, newLit(idx))
  concatImpl()

proc concat*(a: tuple; b: int): auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let aType = aNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    for idx in 0 ..< aType.len:
      result.add newTree(nnkBracketExpr, aNode, newLit(idx))
    result.add bNode
  concatImpl()

proc concat*(a: tuple; b: static int): auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let aType = aNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    for idx in 0 ..< aType.len:
      result.add newTree(nnkBracketExpr, aNode, newLit(idx))
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", bNode))
  concatImpl()

proc concat*[V: static int](a: tuple; b: Int[V]): auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let aType = aNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    for idx in 0 ..< aType.len:
      result.add newTree(nnkBracketExpr, aNode, newLit(idx))
    result.add bNode
  concatImpl()

proc concat*(a, b: tuple): auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    let aType = aNode.getTypeImpl(); let bType = bNode.getTypeImpl()
    result = newNimNode(nnkTupleConstr)
    for idx in 0 ..< aType.len:
      result.add newTree(nnkBracketExpr, aNode, newLit(idx))
    for idx in 0 ..< bType.len:
      result.add newTree(nnkBracketExpr, bNode, newLit(idx))
  concatImpl()

proc concat*(a, b: int): auto {.inline, noInit.} =
  ## int + int → (int, int) tuple
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    result.add bNode
  concatImpl()

proc concat*(a: int; b: static int): auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", bNode))
  concatImpl()

proc concat*(a: static int; b: int): auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", aNode))
    result.add bNode
  concatImpl()

proc concat*(a, b: static int): auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", aNode))
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", bNode))
  concatImpl()

proc concat*[V1, V2: static int](a: Int[V1]; b: Int[V2]): static auto {.inline, noInit.} =
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    result.add bNode
  concatImpl()

proc concat*[V: static int](a: Int[V] or int; b: int or Int[V]): auto {.inline, noInit.} =
  ## Int[N] + int tuple
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    result.add bNode
  concatImpl()

proc concat*[V: static int](a: Int[V]; b: static int): static auto {.inline, noInit.} =
  ## Int[N] + static int tuple (static int → Int[b])
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add aNode
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", bNode))
  concatImpl()

proc concat*[V: static int](a: static int; b: Int[V]): static auto {.inline, noInit.} =
  ## static int + Int[N] tuple (static int → Int[a])
  macro concatImpl(): untyped =
    let aNode = bindSym"a"; let bNode = bindSym"b"
    result = newNimNode(nnkTupleConstr)
    result.add newNimNode(nnkObjConstr).add(
      newNimNode(nnkBracketExpr).add(ident"Int", aNode))
    result.add bNode
  concatImpl()
