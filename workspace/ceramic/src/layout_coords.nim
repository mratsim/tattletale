# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ═══════════════════════════════════════════════════════════════
#  Joker — CuTe Underscore: keep/slice marker for coordinates
# ═══════════════════════════════════════════════════════════════

type Joker* = object

type CoordType* = int | Int | Joker | tuple
  ## Marker type for "keep this dimension" in slice/dice.
  ## Analogous to CuTe's `Underscore` / `_`.
  ## Use `_` as marker in macro/template context.

template `_`*: Joker = Joker()

func `$`*(x: Joker): string = "_"
func crd2idx*(coord: Joker; shape: int): int = 0
  ## Joker contributes 0 to indexing.
func crd2idx*[V: static int](coord: Joker; shape: Int[V]): int = 0
func crd2idx*(coord: Joker; shape, stride: int): int = 0
func crd2idx*[V: static int](coord: Joker; shape, stride: Int[V]): int = 0
func `*`*(c: Joker; s: int): int = 0
func `*`*(s: int; c: Joker): int = 0
func `*`*[V: static int](c: Joker; s: Int[V]): int = 0
func `*`*[V: static int](s: Int[V]; c: Joker): int = 0
func `+`*(c: Joker; s: int): int = s
func `+`*(s: int; c: Joker): int = s
func `+`*[V: static int](c: Joker; s: Int[V]): int = V
func `+`*[V: static int](s: Int[V]; c: Joker): int = V


func isJokerNode*(n: NimNode): bool {.compileTime.} =
  ## True if NimNode represents a Joker value.
  let t = n.getTypeInst()
  t.kind == nnkSym and $t == "Joker"

# ═══════════════════════════════════════════════════════════════
#  slice(coord, target) — keep elements paired with Joker
#  dice(coord, target)  — keep elements paired with int
# ═══════════════════════════════════════════════════════════════
##
## Both are compile-time tuple filtering operations.
## CuTe C++: underscore.hpp
##
## slice(_, b)           → b        (bare scalar, joker on scalar)
## slice(0, b)           → ()       (empty tuple)
## slice((_, 0), (a,b))  → (a,)     (keep a, drop b)
## dice(_, b)            → ()       (empty tuple)
## dice(0, b)            → b        (bare scalar, int on scalar)
## dice((0, _), (a,b))   → a        (keep a, drop b)

macro slice*(coord: CoordType; target: IntOrIntTuple): untyped =
  ## Walk `coord` and `target` in parallel. For each leaf:
  ##   - if coord is `_` → include that target element in the result
  ##   - if coord is an int → skip it
  ##
  ## runnableExamples:
  ##   slice((_, 0), (3, 4))    → keeps 3, drops 4 → (3,)
  ##   slice(((_, 0), (0, _)), ((2, 3), (4, 5)))
  ##     → keeps 2, then 5      → ((2,), 5)
  ##   slice(_, (42,))          → bare joker on 1-tuple → 42

  # Replace `_` identifiers with Joker() so `_` syntax works
  proc clense(n: NimNode): NimNode =
    if n.kind == nnkIdent and n.eqIdent("_"):
      result = newCall(bindSym"Joker")
    else:
      result = n.copyNimTree()
      for i in 0 ..< n.len:
        result[i] = clense(n[i])
  let c = clense(coord)
  let t = target

  # Collect all (coord_leaf, target_leaf_index_path) pairs
  # where target_leaf_index_path is the sequence of indices needed to access it
  proc collectLeaves(cNode: NimNode; path: seq[int]): seq[(NimNode, seq[int])] =
    if cNode.kind == nnkTupleConstr:
      for i in 0 ..< cNode.len:
        for pair in collectLeaves(cNode[i], path & i):
          result.add pair
    else:
      result.add (cNode, path)

  let leaves = collectLeaves(c, @[])

  # Build a flat tuple: for each joker leaf, add the target element at the path
  var parts: seq[NimNode] = @[]
  for (coordLeaf, path) in leaves:
    if isJokerNode(coordLeaf):
      # Keep: construct target path access like t[i][j]...
      var access = t
      for idx in path:
        access = newCall(bindSym"[]", access, newLit(idx))
      parts.add access
    # else: int -> drop (don't add)

  if parts.len == 0:
    result = nnkPar.newTree()  # empty tuple ()
  elif parts.len == 1 and c.kind != nnkTupleConstr:
    # Bare joker on scalar: return bare
    result = parts[0]
  else:
    # Tuple coord -> always tuple result
    result = nnkTupleConstr.newTree(parts)

macro dice*(coord: CoordType; target: IntOrIntTuple): untyped =
  ## Walk `coord` and `target` in parallel. For each leaf:
  ##   - if coord is an int → include that target element in the result
  ##   - if coord is `_` → skip it
  ##
  ## runnableExamples:
  ##   dice((_, 0), (3, 4))    → drops 3, keeps 4 → 4
  ##   dice(((0, _), (_, 0)), ((2, 3), (4, 5)))
  ##     → keeps 2, then 5    → (2, 5)
  ##   dice(0, (42,))          → bare int on 1-tuple → 42

  # Replace `_` identifiers with Joker()
  proc clense(n: NimNode): NimNode =
    if n.kind == nnkIdent and n.eqIdent("_"):
      result = newCall(bindSym"Joker")
    else:
      result = n.copyNimTree()
      for i in 0 ..< n.len:
        result[i] = clense(n[i])
  let c = clense(coord)
  let t = target
  # Validate: coord must contain Joker, int, Int[N], or tuples thereof
  # target must contain int, Int[N], or tuples thereof
  # (Implicitly checked by the fact we only generate valid accesses)

  # Collect all (coord_leaf, target_leaf_index_path) pairs
  proc collectLeaves(cNode: NimNode; path: seq[int]): seq[(NimNode, seq[int])] =
    if cNode.kind == nnkTupleConstr:
      for i in 0 ..< cNode.len:
        for pair in collectLeaves(cNode[i], path & i):
          result.add pair
    else:
      result.add (cNode, path)

  let leaves = collectLeaves(c, @[])

  # Build result: for each int leaf, add target element; for joker, drop
  var parts: seq[NimNode] = @[]
  for (coordLeaf, path) in leaves:
    if not isJokerNode(coordLeaf):
      # Keep: construct target path access
      var access = t
      for idx in path:
        access = newCall(bindSym"[]", access, newLit(idx))
      parts.add access

  if parts.len == 0:
    result = nnkPar.newTree()  # empty
  elif parts.len == 1 and c.kind != nnkTupleConstr:
    # Bare int/joker on scalar: return bare
    result = parts[0]
  else:
    # Tuple coord -> always tuple result
    result = nnkTupleConstr.newTree(parts)
