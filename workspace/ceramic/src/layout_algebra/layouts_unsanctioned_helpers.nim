## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Unsanctioned helpers, pending rationalization.
##
## Helpers landing here bypass the sanctioned op surface, the funnel pass
## later promotes, keeps, or kills each one.
##
## - sanctioned modules may import this file while a helper awaits its home
## - what survives rationalization is re-exported through the family module
import std/macros
import workspace/ceramic/src/int_tuples

proc shapeRank*(shTyp: NimNode): int {.compileTime.} =
  ## Rank of a layout given its shape type node, tuple constr = element count, scalar = 1.
  if shTyp.kind == nnkTupleConstr:
    shTyp.len
  else:
    1

proc dimCount*(ty: NimNode): int {.compileTime.} =
  ## Top-level dimension count of a shape or layout type node.
  if ty.isTupleTy():
    ty.len
  else:
    1

# ── AST-level helpers (compile-time value extraction) ──

proc flattenAst*(n: NimNode): seq[NimNode] {.compileTime.} =
  case n.kind
  of nnkIntLit, nnkUIntLit:
    result.add n
  of nnkCall, nnkBracketExpr:
    # an Int[N]() call leaf stays whole, deeper call shapes drop silently
    if n.len >= 1 and $n[0] == "Int" and n[1].kind == nnkIntLit: result.add n
    else: discard
  of nnkPar, nnkTupleConstr, nnkArgList:
    for child in n:
      for leaf in flattenAst(child):
        result.add leaf
  else:
    discard

proc flattenType*(t: NimNode): seq[NimNode] {.compileTime.} =
  case t.kind
  of nnkTupleConstr:
    for child in t:
      for leaf in flattenType(child):
        result.add leaf
  else:
    result.add t

proc typeIntVals*(t: NimNode): seq[int] {.compileTime.} =
  ## Flattened leaf values of an Int tuple type, DynamicSentinel where a leaf is not a static Int.
  for leaf in flattenType(t):
    result.add leaf.getStaticInt()

proc litTuple*(vals: seq[int]): NimNode {.compileTime.} =
  ## Int literal tuple expression, scalar when single-valued.
  result = nnkPar.newNimNode()
  for v in vals:
    result.add newLit(v)
