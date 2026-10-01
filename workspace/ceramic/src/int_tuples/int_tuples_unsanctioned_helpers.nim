## Unsanctioned helpers, pending rationalization.
##
## Helpers landing here bypass the sanctioned op surface, the funnel pass
## later promotes, keeps, or kills each one.
##
## - sanctioned modules may import this file while a helper awaits its home
## - what survives rationalization is re-exported through the family module
import std/macros, std/typetraits
import ./int_tuples_compiletime
import ./int_tuples_streams

macro groupedHead*(head, tail: typed): untyped =
  ## Tuple (head, tail[0], tail[1], ...) with head verbatim so nesting survives,
  ## tail top-level elements unpacked one level, a scalar tail kept whole.
  ## tiled_divide/tiled_product reassembly, CuTe `result(_, repeat<R1>(_))` as a slice.
  result = nnkTupleConstr.newTree(head)
  let tt = tail.getTypeInst()
  if tt.kind notin {nnkTupleTy, nnkTupleConstr} or tt.len == 0:
    result.add tail
    return
  for i in 0 ..< tt.len:
    result.add nnkBracketExpr.newTree(tail, newLit(i))

func tupleType*(n: NimNode): NimNode {.compileTime.} =
  ## Resolve to the underlying TupleConstr node, handling values, consts,
  ## and type aliases uniformly.
  let t = n.getType()
  let inner =
    if t.kind == nnkBracketExpr and t[0].eqIdent("typeDesc"):
      t[1]
    else:
      t
  inner.getTypeImpl()

func tupleTypeLen*(n: NimNode): int {.compileTime.} =
  ## Length of the resolved tuple type node, callers guard with the tuple-kind check first.
  let t = n.tupleType()
  result = t.len

func toSeqStaticInts*(t: NimNode): seq[int] {.compileTime.} =
  ## Recursively extract Int[N] values from a (possibly nested) tuple type AST node.
  ## Returns low(int) (DynamicSentinel) for non-static (dynamic int) elements.
  ##
  ## Handles:
  ## - ((Int[1], Int[16]), (Int[512], Int[64]))  → @[1, 16, 512, 64]
  ## - ((int, int), (int, int))                  → @[DynamicSentinel, DynamicSentinel, ...]
  ## - Int[64]                                   → @[64]
  for (_, ty) in t.tupleStream().leaves():
    result.add ty.getStaticInt()
