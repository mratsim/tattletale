## Unsanctioned helpers, pending rationalization.
##
## Helpers landing here bypass the sanctioned op surface, the funnel pass
## later promotes, keeps, or kills each one.
##
## - nothing outside int_tuples imports this file
## - sanctioned modules re-export what survives rationalization
import std/macros, std/typetraits

macro groupedHead(head, tail: typed): untyped =
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
