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

# ═══════════════════════════════════════════════════════════════
#  layoutTypeArgs, layout dimensions+types extraction
# ═══════════════════════════════════════════════════════════════

func layoutTypeArgs*(layout: NimNode): tuple[shapeTy, strideTy: NimNode] {.compileTime.} =
  ## Extract the Layout type's shape and stride type nodes from a typed expression, resolving type aliases.
  ##
  ## Removability: a macro can often avoid this helper by destructuring
  ## the layout first and passing
  ## the destructured shape and stride to a typed macro:
  ##
  ## - the destructured nodes arrive semchecked
  ##   (the nested call's argument semcheck)
  ## - the inner fold works on nodes and per-leaf values directly
  ##   (coalesce took this route, complement's transplant candidate)
  ##
  ## Prefer the destructure route when the fold does not need
  ## the whole-profile type peek in one shot.
  let typ = layout.getTypeInst()
  if typ.kind == nnkBracketExpr and typ[0].eqIdent("Layout"):
    return (typ[1], typ[2])
  if typ.kind == nnkSym:
    let objTy = typ.getTypeImpl()
    if objTy.kind == nnkObjectTy:
      for field in objTy[2]:
        if field.kind == nnkIdentDefs and field[0].eqIdent("shape"):
          result.shapeTy = field[1]
        elif field.kind == nnkIdentDefs and field[0].eqIdent("stride"):
          result.strideTy = field[1]
      if result.shapeTy != nil and result.strideTy != nil:
        return
  error("layoutTypeArgs: cannot recover Layout type args from " & typ.repr)

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
