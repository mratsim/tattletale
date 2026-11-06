
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers
import ./passes_low_views

# ############################################################
#
#            Shared lowering passes
#
# ############################################################

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerFoldMaxMinToBuiltins*(reg: var PassRegistry) =
  reg.register("foldMaxMinToBuiltins", pkTransform,
    "Folds ternary max/min idioms to the native max/min builtins",
    run = foldMaxMinToBuiltins)

func registerSharedLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerSpans()

proc sameExpr(a, b: CirExpr): bool =
  ## Structural equality of two idiom operands: symbols and literals by value,
  ## compound operands on kind and children.
  if a.isNil or b.isNil or a.kind != b.kind:
    return false
  case a.kind
  of eSym:
    result = a.name == b.name
  of eIntLit:
    result = a.value == b.value
  of eFloatLit:
    result = a.fValue == b.fValue
  of eDot:
    result = a.fieldName == b.fieldName and sameExpr(a.dotBase, b.dotBase)
  else:
    let ca = getExprChildren(a)
    let cb = getExprChildren(b)
    if ca.len != cb.len:
      return false
    result = true
    for i in 0 ..< ca.len:
      if not sameExpr(ca[i], cb[i]):
        return false

proc leOfCond(cond: CirExpr): tuple[le: CirExpr, guarded: bool] =
  ## Returns the `<=` comparison of a max/min idiom condition.
  ## `guarded` reports the NaN-guard form `((P <= Q) || (!(B == B)))` with the guard on either side.
  ## `!(y == y)` over structurally equal operands is Nim's float max/min guard.
  if cond.kind == eLe:
    return (cond, false)
  if cond.kind == eOr:
    let guard = proc(x: CirExpr): bool =
      x != nil and x.kind == eNot and x.notOperand != nil and
        x.notOperand.kind == eEq and sameExpr(x.notOperand.a, x.notOperand.b)
    let l = cond.andLeft
    let r = cond.andRight
    if l.kind == eLe and guard(r):
      return (l, true)
    if r.kind == eLe and guard(l):
      return (r, true)
  return (nil, false)

proc foldMinMax(e: CirExpr): bool =
  ## Replaces a max/min-idiom ternary with a direct `max`/`min` call, in place.
  ## max folds `(B <= A) ? A : B`, min folds `(A <= B) ? A : B`.
  ## Both operands are scalars of one basic numeric kind, bf16 included.
  ## Float operands require the NaN guard.
  ##
  ## Accepted delta against the guard form: `fmax(+0.0, -0.0)` keeps +0.0.
  ## The guard keeps -0.0. A NaN operand keeps the target's native max/min NaN semantics.
  if e.kind != eTernary:
    return false
  let (le, guarded) = leOfCond(e.cond)
  if le.isNil:
    return false
  var name = ""
  if sameExpr(e.elseE, le.a) and sameExpr(e.thenE, le.b):
    name = "max"
  elif sameExpr(e.thenE, le.a) and sameExpr(e.elseE, le.b):
    name = "min"
  else:
    return false
  let tThen = getExprType(e.thenE)
  let tElse = getExprType(e.elseE)
  if tThen.isNil or tElse.isNil or tThen.kind != tElse.kind or
      tThen.kind notin {ctInt, ctUint, ctFloat, ctBf16} or
      (tThen.kind in {ctFloat, ctBf16} and not guarded):
    return false
  e[] = callE(sym(name, tProcT([tThen, tElse], e.ternType)), e.thenE, e.elseE)[]
  return true

proc foldMaxMinToBuiltins(m: CirModule): string {.nimcall.} =
  let fold = proc(e: CirExpr) =
    discard foldMinMax(e)
  for fn in m.procs:
    fn.body.walkStmts(nil, fold)
