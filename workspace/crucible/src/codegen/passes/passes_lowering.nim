
import std/[tables, sets]
import ../ir/cirv2_build
import ../ir/cirv2_lang
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

func registerLowerByrefParams*(reg: var PassRegistry) =
  reg.register("lowerByrefParams", pkTransform,
    "Rewrites passByRef params into const Type* ptr params",
    run = lowerByrefParams)

func registerSharedLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerSpans()

proc sameExpr(a, b: CirExpr): bool =
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

proc leOfCond(cond: CirExpr): CirExpr =
  ## Returns the `<=` comparison of a max/min idiom condition, Nim's float
  ## NaN guard `!(x == x)` stripped when present.
  if cond.kind == eLe:
    return cond
  if cond.kind == eOr:
    let guard = proc(x: CirExpr): bool =
      x != nil and x.kind == eNot and x.notOperand != nil and
        x.notOperand.kind == eEq and sameExpr(x.notOperand.a, x.notOperand.b)
    let l = cond.andLeft
    let r = cond.andRight
    if l.kind == eLe and guard(r):
      return l
    if r.kind == eLe and guard(l):
      return r
  return nil

proc foldMinMax(e: CirExpr): bool =
  ## Replaces a max/min-idiom ternary with a direct `max`/`min` call, in place.
  ## max folds `(B <= A) ? A : B`, min folds `(A <= B) ? A : B`.
  if e.kind != eTernary:
    return false
  let le = leOfCond(e.cond)
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
      tThen.kind notin {ctInt, ctUint, ctFloat, ctBf16}:
    return false
  e[] = callE(sym(name, tProcT([tThen, tElse], e.ternType)), e.thenE, e.elseE)[]
  return true

proc foldMaxMinToBuiltins(m: CirModule) {.nimcall.} =
  let fold = proc(e: CirExpr) =
    discard foldMinMax(e)
  for fn in m.procs:
    fn.body.walkStmts(nil, fold)

# ═══════════════════════════════════════════════════════════════════════
#  By-ref parameter lowering
# ═══════════════════════════════════════════════════════════════════════

proc addrWrapByrefArgs(pre: var seq[CirStmt], tmpCount: var int, e: CirExpr, byref: Table[string, seq[int]], globalFnParams: seq[string], taken: var HashSet[string]) =
  ## A passByRef parameter takes the struct by reference, so the call passes a pointer.
  ## The argument in the IR is still the struct value, the pass supplies a pointer:
  ##
  ## | Before (IR)                                | After (IR)                                   |
  ## | ------------------------------------------ | -------------------------------------------- |
  ## | `call readCount(w)`, `w` a local           | `call readCount(&w)`                         |
  ## | `call readCount(w)`, `w` a global fn param | `var _t0 = w` + `call readCount(&_t0)`       |
  ## | `call readCount(makeW())`, a call result   | `var _t0 = makeW()` + `call readCount(&_t0)` |
  ##
  ## Hoisted declarations insert in front of the statement, nested calls hoist first, so a call node sees its arguments already rewritten.
  if e.isNil():
    return
  for ch in getExprChildren(e):
    addrWrapByrefArgs(pre, tmpCount, ch, byref, globalFnParams, taken)
  if e.kind != eCall:
    return
  for i in byref.getOrDefault(e.callee.name):
    if i >= e.args.len:
      break
    let at = getExprType(e.args[i])
    if at != nil and at.kind in {ctStruct, ctArray}:
      let ptrTy = tPtr(caThread, at)
      let argIsGlobalFnParam = e.args[i].kind == eSym and e.args[i].name in globalFnParams
      if isLvalue(e.args[i]) and not argIsGlobalFnParam:
        e.args[i] = addrE(ptrTy, e.args[i])
      else:
        inc tmpCount
        var tmpName = "byref_tmp" & $tmpCount
        while tmpName in taken:
          inc tmpCount
          tmpName = "byref_tmp" & $tmpCount
        taken.incl tmpName
        let tmp = sym(tmpName, at)
        pre.add(varD(tmpName, at))
        pre.add(asgnS(tmp, e.args[i]))
        e.args[i] = addrE(ptrTy, tmp)

proc addrWrapByrefStmts(stmts: var seq[CirStmt], tmpCount: var int, byref: Table[string, seq[int]], globalFnParams: seq[string], taken: var HashSet[string])

proc addrWrapByrefStmt(pre: var seq[CirStmt], tmpCount: var int, s: CirStmt, byref: Table[string, seq[int]], globalFnParams: seq[string], taken: var HashSet[string]) =
  case s.kind
  of sAsgn:
    addrWrapByrefArgs(pre, tmpCount, s.lhs, byref, globalFnParams, taken)
    addrWrapByrefArgs(pre, tmpCount, s.rhs, byref, globalFnParams, taken)
  of sIte:
    addrWrapByrefArgs(pre, tmpCount, s.cond, byref, globalFnParams, taken)
    addrWrapByrefStmts(s.thenBody, tmpCount, byref, globalFnParams, taken)
    addrWrapByrefStmts(s.elseBody, tmpCount, byref, globalFnParams, taken)
  of sWhile:
    addrWrapByrefArgs(pre, tmpCount, s.whileCond, byref, globalFnParams, taken)
    addrWrapByrefStmts(s.whileBody, tmpCount, byref, globalFnParams, taken)
    # A while loop tests its condition after every body run: each hoisted argument
    # assignment repeats at the end of the body, its declaration stays in front
    for h in pre:
      if h.kind == sAsgn:
        s.whileBody.add(cloneStmts(@[h])[0])
  of sRet:
    addrWrapByrefArgs(pre, tmpCount, s.retVal, byref, globalFnParams, taken)
  of sDiscard:
    addrWrapByrefArgs(pre, tmpCount, s.discardVal, byref, globalFnParams, taken)
  of sCall:
    let callE = CirExpr(kind: eCall)
    callE.callee = s.callCallee
    callE.args = s.callArgs
    addrWrapByrefArgs(pre, tmpCount, callE, byref, globalFnParams, taken)
    s.callCallee = callE.callee
    s.callArgs = callE.args
  of sVar:
    addrWrapByrefArgs(pre, tmpCount, s.init, byref, globalFnParams, taken)
  of sAsm, sEmit:
    discard

proc addrWrapByrefStmts(stmts: var seq[CirStmt], tmpCount: var int, byref: Table[string, seq[int]], globalFnParams: seq[string], taken: var HashSet[string]) =
  var i = 0
  while i < stmts.len:
    var pre: seq[CirStmt] = @[]
    addrWrapByrefStmt(pre, tmpCount, stmts[i], byref, globalFnParams, taken)
    for h in pre:
      stmts.insert(h, i)
      i.inc
    i.inc

proc collectVarNames(stmts: seq[CirStmt], taken: var HashSet[string]) =
  for s in stmts:
    case s.kind
    of sVar:
      taken.incl s.name
    of sIte:
      collectVarNames(s.thenBody, taken)
      collectVarNames(s.elseBody, taken)
    of sWhile:
      collectVarNames(s.whileBody, taken)
    else:
      discard

proc lowerByrefParams(m: CirModule) =
  var byref = initTable[string, seq[int]]()
  for fn in m.procs:
    if fn.isGlobalFn:
      continue
    for i, p in fn.params.mpairs:
      if not p.passByRef:
        continue
      let oldName = p.name
      p.name = "_p_" & oldName
      let ptrSym = sym(p.name, tPtr(caThread, p.typ))
      fn.body.walkStmts(nil, proc(e: CirExpr) =
        if e.kind == eSym and e.name == oldName:
          e[] = derefE(ptrSym)[])
      byref.mgetOrPut(fn.name, @[]).add(i)
  for fn in m.procs:
    var tmpCount = 0
    var globalFnParams: seq[string]
    if fn.isGlobalFn:
      for p in fn.params:
        globalFnParams.add p.name
    var taken: HashSet[string]
    for p in fn.params:
      taken.incl p.name
    fn.body.collectVarNames(taken)
    addrWrapByrefStmts(fn.body, tmpCount, byref, globalFnParams, taken)
