# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

#
# openArray spans split into pointer + length pairs:
#
# | Before (IR)                                  | After (IR)                                                               |
# | -------------------------------------------- | ------------------------------------------------------------------------ |
# | `f(s: span T)`                               | `f(s: aptr T, s_len: int32)`                                             |
# | `f(toOpenArray(p, a, b))`                    | `f(p + a, b - a + 1)`                                                    |
# | `var v: span T` + `v = toOpenArray(p, a, b)` | `var v: aptr T` + `v = p + a` + `var v_len: int32` + `v_len = b - a + 1` |
# | `len(s)` / `len(toOpenArray(p, a, b))`       | `s_len` / `b - a + 1`                                                    |
#
# Every bundle registers the pass:
#
# - the C-family targets print the split pointer with its address space
# - the per-address-space callee instantiation, registered after this pass,
#   re-spaces the split pointer per call site
# - the pass refuses any span that survives its own rewrite, so the printers
#   never see one

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ############################################################
#
#            View lowering
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════


func registerLowerSpans*(reg: var PassRegistry) =
  ## Placement contract: the shared lowering stage, before every per-target transform,
  ## so the per-address-space callee instantiation sees the split
  ## pointer + length call sites.
  reg.register("lowerSpans", pkTransform,
    "Splits openArray spans into pointer + length pairs, rewrites" &
    " toOpenArray call arguments, folds len reads",
    run = lowerSpansPass)

# ═══════════════════════════════════════════════════════════════════════
#  Span signatures
# ═══════════════════════════════════════════════════════════════════════


type
  SpanSig = object
    paramIdx: int
    ptrTyp: CirType
    ## Element pointer the span splits to, in the span's own space.
    lenTyp: CirType

  SpanSigs = Table[string, seq[SpanSig]]
  ## Callee name -> the span parameters that split, in parameter order.
  ##
  ## - one proc table exists in CIRv2
  ## - call sites reference the callee's exact instantiation name, so no name
  ##   mangling or prefix matching applies

proc collectSpanSigs(m: CirModule): SpanSigs =
  for fn in m.procs:
    var sigs: seq[SpanSig]
    for i, p in fn.params:
      if p.typ != nil and p.typ.kind == ctSpan:
        sigs.add SpanSig(
          paramIdx: i,
          ptrTyp: tAptr(p.typ.sSpace, p.typ.sElem),
          lenTyp: tInt(32))
    if sigs.len > 0:
      result[fn.name] = sigs

proc splitSpanParams(fn: CirProc, sigs: seq[SpanSig]) =
  ## Replaces every span parameter of `fn` with its pointer + length pair,
  ## the pointer is named for the parameter and the length is `<name>_len`.
  var newParams: seq[CirParam]
  var sigIdx = 0
  for i, p in fn.params:
    if sigIdx < sigs.len and sigs[sigIdx].paramIdx == i:
      newParams.add param(p.name, sigs[sigIdx].ptrTyp)
      newParams.add param(p.name & "_len", sigs[sigIdx].lenTyp)
      sigIdx.inc
    else:
      newParams.add p
  fn.params = newParams

# ═══════════════════════════════════════════════════════════════════════
#  View splitting
# ═══════════════════════════════════════════════════════════════════════


proc spanPtrLenParts(base, first, last: CirExpr): tuple[ptrPlus, lenExpr: CirExpr] =
  ## toOpenArray(base, first, last) -> (base + first, last - first + 1).
  let ptrTyp = getExprType(base)
  let idxTyp = getExprType(first)
  let lenSub = subE(idxTyp, last, first)
  result = (addE(ptrTyp, base, first), addE(idxTyp, lenSub, intLit(1, idxTyp)))

proc isToOpenArrayCall(e: CirExpr): bool =
  ## True for the span construction call the frontend emits.
  e.kind == eCall and e.callee.kind == eSym and
    e.callee.name == "toOpenArray" and e.args.len == 3

type
  SpanCtx = object
    ## Per-proc rewrite state.
    sigs: SpanSigs
    ## Span parameters of every module proc, keyed by callee name.
    procs: Table[string, CirProc]
    ## Module procs by name, each with the split signatures its call sites adopt.
    lenNames: HashSet[string]
    ## Span names of the proc under rewrite with a length binding:
    ## split parameters and decomposed view bindings.

proc splitSpanCall(callee: CirExpr, args: var seq[CirExpr], ctx: var SpanCtx, err: var string) =
  ## A `toOpenArray` argument expands into `base + first` and `last - first + 1`,
  ## a span symbol into the pointer and length bindings of its name.
  ## The rewrite refuses a callee outside the module, its span parameter would reach the printer unsplit.
  if callee.kind != eSym or err.len > 0:
    return
  let sigs = ctx.sigs.getOrDefault(callee.name)
  if sigs.len == 0:
    return
  let def = ctx.procs.getOrDefault(callee.name)
  if def.isNil:
    err = "span argument to " & callee.name &
      ", a callee outside the module has no split signature"
    return
  var newArgs: seq[CirExpr]
  var argIdx = 0
  var sigIdx = 0
  while argIdx < args.len:
    if sigIdx < sigs.len and sigs[sigIdx].paramIdx == argIdx:
      let arg = args[argIdx]
      if arg.isToOpenArrayCall:
        let (ptrPlus, lenExpr) = spanPtrLenParts(arg.args[0], arg.args[1], arg.args[2])
        newArgs.add ptrPlus
        newArgs.add lenExpr
      elif arg.kind == eSym and arg.typ != nil and arg.typ.kind == ctSpan:
        newArgs.add sym(arg.name, tAptr(arg.typ.sSpace, arg.typ.sElem))
        newArgs.add sym(arg.name & "_len", tInt(32))
      else:
        newArgs.add arg
      sigIdx.inc
    else:
      newArgs.add args[argIdx]
    argIdx.inc
  args = newArgs
  var paramTypes: seq[CirType]
  for p in def.params:
    paramTypes.add p.typ
  callee.typ = CirType(kind: ctProc, params: paramTypes, ret: def.ret)

proc rewriteSpanUsesExpr(e: var CirExpr, ctx: var SpanCtx, err: var string)
  ## Forward declaration, statement lists recurse into their expressions.

proc rewriteSpanUsesExpr(e: var CirExpr, ctx: var SpanCtx, err: var string) =
  ## Surviving `toOpenArray` calls: the pass refuses one, the frontend emits one only in a view binding, a call argument, or a `len` read.
  ##
  ## A folded node replaces the node slot its parent references (`e` is a `var` param).
  if err.len > 0 or e.isNil:
    return
  case e.kind
  of eSym:
    if e.typ != nil and e.typ.kind == ctSpan:
      if e.name notin ctx.lenNames:
        err = "span symbol " & e.name & " has no split length binding"
        return
      e.typ = tAptr(e.typ.sSpace, e.typ.sElem)
  of eCall:
    let calleeName = if e.callee.kind == eSym: e.callee.name else: ""
    if calleeName == "toOpenArray":
      err = "toOpenArray result used outside a span argument, a len read," &
        " or a view binding"
      return
    if calleeName == "len" and e.args.len == 1:
      let arg = e.args[0]
      if arg.kind == eSym and arg.typ != nil and arg.typ.kind == ctSpan:
        if arg.name notin ctx.lenNames:
          err = "span symbol " & arg.name & " has no split length binding"
          return
        e = sym(arg.name & "_len", tInt(32))
        return
      if arg.isToOpenArrayCall:
        e = spanPtrLenParts(arg.args[0], arg.args[1], arg.args[2]).lenExpr
        return
    else:
      splitSpanCall(e.callee, e.args, ctx, err)
      if err.len > 0:
        return
    e.callee.rewriteSpanUsesExpr(ctx, err)
    if err.len > 0:
      return
    for a in e.args.mitems:
      a.rewriteSpanUsesExpr(ctx, err)
  of eCast, eConv:
    # a conversion over a span read retains the read's fold: the operand is
    # a settable field, a folded read replaces it in the conversion
    e.castOperand.rewriteSpanUsesExpr(ctx, err)
  else:
    # children are shared refs: the walk retypes span symbols in place
    # and no child position replaces its node
    var children = getExprChildren(e)
    for i in 0 ..< children.len:
      rewriteSpanUsesExpr(children[i], ctx, err)

proc rewriteSpanUsesStmts(ss: seq[CirStmt], ctx: var SpanCtx, err: var string) =
  for s in ss:
    case s.kind
    of sVar, sConst:
      s.init.rewriteSpanUsesExpr(ctx, err)
    of sAsgn:
      s.lhs.rewriteSpanUsesExpr(ctx, err)
      s.rhs.rewriteSpanUsesExpr(ctx, err)
    of sIte:
      s.cond.rewriteSpanUsesExpr(ctx, err)
      s.thenBody.rewriteSpanUsesStmts(ctx, err)
      s.elseBody.rewriteSpanUsesStmts(ctx, err)
    of sWhile:
      s.whileCond.rewriteSpanUsesExpr(ctx, err)
      s.whileBody.rewriteSpanUsesStmts(ctx, err)
    of sRet:
      s.retVal.rewriteSpanUsesExpr(ctx, err)
    of sDiscard:
      s.discardVal.rewriteSpanUsesExpr(ctx, err)
    of sCall:
      splitSpanCall(s.callCallee, s.callArgs, ctx, err)
      if err.len > 0:
        return
      for a in s.callArgs.mitems:
        a.rewriteSpanUsesExpr(ctx, err)
    of sAsm, sEmit:
      discard
    if err.len > 0:
      return

# ═══════════════════════════════════════════════════════════════════════
#  View binding decomposition
# ═══════════════════════════════════════════════════════════════════════


proc decomposeSpanBindings(ss: var seq[CirStmt], ctx: var SpanCtx): string =
  ## A `toOpenArray(p, a, b)` assignment to a preceding `var v: span T`
  ## declaration retypes the binding to `p + a` and appends the matching
  ## `v_len` declaration bound to `b - a + 1`.
  var lowered: seq[CirStmt]
  var i = 0
  while i < ss.len:
    let s = ss[i]
    case s.kind
    of sIte:
      var thenErr = decomposeSpanBindings(s.thenBody, ctx)
      if thenErr.len == 0:
        thenErr = decomposeSpanBindings(s.elseBody, ctx)
      if thenErr.len > 0:
        return thenErr
    of sWhile:
      let loopErr = decomposeSpanBindings(s.whileBody, ctx)
      if loopErr.len > 0:
        return loopErr
    else:
      discard
    if s.kind == sVar and s.declType != nil and s.declType.kind == ctSpan and
        s.init == nil and i + 1 < ss.len:
      let bindStmt = ss[i + 1]
      if bindStmt.kind == sAsgn and bindStmt.lhs.kind == eSym and
          bindStmt.lhs.name == s.name and bindStmt.rhs.isToOpenArrayCall:
        let call = bindStmt.rhs
        let (ptrPlus, lenExpr) = spanPtrLenParts(call.args[0], call.args[1], call.args[2])
        s.declType = tAptr(s.declType.sSpace, s.declType.sElem)
        bindStmt.lhs = sym(s.name, s.declType)
        bindStmt.rhs = ptrPlus
        ctx.lenNames.incl s.name
        lowered.add s
        lowered.add bindStmt
        lowered.add varD(s.name & "_len", tInt(32))
        lowered.add asgnS(sym(s.name & "_len", tInt(32)), lenExpr)
        i.inc 2
        continue
    lowered.add s
    i.inc
  ss = lowered
  return ""

# ═══════════════════════════════════════════════════════════════════════
#  Survivor guard
# ═══════════════════════════════════════════════════════════════════════


proc typeHoldsSpan(t: CirType): bool =
  if t.isNil:
    return false
  case t.kind
  of ctSpan:
    result = true
  of ctPtr, ctAptr:
    result = typeHoldsSpan(t.pElem)
  of ctArray:
    result = typeHoldsSpan(t.aElem)
  of ctVector:
    result = typeHoldsSpan(t.vElem)
  of ctStruct:
    result = false
    for f in t.fields:
      if typeHoldsSpan(f.typ):
        return true
  of ctProc:
    result = typeHoldsSpan(t.ret)
    if result:
      return true
    for p in t.params:
      if typeHoldsSpan(p):
        return true
  else:
    result = false

proc survivingSpanRejection(fn: CirProc): string =
  ## Contract:
  ## - the guard checks every position, including the ones the rewrite skips
  ##   (`sAsm` and `sEmit` operands)
  for p in fn.params:
    if typeHoldsSpan(p.typ):
      return "parameter " & p.name & " keeps an openArray span"
  if typeHoldsSpan(fn.ret):
    return "return type keeps an openArray span"
  var err = ""
  let checkStmt = proc(s: CirStmt) =
    if err.len > 0:
      return
    if s.kind in {sVar, sConst} and typeHoldsSpan(s.declType):
      err = "declaration " & s.name & " keeps an openArray span"
  let checkExpr = proc(e: CirExpr) =
    if err.len > 0:
      return
    if typeHoldsSpan(e.getExprType):
      err = "an expression resolves to an openArray span type"
  fn.body.walkStmts(checkStmt, checkExpr)
  return err

# ═══════════════════════════════════════════════════════════════════════
#  Pass runner
# ═══════════════════════════════════════════════════════════════════════


proc lowerSpansPass(m: CirModule): string =
  ## Parameter splits run before the body rewrites, which rebuild call-site
  ## callee types from the split signatures.
  let sigs = collectSpanSigs(m)
  var procs: Table[string, CirProc]
  for fn in m.procs:
    procs[fn.name] = fn
  for fn in m.procs:
    let fnSigs = sigs.getOrDefault(fn.name)
    if fnSigs.len > 0:
      splitSpanParams(fn, fnSigs)
  for fn in m.procs:
    var ctx = SpanCtx(sigs: sigs, procs: procs)
    for s in sigs.getOrDefault(fn.name):
      ctx.lenNames.incl fn.params[s.paramIdx].name
    var err = decomposeSpanBindings(fn.body, ctx)
    if err.len == 0:
      rewriteSpanUsesStmts(fn.body, ctx, err)
    if err.len > 0:
      return "in proc " & fn.name & ": " & err
    let spanErr = fn.survivingSpanRejection()
    if spanErr.len > 0:
      return "in proc " & fn.name & ": " & spanErr
  for g in m.globals:
    if g.kind in {sVar, sConst} and typeHoldsSpan(g.declType):
      return "in global scope: declaration " & g.name & " keeps an openArray span"
  return ""
