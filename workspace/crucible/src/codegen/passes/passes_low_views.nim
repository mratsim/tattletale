# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

# ############################################################
#
# Span and view lowering
#
# ############################################################
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
# - the normal-form verifier refuses any span that survives, so the printers
#   never see one

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerLowerSpans*(reg: var PassRegistry) =
  ## Registers the openArray span split of the lowering stage.
  ##
  ## Returns the registration on `reg`. Position: the shared lowering stage,
  ## before every per-target transform, so the per-address-space callee
  ## instantiation sees the split pointer + length call sites.
  reg.register("lowerSpans", pkTransform, phaseMain,
    "Splits openArray spans into pointer + length pairs, rewrites" &
    " toOpenArray call arguments, folds len reads",
    run = lowerSpansPass)

# ═══════════════════════════════════════════════════════════════════════
# Span signatures
# ═══════════════════════════════════════════════════════════════════════

type
  SpanSig = object
    paramIdx: int
    ## Position of the span parameter in the callee's parameter list.
    ptrTyp: CirType
    ## Element pointer the span splits to, the span's own space.
    lenTyp: CirType
    ## int32, the split length's type.

  SpanSigs = Table[string, seq[SpanSig]]
  ## Callee name -> the span parameters that split, in parameter order.
  ##
  ## - one proc table exists in CIRv2
  ## - call sites reference the callee's exact instantiation name, so no name
  ##   mangling or prefix matching applies

proc collectSpanSigs(m: CirModule): SpanSigs =
  ## Span parameters of every module proc, keyed by proc name.
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
  ## the pointer keeps the parameter's name, the length is `<name>_len`.
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
# View splitting
# ═══════════════════════════════════════════════════════════════════════

proc spanPtrLenParts(base, first, last: CirExpr): tuple[ptrPlus, lenExpr: CirExpr] =
  ## toOpenArray(base, first, last) -> (base + first, last - first + 1).
  ##
  ## - the pointer side states the base pointer's type
  ## - the length side states the bounds' integer type
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
    ## Module procs by name, the split signatures a call site adopts.
    lenNames: HashSet[string]
    ## Span names of the proc under rewrite with a length binding:
    ## split parameters and decomposed view bindings.

proc splitSpanCall(callee: CirExpr, args: var seq[CirExpr], ctx: var SpanCtx, err: var string) =
  ## Splits the span arguments of one call site and retypes the callee to the split signature:
  ##
  ## - a `toOpenArray` argument expands into `base + first` and `last - first + 1`
  ## - a span symbol expands into the pointer and length bindings of its name
  ## - a callee outside the module refuses, its span parameter would reach
  ##   the printer unsplit
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
  ## Rewrites the span uses of one expression in place:
  ##
  ## - a span symbol retypes to the element pointer of its own name
  ## - `len(s)` over a span binding folds to the length binding `s_len`
  ## - `len(toOpenArray(p, a, b))` folds to `b - a + 1`
  ## - a call site with span arguments splits them and retypes its callee
  ##
  ## A surviving `toOpenArray` call refuses. The frontend emits one only in three positions: a view binding, a call argument, a `len` read.
  ##
  ## Children descend through their own fields. A folded node replaces the node slot its parent references.
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
  else:
    # children are shared refs: the walk retypes span symbols in place
    # and no child position replaces its node
    var children = getExprChildren(e)
    for i in 0 ..< children.len:
      rewriteSpanUsesExpr(children[i], ctx, err)

proc rewriteSpanUsesStmts(ss: seq[CirStmt], ctx: var SpanCtx, err: var string) =
  ## Statement-walk overload of `rewriteSpanUsesExpr`, nested bodies included.
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
    of sLoop:
      s.loopBody.rewriteSpanUsesStmts(ctx, err)
    of sCase:
      s.selector.rewriteSpanUsesExpr(ctx, err)
      for br in s.branches:
        br.body.rewriteSpanUsesStmts(ctx, err)
      s.caseElse.rewriteSpanUsesStmts(ctx, err)
    of sScope:
      s.scopeBody.rewriteSpanUsesStmts(ctx, err)
    of sAssume:
      s.assumeCond.rewriteSpanUsesExpr(ctx, err)
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
    of sLab, sJmp, sAsm, sEmit:
      discard
    if err.len > 0:
      return

# ═══════════════════════════════════════════════════════════════════════
# View binding decomposition
# ═══════════════════════════════════════════════════════════════════════

proc decomposeSpanBindings(ss: var seq[CirStmt], ctx: var SpanCtx): string =
  ## Decomposes the view bindings of one statement list in place:
  ##
  ## - the assignment of a `toOpenArray(p, a, b)` call to a preceding
  ##   `var v: span T` declaration retypes the binding, its value is `p + a`
  ## - the decomposition appends the `v_len` declaration bound to `b - a + 1`
  ##
  ## Nested statement lists recurse.
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
    of sLoop:
      let loopErr = decomposeSpanBindings(s.loopBody, ctx)
      if loopErr.len > 0:
        return loopErr
    of sCase:
      for br in s.branches.mitems:
        let brErr = decomposeSpanBindings(br.body, ctx)
        if brErr.len > 0:
          return brErr
      let elseErr = decomposeSpanBindings(s.caseElse, ctx)
      if elseErr.len > 0:
        return elseErr
    of sScope:
      let scopeErr = decomposeSpanBindings(s.scopeBody, ctx)
      if scopeErr.len > 0:
        return scopeErr
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
# Pass runner
# ═══════════════════════════════════════════════════════════════════════

proc lowerSpansPass(m: CirModule): string =
  ## Splits the openArray spans of every proc into pointer + length pairs.
  ##
  ## Returns "" when clean, else the first refusal.
  ##
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
  return ""
