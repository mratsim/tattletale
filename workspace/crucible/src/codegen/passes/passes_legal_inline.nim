# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Legalization: splicing the `{.inline.}` and `{.forceinline.}` call sites.
##
## A device function cannot take the thread builtins as attributed parameters,
## so an `{.inline.}` proc that reads them splices into its caller. Per call site:
##
## - arguments bind to fresh temps, immutable ones and pointer formals substitute directly
## - the callee's locals and labels rename fresh, its `result` maps to the call's destination
## - the trailing `ret` drops after moving a carried value into the call's destination
## - the spliced lab/jmp chain then joins the caller's structuring
##
## Refusals name the proc: an inline proc with an early `ret`, or a `ret` that is
## not the trailing statement, has no single-entry splice.

import ../ir/cirv2_lang
import std/[sets, tables, sequtils]
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerInlineSplice*(reg: var PassRegistry) =
  ## Registers the inline-call splicing pass, the legalization stage.
  ##
  ## Ordering:
  ## - depends on the builtin kind assignment, the kernel-only-reads check
  ##   below keys on the typed coordinate kinds that pass assigns
  reg.register("inlineProcs", pkTransform, phaseEarly,
    "Splices the inline and forceinline call sites, deletes the uncalled inline procs",
    dependsOn = @["builtinKindAssignment"],
    run = inlineProcs)

type
  InlineError* = object
    ## Refusal of the inline pass, names the proc and the cause.
    procName*: string
    msg*: string

  InlineCtx = ref object
    fnName: string
    err: string
    counter: int
    splicedAway: HashSet[string]
      ## Procs spliced at least once. A spliced proc with no surviving caller
      ## deletes from the module, its body lives in the splicers.

proc appendBuiltinReads(e: CirExpr, found: var HashSet[string]) =
  ## Collects the coordinate builtin names one expression reads.
  ## Recognition is by the typed coordinate kind the alias canonicalization
  ## pass assigned: a local or parameter that happens to share a builtin's
  ## name has no kind and counts as an ordinary symbol.
  if e.isNil:
    return
  if e.kind == eSym and e.coordBuiltin != cbkNone:
    found.incl(e.name)
    return
  for ch in getExprChildren(e):
    appendBuiltinReads(ch, found)

proc appendBuiltinReads(ss: seq[CirStmt], found: var HashSet[string]) =
  ## Collects the coordinate builtin names one statement list reads.
  for s in ss:
    case s.kind
    of sVar, sConst:
      appendBuiltinReads(s.init, found)
    of sAsgn:
      appendBuiltinReads(s.lhs, found)
      appendBuiltinReads(s.rhs, found)
    of sIte:
      appendBuiltinReads(s.cond, found)
      appendBuiltinReads(s.thenBody, found)
      appendBuiltinReads(s.elseBody, found)
    of sLoop:
      appendBuiltinReads(s.loopBody, found)
    of sScope:
      appendBuiltinReads(s.scopeBody, found)
    of sAssume:
      appendBuiltinReads(s.assumeCond, found)
    of sRet:
      appendBuiltinReads(s.retVal, found)
    of sDiscard:
      appendBuiltinReads(s.discardVal, found)
    of sCall:
      appendBuiltinReads(s.callCallee, found)
      for a in s.callArgs:
        appendBuiltinReads(a, found)
    else:
      discard

func readsCoordBuiltin(ss: seq[CirStmt]): bool =
  ## True when the statement list reads a coordinate builtin symbol.
  ## A device function cannot evaluate one: the call must splice.
  var found: HashSet[string]
  ss.appendBuiltinReads(found)
  result = found.len > 0

proc failing(ctx: InlineCtx): bool =
  ctx.err.len > 0

proc fail(ctx: InlineCtx, detail: string) =
  if ctx.err.len == 0:
    ctx.err = "cirv2 inline: at " & ctx.fnName & ": " & detail

proc freshInlineName(ctx: InlineCtx, name: string): string =
  ctx.counter.inc
  result = "inl" & $ctx.counter & "_" & name

proc renameExpr(ctx: InlineCtx, e: var CirExpr, renames: Table[string, CirExpr], labels: Table[string, string]) =
  if e.isNil:
    return
  case e.kind
  of eSym:
    if renames.hasKey(e.name):
      e = renames[e.name]
    return
  of eCall:
    ctx.renameExpr(e.callee, renames, labels)
    for i in 0 ..< e.args.len:
      ctx.renameExpr(e.args[i], renames, labels)
    return
  else:
    discard
  case e.kind
  of eAt:
    ctx.renameExpr(e.atBase, renames, labels)
    ctx.renameExpr(e.atIndex, renames, labels)
  of ePat:
    ctx.renameExpr(e.patBase, renames, labels)
    ctx.renameExpr(e.patIndex, renames, labels)
  of eDot:
    ctx.renameExpr(e.dotBase, renames, labels)
  of eDeref:
    ctx.renameExpr(e.derefOperand, renames, labels)
  of eAddr:
    ctx.renameExpr(e.addrOperand, renames, labels)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    ctx.renameExpr(e.left, renames, labels)
    ctx.renameExpr(e.right, renames, labels)
  of eEq, eNeq, eLt, eLe:
    ctx.renameExpr(e.a, renames, labels)
    ctx.renameExpr(e.b, renames, labels)
  of eAnd, eOr:
    ctx.renameExpr(e.andLeft, renames, labels)
    ctx.renameExpr(e.andRight, renames, labels)
  of eNot:
    ctx.renameExpr(e.notOperand, renames, labels)
  of eNeg, eBitNot:
    ctx.renameExpr(e.unOperand, renames, labels)
  of eCast, eConv:
    ctx.renameExpr(e.castOperand, renames, labels)
  of eOconstr:
    for i in 0 ..< e.kvFields.len:
      ctx.renameExpr(e.kvFields[i].value, renames, labels)
  of eAconstr:
    for i in 0 ..< e.elems.len:
      ctx.renameExpr(e.elems[i], renames, labels)
  else:
    discard

proc renameStmts(ctx: InlineCtx, stmts: var seq[CirStmt], renames: Table[string, CirExpr], labels: var Table[string, string]) =
  for i in 0 ..< stmts.len:
    let s = addr stmts[i]
    case s[].kind
    of sVar, sConst:
      if renames.hasKey(s[].name) and renames[s[].name] != nil and
          renames[s[].name].kind == eSym:
        s[].name = renames[s[].name].name
      ctx.renameExpr(s[].init, renames, labels)
    of sAsgn:
      ctx.renameExpr(s[].lhs, renames, labels)
      ctx.renameExpr(s[].rhs, renames, labels)
    of sIte:
      ctx.renameExpr(s[].cond, renames, labels)
      ctx.renameStmts(s[].thenBody, renames, labels)
      ctx.renameStmts(s[].elseBody, renames, labels)
    of sLoop:
      ctx.renameStmts(s[].loopBody, renames, labels)
    of sLab:
      if labels.hasKey(s[].labName):
        s[].labName = labels[s[].labName]
    of sJmp:
      if labels.hasKey(s[].jmpTarget):
        s[].jmpTarget = labels[s[].jmpTarget]
    of sScope:
      ctx.renameStmts(s[].scopeBody, renames, labels)
    of sAssume:
      ctx.renameExpr(s[].assumeCond, renames, labels)
    of sRet:
      ctx.renameExpr(s[].retVal, renames, labels)
    of sCall:
      ctx.renameExpr(s[].callCallee, renames, labels)
      for a in mitems(s[].callArgs):
        ctx.renameExpr(a, renames, labels)
    else:
      discard


proc spliceCall(ctx: InlineCtx, callee: CirProc, args: seq[CirExpr], dest: CirExpr): seq[CirStmt]

proc argsMatchParams(callee: CirProc, args: seq[CirExpr]): bool =
  ## True when every argument binds to its formal under the re-space rule:
  ## same lattice type, a pointer formal accepts another-space pointer
  ## with the same pointee.
  ##
  ## A mismatch keeps the call, the spliced body indexes through the formal's
  ## shape while an array argument reads through whatever node kind the pointer formal compiled against.
  if args.len != callee.params.len:
    return false
  for i, p in callee.params:
    let at = getExprType(args[i])
    if at == nil or sameType(at, p.typ):
      continue
    if p.typ.kind == at.kind and p.typ.kind in {ctPtr, ctAptr} and
        sameType(p.typ.pElem, at.pElem):
      continue
    return false
  return true

proc inlineCallsIn(ctx: InlineCtx, stmts: var seq[CirStmt], procsByName: Table[string, CirProc], depth: int): bool =
  ## Splices one inline call per invocation, returns true when one was spliced.
  ## The caller iterates to a fixpoint.
  if depth > 32:
    ctx.fail("inline recursion deeper than 32, a self-inlining proc")
    return false
  for i in 0 ..< stmts.len:
    let s = addr stmts[i]
    case s[].kind
    of sAsgn:
      if s[].rhs != nil and s[].rhs.kind == eCall and
          s[].rhs.callee.kind == eSym and procsByName.hasKey(s[].rhs.callee.name):
        let callee = procsByName[s[].rhs.callee.name]
        let mustSplice = callee.forceInline or readsCoordBuiltin(callee.body)
        if not mustSplice:
          continue
        if not readsCoordBuiltin(callee.body) and
            not argsMatchParams(callee, s[].rhs.args):
          continue
        let dest = s[].lhs
        let spliced = ctx.spliceCall(callee, s[].rhs.args, dest)
        if ctx.failing():
          return false
        stmts.delete(i)
        for j, st in spliced:
          stmts.insert(st, i + j)
        return true
    of sCall:
      if s[].callCallee.kind == eSym and procsByName.hasKey(s[].callCallee.name):
        let callee = procsByName[s[].callCallee.name]
        let mustSplice = callee.forceInline or readsCoordBuiltin(callee.body)
        if not mustSplice:
          continue
        if not readsCoordBuiltin(callee.body) and
            not argsMatchParams(callee, s[].callArgs):
          continue
        let spliced = ctx.spliceCall(callee, s[].callArgs, nil)
        if ctx.failing():
          return false
        stmts.delete(i)
        for j, st in spliced:
          stmts.insert(st, i + j)
        return true
    of sIte:
      if ctx.inlineCallsIn(s[].thenBody, procsByName, depth + 1):
        return true
      if ctx.inlineCallsIn(s[].elseBody, procsByName, depth + 1):
        return true
    of sLoop:
      if ctx.inlineCallsIn(s[].loopBody, procsByName, depth + 1):
        return true
    of sScope:
      if ctx.inlineCallsIn(s[].scopeBody, procsByName, depth + 1):
        return true
    else:
      discard
  return false

proc spliceCall(ctx: InlineCtx, callee: CirProc, args: seq[CirExpr], dest: CirExpr): seq[CirStmt] =
  ## Builds the spliced body of one call:
  ##
  ## - arguments bind to fresh temps unless immutable or a pointer formal
  ## - the callee's locals rename fresh, `result` maps to `dest`
  ## - the trailing `ret` drops, its value moves into `dest` first
  ## - an early `ret` refuses: a `ret` left mid-body would return from the caller
  # single exit: only the trailing ret may return
  var rets = 0
  proc countRets(ss: seq[CirStmt]) =
    for s in ss:
      if s.kind == sRet:
        rets.inc
      case s.kind
      of sIte:
        countRets(s.thenBody)
        countRets(s.elseBody)
      of sLoop:
        countRets(s.loopBody)
      of sScope:
        countRets(s.scopeBody)
      of sCase:
        for b in s.branches:
          countRets(b.body)
        countRets(s.caseElse)
      else:
        discard
  countRets(callee.body)
  if rets > 1:
    ctx.fail("inline proc " & callee.name & " has an early return")
    return @[]
  # a single `ret` is splicable only as the trailing statement: a `ret` left
  # mid-body would return from the caller and skip the spliced call site
  if rets == 1 and (callee.body.len == 0 or callee.body[^1].kind != sRet):
    ctx.fail("inline proc " & callee.name & " has an early return")
    return @[]
  if dest.isNil and callee.ret.kind != ctVoid:
    ctx.fail("inline proc " & callee.name & " returns a value with no call destination")
    return @[]

  ctx.splicedAway.incl(callee.name)
  var renames = initTable[string, CirExpr]()
  var pre: seq[CirStmt]
  for i, p in callee.params:
    if i >= args.len:
      ctx.fail("inline proc " & callee.name & " call has " & $args.len &
        " arguments for " & $callee.params.len & " parameters")
      return @[]
    let arg = args[i]
    if arg.kind in {eIntLit, eFloatLit, eTrue, eFalse, eNil} or
        (arg.kind == eSym and p.typ.kind in {ctPtr, ctAptr}):
      # a pointer formal aliases the caller's object by contract
      renames[p.name] = arg
    else:
      # a value argument binds to a fresh temp, a spliced write to the formal
      # must not reach the caller's own variable
      # a complex argument evaluates once, its body reads the temp
      let tname = ctx.freshInlineName(p.name)
      pre.add varD(tname, p.typ)
      pre.add asgnS(sym(tname, p.typ), arg)
      renames[p.name] = sym(tname, p.typ)
  var locals: seq[tuple[name: string, typ: CirType]]
  callee.body.collectDeclaredLocals(locals)
  for l in locals:
    renames[l.name] = sym(ctx.freshInlineName(l.name), l.typ)
  renames["result"] = dest

  # labels rename fresh so the spliced chain structures with the caller's
  var labels = initTable[string, string]()
  proc collectLabels(ss: seq[CirStmt]) =
    for s in ss:
      if s.kind == sLab:
        labels[s.labName] = ctx.freshInlineName(s.labName)
      case s.kind
      of sIte:
        collectLabels(s.thenBody)
        collectLabels(s.elseBody)
      of sLoop:
        collectLabels(s.loopBody)
      of sScope:
        collectLabels(s.scopeBody)
      else:
        discard
  collectLabels(callee.body)

  var body = cloneStmts(callee.body)
  # callee's own `result` declaration drops, the call's destination variable serves
  body = body.filterIt(not (it.kind in {sVar, sConst} and it.name == "result"))
  # a bare `ret result` needs no move, every `result` write already renamed to dest
  let retIsBareResult = body.len > 0 and body[^1].kind == sRet and
    body[^1].retVal != nil and body[^1].retVal.kind == eSym and
    body[^1].retVal.name == "result"
  ctx.renameStmts(body, renames, labels)
  if body.len > 0 and body[^1].kind == sRet:
    let retVal = body[^1].retVal
    body.setLen(body.len - 1)
    if retVal != nil and not retIsBareResult:
      body.add asgnS(dest, retVal)
  return pre & body

proc inlineProcs(m: CirModule): string =
  ## Splices every `{.inline.}`/`{.forceinline.}` call site to a fixpoint.
  ## Deletes the inline procs nothing calls.
  ## Returns "" when clean, else the first refusal.
  ##
  ## Registered by `registerInlineSplice` in the legalization stage.
  var ctx = InlineCtx(fnName: "<module>")
  var byName = initTable[string, CirProc]()
  for fn in m.procs:
    byName[fn.name] = fn
  const maxRounds = 256
  for round in 0 ..< maxRounds:
    var spliced = false
    for fn in m.procs:
      if fn.forceInline and not fn.isKernel:
        continue
      ctx.fnName = fn.name
      if ctx.inlineCallsIn(fn.body, byName, 0):
        spliced = true
        break
      if ctx.failing():
        return ctx.err
    if not spliced:
      break
  var surviving: HashSet[string]
  for fn in m.procs:
    fn.body.appendCallTargetsStmts(surviving)
  # Dead-proc elimination to a fixpoint: kernels are the entry points and any
  # non-kernel proc nothing calls is unreachable.
  # Survivors are recomputed each round since a deletion can orphan callees.
  while true:
    var reachable: HashSet[string]
    for fn in m.procs:
      fn.body.appendCallTargetsStmts(reachable)
    let before = m.procs.len
    m.procs = m.procs.filterIt(it.isKernel or it.name in reachable)
    if m.procs.len == before:
      break
  return ctx.err
