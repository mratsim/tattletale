## Hand-built CIRv2: the openArray span split pass.
##
## Test that `lowerSpans` splits openArray spans into pointer + length pairs
## and the per-address-space callee instantiation re-spaces the split pointer:
##
## - a span parameter splits to `aptr` + `int32` in the callee's signature
## - a `toOpenArray` call argument splits into `base + first` and `last - first + 1`, a `len` read folds the same way
## - a threadgroup-space call site names a clone whose split pointer formal re-types to the threadgroup space
## - no `toOpenArray` call survives, and the verifier refuses a surviving span
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_lower_spans.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/ir/cirv2_verify
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_lowering
import workspace/crucible/src/codegen/passes/passes_lowering_msl

proc runLowering(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerSharedLoweringPasses()
  reg.registerMslLoweringPasses()
  result = m.runPasses(reg)

proc holdsToOpenArrayCall(e: CirExpr): bool =
  ## True when a `toOpenArray` construction call appears anywhere inside the expression.
  if e.isNil:
    return false
  if e.kind == eCall and e.callee.kind == eSym and e.callee.name == "toOpenArray":
    return true
  for ch in getExprChildren(e):
    if ch.holdsToOpenArrayCall:
      return true
  result = false

proc holdsToOpenArrayCall(body: seq[CirStmt]): bool =
  ## Statement-walk overload of `holdsToOpenArrayCall`, nested bodies included.
  for s in body:
    case s.kind
    of sVar, sConst:
      if s.init.holdsToOpenArrayCall:
        return true
    of sAsgn:
      if s.lhs.holdsToOpenArrayCall or s.rhs.holdsToOpenArrayCall:
        return true
    of sIte:
      if s.cond.holdsToOpenArrayCall or s.thenBody.holdsToOpenArrayCall or
          s.elseBody.holdsToOpenArrayCall:
        return true
    of sLoop:
      if s.loopBody.holdsToOpenArrayCall:
        return true
    of sCase:
      if s.selector.holdsToOpenArrayCall:
        return true
      for br in s.branches:
        if br.body.holdsToOpenArrayCall:
          return true
      if s.caseElse.holdsToOpenArrayCall:
        return true
    of sScope:
      if s.scopeBody.holdsToOpenArrayCall:
        return true
    of sAssume:
      if s.assumeCond.holdsToOpenArrayCall:
        return true
    of sRet:
      if s.retVal.holdsToOpenArrayCall:
        return true
    of sDiscard:
      if s.discardVal.holdsToOpenArrayCall:
        return true
    of sCall:
      for a in s.callArgs:
        if a.holdsToOpenArrayCall:
          return true
    of sLab, sJmp, sAsm, sEmit:
      discard
  result = false

proc callArgAsgn(body: seq[CirStmt], idx: int): CirStmt =
  ## Returns the `asgn` whose right side is a value call, at `body[idx]`.
  doAssert body[idx].kind == sAsgn and body[idx].rhs.kind == eCall,
    "statement " & $idx & " must bind a call to a destination"
  body[idx]

static:
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let spanDev = tSpan(caDevice, f32)
  let spanTg = tSpan(caThreadgroup, f32)
  let outT = tAptr(caDevice, f32)
  let arrT = tArray(8, f32)

  proc toOpenArrayE(base: CirExpr, space: CirAddrSpace, first, last: int64): CirExpr =
    ## A `toOpenArray(base, first, last)` call over `base`, the construction form the frontend emits for the span split pass to consume.
    callE(sym("toOpenArray", tProcT(getExprType(base), i32, i32,
        tSpan(space, f32))),
      base, intLit(first, i32), intLit(last, i32))

  # callee: `f(s: openArray[float32], i: int32) -> float32`, reads s[i]
  let fFn = procD("f", @[param("s", spanDev), param("i", i32)], f32, @[
    retS(patE(sym("s", spanDev), sym("i", i32))),
  ])
  # kernel: one device view, one len fold, one threadgroup view call site
  let kern = procD("k", @[param("out", outT)], tVoid(), @[
    varD("scratch", arrT, nil, caThreadgroup),
    varD("n", i32),
    asgnS(patE(sym("out", outT), intLit(0, i32)),
      callE(sym("f", tProcT(spanDev, i32, f32)),
        toOpenArrayE(sym("out", outT), caDevice, 1, 4), intLit(2, i32))),
    asgnS(sym("n", i32),
      callE(sym("len", tProcT(spanDev, i32)),
        toOpenArrayE(sym("out", outT), caDevice, 2, 6))),
    asgnS(patE(sym("out", outT), intLit(2, i32)),
      callE(sym("f", tProcT(spanTg, i32, f32)),
        toOpenArrayE(castE(tAptr(caThreadgroup, f32),
          addrE(tPtr(caThreadgroup, f32),
            atE(sym("scratch", arrT), intLit(0, i32)))), caThreadgroup, 0, 7),
        intLit(5, i32))),
  ])
  kern.isKernel = true
  let m = moduleD("probe", @[], @[fFn, kern])

  let err = m.runLowering()
  doAssert err.len == 0, "the lowering must succeed: " & err

  # ═══ the callee's span parameter split into pointer + length ═══
  var fSplit: CirProc
  for fn in m.procs:
    if fn.name == "f":
      fSplit = fn
  doAssert fSplit != nil, "the callee must keep its base instantiation"
  doAssert fSplit.params.len == 3, "the span param splits to ptr + len plus i, got " &
    $fSplit.params.len
  doAssert fSplit.params[0].name == "s" and
    fSplit.params[0].typ.kind == ctAptr and
    fSplit.params[0].typ.space == caDevice and
    sameType(fSplit.params[0].typ.pElem, f32),
    "the split pointer keeps the parameter name and the span's device space"
  doAssert fSplit.params[1].name == "s_len" and
    sameType(fSplit.params[1].typ, i32), "the split length is int32 s_len"
  doAssert fSplit.params[2].name == "i" and sameType(fSplit.params[2].typ, i32)

  # ═══ the view argument split into ptr + len at the device call site ═══
  var kernBody: seq[CirStmt]
  for fn in m.procs:
    if fn.name == "k":
      kernBody = fn.body
  let devSite = callArgAsgn(kernBody, 2)
  doAssert devSite.rhs.args.len == 3, "the view argument expands to ptr + len," &
    " got " & $devSite.rhs.args.len
  let ptrPlus = devSite.rhs.args[0]
  doAssert ptrPlus.kind == eAdd and ptrPlus.opType.kind == ctAptr and
    ptrPlus.opType.space == caDevice and
    ptrPlus.left.kind == eSym and ptrPlus.left.name == "out" and
    ptrPlus.right.kind == eIntLit and ptrPlus.right.value == 1,
    "the pointer side is out + 1 over the device space"
  let lenExpr = devSite.rhs.args[1]
  doAssert lenExpr.kind == eAdd and lenExpr.left.kind == eSub and
    lenExpr.left.left.kind == eIntLit and lenExpr.left.left.value == 4 and
    lenExpr.left.right.kind == eIntLit and lenExpr.left.right.value == 1 and
    lenExpr.right.kind == eIntLit and lenExpr.right.value == 1,
    "the length side is (last - first) + 1 = 4"
  doAssert devSite.rhs.callee.typ.params.len == 3 and
    sameType(devSite.rhs.callee.typ.params[0], fSplit.params[0].typ) and
    sameType(devSite.rhs.callee.typ.params[1], fSplit.params[1].typ) and
    sameType(devSite.rhs.callee.typ.ret, fSplit.ret),
    "the call site's callee re-types to the split signature"

  # ═══ the len read folds to (last - first) + 1 ═══
  let lenSite = kernBody[3]
  doAssert lenSite.kind == sAsgn and lenSite.rhs.kind == eAdd and
    lenSite.rhs.left.kind == eSub and
    lenSite.rhs.left.left.kind == eIntLit and lenSite.rhs.left.left.value == 6 and
    lenSite.rhs.left.right.kind == eIntLit and lenSite.rhs.left.right.value == 2 and
    lenSite.rhs.right.kind == eIntLit and lenSite.rhs.right.value == 1,
    "len(toOpenArray(out, 2, 6)) folds to (6 - 2) + 1"
  doAssert sameType(getExprType(lenSite.rhs), i32)

  # ═══ the threadgroup call site names a clone with a threadgroup pointer formal ═══
  let tgSite = callArgAsgn(kernBody, 4)
  let cloneName = tgSite.rhs.callee.name
  doAssert cloneName != "f" and cloneName.startsWith("f_"),
    "the threadgroup site must name a clone, got " & cloneName
  var clone: CirProc
  for fn in m.procs:
    if fn.name == cloneName:
      clone = fn
  doAssert clone != nil, "the clone must exist in the proc set"
  doAssert clone.params[0].typ.kind == ctAptr and
    clone.params[0].typ.space == caThreadgroup,
    "the clone's split pointer re-types to the threadgroup space"
  doAssert clone.body.len == 1 and clone.body[0].kind == sRet and
    clone.body[0].retVal.kind == ePat and
    clone.body[0].retVal.patBase.kind == eSym and
    clone.body[0].retVal.patBase.typ.space == caThreadgroup,
    "the clone body reads the re-typed split pointer"

  # ═══ no toOpenArray call survives the pass ═══
  for fn in m.procs:
    doAssert not fn.body.holdsToOpenArrayCall,
      "no toOpenArray call may survive the split in " & fn.name

  # ═══ the post-pass module verifies clean ═══
  doAssert verifyMessage(m).len == 0,
    "the post-pass module must verify clean: " & verifyMessage(m)

  # ═══ the verifier refuses a span that survived lowering ═══
  let survivor = moduleD("survivor", @[], @[procD("g",
    @[param("s", spanDev)], f32, @[retS(patE(sym("s", spanDev), intLit(0, i32)))]),
    procD("h", @[param("out", outT)], tVoid(), @[asgnS(
      patE(sym("out", outT), intLit(0, i32)),
      callE(sym("g", tProcT(spanDev, f32)), toOpenArrayE(sym("out", outT),
        caDevice, 0, 3)))])])
  let vmsg = verifyMessage(survivor)
  doAssert vmsg.contains("span"), "the verifier must refuse a surviving span, got: " & vmsg

  # ═══ an unconsumed toOpenArray call refuses in the pass ═══
  let stray = moduleD("stray", @[], @[procD("h2", @[param("out", outT)], tVoid(),
    @[asgnS(patE(sym("out", outT), intLit(0, i32)),
      callE(sym("toOpenArray", tProcT(outT, i32, i32, spanDev)),
        sym("out", outT), intLit(0, i32), intLit(3, i32)))])])
  stray.procs[0].isKernel = true
  var strayReg = PassRegistry.new()
  strayReg.registerSharedLoweringPasses()
  let serr = stray.runPasses(strayReg)
  doAssert serr.contains("toOpenArray result used outside"),
    "an unconsumed toOpenArray call must refuse, got: " & serr

echo "[OK] t_ir_lower_spans"
