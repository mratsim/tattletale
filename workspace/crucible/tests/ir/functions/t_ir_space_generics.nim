## Hand-built CIRv2: the per-address-space callee instantiation registration.
##
## Test that the MSL lowering instantiates a module callee once per distinct
## argument-address-space combination and rewrites each call site onto its selected instantiation:
##
## - a matching-space call site keeps the callee
## - a differing call site names a clone whose pointer formal re-types to the argument's space
## - same-combination call sites share one clone, and callees outside the module never specialize
## - the verifier rejects a surviving module-callee space mismatch
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_space_generics.nim

import std/[sets, strutils]
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/ir/cirv2_verify
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_lowering_msl

proc runMslLowering(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerMslLoweringPasses()
  result = m.runPasses(reg)

proc fCall(arg: CirExpr): CirStmt =
  ## `out[0] = f(arg)` over the `f` device function's signature.
  asgnS(patE(sym("out", tAptr(caDevice, tFloat(32))), intLit(0, tInt(32))),
    callE(sym("f", tProcT(getExprType(arg), tInt(32), tFloat(32))),
      arg, intLit(0, tInt(32))))

proc callCalleeName(s: CirStmt): string =
  ## Callee name of an `asgn`-bound value call, "" otherwise.
  if s.kind == sAsgn and s.rhs.kind == eCall and s.rhs.callee.kind == eSym:
    return s.rhs.callee.name

static:
  let f32 = tFloat(32)
  let devP = tPtr(caDevice, f32)
  let tgP = tPtr(caThreadgroup, f32)
  let thP = tPtr(caThread, f32)
  let arrT = tArray(8, f32)
  let fFn = procD("f", @[param("p", devP), param("i", tInt(32))], f32, @[
    retS(derefE(sym("p", devP))),
  ])
  # a device function with a threadgroup-space formal names the storage it declares
  # a matching call site binds its own space
  let gFn = procD("g", @[param("p", tgP)], tVoid(), @[
    asgnS(derefE(sym("p", tgP)), fltLit(1.0, f32)),
  ])
  let kern = procD("k", @[param("out", tAptr(caDevice, f32))], tVoid(), @[
    varD("scratch", arrT, nil, caThreadgroup),
    varD("priv", arrT, nil, caThread),
    # argument space matches the formal: no instantiation
    fCall(addrE(devP, patE(sym("out", tAptr(caDevice, f32)),
      intLit(0, tInt(32))))),
    # threadgroup argument, then a second site with the same space:
    # one clone, both sites rewritten
    fCall(addrE(tgP, atE(sym("scratch", arrT), intLit(0, tInt(32))))),
    fCall(addrE(tgP, atE(sym("scratch", arrT), intLit(1, tInt(32))))),
    # thread argument: a second distinct instantiation of f
    fCall(addrE(thP, atE(sym("priv", arrT), intLit(0, tInt(32))))),
    # g's threadgroup formal binds a matching threadgroup argument
    callS(sym("g", tProcT(tgP, tVoid())),
      addrE(tgP, atE(sym("scratch", arrT), intLit(2, tInt(32))))),
    # a foreign callee with a mismatching space: builtins state a nominal formal space.
    # printers re-space that nominal space at the call site
    callS(sym("foreign_copy", tProcT(thP, devP, tVoid())),
      addrE(thP, sym("priv", arrT)),
      addrE(devP, patE(sym("out", tAptr(caDevice, f32)),
        intLit(0, tInt(32))))),
  ])
  kern.isKernel = true
  let m = moduleD("probe", @[], @[fFn, gFn, kern])

  let err = m.runMslLowering()
  doAssert err.len == 0, "the MSL lowering must succeed: " & err

  proc findProcs(m: CirModule, base: string): seq[CirProc] =
    for fn in m.procs:
      if fn.name == base or fn.name.startsWith(base & "_"):
        result.add fn

  # f, g, k keep their bodies plus one clone per distinct argument space
  doAssert m.procs.len == 5, "expected f, g, k plus two f instantiations, got " &
    $m.procs.len & ": " & (block:
      var names = ""
      for fn in m.procs: names.add fn.name & " "
      names)

  let fInsts = m.findProcs("f")
  doAssert fInsts.len == 3, "f must keep one instantiation per argument space, got " &
    $fInsts.len
  var spaces: HashSet[CirAddrSpace]
  for inst in fInsts:
    doAssert inst.params.len == 2
    spaces.incl inst.params[0].typ.space
    # clone body reads of the formal show the re-typed symbol
    let ret = inst.body[0]
    doAssert ret.kind == sRet and ret.retVal.kind == eDeref and
      ret.retVal.derefOperand.kind == eSym and
      sameType(ret.retVal.derefOperand.typ, inst.params[0].typ),
      "the instantiation " & inst.name & " must read its re-typed formal"
  doAssert spaces == [caDevice, caThreadgroup, caThread].toHashSet,
    "the f instantiations must cover the device, threadgroup, and thread spaces"

  # both threadgroup call sites share one instantiation name
  var kernBody: seq[CirStmt]
  for fn in m.procs:
    if fn.name == "k":
      kernBody = fn.body
  doAssert callCalleeName(kernBody[2]) == "f",
    "the matching-space call site keeps the callee"
  doAssert callCalleeName(kernBody[3]).len > 0 and
    callCalleeName(kernBody[3]) == callCalleeName(kernBody[4]),
    "the two threadgroup call sites must name one clone"
  doAssert callCalleeName(kernBody[5]) != callCalleeName(kernBody[3]),
    "the thread-space call site names a second clone"

  # g's threadgroup formal never specializes
  doAssert m.findProcs("g").len == 1

  # ═══ the verifier rejects a surviving module-callee space mismatch ═══
  let mismatchKern = procD("k2", @[param("out", tAptr(caDevice, f32))], tVoid(), @[
    asgnS(patE(sym("out", tAptr(caDevice, f32)), intLit(0, tInt(32))),
      callE(sym("h", tProcT(devP, tInt(32), f32)),
        addrE(tgP, atE(sym("scratch", arrT), intLit(0, tInt(32)))),
        intLit(0, tInt(32)))),
  ])
  mismatchKern.isKernel = true
  let mismatch = moduleD("mismatch", @[], @[
    procD("h", @[param("p", devP), param("i", tInt(32))], f32,
      @[retS(derefE(sym("p", devP)))]),
    mismatchKern,
  ])
  let msg = verifyMessage(mismatch)
  doAssert msg.contains("address space"),
    "the verifier must reject a module-callee space mismatch, got: " & msg
  doAssert msg.contains("h"), "the rejection must name the callee, got: " & msg

  # a foreign callee with a pointee-matching space mismatch stays legal
  let foreignKern = procD("k3", @[param("out", tAptr(caDevice, f32))], tVoid(), @[
    varD("priv", arrT, nil, caThread),
    callS(sym("foreign_copy", tProcT(thP, devP, tVoid())),
      addrE(thP, sym("priv", arrT)),
      addrE(devP, patE(sym("out", tAptr(caDevice, f32)),
        intLit(0, tInt(32))))),
  ])
  foreignKern.isKernel = true
  let foreign = moduleD("foreign", @[], @[foreignKern])
  doAssert verifyMessage(foreign).len == 0,
    "a foreign callee's pointee-matching space mismatch stays legal: " &
    verifyMessage(foreign)

echo "[OK] t_ir_space_generics"
