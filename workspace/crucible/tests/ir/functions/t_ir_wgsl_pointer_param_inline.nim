## Hand-built CIRv2: the WGSL pointer-param call inlining pass.
##
## Test that `wgslInlinePointerParamCalls` inlines the statement calls whose
## callee takes a pointer that cannot cross a WGSL function signature:
##
## - a buffer-pointer callee splices over the call with the arguments
##   substituted at the parameter's uses, and drops from the module
## - a function-space pointer callee keeps a real call
## - a returning callee, a value-position call, an argument holding a call,
##   a transitively self-calling callee, each refuses
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_wgsl_pointer_param_inline.nim

import std/[sets, strutils]
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/ir/cirv2_verify
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_lowering_wgsl {.all.}

proc runWgslLowering(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerWgslLoweringPasses()
  result = m.runPasses(reg)

proc procNamed(m: CirModule, name: string): CirProc =
  ## Returns the module proc named `name`, nil when absent.
  for fn in m.procs:
    if fn.name == name:
      return fn

static:
  let i32 = tInt(32)
  let outT = tAptr(caDevice, i32)
  let fillT = tAptr(caDevice, i32)
  let fnPtrT = tPtr(caThread, i32)

  # crossing callee: `fill(p: ptr UncheckedArray[float32], n: int32)`,
  # body writes p[0] = 7 and p[n] = 9 through the buffer pointer,
  # an aptr parameter indexes with pat directly, the frontend's shape
  let fill = procD("fill", @[param("p", fillT), param("n", i32)], tVoid(), @[
    asgnS(patE(sym("p", fillT), intLit(0, i32)), intLit(7, i32)),
    asgnS(patE(sym("p", fillT), sym("n", i32)), intLit(9, i32)),
    retS(),
  ])
  # function-space callee: `bump(x: var int32)`, body `x = x + 1`
  let bump = procD("bump", @[param("x", fnPtrT)], tVoid(), @[
    asgnS(derefE(sym("x", fnPtrT)),
      addE(i32, derefE(sym("x", fnPtrT)), intLit(1, i32))),
    retS(),
  ])
  # kernel: the buffer-pointer call site and the function-space call site
  let kern = procD("k", @[param("out", outT)], tVoid(), @[
    varD("v", i32, intLit(41, i32)),
    callS(sym("fill", tProcT([fillT, i32], tVoid())),
      sym("out", outT), intLit(3, i32)),
    callS(sym("bump", tProcT([fnPtrT], tVoid())),
      addrE(fnPtrT, sym("v", i32))),
    asgnS(patE(sym("out", outT), intLit(2, i32)), sym("v", i32)),
  ])
  kern.isKernel = true
  let m = moduleD("probe", @[], @[fill, bump, kern])

  let err = m.runWgslLowering()
  doAssert err.len == 0, "the WGSL lowering must succeed: " & err

  # ═══ the crossing callee inlined and dropped from the module ═══
  doAssert m.procNamed("fill").isNil,
    "the buffer-pointer callee must drop from the module"

  # ═══ the spliced body substituted its arguments at the parameter's uses ═══
  var kernBody: seq[CirStmt]
  for fn in m.procs:
    if fn.name == "k":
      kernBody = fn.body
  doAssert kernBody.len == 5,
    "the call spliced two statements over its site, one local, the bump call" &
    " and the readback stay"
  let splice0 = kernBody[1]
  doAssert splice0.kind == sAsgn and splice0.lhs.kind == ePat and
    splice0.lhs.patBase.kind == eSym and splice0.lhs.patBase.name == "out" and
    sameType(splice0.lhs.patBase.typ, outT) and
    splice0.rhs.kind == eIntLit and splice0.rhs.value == 7,
    "the first spliced write reads p[0] = 7 through the substituted buffer"
  let splice1 = kernBody[2]
  doAssert splice1.kind == sAsgn and splice1.lhs.kind == ePat and
    splice1.lhs.patIndex.kind == eIntLit and splice1.lhs.patIndex.value == 3,
    "the second spliced write reads p[n] with n bound to the argument 3"

  # ═══ the function-space callee kept a real call ═══
  doAssert m.procNamed("bump") != nil,
    "a function-space pointer callee keeps a real call"
  let call = kernBody[3]
  doAssert call.kind == sCall and call.callCallee.kind == eSym and
    call.callCallee.name == "bump" and call.callArgs[0].kind == eAddr,
    "the function-space call site keeps its address-of argument"

  # ═══ the post-pass module verifies clean ═══
  doAssert verifyMessage(m).len == 0,
    "the post-pass module must verify clean: " & verifyMessage(m)

  # ═══ a callee whose body returns refuses ═══
  block:
    let rfill = procD("fill", @[param("p", fillT), param("n", i32)], tVoid(), @[
      asgnS(patE(sym("p", fillT), intLit(0, i32)), intLit(7, i32)),
      retS(),
      asgnS(patE(sym("p", fillT), intLit(1, i32)), intLit(8, i32)),
      retS(),
    ])
    let rkern = procD("k", @[param("out", outT)], tVoid(), @[
      callS(sym("fill", tProcT([fillT, i32], tVoid())),
        sym("out", outT), intLit(3, i32)),
    ])
    rkern.isKernel = true
    let rm = moduleD("ret", @[], @[rfill, rkern])
    let rerr = rm.runWgslLowering()
    doAssert rerr.contains("must not return"),
      "a callee with a mid-body return must refuse, got: " & rerr

  # ═══ a value-position call to a crossing callee refuses ═══
  block:
    let vfill = procD("fill", @[param("p", fillT), param("n", i32)], tVoid(), @[
      asgnS(patE(sym("p", fillT), intLit(0, i32)), intLit(7, i32)),
      retS(),
    ])
    let vkern = procD("k", @[param("out", outT)], tVoid(), @[
      varD("t", tVoid(), callE(sym("fill", tProcT([fillT, i32], tVoid())),
        sym("out", outT), intLit(3, i32))),
    ])
    vkern.isKernel = true
    let vm = moduleD("value", @[], @[vfill, vkern])
    let verr = vm.runWgslLowering()
    doAssert verr.contains("value-position"),
      "a value-position call must refuse, got: " & verr

  # ═══ an argument holding a call refuses ═══
  block:
    let ifill = procD("fill", @[param("p", fillT), param("n", i32)], tVoid(), @[
      asgnS(patE(sym("p", fillT), intLit(0, i32)),
        sym("n", i32)),
      retS(),
    ])
    let lenFn = procD("len", @[], i32, @[retS(intLit(3, i32))])
    let ikern = procD("k", @[param("out", outT)], tVoid(), @[
      callS(sym("fill", tProcT([fillT, i32], tVoid())),
        sym("out", outT), callE(sym("len", tProcT([], i32)))),
    ])
    ikern.isKernel = true
    let im = moduleD("impure", @[], @[ifill, lenFn, ikern])
    let ierr = im.runWgslLowering()
    doAssert ierr.contains("re-evaluate"),
      "an argument holding a call must refuse, got: " & ierr

  # ═══ a transitively self-calling crossing callee refuses ═══
  block:
    let rec = procD("rec", @[param("p", fillT), param("n", i32)], tVoid(), @[
      callS(sym("rec", tProcT([fillT, i32], tVoid())),
        sym("p", fillT), sym("n", i32)),
      retS(),
    ])
    let recurKern = procD("k", @[param("out", outT)], tVoid(), @[
      callS(sym("rec", tProcT([fillT, i32], tVoid())),
        sym("out", outT), intLit(3, i32)),
    ])
    recurKern.isKernel = true
    let recurM = moduleD("recur", @[], @[rec, recurKern])
    let recurErr = recurM.runWgslLowering()
    doAssert recurErr.contains("not terminate"),
      "a self-calling crossing callee must refuse, got: " & recurErr

echo "[OK] t_ir_wgsl_pointer_param_inline"
