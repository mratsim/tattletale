## Hand-built CIRv2: the whole-array assignment decomposition registration.
##
## Test that the registered CUDA lowering passes rewrite every whole-array
## copy into a `__builtin_memcpy(&dst, &src, sizeof(T))` call and refuse
## the shapes the form cannot express:
##
## - an array `sAsgn` becomes the memcpy call over both addresses and the size
## - an array `sVar` with a value initializer splits into the plain
##   declaration plus the memcpy call
## - constructor inits, struct assignments, and scalar assignments pass through
## - a call or non-lvalue right-hand side refuses loudly
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_decompose_array_assign.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_lowering_cuda

proc runCudaLowering(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerCudaLoweringPasses()
  result = m.runPasses(reg)

proc isMemcpyCall(s: CirStmt, dst, src: string, arrType: CirType): bool =
  if s.kind != sCall:
    return false
  if s.callCallee.kind != eSym or s.callCallee.name != "__builtin_memcpy":
    return false
  let calleeType = s.callCallee.typ
  if calleeType.kind != ctProc or calleeType.ret.kind != ctVoid:
    return false
  if calleeType.params.len != 3:
    return false
  if s.callArgs.len != 3:
    return false
  let dstAddr = s.callArgs[0]
  let srcAddr = s.callArgs[1]
  let sizeExpr = s.callArgs[2]
  result = dstAddr.kind == eAddr and dstAddr.addrOperand.kind == eSym and
    dstAddr.addrOperand.name == dst and
    srcAddr.kind == eAddr and srcAddr.addrOperand.kind == eSym and
    srcAddr.addrOperand.name == src and
    sizeExpr.kind == eSizeof and sameType(sizeExpr.ofType, arrType)

static:
  let f32 = tFloat(32)
  let arrT = tArray(8, f32)
  let structT = tStruct("Pair", fld("x", f32))
  let outT = tAptr(caDevice, f32)
  let kernelFn = procD("k", @[param("output", outT)], tVoid(), @[
    varD("b", arrT, nil),
    varD("a", arrT, nil),
    asgnS(sym("a", arrT), sym("b", arrT)),
    varD("c", arrT, sym("b", arrT)),
    varD("lit", arrT, aconstrE(arrT, fltLit(1.0, f32), fltLit(2.0, f32))),
    varD("s", structT, nil),
    varD("s2", structT, nil),
    asgnS(sym("s", structT), sym("s2", structT)),
    varD("x", f32, nil),
    varD("y", f32, nil),
    asgnS(sym("x", f32), sym("y", f32)),
    iteS(trueE(), @[asgnS(sym("a", arrT), sym("b", arrT))], @[]),
    retS(),
  ])
  kernelFn.isKernel = true
  let m = moduleD("probe", @[], @[kernelFn])

  let err = m.runCudaLowering()
  doAssert err.len == 0, "the CUDA lowering must succeed: " & err
  let body = m.procs[0].body

  # array assignment decomposes into the memcpy call
  doAssert body[2].isMemcpyCall("a", "b", arrT),
    "the array assignment must decompose into the memcpy call over `a` and `b`"
  # array declaration init split into decl + memcpy
  doAssert body[3].kind == sVar and body[3].name == "c" and body[3].init == nil,
    "the array declaration init must split into a plain declaration"
  doAssert body[4].isMemcpyCall("c", "b", arrT),
    "the split declaration must copy through the memcpy call over `c` and `b`"
  # an array constructor initializer survives untouched
  doAssert body[5].kind == sVar and body[5].name == "lit" and
    body[5].init != nil and body[5].init.kind == eAconstr,
    "an array constructor initializer must stay on the declaration"
  # struct assignment is a legal whole-value copy
  doAssert body[8].kind == sAsgn and body[8].lhs.kind == eSym and
    body[8].lhs.name == "s",
    "a struct assignment must pass through untouched"
  # scalar assignment is untouched
  doAssert body[11].kind == sAsgn and body[11].lhs.name == "x",
    "a scalar assignment must pass through untouched"
  # a copy nested in a branch decomposes too
  let branch = body[12]
  doAssert branch.kind == sIte and branch.thenBody.len == 1 and
    branch.thenBody[0].isMemcpyCall("a", "b", arrT),
    "an array copy nested in a branch must decompose into the memcpy call"

  # refusal: a call right-hand side has no address to copy
  let callRhsFn = procD("kc", @[param("output", outT)], tVoid(), @[
    varD("a", arrT, nil),
    asgnS(sym("a", arrT), callE(sym("make", tProcT(@[], arrT)))),
    retS(),
  ])
  callRhsFn.isKernel = true
  let callRhsErr = moduleD("probe2", @[], @[callRhsFn]).runCudaLowering()
  doAssert callRhsErr.contains("call result"),
    "a whole-array assignment binding a call result must refuse, got: " & callRhsErr

  # refusals: an array constructor right-hand side is not an lvalue
  let constrRhsFn = procD("kd", @[param("output", outT)], tVoid(), @[
    varD("a", arrT, nil),
    asgnS(sym("a", arrT), aconstrE(arrT, fltLit(1.0, f32))),
    retS(),
  ])
  constrRhsFn.isKernel = true
  let constrRhsErr = moduleD("probe3", @[], @[constrRhsFn]).runCudaLowering()
  doAssert constrRhsErr.contains("not an lvalue"),
    "a whole-array assignment from a constructor must refuse, got: " & constrRhsErr

echo "  OK — whole-array copies decompose into memcpy calls, constructor inits," &
  " struct and scalar assignments pass through, unaddressable shapes refuse"
