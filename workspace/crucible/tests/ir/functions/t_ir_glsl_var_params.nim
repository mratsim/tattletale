## Hand-built CIRv2: the GLSL var-parameter qualification pass.
##
## Test that `glslQualifyVarParams` retypes `var T` parameters to their pointee
## for the GLSL inout binding and the GLSL bundle then dissolves cleanly:
##
## - a var parameter retypes to its pointee, keeps the `varParam` mark, its body derefs unwrap to the plain symbol
## - a call-site address-of argument unwraps to the plain lvalue, the callee
##   type at that call site re-types to the retyped signature
## - a var parameter passed through to another device function re-types at the call boundary and stays a plain symbol
## - a non-lvalue argument at a var parameter position refuses
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_glsl_var_params.nim

import std/[sets, strutils]
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/ir/cirv2_verify
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_lowering_glsl {.all.}

proc runGlslLowering(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerGlslLoweringPasses()
  result = m.runPasses(reg)

proc varParamProc(name: string, params: seq[CirParam], body: seq[CirStmt]): CirProc =
  ## A device function whose var parameters carry the frontend's mark.
  result = procD(name, params, tVoid(), body)
  for i in 0 ..< result.params.len:
    if result.params[i].typ.kind == ctPtr:
      result.params[i].varParam = true

static:
  let i32 = tInt(32)
  let xPtr = tPtr(caThread, i32)
  let outT = tAptr(caDevice, i32)

  # callee: `bump(x: var int32)`, the frontend lowers `x = x + 1` into param derefs
  let bump = varParamProc("bump", @[param("x", xPtr)], @[
    asgnS(derefE(sym("x", xPtr)),
      addE(i32, derefE(sym("x", xPtr)), intLit(1, i32))),
    retS(),
  ])
  # inner callee of the pass-through check: `g(x: var int32)`, body `x = 5`
  let g = varParamProc("g", @[param("x", xPtr)], @[
    asgnS(derefE(sym("x", xPtr)), intLit(5, i32)),
    retS(),
  ])
  # pass-through callee: `f(x: var int32)`, body `g(x)` passes the var parameter through
  let f = varParamProc("f", @[param("x", xPtr)], @[
    callS(sym("g", tProcT(xPtr, tVoid())), sym("x", xPtr)),
    retS(),
  ])
  # kernel: one local, one call with the address-of arg, one readback
  let kern = procD("k", @[param("out", outT)], tVoid(), @[
    varD("v", i32, intLit(3, i32)),
    callS(sym("bump", tProcT(xPtr, tVoid())),
      addrE(xPtr, sym("v", i32))),
    asgnS(patE(sym("out", outT), intLit(0, i32)), sym("v", i32)),
  ])
  kern.isKernel = true
  let m = moduleD("probe", @[], @[bump, f, g, kern])

  let err = m.runGlslLowering()
  doAssert err.len == 0, "the GLSL lowering must succeed: " & err

  # ═══ the var parameter retyped to its pointee, mark kept ═══
  var bumpOut: CirProc
  var fOut: CirProc
  var kernBody: seq[CirStmt]
  for fn in m.procs:
    case fn.name
    of "bump": bumpOut = fn
    of "f": fOut = fn
    of "k": kernBody = fn.body
    else: discard
  doAssert bumpOut != nil and fOut != nil
  doAssert bumpOut.params[0].typ.kind == ctInt and
    sameType(bumpOut.params[0].typ, i32),
    "the var parameter retypes to its pointee"
  doAssert bumpOut.params[0].varParam,
    "the var parameter keeps the inout mark for the printer"

  # ═══ the body derefs unwrapped to the plain symbol ═══
  let asgn = bumpOut.body[0]
  doAssert asgn.kind == sAsgn and asgn.lhs.kind == eSym and
    asgn.lhs.name == "x" and sameType(asgn.lhs.typ, i32),
    "the deref of the var parameter unwraps to the plain symbol"
  doAssert asgn.rhs.kind == eAdd and asgn.rhs.left.kind == eSym and
    asgn.rhs.left.name == "x", "the read side unwraps the same way"

  # ═══ the call-site address-of arg unwrapped, the callee type re-typed ═══
  let call = kernBody[1]
  doAssert call.kind == sCall and call.callArgs[0].kind == eSym and
    call.callArgs[0].name == "v" and sameType(call.callArgs[0].typ, i32),
    "the address-of call argument unwraps to the plain lvalue"
  doAssert call.callCallee.typ.kind == ctProc and
    call.callCallee.typ.params.len == 1 and
    sameType(call.callCallee.typ.params[0], i32),
    "the call site's callee type re-types to the retyped signature"

  # ═══ the pass-through var argument re-types at the call boundary ═══
  let fBody = fOut.body[0]
  doAssert fBody.kind == sCall and fBody.callArgs[0].kind == eSym and
    fBody.callArgs[0].name == "x" and sameType(fBody.callArgs[0].typ, i32),
    "a var parameter passed through re-types to the pointee at the call boundary"
  doAssert fBody.callCallee.typ.kind == ctProc and
    sameType(fBody.callCallee.typ.params[0], i32)

  # ═══ the post-pass module verifies clean ═══
  doAssert verifyMessage(m).len == 0,
    "the post-pass module must verify clean: " & verifyMessage(m)

  # ═══ a non-lvalue argument at a var parameter position refuses ═══
  let badBump = varParamProc("bump", @[param("x", xPtr)], @[
    asgnS(derefE(sym("x", xPtr)), intLit(1, i32)),
    retS(),
  ])
  let badKern = procD("k", @[param("out", outT)], tVoid(), @[
    callS(sym("bump", tProcT(xPtr, tVoid())), intLit(3, i32)),
  ])
  badKern.isKernel = true
  let bad = moduleD("bad", @[], @[badBump, badKern])
  let badErr = bad.runGlslLowering()
  doAssert badErr.contains("must be an lvalue"),
    "a non-lvalue var argument must refuse, got: " & badErr

echo "[OK] t_ir_glsl_var_params"
