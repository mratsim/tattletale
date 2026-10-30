## Hand-built CIRv2: the inline splice pass over return-carrying `ret`.
##
## Test that a trailing value-carrying `ret` moves its value into the call destination
## and that a single `ret` in a non-trailing position refuses.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/manual_ir_inline_ret_splice.nim

import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_norm_aliases
import workspace/crucible/src/codegen/passes/passes_legal_inline

proc splice(m: CirModule): tuple[m: CirModule, err: string] =
  ## Runs the inline splice pass over `m` through the registry.
  var reg = PassRegistry.new()
  reg.registerBuiltinKindAssignment()
  reg.registerInlineSplice()
  let err = m.runPasses(reg)
  result = (m, err)

static:
  # trailing `ret(expr)`: the value must reach the call destination
  let valueFn = procD("caller", @[], tInt(32), @[

    varD("r", tInt(32)),
    asgnS(sym("r", tInt(32)),
      callE(sym("addv", tProcT([tInt(32), tInt(32)], tInt(32))),
        intLit(3, tInt(32)), intLit(4, tInt(32)))),
    retS(sym("r", tInt(32))),
  ])
  valueFn.isKernel = true
  let addvFn = procD("addv", @[param("a", tInt(32)), param("b", tInt(32))],
    tInt(32), @[retS(addE(tInt(32), sym("a", tInt(32)), sym("b", tInt(32))))])
  addvFn.forceInline = true
  let (valueOut, valueErr) = splice(moduleD("probe", @[], @[valueFn, addvFn]))
  doAssert valueErr.len == 0, "trailing ret(expr) must splice: " & valueErr
  # spliced body: the call with the body plus `asgn r, a + b`
  var moved = false
  for st in valueOut.procs[0].body:
    if st.kind == sAsgn and st.lhs.kind == eSym and st.lhs.name == "r" and
        st.rhs.kind == eAdd:
      moved = true
  doAssert moved, "expected the returned value assigned to the call destination"

  # mid-body `ret`: no single-entry splice, the `ret` would return from the caller
  let midFn = procD("caller", @[], tVoid(), @[

    callS(sym("early", tProcT([], tVoid()))),
    asgnS(sym("r", tInt(32)), intLit(1, tInt(32))),
  ])
  midFn.isKernel = true
  let earlyFn = procD("early", @[], tVoid(), @[
    retS(nil),
    asgnS(sym("r", tInt(32)), intLit(2, tInt(32))),
  ])
  earlyFn.forceInline = true
  let (midOut, midErr) = splice(moduleD("probe", @[], @[midFn, earlyFn]))
  doAssert midErr.len > 0, "a mid-body ret must refuse, got: " & $midOut.procs[0].body.len
