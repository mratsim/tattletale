## Hand-built CIRv2: the inline splice pass binding value arguments.
##
## Test that a value argument binds to a fresh temp, so the spliced body's
## writes to the formal never reach the caller's variable.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_inline_value_param_temp.nim

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
  # callee assigns to its value formal: legal CIRv2, unexpressible in Nim source
  # Nim value formals are unassignable
  let bumpFn = procD("bump", @[param("x", tInt(32))], tInt(32), @[
    asgnS(sym("x", tInt(32)),
      addE(tInt(32), sym("x", tInt(32)), intLit(1, tInt(32)))),
    retS(sym("x", tInt(32))),
  ])
  bumpFn.forceInline = true
  let callerFn = procD("caller", @[], tInt(32), @[
    varD("v", tInt(32), intLit(5, tInt(32))),
    asgnS(sym("r", tInt(32)),
      callE(sym("bump", tProcT([tInt(32)], tInt(32))), sym("v", tInt(32)))),
    asgnS(sym("out0", tInt(32)), sym("v", tInt(32))),
    retS(sym("out0", tInt(32))),
  ])
  callerFn.isKernel = true
  let (outM, err) = splice(moduleD("probe", @[], @[callerFn, bumpFn]))
  doAssert err.len == 0, "value-arg splice must succeed: " & err
  # `v` keeps its init: a write to the formal must reach a temp only
  var vWritten = false
  for st in outM.procs[0].body:
    if st.kind == sAsgn and st.lhs.kind == eSym and st.lhs.name == "v" and
        st.rhs.kind == eAdd:
      vWritten = true
  doAssert not vWritten, "the caller's variable must not be written by the spliced body"
