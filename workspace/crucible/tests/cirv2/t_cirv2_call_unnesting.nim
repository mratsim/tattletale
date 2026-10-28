## CIRv2 call unnesting: the destination-binding rule
##
## Test that the verifier rejects every call outside its two legal positions
## (asgn RHS bound to its destination, void statement) and rejects `and`/`or`
## operands carrying calls
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/cirv2 --nimcache:nimcache/tests/cirv2 \
##     workspace/crucible/tests/cirv2/t_cirv2_call_unnesting.nim

import
  workspace/crucible/src/codegen/ir/cirv2_lang,
  workspace/crucible/src/codegen/ir/cirv2_build,
  workspace/crucible/src/codegen/ir/cirv2_verify,
  workspace/crucible/tests/cirv2/cirv2_testutils

proc callEnv(): tuple[f32: CirType, u32t: CirType, fProc: CirType, vProc: CirType] =
  ## f32 scalars plus a `(f +32) -> (f +32)` proc type and a void proc type
  let f32 = tFloat(32)
  result.f32 = f32
  result.u32t = tUint(32)
  result.fProc = tProcT(f32, f32)
  result.vProc = tProcT(f32, tVoid())

static:
  let env = callEnv()
  let f32 = env.f32
  let x = sym("x", f32)

  # a call nested in operand position (RHS of an asgn, under an add)
  block:
    let fn = procD("nested_call", @[param("x", f32)], f32, @[
      asgnS(sym("y", f32), addE(f32, callE(sym("f", env.fProc), @[x]), x)),
    ])
    fn.expectVerifyError("call in operand position")

  # a call operand under `and`: a side effect in expression position
  block:
    let fn = procD("and_call_operand", @[], tVoid(), @[
      iteS(andE(trueE(), eqE(callE(sym("f", env.fProc), @[x]), x)),
        @[retS()]),
    ])
    fn.expectVerifyError("call in operand position")

  # a non-void call as a statement: the value has no destination
  block:
    let fn = procD("nonvoid_call_stmt", @[], tVoid(), @[
      callS(sym("g", env.fProc), @[x]),
    ])
    fn.expectVerifyError("non-void call must bind to a destination")

  # a void call as an asgn RHS: a void value cannot bind
  block:
    let fn = procD("void_call_rhs", @[param("x", f32)], tVoid(), @[
      asgnS(sym("y", f32), callE(sym("g", env.vProc), @[x])),
    ])
    fn.expectVerifyError("void call must be a statement")

  # a call in a var initializer: declarations are not call destinations
  block:
    let fn = procD("call_var_init", @[param("x", f32)], tVoid(), @[
      varD("y", f32, callE(sym("f", env.fProc), @[x])),
    ])
    fn.expectVerifyError("call in operand position")

  # a call in a ret: ret binds no destination
  block:
    let fn = procD("call_in_ret", @[param("x", f32)], f32, @[
      retS(callE(sym("f", env.fProc), @[x])),
    ])
    fn.expectVerifyError("call in operand position")
