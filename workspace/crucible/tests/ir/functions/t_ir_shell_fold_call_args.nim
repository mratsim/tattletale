## Constant-shell folding over calls in argument positions
##
## Test that `foldConstantShells` substitutes a shell call in any value position,
## call arguments included, and deletes the shell definition once no call site remains:
##
## - a zero-arg shell call as a call argument folds to its constant and its definition dies
## - a shell call in a declaration initializer folds the same way
## - a callee with parameters never folds, its call sites survive untouched
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_shell_fold_call_args.nim

import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_helpers
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_normalizations

proc runFold(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerNormalizationPasses()
  result = m.runPasses(reg)

proc holdsCallTo(body: seq[CirStmt], callee: string): bool =
  ## True when a call to `callee` occurs anywhere in the statement list,
  ## nested bodies and nested expressions included.
  var found = false
  let onExpr = proc(e: CirExpr) =
    if e.kind == eCall and e.callee.kind == eSym and e.callee.name == callee:
      found = true
  let onStmt = proc(s: CirStmt) =
    if s.kind == sCall and s.callCallee.kind == eSym and s.callCallee.name == callee:
      found = true
  body.walkStmts(onStmt, onExpr)
  result = found

proc holdsCallTo(procs: seq[CirProc], callee: string): bool =
  for fn in procs:
    if fn.body.holdsCallTo(callee):
      return true
  result = false

proc procNamed(procs: seq[CirProc], name: string): CirProc =
  ## Returns the proc named `name`, nil when absent.
  for fn in procs:
    if fn.name == name:
      return fn
  result = nil

static:
  let u16 = tUint(16)

  # A shell: zero parameters, a single-assignment body returning one constant.
  let zeroShell = procD("zero_q", @[], u16, @[
    varD("result", u16, intLit(0, u16)),
    retS(sym("result", u16))
  ])

  # A proc with parameters is never a shell, its call sites must survive.
  let noisy = procD("noise_q", @[param("x", u16)], u16, @[
    retS(sym("x", u16))
  ])

  let caller = procD("consumer", @[], tVoid(), @[
    # shell call nested in a statement-call argument position
    callS(sym("noise_q", tProcT(u16, u16)),
      @[intLit(7, u16), callE(sym("zero_q", tProcT([], u16)), @[])]),
    # shell call in a declaration initializer
    varD("t", u16, callE(sym("zero_q", tProcT([], u16)), @[]))
  ])

  var m = moduleD("shell_fold_call_args", @[], @[zeroShell, noisy, caller])

  let err = m.runFold()
  doAssert err == "", "fold pass refused: " & err

  let consumer = m.procs.procNamed("consumer")
  doAssert consumer != nil, "the consumer proc disappeared"

  # argument-position shell call folded to its constant
  block:
    let call = consumer.body[0]
    doAssert call.kind == sCall and call.callCallee.name == "noise_q"
    doAssert call.callArgs.len == 2
    doAssert call.callArgs[1].kind == eIntLit and call.callArgs[1].value == 0,
      "call argument did not fold to the shell constant: " & $call.callArgs[1].kind

  # declaration initializer folded to its constant
  block:
    let decl = consumer.body[1]
    doAssert decl.kind == sVar and decl.name == "t"
    doAssert decl.init.kind == eIntLit and decl.init.value == 0,
      "declaration initializer did not fold to the shell constant: " & $decl.init.kind

  # folded shell definition deleted, parameterized proc stays
  doAssert m.procs.procNamed("zero_q").isNil, "the folded shell's definition survived"
  doAssert not m.procs.procNamed("noise_q").isNil, "the parameterized proc was deleted"

  # hazard negative: a call to a proc with parameters survives in argument position
  doAssert m.procs.holdsCallTo("noise_q"), "the noise_q call site was removed"
  doAssert not m.procs.holdsCallTo("zero_q"), "a zero_q call site survived the fold"
