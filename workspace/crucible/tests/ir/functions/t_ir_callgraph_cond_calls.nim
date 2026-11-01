## Hand-built CIRv2: call-graph analysis over calls in condition positions.
##
## Test that the call-graph pass records the argument-flow edge of a call
## sitting in a loop condition or a ternary branch, so a buffer passed there
## classifies read-write when the callee stores through it.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_callgraph_cond_calls.nim

import std/sets
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/callgraph_analysis

proc runCallGraph(m: CirModule): string =
  ## Runs the recursion refusal and the call-graph analysis over `m` through the registry.
  var reg = PassRegistry.new()
  reg.registerDeviceRecursionPass()
  reg.registerHlslCallGraphPass()
  result = m.runPasses(reg)

static:
  let u32 = tUint(32)
  let buf = tAptr(caDevice, u32)
  let zero = intLit(0, u32)
  let one = intLit(1, u32)
  let storeSig = tProcT(buf, tVoid())

  # storeIt stores through its buffer parameter
  let storeFn = procD("storeIt", @[param("buf", buf)], tVoid(), @[
    asgnS(atE(sym("buf", buf), zero), one),
    retS(),
  ])

  # A loop condition calls storeIt with the kernel's buffer
  let condCallFn = procD("kCond", @[param("a", buf)], tVoid(), @[
    whileS(callE(sym("storeIt", storeSig), sym("a", buf)), @[retS()]),
    retS(),
  ])
  condCallFn.isKernel = true

  # a ternary branch value calls storeIt with the kernel's buffer
  let ternCallFn = procD("kTern", @[param("a", buf)], tVoid(), @[
    varD("r", u32),
    asgnS(sym("r", u32),
      ternaryE(u32, intLit(1, u32),
        callE(sym("storeIt", storeSig), sym("a", buf)), zero)),
    retS(),
  ])
  ternCallFn.isKernel = true

  let m = moduleD("condcalls", @[], @[condCallFn, ternCallFn, storeFn])

  block condCallEdge:
    let err = m.runCallGraph()
    doAssert err.len == 0, "the call-graph pass must succeed: " & err
    let reached = m.callGraph.reached
    doAssert reached.contains("kCond/a"),
      "the buffer passed to a storing callee in a loop condition is read-write"
    doAssert reached.contains("kTern/a"),
      "the buffer passed to a storing callee in a ternary branch is read-write"
    let condEdge: tuple[caller, arg, callee, param: string] =
      ("kCond", "a", "storeIt", "buf")
    let ternEdge: tuple[caller, arg, callee, param: string] =
      ("kTern", "a", "storeIt", "buf")
    doAssert m.callGraph.edges.contains(condEdge),
      "the loop-condition call site records one argument-flow edge"
    doAssert m.callGraph.edges.contains(ternEdge),
      "the ternary-branch call site records one argument-flow edge"
