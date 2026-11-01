## Hand-built CIRv2: the call-graph analysis pass read-write classification.
##
## Test that the call-graph pass marks a buffer read-write exactly when a store reaches it, directly or through a callee.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_callgraph_rw_classification.nim

import std/[sets, strutils]
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/callgraph_analysis

proc runCallGraph(m: CirModule): string =
  ## Runs the call-graph analysis pass over `m` through the registry.
  var reg = PassRegistry.new()
  reg.registerHlslCallGraphPass()
  result = m.runPasses(reg)

static:
  let u32 = tUint(32)
  let buf = tAptr(caDevice, u32)
  let zero = intLit(0, u32)
  let one = intLit(1, u32)

  # k reads a, writes b, passes a to a device function that stores through it:
  # a and b are read-write, c is read-only
  let writeFn = procD("writeFirst", @[param("buf", buf)], tVoid(), @[
    asgnS(atE(sym("buf", buf), zero), one),
    retS(),
  ])
  let kernelFn = procD("k", @[param("a", buf), param("b", buf), param("c", buf)],
    tVoid(), @[
    constD("r", u32, atE(sym("a", buf), zero)),
    asgnS(atE(sym("b", buf), zero), one),
    constD("s", u32, atE(sym("c", buf), zero)),
    callS(sym("writeFirst", tProcT(buf, tVoid())), sym("a", buf)),
    retS(),
  ])
  kernelFn.isKernel = true
  let m = moduleD("probe", @[], @[kernelFn, writeFn])

  block classification:
    let err = m.runCallGraph()
    doAssert err.len == 0, "the call-graph pass must succeed: " & err
    let reached = m.callGraph.reached
    doAssert reached.contains("k/b"),
      "the buffer the kernel stores through is read-write"
    doAssert reached.contains("writeFirst/buf"),
      "the device-function parameter the callee stores through is read-write"
    doAssert reached.contains("k/a"),
      "the buffer passed to a storing callee is read-write"
    doAssert "k/c" notin reached,
      "a buffer the kernel only reads is not read-write"
    doAssert m.callGraph.edges == @[(caller: "k", arg: "a",
      callee: "writeFirst", param: "buf")],
      "the call argument rooted at the kernel parameter records one edge"

  block recursionRefusal:
    # a device function calling itself refuses: the frontend accepts
    # recursive source while miscompiling it
    let selfFn = procD("f", @[param("x", u32)], u32, @[
      callS(sym("f", tProcT(u32, u32)), sym("x", u32)),
      retS(sym("x", u32)),
    ])
    let recursionM = moduleD("probe2", @[], @[selfFn])
    let err = recursionM.runCallGraph()
    doAssert err.contains("recursive device function `f`"),
      "recursion must refuse, got: " & err

  block acyclicClean:
    # hazard negative: two device functions calling each other's caller
    # in a chain stay clean, only a cycle refuses
    let g = procD("g", @[param("x", u32)], u32, @[retS(sym("x", u32))])
    let h = procD("h", @[param("x", u32)], u32, @[
      asgnS(sym("x", u32), callE(sym("g", tProcT(u32, u32)), sym("x", u32))),
      retS(sym("x", u32)),
    ])
    let cleanM = moduleD("probe3", @[], @[g, h])
    let err = cleanM.runCallGraph()
    doAssert err.len == 0, "an acyclic call graph must stay clean: " & err
