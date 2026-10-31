## Vulkan (GLSL): the `not` operator on a bool expression
##
## Test that a `glsl:` kernel applying `not` to a bool comparison writes
## the negated result through the Vulkan engine.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/app/t_vulkan_glsl_not_expr.nim

import workspace/crucible
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/glsl_lang

proc buildNotModule(): CirModule =
  ## One kernel with a bool local bound to `not (a == 0)`
  let i32 = tInt(32)
  let kernel = procD("notKernel", @[param("a", i32)], tVoid(), @[
    varD("nb", tBool(), notE(eqE(sym("a", i32), intLit(0, i32)))),
    retS(),
  ])
  kernel.isKernel = true
  moduleD("notModule", @[], @[kernel])

const emission = static:
  printGlslModule(buildNotModule())

const notVk = glsl:
  proc notKernel(output, a: ptr UncheckedArray[uint32]) {.global.} =
    let flag = not (a[0] == 0'u32)
    if flag:
      output[0] = 1'u32
    else:
      output[0] = 0'u32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # printer check first: the GLSL module prints and the `not` operator survives
  doAssert emission.err.len == 0,
    "GLSL emission refused the `not` expression: " & emission.err

  # device check next: the kernel runs and the negation reaches the output.
  # Sentinel output: only a real kernel write can replace 9
  let sentinel: uint32 = 9

  block: # not(0 == 0) is false
    var engine = newEngine(runVulkan)
    engine.ingest(notVk)
    var input: array[1, uint32] = [0'u32]
    let aDev = engine.allocDevice(uint32, 1)
    let oDev = engine.allocDevice(uint32, 1)
    engine.copyToDeviceFromHost(aDev, addr input[0], 1)
    engine.copyToDeviceFromHost(oDev, addr sentinel, 1)
    engine.launchKernel("notKernel", oDev, (aDev,))
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    doAssert res[0] == 0'u32, "not(0 == 0) must be false, got " & $res[0]

  block: # not(5 == 0) is true
    var engine = newEngine(runVulkan)
    engine.ingest(notVk)
    var input: array[1, uint32] = [5'u32]
    let aDev = engine.allocDevice(uint32, 1)
    let oDev = engine.allocDevice(uint32, 1)
    engine.copyToDeviceFromHost(aDev, addr input[0], 1)
    engine.copyToDeviceFromHost(oDev, addr sentinel, 1)
    engine.launchKernel("notKernel", oDev, (aDev,))
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    doAssert res[0] == 1'u32, "not(5 == 0) must be true, got " & $res[0]

when isMainModule:
  runTest()
