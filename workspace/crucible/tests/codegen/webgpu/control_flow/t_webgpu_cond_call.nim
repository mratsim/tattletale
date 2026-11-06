## WebGPU: a device-function call in a condition launches and evaluates in place.
##
## Test that call trees in `if`, `while`, and ternary conditions compile
## through the real frontend and run on the device.
##
## - the if condition's call evaluates to the taken branch
## - the while condition's call re-evaluates per iteration until the call's value turns the condition false
## - the ternary condition's call selects one variant, WGSL's `select` evaluates both variants so both stay call-free
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/control_flow/t_webgpu_cond_call.nim

import std/unittest
import workspace/crucible

const code = webgpu:
  proc square(x: int32): int32 =
    x * x

  proc condKernel(output: ptr UncheckedArray[int32], n: int32) {.global.} =
    if square(n) > 4:
      output[0] = 1
    else:
      output[0] = 2
    var i = 0'i32
    var steps = 0'i32
    while square(i) < n:
      steps += 1
      i += 1
    output[1] = steps
    output[2] = if square(n) > 9: 10 else: 20

proc runCond(n: int32): array[3, int32] =
  ## One launch per call: device buffers serve a single launch so handles never need a copy.
  var engine = newEngine(runWebGPU)
  engine.ingest(code)
  let oDev = engine.allocDevice(int32, 3)
  engine.launchKernel("condKernel", oDev, (n,))
  engine.copyToHostFromDevice(addr result[0], oDev, 3)

proc runTest() =
  suite "WebGPU - device calls in conditions":
    test "if, while, and ternary conditions carry the call and evaluate in place":
      # square(3) = 9: if takes the > 4 branch, while steps i = 0, 1
      # (0 < 3, 1 < 3, then square(2) = 4 stops), the ternary's 9 > 9 is
      # false and takes the else variant
      let hot = runCond(3)
      check hot[0] == 1
      check hot[1] == 2
      check hot[2] == 20
      # square(1) = 1: if takes the else branch, while steps once
      # (square(0) = 0 < 1 fires and square(1) = 1 < 1 stops),
      # ternary takes the else variant
      let cold = runCond(1)
      check cold[0] == 2
      check cold[1] == 1
      check cold[2] == 20
      # square(0) = 0: while condition false immediately, the loop's call
      # never turns it true
      let zero = runCond(0)
      check zero[0] == 2
      check zero[1] == 0
      check zero[2] == 20

when isMainModule:
  runTest()
