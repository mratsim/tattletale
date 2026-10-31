## Vulkan: var-parameter device functions bind as GLSL inout (v1 port).
##
## Test that a device function taking `var T` parameters mutates the caller's
## storage through the inout bindings:
##
## - a scalar local, a local array element and a buffer element read back written
## - a var parameter passed through to another device function stays by reference
## - a var array parameter composes the GLSL array parameter form
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/functions/t_vulkan_var_param.nim

import std/unittest
import workspace/crucible

proc bump(x: var int32) =
  x = x + 1

proc scale(x: var int32, a: var array[4, float32]) =
  bump(x)
  a[1] = float32(x) * 2.0'f32

proc accumulate(c: ptr UncheckedArray[float32], i: int32, acc: var float32) =
  acc = acc + c[i]

const kernel = glsl:
  proc k(outBuf: ptr UncheckedArray[float32]) {.global.} =
    var v: int32 = 3
    var arr: array[4, float32]
    scale(v, arr)
    var acc: float32 = 0.0'f32
    accumulate(outBuf, 2, acc)
    accumulate(outBuf, 2, acc)
    # 3 + 1 = 4 through the pass-through bump, arr[1] = 8, acc = 2 * outBuf[2]
    outBuf[0] = float32(v) + arr[1] + acc

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernel

  suite "Vulkan - var-parameter device functions":

    test "inout bindings mutate locals, array elements and buffer elements":
      var engine = newEngine(runVulkan)
      engine.ingest(kernel)
      let oDev = engine.allocDevice(float32, 8)
      var seed: array[8, float32]
      for i in 0 ..< 8:
        seed[i] = 100.0'f32 + float32(i)
      engine.copyToDeviceFromHost(oDev, addr seed[0], seed.len)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("k", oDev, ())
      var res: array[8, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 8)
      # v: 3 -> 4 through the pass-through bump, arr[1] = 8, acc = 2 * outBuf[2]
      check res[0] == 4.0'f32 + 8.0'f32 + 2.0'f32 * 102.0'f32
      check res[2] == 102.0'f32

when isMainModule:
  runTest()
