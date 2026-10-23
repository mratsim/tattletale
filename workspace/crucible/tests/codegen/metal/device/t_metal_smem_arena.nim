## Metal: `{.smem.}` arena pass. A device function's `{.smem.}` var hoists
## to a `threadgroup T*` parameter, the caller supplies the arena.
##
## - raw ingest: a user kernel declares the arena and calls the device fn
## - launchDeviceFn: the generated wrapper kernel synthesizes the arena
## - the output is byte-exact against the host reference
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_smem_arena.nim

import std/unittest
import workspace/crucible

type Stage = object
  ## Shared-storage object: the arena wraps its scratch in a named struct,
  ## CUTLASS `SharedStorage` style, fields grow without resizing the launch
  data: array[64, float32]

const smemArenaMsl = metal:
  proc fold(input: ptr UncheckedArray[float32], output: ptr UncheckedArray[float32]) {.inline.} =
    ## Device function owning its scratch: hoisted to a `threadgroup float*` param
    var scratch {.smem.}: array[64, float32]
    let t = thread_position_in_threadgroup.x
    scratch[t] = input[t]
    threadgroup_barrier()
    var acc = 0.0'f32
    for i in 0 ..< 64:
      acc += scratch[i]
    output[t] = acc
  proc sumSmem(output: ptr UncheckedArray[float32], input: ptr UncheckedArray[float32]) {.global.} =
    # Kernel entry owning the arena, matched to fold's hoisted var by name
    var scratch {.smem.}: array[64, float32]
    fold(input, output)

proc foldKernel(output, input: ptr UncheckedArray[float32]) {.inline.} =
  ## Device proc declaring its own scratch, launched through `launchDeviceFn`,
  ## the generated wrapper kernel synthesizes the arena
  var scratch {.smem.}: array[64, float32]
  let t = thread_position_in_threadgroup.x
  scratch[t] = input[t]
  threadgroup_barrier()
  var acc = 0.0'f32
  for i in 0 ..< 64:
    acc += scratch[i]
  output[t] = acc

proc runTest() =
  suite "Metal - {.smem.} arena pass":
    test "device fn scratch hoists to a threadgroup param, kernel owns the arena":
      var engine = newEngine(runMetal)
      engine.ingest(smemArenaMsl)
      let iDev = engine.allocDevice(float32, 64)
      let oDev = engine.allocDevice(float32, 64)
      var input: array[64, float32]
      for i in 0 ..< 64:
        input[i] = float32(i)
      engine.copyToDeviceFromHost(iDev, addr input[0], 64)
      engine.launchKernel<<(grid: (1, 1), blk: (64, 1))>>("sumSmem", oDev, (iDev,))
      var res: array[64, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 64)
      for i in 0 ..< 64:
        check res[i] == 2016.0'f32

    test "an object-wrapped arena hoists as a threadgroup reference":
      const smemObjMsl = metal:
        proc foldObj(input: ptr UncheckedArray[float32], output: ptr UncheckedArray[float32]) {.inline.} =
          var scratch {.smem.}: Stage
          let t = thread_position_in_threadgroup.x
          scratch.data[t] = input[t]
          threadgroup_barrier()
          var acc = 0.0'f32
          for i in 0 ..< 64:
            acc += scratch.data[i]
          output[t] = acc
        proc sumSmemObj(output: ptr UncheckedArray[float32], input: ptr UncheckedArray[float32]) {.global.} =
          var scratch {.smem.}: Stage
          foldObj(input, output)
      var engine = newEngine(runMetal)
      engine.ingest(smemObjMsl)
      let iDev = engine.allocDevice(float32, 64)
      let oDev = engine.allocDevice(float32, 64)
      var input: array[64, float32]
      for i in 0 ..< 64:
        input[i] = float32(i)
      engine.copyToDeviceFromHost(iDev, addr input[0], 64)
      engine.launchKernel<<(grid: (1, 1), blk: (64, 1))>>("sumSmemObj", oDev, (iDev,))
      var res: array[64, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 64)
      for i in 0 ..< 64:
        check res[i] == 2016.0'f32

    test "launchDeviceFn synthesizes the arena for a hoisting device proc":
      var engine = newEngine(runMetal)
      let iDev = engine.allocDevice(float32, 64)
      let oDev = engine.allocDevice(float32, 64)
      var input: array[64, float32]
      for i in 0 ..< 64:
        input[i] = float32(i)
      engine.copyToDeviceFromHost(iDev, addr input[0], 64)
      engine.launchDeviceFn foldKernel(oDev, iDev) do:
        grid = (1, 1)
        blk = (64, 1)
      var res: array[64, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 64)
      for i in 0 ..< 64:
        check res[i] == 2016.0'f32

when isMainModule:
  runTest()
