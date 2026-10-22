## Metal: `+%` through a struct's pointer field preserves the instance's
## address space when the struct has variants over two spaces.
##
## One `View` type, two constructions:
## - device memory in `viewDeviceKernel`
## - threadgroup memory in `viewSmemKernel`
##
## The `+%` cast qualifier resolves from the dot expression, it must
## match the instance's space in both kernels.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_view_variant_dot_space.nim

import std/unittest
import workspace/crucible

type
  View = object
    data: ptr UncheckedArray[uint32]
    len: uint32

template `+%`(p: ptr UncheckedArray[uint32], offset: SomeInteger): ptr UncheckedArray[uint32] =
  cast[ptr UncheckedArray[uint32]](cast[uint64](p) + uint64(offset) * 4)

const variantMsl = metal:
  proc viewDeviceKernel(output: ptr UncheckedArray[uint32], input: ptr UncheckedArray[uint32]) {.global.} =
    let v = View(data: cast[ptr UncheckedArray[uint32]](input), len: 4'u32)
    for i in 0 ..< v.len:
      output[i] = (v.data +% 1)[i]

  proc viewSmemKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[8, uint32]
    let t = thread_position_in_threadgroup.x
    scratch[t] = uint32(t * 3)
    syncthreads()
    let v = View(data: cast[ptr UncheckedArray[uint32]](addr scratch[0]),
                 len: 4'u32)
    if t < 4:
      output[t] = (v.data +% 1)[t]

proc runTest() =
  suite "Metal - view variant dot resolution keeps the instance address space":
    test "device view: +% through the field reads input[i + 1]":
      var engine = bkMetal.newEngine()
      engine.ingest(variantMsl)
      var input = [10'u32, 20'u32, 30'u32, 40'u32, 50'u32]
      let iDev = engine.allocDevice(uint32, input.len)
      engine.copyToDeviceFromHost(iDev, addr input[0], input.len)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel << (grid: (1, 1), blk: (1, 1)) >> (
        "viewDeviceKernel", oDev, (iDev,))
      var output: array[4, uint32]
      engine.copyToHostFromDevice(addr output[0], oDev, 4)
      for i in 0 ..< 4:
        check output[i] == input[i + 1]
    test "threadgroup view: +% through the field reads scratch[t + 1]":
      var engine = bkMetal.newEngine()
      engine.ingest(variantMsl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel << (grid: (1, 1), blk: (8, 1)) >> (
        "viewSmemKernel", oDev, ())
      var output: array[4, uint32]
      engine.copyToHostFromDevice(addr output[0], oDev, 4)
      for i in 0 ..< 4:
        check output[i] == uint32((i + 1) * 3)

when isMainModule:
  runTest()
