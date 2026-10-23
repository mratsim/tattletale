## Metal: pointer struct fields and casts carry the resolved address-space
## qualifier (device view over a buffer param, smem view over a `{.smem.}`
## array). Both kernels run on-device and assert byte-exact values.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_ptr_struct_field_cast.nim

import std/[unittest]
import workspace/crucible

type
  DeviceView = object
    ## Non-owning view over device memory: the pointer field must emit
    ## `device`, resolved from the kernel buffer param at construction.
    data: ptr UncheckedArray[uint32]
    len: uint32

  SmemView = object
    ## Non-owning view over threadgroup memory: the pointer field must emit
    ## `threadgroup`, resolved from the `{.smem.}` var at construction.
    data: ptr UncheckedArray[uint32]
    len: uint32

  View = object
    ## One view type over both device and threadgroup memory: the printer
    ## must emit a distinct struct variant per address space.
    data: ptr UncheckedArray[uint32]
    len: uint32

proc makeDeviceView(data: ptr UncheckedArray[uint32], len: uint32): DeviceView {.inline.} =
  ## Mirror of ceramic `make_view`: the pointer-typed field value is an
  ## explicit cast of the `data` parameter.
  DeviceView(data: cast[ptr UncheckedArray[uint32]](data), len: len)

const msl = metal:
  proc ptrFieldDeviceKernel(output: ptr UncheckedArray[uint32];
                            input: ptr UncheckedArray[uint32]) {.global.} =
    let v = makeDeviceView(input, 4'u32)
    for i in 0 ..< v.len:
      output[i] = v.data[i]

  proc ptrFieldSmemKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[4, uint32]
    let t = thread_position_in_threadgroup.x
    scratch[t] = uint32(t * 3)
    syncthreads()
    let v = SmemView(data: cast[ptr UncheckedArray[uint32]](addr scratch[0]),
                     len: 4'u32)
    output[t] = v.data[3 - t]

  proc viewDeviceKernel(output: ptr UncheckedArray[uint32];
                        input: ptr UncheckedArray[uint32]) {.global.} =
    let v = View(data: cast[ptr UncheckedArray[uint32]](input), len: 4'u32)
    for i in 0 ..< v.len:
      output[i] = v.data[i]

  proc viewSmemKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[4, uint32]
    let t = thread_position_in_threadgroup.x
    scratch[t] = uint32(t * 3)
    syncthreads()
    let v = View(data: cast[ptr UncheckedArray[uint32]](addr scratch[0]),
                 len: 4'u32)
    output[t] = v.data[3 - t]

proc runTest() =
  suite "Metal - pointer struct fields and casts carry address spaces":
    test "device view field + cast qualify as `device` and run":
      var engine = bkMetal.newEngine()
      engine.ingest(msl)
      var input = [10'u32, 20'u32, 30'u32, 40'u32]
      let iDev = engine.allocDevice(uint32, 4)
      engine.copyToDeviceFromHost(iDev, addr input[0], 4)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel << (grid: (1, 1), blk: (1, 1)) >> (
        "ptrFieldDeviceKernel", oDev, (iDev,))
      var output: array[4, uint32]
      engine.copyToHostFromDevice(addr output[0], oDev, 4)
      for i in 0 ..< 4:
        check output[i] == input[i]
    test "smem view field + cast qualify as `threadgroup` and run":
      var engine = bkMetal.newEngine()
      engine.ingest(msl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel << (grid: (4, 1), blk: (4, 1)) >> (
        "ptrFieldSmemKernel", oDev, ())
      var output: array[4, uint32]
      engine.copyToHostFromDevice(addr output[0], oDev, 4)
      # lane L reads the slot written by lane 3-L (reverse through smem)
      for i in 0 ..< 4:
        check output[i] == uint32((3 - i) * 3)
    test "one view type over device and threadgroup memory specializes per space":
      var engine = bkMetal.newEngine()
      engine.ingest(msl)
      var input = [10'u32, 20'u32, 30'u32, 40'u32]
      let iDev = engine.allocDevice(uint32, 4)
      engine.copyToDeviceFromHost(iDev, addr input[0], 4)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel << (grid: (1, 1), blk: (1, 1)) >> (
        "viewDeviceKernel", oDev, (iDev,))
      var output: array[4, uint32]
      engine.copyToHostFromDevice(addr output[0], oDev, 4)
      for i in 0 ..< 4:
        check output[i] == input[i]
      engine.launchKernel << (grid: (4, 1), blk: (4, 1)) >> (
        "viewSmemKernel", oDev, ())
      engine.copyToHostFromDevice(addr output[0], oDev, 4)
      # lane L reads the slot written by lane 3-L (reverse through smem)
      for i in 0 ..< 4:
        check output[i] == uint32((3 - i) * 3)

when isMainModule:
  runTest()
