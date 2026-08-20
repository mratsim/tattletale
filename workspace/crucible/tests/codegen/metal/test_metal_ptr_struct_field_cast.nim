## Metal: pointer-typed struct fields and pointer casts carry the resolved
## address-space qualifier.
##
## MSL rejects pointer spellings without an explicit address space: a
## `ptr T` struct field rendered bare (`ushort* data;`) and a
## `cast[ptr T]` rendered bare (`(ushort*)data`) both fail to compile. The
## printer resolves the space from the value's dataflow — the var's
## `{.smem.}`/`{.rmem.}`/`{.const_mem.}` pragma or the kernel buffer param,
## propagated through `addr`/casts/object construction, `asDevice` default —
## and emits it on both spellings.
##
## The device kernel builds a view over a kernel buffer param (the ceramic
## make_view shape); the smem kernel builds a view over a `{.smem.}` array
## (the threadgroup branch of the same propagation). Both run on-device and
## assert byte-exact values.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_ptr_struct_field_cast.nim

import std/[unittest, strutils]
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

proc makeDeviceView(data: ptr UncheckedArray[uint32]; len: uint32): DeviceView {.inline.} =
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
    scratch[thread_position_in_threadgroup.x] =
      thread_position_in_threadgroup.x * 3'u32
    syncthreads()
    let v = SmemView(data: cast[ptr UncheckedArray[uint32]](addr scratch[0]),
                     len: 4'u32)
    output[thread_position_in_threadgroup.x] =
      v.data[3'u32 - thread_position_in_threadgroup.x]

proc runTest() =
  suite "Metal - pointer struct fields and casts carry address spaces":
    test "device view field + cast qualify as `device` and run":
      var engine = bkMetal.init()
      engine.ingest(msl)
      let src = engine.getArtifact()
      check "device uint* data;" in src
      check "(device uint*)" in src
      var input = [10'u32, 20'u32, 30'u32, 40'u32]
      var output: array[4, uint32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> (
        "ptrFieldDeviceKernel", output, input)
      for i in 0 ..< 4:
        check output[i] == input[i]
    test "smem view field + cast qualify as `threadgroup` and run":
      var engine = bkMetal.init()
      engine.ingest(msl)
      let src = engine.getArtifact()
      check "threadgroup uint* data;" in src
      check "(threadgroup uint*)" in src
      var output: array[4, uint32]
      engine.run << (grid: (4, 1), blk: (4, 1)) >> (
        "ptrFieldSmemKernel", output, ())
      # lane L reads the slot written by lane 3-L (reverse through smem)
      for i in 0 ..< 4:
        check output[i] == uint32((3 - i) * 3)

when isMainModule:
  runTest()
