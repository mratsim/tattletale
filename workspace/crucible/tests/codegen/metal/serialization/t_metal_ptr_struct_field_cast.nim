## Metal: pointer struct fields and casts carry the resolved address-space
## qualifier (device view over a buffer param, smem view over a `{.smem.}`
## array). Both kernels run on-device and assert byte-exact values.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_ptr_struct_field_cast.nim

import std/[unittest, strutils, sequtils]
import workspace/crucible

proc structBodies(msl: string): seq[(string, string)] =
  ## (def line, body text) pairs of every struct def, emission order.
  ## Body text runs from the def line through the closing brace
  let lines = msl.split('\n')
  var i = 0
  while i < lines.len:
    let s = lines[i].strip()
    if s.startsWith("struct ") and s.endsWith("{"):
      var body = s & "\n"
      inc i
      while i < lines.len and not lines[i].strip().startsWith("};"):
        body.add lines[i] & "\n"
        inc i
      if i < lines.len:
        body.add lines[i]
      result.add (s, body)
    inc i

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
      let src = engine.getArtifact()
      # the construction space equals the declared field's birth space
      # (device), the same-space rule keeps one def.
      # A cross-space construction spawns a `_<typehash>` instantiated def
      let viewBodies = src.structBodies.filterIt(it[0].startsWith("struct DeviceView"))
      check viewBodies.len == 1
      check viewBodies[0][0] == "struct DeviceView{"
      check "device uint* data;" in viewBodies[0][1]
      check "threadgroup" notin viewBodies[0][1]
      check src.split('\n').anyIt("(device uint*)" in it and "input" in it)
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
      let src = engine.getArtifact()
      # the instantiated def carries the construction's space (threadgroup),
      # the kernel uses it.
      # The declared def keeps the birth space
      let viewBodies = src.structBodies.filterIt(it[0].startsWith("struct SmemView"))
      check viewBodies.len == 2
      check viewBodies.anyIt(it[0] == "struct SmemView{")
      check viewBodies.anyIt(it[0].startsWith("struct SmemView_"))
      let inst = viewBodies.filterIt(it[0].startsWith("struct SmemView_"))
      check inst.len == 1 and "threadgroup uint* data;" in inst[0][1]
      check "device uint* data;" notin inst[0][1]
      check src.split('\n').anyIt("(threadgroup uint*)" in it and "scratch" in it)
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
      let src = engine.getArtifact()
      # the per-space variants are distinct defs.
      # The declared def keeps the plain name.
      # The instantiated def carries the type-hash instance name.
      # Each body carries exactly one space qualifier
      let viewBodies = src.structBodies.filterIt(it[0].startsWith("struct View"))
      check viewBodies.len == 2
      check viewBodies.anyIt(it[0] == "struct View{")
      check viewBodies.anyIt(it[0].startsWith("struct View_"))
      check viewBodies.countIt("device uint* data;" in it[1]) == 1
      check viewBodies.countIt("threadgroup uint* data;" in it[1]) == 1
      check viewBodies.allIt(not ("device uint* data;" in it[1] and
                                  "threadgroup uint* data;" in it[1]))
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
