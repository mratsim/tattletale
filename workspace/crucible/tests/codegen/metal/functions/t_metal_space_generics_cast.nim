## Metal: per-address-space callee instantiation, cast chains inside the callee body.
##
## Test that every cast and pointer declaration inside a clone re-derives
## the re-spaced formal's storage, at any cast depth:
##
## - a cast directly over the formal, and a cast chain over a local alias of it
## - device and threadgroup call sites, launch + byte-exact readback
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_space_generics_cast.nim

import workspace/crucible

proc bitPattern(p: ptr UncheckedArray[float32]): uint32 {.device.} =
  let words = cast[ptr UncheckedArray[uint32]](p)
  words[0]

proc bitPatternChained(p: ptr UncheckedArray[float32]): uint32 {.device.} =
  let alias = p
  let words = cast[ptr UncheckedArray[uint32]](
    cast[ptr UncheckedArray[float32]](alias))
  words[0]

const spaceCastMsl = metal:
  proc kern(outp: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[4, float32]
    let t = thread_position_in_threadgroup.x
    if t == 0:
      scratch[0] = 1.5'f32
    threadgroup_barrier()
    if t == 0:
      outp[2] = 1069547520'u32
      outp[0] = bitPattern(cast[ptr UncheckedArray[float32]](addr outp[2]))
      outp[1] = bitPattern(cast[ptr UncheckedArray[float32]](addr scratch[0]))
      outp[3] = bitPatternChained(cast[ptr UncheckedArray[float32]](addr outp[2]))
      outp[4] = bitPatternChained(cast[ptr UncheckedArray[float32]](addr scratch[0]))

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(spaceCastMsl)
  let dev = engine.allocDevice(uint32, 5)
  engine.launchKernel<<(grid: (1, 1), blk: (4, 1))>>("kern", dev, ())
  var got: array[5, uint32]
  engine.copyToHostFromDevice(addr got[0], dev, 5)
  # 1.5'f32 is 0x3FC00000. Each call site must read those bits back through
  # its own instantiation: base callee on device space, clone on threadgroup
  for i in 0 ..< 5:
    doAssert got[i] == 1069547520'u32,
      "kern readback: element " & $i & " got " & $got[i] &
      " want 1069547520"
  echo "[OK] Metal: a clone's casts and declarations adopt the" &
    " re-spaced formal's storage at any cast depth, both call sites read" &
    " the same bits"

when isMainModule:
  runTest()
