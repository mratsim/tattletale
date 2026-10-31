## Metal: generic device procs in a comment-shifted source each run
## their own static parameter at runtime.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/generics/t_metal_generic_inst_name_stability.nim

import std/unittest
import workspace/crucible

proc scaleB[Factor: static int](
    outp, inp: ptr UncheckedArray[float32]) {.device.} =
  # Filler comment line one.
  # Filler comment line two.

  outp[0] = inp[0] * float32(Factor)
  # Trailing filler comment.

  # Blank filler line.

const artB = metal:
  # Filler comment before the kernels.

  proc kB16(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    # Filler comment inside the kernel.
    scaleB[16](outp, inp)

  proc kB32(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scaleB[32](outp, inp)
    # Trailing filler comment.

  proc kB7(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scaleB[7](outp, inp)

proc runKernel(artifact: string, launcher: string): seq[float32] =
  var engine = newEngine(runMetal)
  engine.ingest(artifact)
  var inp = newSeq[float32](1)
  inp[0] = 1.0'f32
  var outp = newSeq[float32](1)
  let iDev = engine.allocDevice(float32, 1)
  engine.copyToDeviceFromHost(iDev, addr inp[0], 1)
  let oDev = engine.allocDevice(float32, 1)
  engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>(launcher, oDev, (iDev,))
  engine.copyToHostFromDevice(addr outp[0], oDev, 1)
  result = outp

proc runTest() =
  suite "Metal - generic instantiation under comment-shifted sources":
    test "the comment-shifted variant computes its own factors at runtime":
      check runKernel(artB, "kB16") == @[16.0'f32]
      check runKernel(artB, "kB32") == @[32.0'f32]
      check runKernel(artB, "kB7") == @[7.0'f32]

when isMainModule:
  runTest()
