## WebGPU (WGSL) generic device procs instantiated at multiple
## static-parameter values must emit one body per instantiation, since
## `signatureHash` erases static bindings and dedups all instantiations.
##
## Run:
##   nim c -r --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/webgpu/test_webgpu_generic_inst_dedup.nim

import std/unittest
import workspace/crucible

proc scale1[Factor: static int](outp, inp: ptr UncheckedArray[float32]) {.device.} =
  outp[0] = inp[0] * float32(Factor)

proc scale2[Factor: static int, Width: static int](
    outp, inp: ptr UncheckedArray[float32]) {.device.} =
  for c in 0 ..< Width:
    outp[c] = inp[c] * float32(Factor)

proc inner[Factor: static int](outp, inp: ptr UncheckedArray[float32]) {.device.} =
  outp[0] = inp[0] * float32(Factor)
  outp[1] = inp[1] * float32(Factor)

proc wrapper[N: static int](outp, inp: ptr UncheckedArray[float32]) {.device.} =
  inner[N](outp, inp)

func fscale[Factor: static int](outp, inp: ptr UncheckedArray[float32]) {.device.} =
  outp[0] = inp[0] * float32(Factor)

proc scaleSwap[N: static int, M: static int](
    outp, inp: ptr UncheckedArray[float32]) {.device.} =
  outp[0] = inp[0] * float32(N) + inp[1] * float32(M)

const dedupSrc = webgpu:
  proc k16(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scale1[16](outp, inp)
  proc k32(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scale1[32](outp, inp)
  proc k7(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scale1[7](outp, inp)
  proc k2x4(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scale2[2, 4](outp, inp)
  proc k3x4(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scale2[3, 4](outp, inp)
  # Both `inner[N]` instantiations are created at one shared source line
  # inside the generic `wrapper` body.
  proc kw16(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    wrapper[16](outp, inp)
  proc kw32(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    wrapper[32](outp, inp)
  proc kf16(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    fscale[16](outp, inp)
  proc kf32(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    fscale[32](outp, inp)
  proc k57(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scaleSwap[5, 7](outp, inp)
  proc k75(outp, inp: ptr UncheckedArray[float32]) {.global.} =
    scaleSwap[7, 5](outp, inp)

proc runScale(launcher: string, width = 1, outWidth = -1): seq[float32] =
  var engine = bkWGSL.init()
  engine.ingest(dedupSrc)
  let outW = if outWidth < 0: width else: outWidth
  var inp = newSeq[float32](width)
  for i in 0 ..< width:
    inp[i] = float32(i + 1)
  var outp = newSeq[float32](outW)
  engine.run<<(grid: (1, 1), blk: (32, 1))>>(launcher, outp, (inp,))
  result = outp

proc runTest() =
  suite "WebGPU (WGSL) - generic device proc static-param instantiations":
    test "each single-static instantiation runs its own body":
      check runScale("k16") == @[16.0'f32]
      check runScale("k32") == @[32.0'f32]
      check runScale("k7") == @[7.0'f32]

    test "each two-static instantiation runs its own body":
      check runScale("k2x4", width = 4) == @[2.0'f32, 4.0, 6.0, 8.0]
      check runScale("k3x4", width = 4) == @[3.0'f32, 6.0, 9.0, 12.0]

    test "nested generic instantiations at a shared source line keep their own bodies":
      check runScale("kw16", width = 2) == @[16.0'f32, 32.0]
      check runScale("kw32", width = 2) == @[32.0'f32, 64.0]

    test "generic device func instantiations run their own bodies":
      check runScale("kf16") == @[16.0'f32]
      check runScale("kf32") == @[32.0'f32]

    test "swapped statics at distinct body positions keep their own bodies":
      check runScale("k57", width = 2, outWidth = 1) == @[19.0'f32]
      check runScale("k75", width = 2, outWidth = 1) == @[17.0'f32]

when isMainModule:
  runTest()
