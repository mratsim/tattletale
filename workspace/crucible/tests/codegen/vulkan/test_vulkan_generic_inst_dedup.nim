## Vulkan generic device procs instantiated at multiple static-parameter
## values must emit one body per instantiation, since `signatureHash` erases
## static bindings and dedups all instantiations.
##
## Vulkan device procs here take scalars and return values because SSBO
## lowering does not apply to non-global device procs, and kernels carry
## `{.workgroup: (32, 1, 1).}` to match the launch `blk`.
##
## Run:
##   nim c -r --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/vulkan/test_vulkan_generic_inst_dedup.nim

import std/unittest
import workspace/crucible

proc scale1[Factor: static int](x: float32): float32 {.device.} =
  x * float32(Factor)

proc scale2[Factor: static int, Width: static int](x: float32): float32 {.device.} =
  x * float32(Factor) + float32(Width)

proc inner[Factor: static int](x: float32): float32 {.device.} =
  x * float32(Factor)

proc wrapper[N: static int](x: float32): float32 {.device.} =
  inner[N](x)

func fscale[Factor: static int](x: float32): float32 {.device.} =
  x * float32(Factor)

proc scaleSwap[N: static int, M: static int](x: float32): float32 {.device.} =
  x * float32(N) + float32(M)

const dedupSrc = vulkan:
  proc k16(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = scale1[16](inp[0])
  proc k32(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = scale1[32](inp[0])
  proc k7(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = scale1[7](inp[0])
  proc k2x4(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = scale2[2, 4](inp[0])
  proc k3x4(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = scale2[3, 4](inp[0])
  # Both `inner[N]` instantiations are created at one shared source line
  # inside the generic `wrapper` body.
  proc kw16(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = wrapper[16](inp[0])
  proc kw32(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = wrapper[32](inp[0])
  proc kf16(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = fscale[16](inp[0])
  proc kf32(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = fscale[32](inp[0])
  proc k57(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = scaleSwap[5, 7](inp[0])
  proc k75(outp, inp: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    outp[0] = scaleSwap[7, 5](inp[0])

proc runScale(launcher: string, inpVal = 1.0'f32): float32 =
  var engine = bkVulkan.init()
  engine.ingest(dedupSrc)
  var inp = newSeq[float32](1)
  inp[0] = inpVal
  var outp = newSeq[float32](1)
  engine.run<<(grid: (1, 1), blk: (32, 1))>>(launcher, outp, (inp,))
  result = outp[0]

proc runTest() =
  suite "Vulkan - generic device proc static-param instantiations":
    test "each single-static instantiation runs its own body":
      check runScale("k16") == 16.0'f32
      check runScale("k32") == 32.0'f32
      check runScale("k7") == 7.0'f32

    test "each two-static instantiation runs its own body":
      check runScale("k2x4") == 6.0'f32
      check runScale("k3x4") == 7.0'f32

    test "nested generic instantiations at a shared source line keep their own bodies":
      check runScale("kw16") == 16.0'f32
      check runScale("kw32") == 32.0'f32

    test "generic device func instantiations run their own bodies":
      check runScale("kf16") == 16.0'f32
      check runScale("kf32") == 32.0'f32

    test "swapped statics at distinct body positions keep their own bodies":
      check runScale("k57", inpVal = 2.0) == 17.0'f32
      check runScale("k75", inpVal = 2.0) == 19.0'f32

when isMainModule:
  runTest()
