## Vulkan: fp16 SSBO arithmetic through the GPU DSL, end to end.
##
## Requires a driver with VK_KHR_shader_float16_int8 + 16-bit storage
## (MoltenVK on Apple Silicon). Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_f16.nim

import std/[strutils, random]
import workspace/crucible

func f16ToF32(h: uint16): float32 =
  ## Exact binary16 → binary32: every f16 value is representable in f32.
  let sign = uint32(h and 0x8000'u16) shl 16
  let e = (h shr 10) and 0x1F'u16
  let m = h and 0x3FF'u16
  if e == 0:
    if m == 0:
      return cast[float32](sign)
    var m2 = uint32(m)
    var e2 = 113
    while (m2 and 0x400'u32) == 0:
      m2 = m2 shl 1
      dec e2
    return cast[float32](sign or (uint32(e2) shl 23) or ((m2 and 0x3FF'u32) shl 13))
  if e == 0x1F:
    return cast[float32](sign or 0x7F800000'u32 or (uint32(m) shl 13))
  return cast[float32](sign or ((uint32(e) + 112) shl 23) or (uint32(m) shl 13))

func f16OfInt(i: int): uint16 =
  ## Exact f16 encoding of the integers −15..15.
  if i == 0:
    return 0
  let neg = i < 0
  var mag = abs(i)
  var e = 0
  while mag > 1:
    mag = mag shr 1
    inc e
  let mant = ((abs(i) - (1 shl e)) shl 10) shr e
  var bits = uint16((e + 15) shl 10) or uint16(mant)
  if neg:
    bits = bits or 0x8000'u16
  return bits

const f16Vk = vulkan:
  proc f16MulKernel(C: ptr UncheckedArray[float16];
                    A, B: ptr UncheckedArray[float16]) {.global, workgroup: (64, 1, 1).} =
    let i = int(thread_position_in_grid.x)
    C[i] = A[i] * B[i]

proc runTest() =
  var engine = bkVulkan.init()
  engine.ingest(f16Vk)
  const N = 64
  var rng = initRand(0xF16)
  var A = newSeq[uint16](N)
  var B = newSeq[uint16](N)
  for i in 0 ..< N:
    A[i] = f16OfInt(rng.rand(-15 .. 15))
    B[i] = f16OfInt(rng.rand(-15 .. 15))
  var C = newSeq[uint16](N)
  engine.run("f16MulKernel", C, (A, B))
  # Products ≤ 225 are exact in f16 and f32: the gate is bit-exact.
  for i in 0 ..< N:
    let refVal = f16ToF32(A[i]) * f16ToF32(B[i])
    doAssert f16ToF32(C[i]) == refVal,
      "C[" & $i & "] = " & $f16ToF32(C[i]) & ", expected " & $refVal
  echo "  OK: fp16 SSBO multiply, 64 lanes, bit-exact vs the f32 reference"

when isMainModule:
  runTest()
