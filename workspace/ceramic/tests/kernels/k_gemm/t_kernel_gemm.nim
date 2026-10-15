# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test:
##   k_tile_gemm matmul/gemm on Metal
##
## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_gemm/t_kernel_gemm.nim (from tattletale)
##
## - Cases: stride permutations at the atom tile size
## - Cases: tile dims not multiples of (MT, NT, KT)
## - Cases: excess squadrons, alpha scaling, alpha+beta with prefilled C

import std/[math, strformat]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/ism_coord_strides
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/ceramic/src/tile_algebra

{.experimental: "callOperator".}

const MT = 32
const NT = 32
const KT = 16

# ═══════════════════════════════════════════════════════════════
#  Kernels
# ═══════════════════════════════════════════════════════════════

proc gemmKernels(): string =
  metal:
    proc gemmK(D, A, B, C: ptr UncheckedArray[float32], M, N, K, sA0, sA1, sB0, sB1, sC0, sC1: int) {.global.} =
      const mma = tiledMma(float32, float16, float16, 32, 32, 16)
      let aView = make_view(cast[ptr UncheckedArray[float16]](A),
                            makeIntTuple((M, K)), makeIntTuple((sA0, sA1)))
      let bView = make_view(cast[ptr UncheckedArray[float16]](B),
                            makeIntTuple((N, K)), makeIntTuple((sB0, sB1)))
      let dView = make_view(D, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      matmul(mma, dView, aView, bView)
    proc gemmAlphaK(D, A, B, C: ptr UncheckedArray[float32], M, N, K, sA0, sA1, sB0, sB1, sC0, sC1: int) {.global.} =
      const mma = tiledMma(float32, float16, float16, 32, 32, 16)
      let aView = make_view(cast[ptr UncheckedArray[float16]](A),
                            makeIntTuple((M, K)), makeIntTuple((sA0, sA1)))
      let bView = make_view(cast[ptr UncheckedArray[float16]](B),
                            makeIntTuple((N, K)), makeIntTuple((sB0, sB1)))
      let dView = make_view(D, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      let cView = make_view(C, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      gemm(mma, dView, 2.0'f32, aView, bView, 0.0'f32, cView)
    proc gemmAlphaBetaK(D, A, B, C: ptr UncheckedArray[float32], M, N, K, sA0, sA1, sB0, sB1, sC0, sC1: int) {.global.} =
      const mma = tiledMma(float32, float16, float16, 32, 32, 16)
      let aView = make_view(cast[ptr UncheckedArray[float16]](A),
                            makeIntTuple((M, K)), makeIntTuple((sA0, sA1)))
      let bView = make_view(cast[ptr UncheckedArray[float16]](B),
                            makeIntTuple((N, K)), makeIntTuple((sB0, sB1)))
      let dView = make_view(D, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      let cView = make_view(C, makeIntTuple((M, N)), makeIntTuple((sC0, sC1)))
      gemm(mma, dView, 2.0'f32, aView, bView, 1.0'f32, cView)

# ═══════════════════════════════════════════════════════════════
#  binary16 conversion for the host-side reference
# ═══════════════════════════════════════════════════════════════

func fp32ToFp16(x: float32): uint16 =
  let u = cast[uint32](x)
  let sign = uint16((u shr 16) and 0x8000'u32)
  let aexp = int32((u shr 23) and 0xFF)
  let mant = u and 0x7FFFFF'u32
  if aexp == 0xFF:
    return sign or 0x7C00'u16
  var bexp = aexp - 127 + 15
  if bexp >= 31:
    return sign or 0x7C00'u16
  if bexp <= 0:
    if bexp < -10: return sign
    let m2 = mant or 0x800000'u32
    let shift = uint32(126 - aexp)
    let half = 1'u32 shl (shift - 1)
    var r = m2 shr shift
    let rem = m2 and ((1'u32 shl shift) - 1)
    if rem > half or (rem == half and (r and 1) == 1):
      r += 1
    return sign or uint16(r)
  let half = 0x1000'u32
  var r = mant shr 13
  let rem = mant and 0x1FFF'u32
  if rem > half or (rem == half and (r and 1) == 1):
    r += 1
    if r == 0x400:
      bexp += 1
      r = 0
      if bexp >= 31:
        return sign or 0x7C00'u16
  return sign or uint16(bexp shl 10) or uint16(r)

func fp16ToFp32(h: uint16): float32 =
  let sign = uint32(h and 0x8000'u16) shl 16
  let e = (h shr 10) and 0x1F'u16
  let m = h and 0x3FF'u16
  if e == 0:
    if m == 0: return cast[float32](sign)
    var m2 = uint32(m)
    var e2 = 113
    while (m2 and 0x400'u32) == 0:
      m2 = m2 shl 1
      dec e2
    return cast[float32](sign or (uint32(e2) shl 23) or ((m2 and 0x3FF'u32) shl 13))
  if e == 0x1F:
    return cast[float32](sign or 0x7F800000'u32 or (uint32(m) shl 13))
  return cast[float32](sign or (uint32(e + 112) shl 23) or (uint32(m) shl 13))

# ═══════════════════════════════════════════════════════════════
#  Case runner, one (M, N, K, strides, kernel) configuration
# ═══════════════════════════════════════════════════════════════

proc checkMatmul(engine: HwEngine, M, N, K: int, sA, sB, sC: (int, int),
    gxMul = 1, gyMul = 1, kernel = "gemmK", scale = 1.0'f32,
    prefill = false) =
  ## Runs one gemm configuration and asserts every output element
  ## against the host-side reference, worst |d| echoed as info.
  let
    gridX = ((M + MT - 1) div MT) * gxMul
    gridY = ((N + NT - 1) div NT) * gyMul
  var
    aHost = newSeq[uint16](M * K)
    bHost = newSeq[uint16](K * N)
    outHost = newSeq[float32](M * N)
  for m in 0 ..< M:
    for k in 0 ..< K:
      aHost[m * K + k] = fp32ToFp16(float32(1 + (2 * m + 7 * k) mod 13))
  for k in 0 ..< K:
    for n in 0 ..< N:
      bHost[k * N + n] = fp32ToFp16(float32(1 + (3 * k + 11 * n) mod 17))
  let
    aDev = engine.allocDevice(M * K * sizeof(uint16))
    bDev = engine.allocDevice(K * N * sizeof(uint16))
    cDev = engine.allocDevice(M * N * sizeof(float32))
    dDev = engine.allocDevice(M * N * sizeof(float32))
  engine.copyToDeviceFromHost(aDev, addr aHost[0], M * K * sizeof(uint16))
  engine.copyToDeviceFromHost(bDev, addr bHost[0], K * N * sizeof(uint16))
  if prefill:
    # C = A·B, the gemm then reads C from a buffer distinct from D
    engine.run << (grid: (gridX, gridY), blk: (32, 1)) >>
      ("gemmK",
       cDev,
       (aDev, bDev, dDev,
        M, N, K,
        sA[0], sA[1], sB[0], sB[1],
        sC[0], sC[1]))
  engine.run << (grid: (gridX, gridY), blk: (32, 1)) >>
    (kernel,
     dDev,
     (aDev, bDev, cDev,
      M, N, K,
      sA[0], sA[1], sB[0], sB[1],
      sC[0], sC[1]))
  engine.copyToHostFromDevice(addr outHost[0], dDev, M * N * sizeof(float32))
  engine.freeDevice(aDev)
  engine.freeDevice(bDev)
  engine.freeDevice(cDev)
  engine.freeDevice(dDev)
  let
    A = cast[ptr UncheckedArray[uint16]](addr aHost[0])
    B = cast[ptr UncheckedArray[uint16]](addr bHost[0])
    C = cast[ptr UncheckedArray[float32]](addr outHost[0])
  var bad = 0
  var worst = 0.0'f32
  var first = ""
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += fp16ToFp32(A[m * sA[0] + k * sA[1]]) * fp16ToFp32(B[n * sB[0] + k * sB[1]])
      let want = (scale + (if prefill: 1.0'f32 else: 0.0'f32)) * acc
      let got = C[m * sC[0] + n * sC[1]]
      let d = abs(got - want)
      worst = max(worst, d)
      if d > 1e-3'f32:
        inc bad
        if bad == 1: first = &"({m},{n}): got {got} want {want}"
  echo &"  {M}x{N}x{K} A{sA} B{sB} C{sC}" &
       (if gxMul != 1 or gyMul != 1: &" grid x{gxMul}x{gyMul}" else: "") &
       &" worst |d| = {worst}"
  doAssert bad == 0, &"{kernel} {M}x{N}x{K} A{sA} B{sB} C{sC}: " &
    &"{bad} bad, e.g. {first}, worst {worst}"

# ═══════════════════════════════════════════════════════════════
#  Cases
# ═══════════════════════════════════════════════════════════════

proc main =
  var engine = bkMetal.init()
  engine.ingest(gemmKernels())

  block:
    # every stride permutation at the atom tile size
    for sA in [(32, 1), (1, 32)]:
      for sB in [(1, 32), (32, 1)]:
        for sC in [(32, 1), (1, 32)]:
          engine.checkMatmul(32, 32, 32, sA, sB, sC)

  block:
    # ragged edges, tile dims not multiples of (MT, NT, KT)
    engine.checkMatmul(32, 32, 41, (41, 1), (1, 32), (32, 1))
    engine.checkMatmul(40, 32, 32, (32, 1), (1, 32), (32, 1))
    engine.checkMatmul(32, 45, 32, (32, 1), (1, 45), (45, 1))
    engine.checkMatmul(40, 45, 41, (41, 1), (1, 45), (45, 1))
    engine.checkMatmul(70, 50, 100, (100, 1), (1, 50), (50, 1))
    engine.checkMatmul(70, 50, 96, (96, 1), (1, 50), (50, 1))

  block:
    # excess squadrons, the launch grid over-covers the tile grid
    engine.checkMatmul(64, 64, 64, (64, 1), (1, 64), (64, 1), gxMul = 3, gyMul = 2)
    engine.checkMatmul(70, 50, 100, (100, 1), (1, 50), (50, 1), gxMul = 2, gyMul = 3)

  block:
    # alpha and beta, the beta case reads a prefilled C buffer
    engine.checkMatmul(32, 32, 32, (32, 1), (1, 32), (32, 1),
                       kernel = "gemmAlphaK", scale = 2.0'f32)
    engine.checkMatmul(32, 32, 32, (32, 1), (1, 32), (32, 1),
                       kernel = "gemmAlphaBetaK", scale = 2.0'f32, prefill = true)

when isMainModule:
  main()
