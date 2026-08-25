## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/experiments/manual_tile_ragged_gemm_fp16 \
##   --nimcache:nimcache/experiments/manual_tile_ragged_gemm_fp16 \
##   workspace/ceramic/experiments/manual_tile_ragged_gemm_fp16.nim

import std/[strformat, strutils]
import workspace/crucible
import ../tests/tile_test_utils
import ../src/int_tuples
import ../src/layouts
import ../src/layout_constructors
import ../src/layout_indexing
import ../src/tensors
import ../src/ptr_arithmetic
import ../src/tile_algebra

const raggedZfMsl = metal:
  proc ragZf(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
             N, K, M: int32) {.global.} =
    # Zero-fill strategy, composed from the tile ops: the full K-loop over
    # the padded (zero-filled) A rows.
    let gd_a = gd(A, shape = (1, 1, M, K), stride = (0, 0, K, 1))
    let gd_b = gd(B, shape = (1, 1, N, K), stride = (0, 0, 1, N))
    let gd_d = gd(D, shape = (1, 1, M, N), stride = (0, 0, N, 1))
    var a_rtl: rt_l(float16, 32, 16)
    var b_rtr: rt_r(float16, 16, 32)
    var d_rtl: rt_l(float32, 32, 32, getTileConfig(float32, float16))
    d_rtl.zero()
    let OUTPUT_Y = threadgroup_position_in_grid.y
    let OUTPUT_X = threadgroup_position_in_grid.x
    let kLimit = K div int32(16)
    for k in 0'i32 ..< kLimit:
      loadTile(a_rtl, gd_a, (0, 0, OUTPUT_Y, k))
      loadTile(b_rtr, gd_b, (0, 0, OUTPUT_X, k))
      d_rtl.mma_AB(a_rtl, b_rtr)
    storeTile(gd_d, d_rtl, (0, 0, OUTPUT_Y, OUTPUT_X))

const raggedBrMsl = metal:
  proc ragBr(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
             Lengths: ptr UncheckedArray[uint16],
             N, K, M: int32) {.global.} =
    # Branching strategy, composed from the tile ops (the gemm kernels
    # carry no Lengths): the per-tile k-end ladder bounds the K-loop.
    let gd_a = gd(A, shape = (1, 1, M, K), stride = (0, 0, K, 1))
    let gd_b = gd(B, shape = (1, 1, N, K), stride = (0, 0, 1, N))
    let gd_d = gd(D, shape = (1, 1, M, N), stride = (0, 0, N, 1))
    var a_rtl: rt_l(float16, 32, 16)
    var b_rtr: rt_r(float16, 16, 32)
    var d_rtl: rt_l(float32, 32, 32, getTileConfig(float32, float16))
    d_rtl.zero()
    let OUTPUT_Y = threadgroup_position_in_grid.y
    let OUTPUT_X = threadgroup_position_in_grid.x
    let kLimit = min(K div int32(16), int32(tileKMax(Lengths, OUTPUT_Y)))
    for k in 0'i32 ..< kLimit:
      loadTile(a_rtl, gd_a, (0, 0, OUTPUT_Y, k))
      loadTile(b_rtr, gd_b, (0, 0, OUTPUT_X, k))
      d_rtl.mma_AB(a_rtl, b_rtr)
    storeTile(gd_d, d_rtl, (0, 0, OUTPUT_Y, OUTPUT_X))

# ═════════════════════════════════════════════════════════════════════════
#  Host reference: the dumb triple loop (ragged A + full B)
# ═════════════════════════════════════════════════════════════════════════

proc buildRaggedA(M, Kp: int; kEffs: seq[int]): seq[uint16] =
  ## fp32ToFp16(1 + 2m + 7k), exact fp16.
  doAssert kEffs.len == M
  var Ah = newSeq[uint16](M * Kp)
  for m in 0 ..< M:
    for k in 0 ..< kEffs[m]:
      Ah[m * Kp + k] = fp32ToFp16(float32(1 + 2 * m + 7 * k))
  result = Ah

proc buildB(K, Kp, N, Np: int): seq[uint16] =
  ## pattern fp32ToFp16(1 + 3k + 11n)
  result = newSeq[uint16](Kp * Np)
  for k in 0 ..< K:
    for n in 0 ..< N:
      result[k * Np + n] = fp32ToFp16(float32(1 + 3 * k + 11 * n))

func raggedRefGemm(M, N, Np, Kp: int; Ah, Bh: seq[uint16];
                   kEffs: seq[int]): seq[float32] =
  doAssert kEffs.len == M
  result = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< kEffs[m]:
        acc += fp16ToFp32(Ah[m * Kp + k]) * fp16ToFp32(Bh[k * Np + n])
      result[m * N + n] = acc

# ═════════════════════════════════════════════════════════════════════════
#  The on-device checker: one strategy per call, returns the output
# ═════════════════════════════════════════════════════════════════════════

proc checkRagged(
    engine: var auto; kernel: string;
    M, N, K: int; kEffs: seq[int]): seq[float32] =
  let Mp = ((M + 31) div 32) * 32
  let Np = ((N + 31) div 32) * 32
  let Kp = ((K + 15) div 16) * 16

  doAssert Kp mod 16 == 0
  doAssert kEffs.len == M

  let Ah = buildRaggedA(M, Kp, kEffs)
  let Bh = buildB(K, Kp, N, Np)
  var Lengths = newSeq[uint16](Mp)
  for m in 0 ..< Mp:
    if m < M:
      Lengths[m] = uint16(min(kEffs[m], Kp))
    else:
      Lengths[m] = 0'u16          # padded rows are empty

  var outC = newSeq[float32](Mp * Np)
  if kernel == "ragZf":
    engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >> (
      kernel, outC, (Ah, Bh, int32(Np), int32(Kp), int32(Mp)))
  else:
    engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >> (
      kernel, outC, (Ah, Bh, Lengths, int32(Np), int32(Kp), int32(Mp)))

  let refC = raggedRefGemm(M, N, Np, Kp, Ah, Bh, kEffs)
  var ok = true
  var worstD = 0.0'f32
  var nBad = 0
  var firstBad = ""
  for m in 0 ..< M:
    for n in 0 ..< N:
      let d = abs(outC[m * Np + n] - refC[m * N + n])
      if d > worstD: worstD = d
      if outC[m * Np + n] != outC[m * Np + n] or d > 1e-3'f32:
        inc nBad
        if firstBad.len == 0:
          firstBad = &"(m={m}, n={n}): got {outC[m * Np + n]}, want {refC[m * N + n]}"
  echo "  ", kernel, ": ",
       (if nBad == 0: &"PASS — worst |Δ| = {worstD} (tolerance 1e-3)"
        else: &"FAIL — {nBad} bad, e.g. {firstBad}, worst |Δ| = {worstD}")
  if nBad != 0:
    quit 1
  result = outC

proc checkIdentical(name: string; A, B: seq[float32]; M, N, Np: int) =
  ## The two strategies on the same inputs must be bit-identical
  ## over all outputs (the skipped/excess k contributions are exact +0.0).
  var worstD = 0.0'f32
  var nBad = 0
  var firstBad = ""
  for m in 0 ..< M:
    for n in 0 ..< N:
      let d = abs(A[m * Np + n] - B[m * Np + n])
      if d > worstD: worstD = d
      if A[m * Np + n] != B[m * Np + n]:
        inc nBad
        if firstBad.len == 0:
          firstBad = &"(m={m}, n={n}): {A[m * Np + n]} vs {B[m * Np + n]}"
  echo "  ", name, ": ",
       (if nBad == 0: &"bit-identical (all {M * N} outputs, worst |Δ| = {worstD})"
        else: &"DIFFER — {nBad} outputs, e.g. {firstBad}, worst |Δ| = {worstD}")
  if nBad != 0:
    quit 1

proc runTests() =   # engines are RAII, so keep them function-local
  checkDrawPins()          # the hash draws must not drift
  var zfEngine = bkMetal.init()
  var brEngine = bkMetal.init()
  zfEngine.ingest(raggedZfMsl)
  brEngine.ingest(raggedBrMsl)

  var k40 = newSeq[int](32)
  for m in 0 ..< 32:
    k40[m] = 40
  let M0 = 32
  let N0 = 32
  let K0 = 64
  let Np0 = ((N0 + 31) div 32) * 32
  #echo raggedZfMsl      # keep the generated MSL inspectable
  #echo raggedBrMsl      # keep the generated MSL inspectable
  echo &"ragged GEMM {M0}×{N0}×{K0} (K_eff = 40, both strategies):"
  let zf0 = checkRagged(zfEngine, "ragZf", M0, N0, K0, k40)
  let br0 = checkRagged(brEngine, "ragBr", M0, N0, K0, k40)
  checkIdentical("ragZf ≡ ragBr", zf0, br0, M0, N0, Np0)

  var draws = initShapeDraws("ragged")
  for draw in 0 ..< 2:
    let b0 = draws.nextBytes()
    let M = drawInRange(b0, 0, 1, 96)
    let N = drawInRange(b0, 1, 1, 96)
    let K = drawInRange(b0, 2, 1, 96)
    let Np = ((N + 31) div 32) * 32
    # per-row K_eff from the following digest bytes (20 rows per draw)
    var kEffs = newSeq[int](M)
    var i = 0
    var b = draws.nextBytes()
    for m in 0 ..< M:
      if i >= 20:
        i = 0
        b = draws.nextBytes()
      kEffs[m] = int(b[i]) mod (K + 1)
      inc i
    echo &"ragged GEMM {M}×{N}×{K} (K_eff ∈ [{kEffs.min},{kEffs.max}], both strategies):"
    let zf = checkRagged(zfEngine, "ragZf", M, N, K, kEffs)
    let br = checkRagged(brEngine, "ragBr", M, N, K, kEffs)
    checkIdentical("ragZf ≡ ragBr", zf, br, M, N, Np)

when isMainModule:
  runTests()
