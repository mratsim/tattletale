## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## On-device tile-layer ragged GEMM (manual, Metal): the `gemm` kernel's
## strategies (zero-fill and branching) vs the fp32-exact host reference
## (tolerance 1e-3) over hash-randomized per-row lengths.
## The 8×8×8 mma's cross-lane reduction needs subgroup shuffles, which
## Apple's OpenCL-to-Metal translation rejects, so both strategies run
## on Metal on this machine; on OpenCL 2.0+ platforms they run on
## OpenCL. The fp32 accumulation order is deterministic, so the
## ragZf ≡ ragBr bit-identical gate stays valid.
##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/manual_tile_ragged_gemm_fp16 \
##   --nimcache:nimcache/tests/manual_tile_ragged_gemm_fp16 \
##   workspace/ceramic/tests/kernels_tiles/manual_tile_ragged_gemm_fp16.nim

import std/[strformat, strutils]
import workspace/crucible
import ../tile_test_utils
import ../../src/kernels/k_tile_gemm

# ═════════════════════════════════════════════════════════════════════════
#  The thin {.global.} launchers, one per strategy.
#  First param = the output buffer D: engine.run binds the separate outBuf argument to it
#  and the args tuple to the rest (the zero-bug trap).
# ═════════════════════════════════════════════════════════════════════════

const raggedZfMsl = metal:
  proc ragZf(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
             N, K, M: int32) {.global.} =
    # The full K-loop form: the padded A zeros contribute nothing.
    gemm(D, M, N, K, A, K, 1, B, N, 1, nil, false)

const raggedBrMsl = metal:
  proc ragBr(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
             Lengths: ptr UncheckedArray[uint16],
             N, K, M: int32) {.global.} =
    # The branching form: the k-end ladder bounds the loop.
    gemm(D, M, N, K, A, K, 1, B, N, 1, Lengths, true)

# ═════════════════════════════════════════════════════════════════════════
#  Host reference: the dumb triple loop (ragged A + full B)
# ═════════════════════════════════════════════════════════════════════════

proc buildRaggedA(M, Kp: int; kEffs: seq[int]): seq[uint16] =
  ## Builds the ragged A buffer for M rows with per-row effective K:
  ## row m occupies `Ah[m·Kp .. m·Kp + Kp)`, the first kEffs[m]
  ## elements real, the rest zero. The per-row padding makes the
  ## strided load's unconditional read safe and contributes zeros to
  ## the mma (the padding is the load's contract, the mma never
  ## guards). Values use the deterministic pattern
  ## fp32ToFp16(1 + 2m + 7k), exact fp16.
  doAssert kEffs.len == M
  var Ah = newSeq[uint16](M * Kp)
  for m in 0 ..< M:
    for k in 0 ..< kEffs[m]:
      Ah[m * Kp + k] = fp32ToFp16(float32(1 + 2 * m + 7 * k))
  result = Ah

proc buildB(K, Kp, N, Np: int): seq[uint16] =
  ## The full Kp×Np row-major B buffer (pattern fp32ToFp16(1 + 3k + 11n))
  ## with the real K×N data in the top-left. B rows k ≥ every K_eff
  ## are never load-bearing: they multiply the zero-filled A in the zero-fill variant
  ## and are skipped in the branching variant.
  ## The finite pattern keeps their products ±0.
  result = newSeq[uint16](Kp * Np)
  for k in 0 ..< K:
    for n in 0 ..< N:
      result[k * Np + n] = fp32ToFp16(float32(1 + 3 * k + 11 * n))

func raggedRefGemm(M, N, Np, Kp: int; Ah, Bh: seq[uint16];
                   kEffs: seq[int]): seq[float32] =
  ## The fp32-exact host reference is the plain triple loop
  ## D(m, n) = Σ_{k < kEffs[m]} A[m·Kp + k] · B[k·Np + n] (Kp = the
  ## padded row stride, the `K` the kernel sees). It runs over the
  ## real M×N shape; the loop bounds keep the padding out. fp16→fp32
  ## is exact and the fp16 products are exact in fp32. Plain fp32
  ## accumulation is therefore the exact-in-fp32 reference. The
  ## hardware MMA's internal accumulation order may differ, hence the
  ## 1e-3 tolerance.
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
  ## Runs one strategy on the shape and gates it against the fp32-exact
  ## host reference (1e-3, NaN is a hard failure). Returns the padded
  ## output for the cross-strategy bit-identical gate.
  let Mp = ((M + 31) div 32) * 32
  let Np = ((N + 31) div 32) * 32
  let Kp = ((K + 15) div 16) * 16
  # Guard: the kernel's K-loop steps 16-wide k-blocks, so K must stay a multiple
  # of 16 (a non-divisible dim would silently truncate the accumulation).
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

proc gateIdentical(name: string; A, B: seq[float32]; M, N, Np: int) =
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

  # K=64, K_eff = 40 for all rows: tileKMax = 40, block 3 (k=48..63)
  # is skipped. Rows' k=40..47 are zero-filled by the ragged predicate.
  var k40 = newSeq[int](32)
  for m in 0 ..< 32:
    k40[m] = 40
  let M0 = 32
  let N0 = 32
  let K0 = 64
  let Np0 = ((N0 + 31) div 32) * 32
  echo raggedZfMsl      # keep the generated MSL inspectable
  echo raggedBrMsl      # keep the generated MSL inspectable
  echo &"ragged GEMM {M0}×{N0}×{K0} (K_eff = 40, both strategies):"
  let zf0 = checkRagged(zfEngine, "ragZf", M0, N0, K0, k40)
  let br0 = checkRagged(brEngine, "ragBr", M0, N0, K0, k40)
  gateIdentical("ragZf ≡ ragBr", zf0, br0, M0, N0, Np0)

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
    gateIdentical("ragZf ≡ ragBr", zf, br, M, N, Np)

when isMainModule:
  runTests()
