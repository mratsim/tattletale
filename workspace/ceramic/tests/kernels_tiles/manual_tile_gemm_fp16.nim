## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/manual_tile_gemm_fp16 \
##   --nimcache:nimcache/tests/manual_tile_gemm_fp16 \
##   workspace/ceramic/tests/kernels_tiles/manual_tile_gemm_fp16.nim

import std/[strformat]
import workspace/crucible
import ../tile_test_utils
import ../../src/kernels/k_tile_gemm

# ═════════════════════════════════════════════════════════════════════════
#  The thin {.global.} launcher. First param = the output buffer D:
#  engine.run binds the separate outBuf argument to it and the args
#  tuple to the rest (the zero-bug trap).
# ═════════════════════════════════════════════════════════════════════════

const gemmMsl = metal:
  proc fusedGemm(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                 N, K, M: int32, rsa, rsb, rsd: int32) {.global.} =
    # Raw M/N/K are the ragged limits. The strides are the padded buffer
    # pitches (the caller's allocation is padded, the kernel zero-fills the
    # edge lanes in-kernel and masks the store to the real region).
    gemm_with_epilogue(D, rsd, 1, A, rsa, 1, B, rsb, 1, N, K, M,
                       EpiIdentity())

# ═════════════════════════════════════════════════════════════════════════
#  Canary bit patterns for the padded region
# ═════════════════════════════════════════════════════════════════════════

const aCanary = 0xFACA'u16       # A padding lanes (finite ≈ −5.6e4: leaked lanes swamp the accumulator)
const bCanary = 0xDEAD'u16       # B padding lanes (finite ≈ −4.3e2)
const dCanary = 0xBEEFDEAD'u32   # D padding lanes (finite fp32 bit pattern)

# ═════════════════════════════════════════════════════════════════════════
#  Host reference: fp32-exact from f16→f32, over the real shape
# ═════════════════════════════════════════════════════════════════════════

func refGemm(M, N, K, Mp, Np, Kp: int; Ah, Bh: seq[uint16]): seq[float32] =
  ## D(m, n) = Σ_{k < K} A[m·Kp + k]·B[k·Np + n] for m < M, n < N.
  ## The plain triple loop over the real shape indexes the padded buffers
  ## with the padded strides. The loop bounds keep the canary padding out.
  ## f16→f32 is exact and the fp16 products are exact in fp32,
  ## so plain fp32 accumulation is the exact-in-fp32 reference.
  result = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += fp16ToFp32(Ah[m * Kp + k]) * fp16ToFp32(Bh[k * Np + n])
      result[m * N + n] = acc

# ═════════════════════════════════════════════════════════════════════════
#  Tests
# ═════════════════════════════════════════════════════════════════════════

proc checkGemm(engine: var auto; kernel: string; M, N, K: int) =
  ## Fills the padded A (Mp×Kp) and B (Kp×Np) with the deterministic pattern
  ## in the real M×N×K region and canaries in the padding, runs the kernel
  ## with the RAW M/N/K (the grid still covers ceil'd tiles), and compares
  ## the real M×N outputs against the fp32-exact host reference with the 0.0
  ## tolerance. The D padding must still hold the canary after the run
  ## (the masked store leaves it untouched).
  let Mp = ((M + 31) div 32) * 32
  let Np = ((N + 31) div 32) * 32
  # The K-padding floor is 16: a K = 0 shape still uploads non-empty
  # A and B buffers (the harness rejects empty args).
  let Kp = max(((K + 15) div 16) * 16, 16)
  var Ah = newSeq[uint16](Mp * Kp)
  var Bh = newSeq[uint16](Kp * Np)
  for i in 0 ..< Ah.len: Ah[i] = aCanary
  for i in 0 ..< Bh.len: Bh[i] = bCanary
  for m in 0 ..< M:
    for k in 0 ..< K:
      Ah[m * Kp + k] = fp32ToFp16(float32(1 + 2 * m + 7 * k))
  for k in 0 ..< K:
    for n in 0 ..< N:
      Bh[k * Np + n] = fp32ToFp16(float32(1 + 3 * k + 11 * n))
  var C = newSeq[float32](Mp * Np)
  for i in 0 ..< C.len: C[i] = cast[float32](dCanary)
  # Raw dims: the kernel sees M/N/K as-is and handles the ragged edges
  # in-kernel (bounded zero-filled loads, a ceil'd K loop, masked stores).
  # The strides convey the padded buffer pitches (Kp, Np, Np).
  engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >> (
    kernel, C,
    (Ah, Bh, int32(M), int32(K), int32(N),
     int32(Kp), int32(Np), int32(Np)))

  let refC = refGemm(M, N, K, Mp, Np, Kp, Ah, Bh)
  const gemmGate = 0.0'f32
  var ok = true
  var worst = 0.0'f32
  var firstBad = ""
  var nBad = 0
  for m in 0 ..< M:
    for n in 0 ..< N:
      let d = abs(C[m * Np + n] - refC[m * N + n])
      if d > worst: worst = d
      if d != d or d > gemmGate:      # NaN is a failure, not a pass
        ok = false
        inc nBad
        if firstBad.len == 0:
          firstBad = &"(m={m}, n={n}): got {C[m * Np + n]}, want {refC[m * N + n]}"
  # D-canary check: the masked store must leave the padding untouched.
  var nCanary = 0
  var firstCanary = ""
  for m in 0 ..< Mp:
    for n in 0 ..< Np:
      if m >= M or n >= N:
        if cast[uint32](C[m * Np + n]) != dCanary:
          inc nCanary
          if firstCanary.len == 0:
            firstCanary = &"(m={m}, n={n}): got {cast[uint32](C[m * Np + n]):08x}"
  echo &"GEMM {M}×{N}×{K} ({kernel}, grid ({Np div 32},{Mp div 32}), " &
       &"padded {Mp}×{Np}×{Kp}, K-loop {(K + 15) div 16} iters): " &
       (if ok: &"PASS — worst |Δ| = {worst} (tolerance 0.0)"
        else: &"FAIL — {nBad} bad, e.g. {firstBad}, worst |Δ| = {worst}") &
       (if nCanary == 0: ", D-canary intact"
        else: &", D-canary CORRUPTED: {nCanary} cells, e.g. {firstCanary}")
  if not ok or nCanary > 0:
    quit 1

proc runTests() =   # engines are RAII, so keep them function-local
  checkDrawPins()          # the hash draws must not drift
  var engine = bkMetal.init()
  engine.ingest(gemmMsl)
  echo gemmMsl          # keep the generated MSL inspectable

  # The K = 0 empty-tile case: the kernel executes zero k-blocks and
  # the accumulator stays at its zero seed, so the store writes zeros
  # (bit-exact vs the empty reference). The ragged shapes below exercise
  # the masked store against a real canary region.
  checkGemm(engine, "fusedGemm", 32, 64, 0)

  var draws = initShapeDraws("gemm")
  for draw in 0 ..< 2:
    let b = draws.nextBytes()
    let M = drawInRange(b, 0, 1, 96)
    let N = drawInRange(b, 1, 1, 96)
    let K = drawInRange(b, 2, 1, 96)
    checkGemm(engine, "fusedGemm", M, N, K)

  # Truly ragged shapes: M, N and K are all non-divisible by the tile dims.
  checkGemm(engine, "fusedGemm", 37, 55, 50)
  checkGemm(engine, "fusedGemm", 65, 89, 71)

when isMainModule:
  runTests()
