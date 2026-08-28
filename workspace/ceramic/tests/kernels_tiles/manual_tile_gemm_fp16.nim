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

import std/[strformat, strutils]
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
                 N, K, M: int32) {.global.} =
    matmul(D, A, B, N, K, M)

# ═════════════════════════════════════════════════════════════════════════
#  Host reference: fp32-exact from f16→f32, over the real shape
# ═════════════════════════════════════════════════════════════════════════

func refGemm(M, N, K, Mp, Np, Kp: int; Ah, Bh: seq[uint16]): seq[float32] =
  ## D(m, n) = Σ_{k < K} A[m·Kp + k]·B[k·Np + n] for m < M, n < N.
  ## The plain triple loop over the real shape indexes the padded buffers
  ## with the padded strides. The loop bounds keep the padding out.
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
  ## in the real M×N×K region (zeros elsewhere), runs the kernel
  ## with grid (Np/32, Mp/32) × blk 32, and compares the real M×N outputs
  ## against the fp32-exact host reference with the 0.0 tolerance.
  let Mp = ((M + 31) div 32) * 32
  let Np = ((N + 31) div 32) * 32
  # The K-padding floor is 16: a K = 0 shape still uploads non-empty
  # A and B buffers (the harness rejects empty args).
  let Kp = max(((K + 15) div 16) * 16, 16)
  # Guard: the kernel's K-loop steps 16-wide k-blocks, so K must stay a multiple
  # of 16 (a non-divisible dim would silently truncate the accumulation).
  doAssert Kp mod 16 == 0
  var Ah = newSeq[uint16](Mp * Kp)
  var Bh = newSeq[uint16](Kp * Np)
  for m in 0 ..< M:
    for k in 0 ..< K:
      Ah[m * Kp + k] = fp32ToFp16(float32(1 + 2 * m + 7 * k))
  for k in 0 ..< K:
    for n in 0 ..< N:
      Bh[k * Np + n] = fp32ToFp16(float32(1 + 3 * k + 11 * n))
  var C = newSeq[float32](Mp * Np)
  # Scalar semantics: the kernel's (N, K, M) are the A/D row count,
  # the k dim, and the B/D col count. The padded D is Mp×Np.
  # N = Mp, K = Kp, M = Np. A K = 0 shape passes 0 as the kernel K.
  # The buffers stay 16-deep for the harness.
  let kParam = if K == 0: int32(0) else: int32(Kp)
  engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >> (
    kernel, C,
    (Ah, Bh, int32(Mp), kParam, int32(Np)))

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
  echo &"GEMM {M}×{N}×{K} ({kernel}, grid ({Np div 32},{Mp div 32}), " &
       &"padded {Mp}×{Np}×{Kp}, K-loop {kParam div 16} iters): ",
       (if ok: &"PASS — worst |Δ| = {worst} (tolerance 0.0)"
        else: &"FAIL — {nBad} bad, e.g. {firstBad}, worst |Δ| = {worst}")
  if not ok:
    quit 1

proc runTests() =   # engines are RAII, so keep them function-local
  checkCasePins()          # the hash cases must not drift
  var engine = bkMetal.init()
  engine.ingest(gemmMsl)
  echo gemmMsl          # keep the generated MSL inspectable

  # The K = 0 empty-tile case: the kernel executes zero k-blocks and
  # the accumulator stays at its zero seed, so the store writes zeros
  # (bit-exact vs the empty reference).
  checkGemm(engine, "fusedGemm", 32, 64, 0)

  var cases = initShapeCases("gemm")
  for caseIdx in 0 ..< 2:
    let b = cases.nextBytes()
    let M = caseInRange(b, 0, 1, 96)
    let N = caseInRange(b, 1, 1, 96)
    let K = caseInRange(b, 2, 1, 96)
    checkGemm(engine, "fusedGemm", M, N, K)

when isMainModule:
  runTests()
