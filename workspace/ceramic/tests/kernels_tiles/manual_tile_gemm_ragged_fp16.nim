## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run command, from the repo root:
## - nim cpp -r --hints:off --warnings:off workspace/ceramic/tests/kernels_tiles/manual_tile_gemm_ragged_fp16.nim
##
## Ragged GEMM over raw runtime dims, M/N/K off the 32/32/16 tile grid.
##
## All fused kernels are ragged-native, a ceil'd grid, a ceil'd K loop
## whose tail loads zero-fill, a store masked at the real M×N extent,
## and bounded epilogue operand loads.
##
## Every shape runs against the fp32-exact host reference at tolerance 0.0,
## fp16→fp32 is exact and the fixture sums stay inside f32's
## exact-integer domain. A canary band past the real region must stay untouched.

import std/[strformat, math]
import workspace/crucible
import ../tile_test_utils
import ../libtest_epilogues
import ../../src/kernels/k_tile_gemm

const raggedMsl = metal:
  proc fusedMatmul(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16], N, K, M: int32) {.global.} =
    matmul(D, A, B, N, K, M)

  proc fusedRelu(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16], N, K, M: int32) {.global.} =
    gemm_relu(D, A, B, N, K, M)

  proc fusedLinear(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16], Bias: ptr UncheckedArray[float32], N, K, M: int32) {.global.} =
    linear(D, A, B, Bias, N, K, M)

  proc fusedLinearRelu(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16], Bias: ptr UncheckedArray[float32], N, K, M: int32) {.global.} =
    linear_relu(D, A, B, Bias, N, K, M)

  proc fusedGemm(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16], C: ptr UncheckedArray[float32], Alpha, Beta: float32, M, N, K: int32, rsa, csa, rsb, csb, rsc, csc: int32) {.global.} =
    gemm(D, M, N, K, Alpha, A, rsa, csa, B, rsb, csb, Beta, C, rsc, csc)

# ═════════════════════════════════════════════════════════════════════════
#  Fixtures and the fp32-exact references
# ═════════════════════════════════════════════════════════════════════════

const canary = -123456.75'f32

func fillAB(rows, cols, K: int): tuple[Ah, Bh: seq[uint16]] =
  ## A (rows, K) and B (K, cols) row-major, deterministic integer pattern,
  ## A[r,k] = 1+r+k and B[k,c] = 1+k+c. Integers ≤ 2048 are exact in fp16.
  ##
  ## The small coefficients keep K·maxA·maxB inside the f32 exact-integer
  ## domain for every shape in this suite (see exactBound).
  #
  # The K = 0 shape still uploads non-empty buffers, the harness rejects
  # empty args and a 16-cell floor is enough since the bounded loads never
  # read past the raw dims.
  result.Ah = newSeq[uint16](max(rows * K, 16))
  result.Bh = newSeq[uint16](max(K * cols, 16))
  for r in 0 ..< rows:
    for k in 0 ..< K:
      result.Ah[r * K + k] = fp32ToFp16(float32(1 + r + k))
  for k in 0 ..< K:
    for c in 0 ..< cols:
      result.Bh[k * cols + c] = fp32ToFp16(float32(1 + k + c))

proc exactBound(rows, K: int) =
  ## Exactness contract, the suite's 0.0 tolerance is trustworthy only
  ## while every partial sum is an integer below 2^24, f32's
  ## exact-integer mantissa. Assert the fixture domain first.
  let maxA = float32(1 + (rows - 1) + (K - 1))
  let maxB = float32(1 + (K - 1) + 88)  # cols ≤ 89 in this suite
  doAssert float32(K) * maxA * maxB < float32(1 shl 24),
    "fixture leaves the f32 exact-integer domain; the 0.0 tolerance is not trustworthy"

proc refMatmul(rows, cols, K: int; Ah, Bh: seq[uint16]): seq[float32] =
  ## D(r, c) = Σ_k A[r,k]·B[k,c], plain fp32 accumulation of exact fp16→fp32
  ## products:
  ##   exact against the kernel's fp32 accumulator.
  result = newSeq[float32](rows * cols)
  for r in 0 ..< rows:
    for c in 0 ..< cols:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += fp16ToFp32(Ah[r * K + k]) * fp16ToFp32(Bh[k * cols + c])
      result[r * cols + c] = acc

func refAxpby(rows, cols, K: int; alpha, beta: float32;
              rsa, csa, rsb, csb, rsc, csc: int;
              Ah, Bh: seq[uint16]; C: seq[float32]): seq[float32] =
  ## D = α·A·B + β·C over the raw strided views.
  result = newSeq[float32](rows * cols)
  for r in 0 ..< rows:
    for c in 0 ..< cols:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += fp16ToFp32(Ah[r * rsa + k * csa]) * fp16ToFp32(Bh[k * rsb + c * csb])
      result[r * cols + c] = alpha * acc + beta * C[r * rsc + c * csc]

func refBias(rows, cols, K: int; reluBias: bool;
             Ah, Bh: seq[uint16]; Bias: seq[float32]): seq[float32] =
  ## D(r, c) = (Σ_k A·B) + bias[c], clamped at 0 when reluBias.
  result = newSeq[float32](rows * cols)
  for r in 0 ..< rows:
    for c in 0 ..< cols:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += fp16ToFp32(Ah[r * K + k]) * fp16ToFp32(Bh[k * cols + c])
      result[r * cols + c] =
        (if reluBias: max(acc + Bias[c], 0.0'f32) else: acc + Bias[c])

# ═════════════════════════════════════════════════════════════════════════
#  Checks
# ═════════════════════════════════════════════════════════════════════════

proc gridOf(rows, cols: int): tuple[a, b: int] =
  ## Ceil'd grid, one 32×32 tile per threadgroup.
  ((cols + 31) div 32, (rows + 31) div 32)

proc checkD(name: string; D: seq[float32]; expected: seq[float32]; rows, cols: int) =
  ## D's real M×N region must match the reference at tolerance 0.0,
  ## and the canary band past it stay untouched.
  let bandLen = D.len - rows * cols
  doAssert bandLen >= 0, name & ": the D buffer lost its canary band"
  assertAllClose(D[0 ..< rows * cols], expected, 0.0'f32, 0.0'f32)
  for i in rows * cols ..< D.len:
    doAssert D[i] == canary,
      name & ": canary band written at " & $(i - rows * cols) & ": " & $D[i]
  echo &"{name}: PASS (canary band intact)"

proc checkGemmIdentity(engine: var auto; kernel: string;
                       rows, cols, K: int; relu: bool) =
  exactBound(rows, K)
  let (Ah, Bh) = fillAB(rows, cols, K)
  let pA = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](addr Ah[0]), len: Ah.len, off: 0)
  let pB = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](addr Bh[0]), len: Bh.len, off: 0)
  var D = newSeq[float32](rows * cols + 64)
  for i in 0 ..< D.len: D[i] = canary
  engine.run << (grid: gridOf(rows, cols), blk: (32, 1)) >> (kernel, D,
    (pA, pB, int32(rows), int32(K), int32(cols)))
  var expected = refMatmul(rows, cols, K, Ah, Bh)
  if relu:
    for i in 0 ..< expected.len: expected[i] = max(expected[i], 0.0'f32)
  checkD(&"{kernel} {rows}×{cols}×{K}", D, expected, rows, cols)

proc checkLinear(engine: var auto; kernel: string; rows, cols, K: int; relu: bool) =
  exactBound(rows, K)
  let (Ah, Bh) = fillAB(rows, cols, K)
  let pA = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](addr Ah[0]), len: Ah.len, off: 0)
  let pB = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](addr Bh[0]), len: Bh.len, off: 0)
  var Bias = newSeq[float32](cols)
  for c in 0 ..< cols: Bias[c] = float32(1 + 5 * c)
  let pBias = PtrArg[float32](buf: cast[ptr UncheckedArray[float32]](addr Bias[0]), len: cols, off: 0)
  var D = newSeq[float32](rows * cols + 64)
  for i in 0 ..< D.len: D[i] = canary
  engine.run << (grid: gridOf(rows, cols), blk: (32, 1)) >> (kernel, D,
    (pA, pB, pBias, int32(rows), int32(K), int32(cols)))
  checkD(&"{kernel} {rows}×{cols}×{K}", D,
         refBias(rows, cols, K, relu, Ah, Bh, Bias), rows, cols)

proc checkAxpby(engine: var auto; rows, cols, K: int; alpha, beta: float32) =
  ## D = α·A·B + β·C, row-major everywhere, the fused `gemm` surface.
  exactBound(rows, K)
  let (Ah, Bh) = fillAB(rows, cols, K)
  let pA = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](addr Ah[0]), len: Ah.len, off: 0)
  let pB = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](addr Bh[0]), len: Bh.len, off: 0)
  var C = newSeq[float32](rows * cols + 64)
  for i in 0 ..< C.len: C[i] = canary
  if beta != 0.0'f32:
    for r in 0 ..< rows:
      for c in 0 ..< cols:
        C[r * cols + c] = float32(1 + 5 * r + 13 * c)
  # β = 0 leaves the whole C buffer as canary: a C read anywhere lands in
  # the comparison as a canary value and fails the 0.0 tolerance.
  let pC = PtrArg[float32](buf: cast[ptr UncheckedArray[float32]](addr C[0]), len: C.len, off: 0)
  var D = newSeq[float32](rows * cols + 64)
  for i in 0 ..< D.len: D[i] = canary
  engine.run << (grid: gridOf(rows, cols), blk: (32, 1)) >> ("fusedGemm", D,
    (pA, pB, pC, alpha, beta, int32(rows), int32(cols), int32(K),
     int32(K), int32(1), int32(cols), int32(1), int32(cols), int32(1)))
  checkD(&"fusedGemm {rows}×{cols}×{K} (α={alpha}, β={beta})", D,
         refAxpby(rows, cols, K, alpha, beta, K, 1, cols, 1, cols, 1, Ah, Bh, C),
         rows, cols)

proc runTest() =   # engines are RAII, so keep them function-local
  var engine = bkMetal.init()
  engine.ingest(raggedMsl)
  echo raggedMsl          # keep the generated MSL inspectable

  # Ragged shapes, M/N/K off the tile grid. With the exact-bound
  # assert and the 0.0 tolerance, a leaked K-tail, a padded-lane read
  # or an unmasked store becomes a loud failure.
  let ragged = [(61, 37, 17), (61, 37, 64), (64, 89, 32), (37, 89, 32), (64, 33, 32), (61, 37, 0), (64, 64, 32)]
  for (rows, cols, K) in ragged:
    checkGemmIdentity(engine, "fusedMatmul", rows, cols, K, relu = false)
    checkGemmIdentity(engine, "fusedRelu", rows, cols, K, relu = true)
    checkLinear(engine, "fusedLinear", rows, cols, K, relu = false)
    checkLinear(engine, "fusedLinearRelu", rows, cols, K, relu = true)
    checkAxpby(engine, rows, cols, K, 1.5'f32, 2.0'f32)
    checkAxpby(engine, rows, cols, K, 1.0'f32, 0.0'f32)   # β = 0: C never read

  echo "  [OK] ragged tile GEMM: identity/relu/linear/linear_relu/axpby over raw",
       " ragged dims match the fp32-exact reference at tolerance 0.0, canary bands intact"

when isMainModule:
  runTest()
