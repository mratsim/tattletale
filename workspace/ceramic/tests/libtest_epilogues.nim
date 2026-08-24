## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared epilogue-test support: the triple-loop GEMM reference, the
## per-epilogue math helpers, and a seq-based assertAllClose. Not a
## test module (no `test_` prefix): `nim test_ceramic` does not run it.

import std/[strformat]
import ./tile_test_utils

# ═════════════════════════════════════════════════════════════════════════
#  Reference GEMM
# ═════════════════════════════════════════════════════════════════════════

func gemmRef*(M, N, K, Mp, Np, Kp: int; Ah, Bh: seq[uint16]): seq[float32] =
  ## Exact fp32 GEMM D(m, n) = Σ_{k < K} A[m, k]·B[k, n], over the padded
  ## buffers with the padded strides (fp16→fp32 inputs).
  result = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += fp16ToFp32(Ah[m * Kp + k]) * fp16ToFp32(Bh[k * Np + n])
      result[m * N + n] = acc

func gemmRefStrided*(M, N, K: int; rsa, csa, rsb, csb: int;
                     aBase, bBase: int; Ah, Bh: seq[uint16]): seq[float32] =
  ## Exact fp32 GEMM D(m, n) = Σ_{k < K} A[aBase + m·rsa + k·csa]·B[bBase + k·rsb + n·csb],
  ## the strided-view reference (negative strides index the buffer
  ## from the passed base).
  result = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += fp16ToFp32(Ah[aBase + m * rsa + k * csa]) *
               fp16ToFp32(Bh[bBase + k * rsb + n * csb])
      result[m * N + n] = acc

func fillAB*(M, N, K, Mp, Np, Kp: int): tuple[Ah, Bh: seq[uint16]] =
  ## Deterministic input pattern over the real M×N×K region (zeros
  ## elsewhere, exact under the gemm padding).
  var Ah = newSeq[uint16](Mp * Kp)
  var Bh = newSeq[uint16](Kp * Np)
  for m in 0 ..< M:
    for k in 0 ..< K:
      Ah[m * Kp + k] = fp32ToFp16(float32(1 + 2 * m + 7 * k))
  for k in 0 ..< K:
    for n in 0 ..< N:
      Bh[k * Np + n] = fp32ToFp16(float32(1 + 3 * k + 11 * n))
  (Ah, Bh)

# ═════════════════════════════════════════════════════════════════════════
#  Epilogue math helpers
# ═════════════════════════════════════════════════════════════════════════

func relu*(x: seq[float32]): seq[float32] =
  ## max(0, x), element-wise.
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len:
    result[i] = max(x[i], 0.0'f32)

func sum*(a, b: seq[float32]): seq[float32] =
  ## a + b, element-wise.
  result = newSeq[float32](a.len)
  for i in 0 ..< a.len:
    result[i] = a[i] + b[i]

func scale*(x: seq[float32], s: float32): seq[float32] =
  ## s·x, element-wise.
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len:
    result[i] = s * x[i]

# ═════════════════════════════════════════════════════════════════════════
#  Assertion
# ═════════════════════════════════════════════════════════════════════════

proc assertAllClose*(actual, expected: seq[float32],
                     rtol = 1e-4'f32, abstol = 1e-4'f32) =
  ## Element-wise |a − b| ≤ abstol + rtol·|b|, first mismatch reported.
  doAssert actual.len == expected.len, "assertAllClose: length mismatch"
  for i in 0 ..< actual.len:
    if actual[i] != actual[i] or abs(actual[i] - expected[i]) > abstol + rtol * abs(expected[i]):
      raise newException(AssertionDefect,
        &"assertAllClose: mismatch at [{i}]: got {actual[i]}, want {expected[i]}")
