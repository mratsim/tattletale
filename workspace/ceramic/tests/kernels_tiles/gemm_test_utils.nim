## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared strided-GEMM test support: the canary matrices, the fp32 reference,
## the launch helpers and the compare wrappers.
## Not run by `nim test_ceramic` (no `test_` prefix).

import std/[strformat]
import workspace/crucible
import workspace/crucible/src/runtime/engines/metal as metalEngine
import ../tile_test_utils
import ../../src/kernels/k_tile_gemm

# ═════════════════════════════════════════════════════════════════════════
#  Canary bit patterns for the padded region
# ═════════════════════════════════════════════════════════════════════════

const aCanary* = 0xFACA'u16       # A padding lanes (finite ≈ −5.6e4: a leaked lane swamps the accumulator)
const bCanary* = 0xDEAD'u16       # B padding lanes (finite ≈ −4.3e2)
const dCanary* = 0xBEEFDEAD'u32   # C/D padding lanes (finite fp32 pattern)

func canaryD*(Mp, Np: int): seq[float32] =
  ## Padded fp32 buffer, every cell the D canary bit pattern.
  result = newSeq[float32](Mp * Np)
  for i in 0 ..< result.len:
    result[i] = cast[float32](dCanary)

func paddedAB*(M, N, K, Mp, Np, Kp: int;
               rsa, csa, rsb, csb: int): tuple[Ah, Bh: seq[uint16]; aBase, bBase: int] =
  ## Canary-filled A/B buffers over the whole accessed span.
  ## The deterministic in-range pattern A[m,k] = 1+2m+7k, B[k,n] = 1+3k+11n lands at the raw strides.
  ## Negative strides rebase the span via the bases.
  let aLo = min(0, (Mp - 1) * rsa) + min(0, (Kp - 1) * csa)
  let aHi = max(0, (Mp - 1) * rsa) + max(0, (Kp - 1) * csa)
  let bLo = min(0, (Kp - 1) * rsb) + min(0, (Np - 1) * csb)
  let bHi = max(0, (Kp - 1) * rsb) + max(0, (Np - 1) * csb)
  result.aBase = -aLo
  result.bBase = -bLo
  result.Ah = newSeq[uint16](aHi - aLo + 1)
  result.Bh = newSeq[uint16](bHi - bLo + 1)
  for i in 0 ..< result.Ah.len: result.Ah[i] = aCanary
  for i in 0 ..< result.Bh.len: result.Bh[i] = bCanary
  for m in 0 ..< M:
    for k in 0 ..< K:
      result.Ah[result.aBase + m * rsa + k * csa] =
        fp32ToFp16(float32(1 + 2 * m + 7 * k))
  for k in 0 ..< K:
    for n in 0 ..< N:
      result.Bh[result.bBase + k * rsb + n * csb] =
        fp32ToFp16(float32(1 + 3 * k + 11 * n))

func paddedC*(M, N, Mp, Np: int): seq[float32] =
  ## Padded C buffer: canary outside, C[m,n] = 1+5m+13n inside.
  result = canaryD(Mp, Np)
  for m in 0 ..< M:
    for n in 0 ..< N:
      result[m * Np + n] = float32(1 + 5 * m + 13 * n)

#  Buffer spans
# ═════════════════════════════════════════════════════════════════════════

proc boxSpan*(R, C, rs, cs: int): tuple[lo, len: int] =
  ## The accessed element span of an R×C region at (rs, cs) strides.
  ## Returns the min/max corners of the box {r·rs + c·cs}.
  ## `lo` can be negative (negative strides).
  ## The PtrArg binding rebases at -lo.
  ## An empty dimension yields a minimal non-empty span.
  ## The kernel never dereferences it (K = 0 runs zero k-slices).
  ## The engine rejects empty buffers.
  if R == 0 or C == 0:
    return (lo: 0, len: 1)
  let c00 = 0
  let c10 = (R - 1) * rs
  let c01 = (C - 1) * cs
  let c11 = (R - 1) * rs + (C - 1) * cs
  let lo = min(min(c00, c10), min(c01, c11))
  let hi = max(max(c00, c10), max(c01, c11))
  result = (lo: lo, len: hi - lo + 1)
func hostOperands*(M, N, K: int;
                   rsa, csa, rsb, csb, rsc, csc: int): tuple[
                     A, B, C: seq[float32]; aBase, bBase, cBase: int] =
  ## fp32 A/B/C allocated exactly at their accessed spans.
  ## Filled with the deterministic in-range pattern.
  ## Any read past the real region reads out of the PtrArg-bound buffer.
  ## It fails loudly in the engine.
  let aBox = boxSpan(M, K, rsa, csa)
  let bBox = boxSpan(K, N, rsb, csb)
  let cBox = boxSpan(M, N, rsc, csc)
  result.aBase = -aBox.lo
  result.bBase = -bBox.lo
  result.cBase = -cBox.lo
  result.A = newSeq[float32](aBox.len)
  result.B = newSeq[float32](bBox.len)
  result.C = newSeq[float32](cBox.len)
  for m in 0 ..< M:
    for k in 0 ..< K:
      result.A[result.aBase + m * rsa + k * csa] = float32(1 + 2 * m + 7 * k)
  for k in 0 ..< K:
    for n in 0 ..< N:
      result.B[result.bBase + k * rsb + n * csb] = float32(1 + 3 * k + 11 * n)
  for m in 0 ..< M:
    for n in 0 ..< N:
      result.C[result.cBase + m * rsc + n * csc] = float32(1 + 5 * m + 13 * n)

# ═════════════════════════════════════════════════════════════════════════
#  Reference GEMM (fp32 triple loop)
# ═════════════════════════════════════════════════════════════════════════

func gemmRefFp32*(M, N, K: int; alpha, beta: float32;
                  A: seq[float32]; aBase, rsa, csa: int;
                  B: seq[float32]; bBase, rsb, csb: int;
                  C: seq[float32]; cBase, rsc, csc: int): seq[float32] =
  ## fp32 reference over the real M×N×K region. D = α·A·B + β·C.
  result = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      var acc = 0.0'f32
      for k in 0 ..< K:
        acc += A[aBase + m * rsa + k * csa] * B[bBase + k * rsb + n * csb]
      result[m * N + n] = alpha * acc + beta * C[cBase + m * rsc + n * csc]

func worstAbsDiff*(got, want: seq[float32]): float32 =
  ## The largest |got[i] - want[i]| over the shared prefix length.
  result = 0.0'f32
  let n = min(got.len, want.len)
  for i in 0 ..< n:
    let d = abs(got[i] - want[i])
    if d > result: result = d

# ═════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════
#  Compare wrappers
# ═════════════════════════════════════════════════════════════════════════

proc checkStrided*(D: seq[float32]; Np, M, N: int; refC: seq[float32];
                   label: string; tol = 1e-4'f32) =
  ## Real-region compare at tol (abs + rel) plus the D-canary check on the padded region.
  ## The masked store must leave it untouched.
  var ok = true
  var worst = 0.0'f32
  var nBad = 0
  var firstBad = ""
  for m in 0 ..< M:
    for n in 0 ..< N:
      let d = abs(D[m * Np + n] - refC[m * N + n])
      if d > worst: worst = d
      if D[m * Np + n] != D[m * Np + n] or
         d > tol + tol * abs(refC[m * N + n]):
        ok = false
        inc nBad
        if firstBad.len == 0:
          firstBad = &"(m={m}, n={n}): got {D[m * Np + n]}, want {refC[m * N + n]}"
  var nCanary = 0
  var firstCanary = ""
  for m in 0 ..< D.len div Np:
    for n in 0 ..< Np:
      if m >= M or n >= N:
        if cast[uint32](D[m * Np + n]) != dCanary:
          inc nCanary
          if firstCanary.len == 0:
            firstCanary = &"(m={m}, n={n}): got {cast[uint32](D[m * Np + n]):08x}"
  echo label, ": " &
       (if ok: &"PASS — worst |Δ| = {worst}" else: &"FAIL — {nBad} bad, e.g. {firstBad}, worst |Δ| = {worst}") &
       (if nCanary == 0: ", D-canary intact"
        else: &", D-canary CORRUPTED: {nCanary} cells, e.g. {firstCanary}")
  if not ok or nCanary > 0:
    quit 1

proc checkHostStrided*(label: string; C: seq[float32];
                       cBase, rsc, csc, M, N: int;
                       refC: seq[float32]; tol = 1e-3'f32) =
  ## Compares the in-place output C at (rsc, csc) against the reference.
  ## NaN is a failure, not a pass. `label` carries the case description.
  var got = newSeq[float32](M * N)
  var nBad = 0
  var firstBad = ""
  for m in 0 ..< M:
    for n in 0 ..< N:
      let v = C[cBase + m * rsc + n * csc]
      got[m * N + n] = v
      if v != v or abs(v - refC[m * N + n]) > tol:
        nBad += 1
        if firstBad.len == 0:
          firstBad = &"(m={m}, n={n}): got {v}, want {refC[m * N + n]}"
  let worst = worstAbsDiff(got, refC)
  echo label, ": ",
       (if nBad == 0: &"PASS — worst |Δ| = {worst}"
        else: &"FAIL — {nBad} bad, e.g. {firstBad}, worst |Δ| = {worst}")
  if nBad > 0:
    quit 1

# ═════════════════════════════════════════════════════════════════════════
#  Host BLIS gemm_strided launcher
# ═════════════════════════════════════════════════════════════════════════

const gemmStridedLaunchMsl = metal:
  proc gemmStridedLaunch(D: ptr UncheckedArray[float32],
                       A: ptr UncheckedArray[float32], rsa, csa: int32,
                       B: ptr UncheckedArray[float32], rsb, csb: int32,
                       M, N, K: int32, alpha, beta: float32,
                       rsc, csc: int32) {.global.} =
    ## D = α·A·B + β·D, D the (M, N) output at (rsc, csc).
    ## In-place BLIS semantics: β·D is read before the tile write.
    ## The bounded C load keeps a tightly-allocated D safe on ragged edge tiles.
    gemm(D, rsc, csc, M, N, K, alpha, A, rsa, csa, B, rsb, csb, beta, D, rsc, csc)

var gGemmStridedEngine: metalEngine.MetalEngine

proc gemmStridedEngine(): metalEngine.MetalEngine =
  ## The engine backing gemm_strided launches.
  ## Created and fed the launcher source on first use.
  ## Metal context and library compile are lazy.
  if gGemmStridedEngine.isNil:
    gGemmStridedEngine = metalEngine.newMetalEngine()
    gGemmStridedEngine.ingest(gemmStridedLaunchMsl)
  result = gGemmStridedEngine

proc launchGemmStrided*(M, N, K: int; alpha, beta: float32;
                        pA, pB: PtrArg[float32]; pC: var PtrArg[float32];
                        rsa, csa, rsb, csb, rsc, csc: int) =
  ## Launches the gemmStridedLaunch kernel with PtrArg pass-through.
  ## A and B bind the exact accessed span (negative strides rebase via
  ## the PtrArg offset). C is the in-place output
  ## (the engine uploads and read-backs its span).
  let gridX = (N + 31) div 32
  let gridY = (M + 31) div 32
  gemmStridedEngine().run << (grid: (gridX, gridY), blk: (32, 1)) >> (
    "gemmStridedLaunch", pC,
    (pA, int32(rsa), int32(csa), pB, int32(rsb), int32(csb),
     int32(M), int32(N), int32(K), alpha, beta,
     int32(rsc), int32(csc)))

proc gemm_strided*[T: SomeNumber](
    M, N, K: int,
    alpha: T,
    A: ptr T, rowStrideA, colStrideA: int,
    B: ptr T, rowStrideB, colStrideB: int,
    beta: T,
    C: ptr T, rowStrideC, colStrideC: int) =
  ## D = α·A·B + β·C, the BLIS strided GEMM, host-side GPU launcher.
  ##
  ## The BLIS contract surface. Output in place
  ## (C is both the β operand and the D buffer):
  ##   - A (M, K) at (rowStrideA, colStrideA)
  ##   - B (K, N) at (rowStrideB, colStrideB)
  ##   - C (M, N) at (rowStrideC, colStrideC)
  ##
  ## Pure view-building: the launcher forwards the raw strides and dims.
  ## Ragged M/N/K is handled in-kernel:
  ##   - bounded loads zero-fill padded tiles
  ##   - the K loop runs ceil(K / tileK) slices
  ##   - the C load is bounded
  ##   - stores are masked
  ## No caller staging. No caller padding. No staging region of any kind.
  ##
  ## Rejected loudly (compile-time or doAssert, never silent wrong results):
  ##   - non-fp32 T: the kernel accumulates and reads C in fp32.
  ##     The single-T BLIS signature cannot express fp16 A/B with an fp32 C buffer.
  ##     Use the device-level gemm_with_epilogue directly for fp16 inputs.
  ##   - M < 1 or N < 1: the tile grid would be empty.
  ##   - K < 0.
  ##   - negative-strided C: rejected. The output binding uses offset 0
  ##     at the C pointer.
  ##   - negative-strided A/B: allowed. The input PtrArgs carry the span offset.
  ##   - dims, strides or spans that overflow int32.
  static:
    doAssert T is float32,
      "gemm_strided: fp32 operands only. The kernel accumulates in fp32 " &
      "and the fused β·C epilogue reads C as fp32. The BLIS single-T " &
      "signature cannot express fp16 A/B with an fp32 C buffer. Use the " &
      "device-level gemm_with_epilogue directly for fp16 inputs."
  doAssert M >= 1 and N >= 1,
    "gemm_strided: M and N must be >= 1 (the GPU dispatch needs at least one tile CTA)"
  doAssert K >= 0, "gemm_strided: K must be >= 0"
  doAssert rowStrideC >= 0 and colStrideC >= 0,
    "gemm_strided: negative-strided C is rejected. The engine's output " &
    "binding places the buffer base at the C pointer with no offset, " &
    "so a rebased C layout cannot be expressed"
  let aSpan = boxSpan(M, K, rowStrideA, colStrideA)
  let bSpan = boxSpan(K, N, rowStrideB, colStrideB)
  let cSpan = boxSpan(M, N, rowStrideC, colStrideC)
  doAssert M <= int32.high and N <= int32.high and K <= int32.high and
                rowStrideA >= int32.low and rowStrideA <= int32.high and
                colStrideA >= int32.low and colStrideA <= int32.high and
                rowStrideB >= int32.low and rowStrideB <= int32.high and
                colStrideB >= int32.low and colStrideB <= int32.high and
                rowStrideC <= int32.high and colStrideC <= int32.high and
                aSpan.len <= int32.high and bSpan.len <= int32.high and
                cSpan.len <= int32.high,
    "gemm_strided: a dim, stride or accessed span exceeds int32 (the kernel surface is int32)"
  let pA = PtrArg[float32](buf: cast[ptr UncheckedArray[float32]](A),
                           len: aSpan.len, off: -aSpan.lo)
  let pB = PtrArg[float32](buf: cast[ptr UncheckedArray[float32]](B),
                           len: bSpan.len, off: -bSpan.lo)
  var pC = PtrArg[float32](buf: cast[ptr UncheckedArray[float32]](C),
                           len: cSpan.len, off: 0)
  launchGemmStrided(M, N, K,
                    float32(alpha), float32(beta),
                    pA, pB, pC,
                    rowStrideA, colStrideA, rowStrideB, colStrideB,
                    rowStrideC, colStrideC)
