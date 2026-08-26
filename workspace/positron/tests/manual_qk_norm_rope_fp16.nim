## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/wip \
##   --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_qk_norm_rope_fp16.nim

import std/[strformat, strutils, math]
import workspace/crucible
import workspace/ceramic
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/qk_norm_rope

proc worstAbsDiff(actual, expected: seq[float32]): float32 =
  ## Largest |actual[i] - expected[i]| over all elements.
  result = 0.0'f32
  for i in 0 ..< actual.len:
    let d = abs(actual[i] - expected[i])
    if d > result: result = d

# ═════════════════════════════════════════════════════════════════════════
#  The metal block. One launcher forwards the runtime view parameters
#  straight to the kernel (no static params in this kernel's contract).
#  First param = the output buffer Out: engine.run binds the separate
#  outBuf argument to it and the args tuple to the rest. D = 128 baked
#  into every call.
# ═════════════════════════════════════════════════════════════════════════

const qkNormRopeMsl = metal:
  proc qkNormRope(
      Out: ptr UncheckedArray[float16],
      X: ptr UncheckedArray[float16],
      G: ptr UncheckedArray[float16],
      Cos: ptr UncheckedArray[float32],
      Sin: ptr UncheckedArray[float32],
      xTokenStride, cosTokenStride, headBlocks, xColBase: int32,
      eps: float32) {.global.} =
    qk_norm_rope_fwd(Out, X, G, Cos, Sin, xTokenStride, cosTokenStride,
                     headBlocks, xColBase, eps)

# ═════════════════════════════════════════════════════════════════════════
#  The host fma: the reference's rotation add, mirroring the kernel's
#  `fma` builtin (the MSL native fma, one IEEE rounding)
# ═════════════════════════════════════════════════════════════════════════

proc fmaf(a, b, c: float32): float32 {.importc: "fmaf", header: "<math.h>".}
  ## The IEEE fused multiply-add (the kernel's `fma` builtin: the MSL
  ## native fma. The reference spells the same explicit form).

# ═════════════════════════════════════════════════════════════════════════
#  The NEOX cos/sin table builder (tm_exl3_attn.nim's buildRopeCosSin, ported)
# ═════════════════════════════════════════════════════════════════════════

proc buildRopeCosSin(nTokens, rowsPerToken, D: int;
                     positions: seq[int32]; theta: float32):
    tuple[cosT, sinT: seq[float32]] =
  ## Precomputes the NEOX fp32 cos/sin tables: returns one (rows, half)
  ## row-major table per signal, rows = nTokens·rowsPerToken, half =
  ## D div 2. The row r of token m (r = m·rowsPerToken + h) carries
  ## the cos/sin of `inv_freq[t]·pos` with `inv_freq[t] = theta^(−2t/D)`
  ## (fp64 `pow`, then fp32 `cos`/`sin`).
  ## NEOX dims pair: dims t and t + half share the value, the table
  ## stores the half once and the kernel's two half-tiles both read it.
  let half = D shr 1
  var invFreq = newSeq[float32](half)
  for t in 0 ..< half:
    invFreq[t] = float32(pow(float64(theta), -float64(2 * t) / float64(D)))
  let rows = nTokens * rowsPerToken
  result.cosT = newSeq[float32](rows * half)
  result.sinT = newSeq[float32](rows * half)
  for m in 0 ..< nTokens:
    let pos = float32(positions[m])
    for h in 0 ..< rowsPerToken:
      let row = m * rowsPerToken + h
      for t in 0 ..< half:
        let ang = invFreq[t] * pos
        result.cosT[row * half + t] = cos(ang)
        result.sinT[row * half + t] = sin(ang)

# ═════════════════════════════════════════════════════════════════════════
#  Host fp16 arithmetic (single-rounding semantics)
# ═════════════════════════════════════════════════════════════════════════

func f16Mul(a, b: uint16): uint16 =
  ## fp16 multiply with one RNE rounding (the `__hmul2` semantics, the kernel's `mulF16`):
  ## the fp32 product of the promoted operands,
  ## rounded once to fp16.
  fp32ToFp16(fp16ToFp32(a) * fp16ToFp32(b))

# ═════════════════════════════════════════════════════════════════════════
#  The qk-norm reference (hostNorm)
# ═════════════════════════════════════════════════════════════════════════

proc hostNorm(x16: seq[uint16]; rows, D: int; gamma: seq[uint16];
              eps: float32): seq[uint16] =
  ## The qk-norm reference, mirroring the exllamav3 fused-rope norm
  ## (rope.cu apply_norm): per row, the fp32 sum of squares, the fp32
  ## `rmf = rsqrt(sum/D + eps)` (the reference spells 1/sqrt), the fp16
  ## rounding of `x·rmf`, then the fp16 weight multiply (single
  ## rounding each), the two-rounding structure the kernel
  ## reproduces. The sum order is the natural sequential fp32
  ## accumulation. The kernel's shuffle-tree order is the documented fp32-noise class,
  ## invisible after the fp16 roundings except at exact tie boundaries.
  result = newSeq[uint16](rows * D)
  for r in 0 ..< rows:
    var sumSq = 0.0'f32
    for c in 0 ..< D:
      let v = fp16ToFp32(x16[r * D + c])
      sumSq += v * v
    let rmf = 1.0'f32 / sqrt(sumSq / float32(D) + eps)
    for c in 0 ..< D:
      let xr16 = fp32ToFp16(fp16ToFp32(x16[r * D + c]) * rmf)
      result[r * D + c] = f16Mul(xr16, gamma[c])

# ═════════════════════════════════════════════════════════════════════════
#  The host rope reference (hostRope, fma form)
# ═════════════════════════════════════════════════════════════════════════

proc hostRope(x: seq[uint16]; rowsPerToken: int; positions: seq[int32];
              D: int; cosT, sinT: seq[float32]): seq[uint16] =
  ## Applies the exllamav3 NEOX rotary embedding on host (the reference rope).
  ## `x` is (tokens · rowsPerToken, D) row-major fp16. Per dim pair `(t, t + D/2)`:
  ## `t1 = fma(v1, cos, −(v2·sin))`, `t2 = fma(v2, cos, v1·sin)`.
  ## The same explicit fma form the kernel spells (the `x2s` sin product rounded once,
  ## then the fused add), and the result rounded once to fp16. The sin/cos come from the caller's `cosT`/`sinT` tables:
  ## the identical buffers the kernel loads, so the two sides consume structurally identical tables
  ## (buildRopeCosSin with the same arguments would also produce them.
  ## Passing them makes the identity structural).
  let nTokens = positions.len
  doAssert cosT.len == nTokens * rowsPerToken * (D div 2),
    "hostRope: the cos table must match (tokens, rowsPerToken, D div 2)"
  let half = D shr 1
  result = newSeq[uint16](x.len)
  for m in 0 ..< nTokens:
    for h in 0 ..< rowsPerToken:
      let row = m * rowsPerToken + h
      let base = row * D
      for t in 0 ..< half:
        let cosV = cosT[row * half + t]
        let sinV = sinT[row * half + t]
        let v1 = fp16ToFp32(x[base + t])
        let v2 = fp16ToFp32(x[base + t + half])
        result[base + t] = fp32ToFp16(fmaf(v1, cosV, -(v2 * sinV)))
        result[base + t + half] = fp32ToFp16(fmaf(v2, cosV, v1 * sinV))

# ═════════════════════════════════════════════════════════════════════════
#  Deterministic value generators
# ═════════════════════════════════════════════════════════════════════════

func exl3Val(r, c, seed: int; scaleF: float32): float32 =
  ## Deterministic fp16-exact value in [-2, 2)·scaleF: an fp16 grid
  ## point scaled by a power of two (scaleF = 2^k keeps the grid point
  ## exactly representable). The mix carries a (c div 32) column-block
  ## term (never periodic vs the 64-col rotation halves or the 128-col
  ## row) and a (r div 32) row-block term, with distinct rows
  ## (seed·(r+1) mod 32, seed odd).
  let k = (seed * (r + 1) + 7 * c + 11 * seed + 13 * (c div 32) +
           17 * (r div 32)) mod 32
  (float32(k) / 8.0'f32 - 2.0'f32) * scaleF

func exl3ValOdd(r, c, seed: int): float32 =
  ## Deterministic value in (-0.6, 0.6) whose fp16 roundings carry full
  ## mantissas: `exl3Val`'s k-mix scaled by a non-power-of-two factor
  ## (0.3), so the values are not all multiples of 2^-6. The fp32 row sums
  ## of squares therefore genuinely round, and the kernel's shuffle-tree row_sum
  ## order is observable against the reference's sequential accumulation
  ## (the 1/8 grid's partial sums are exact in any order. This generator's are not).
  let k = (seed * (r + 1) + 7 * c + 11 * seed + 13 * (c div 32) +
           17 * (r div 32)) mod 32
  (float32(k) / 8.0'f32 - 2.0'f32) * 0.3'f32

func gammaVal(c, seed: int): float32 =
  ## Deterministic fp16-exact norm weight in [0.5, 1.5): 0.5 + k/16 on the (c div 32) block term,
  ## k ∈ 0..15.
  let k = (7 * c + 11 * seed + 13 * (c div 32)) mod 16
  0.5'f32 + float32(k) / 16.0'f32

func buildGamma(seed: int; D: int): seq[uint16] =
  ## The (D,) fp16 norm weight from `gammaVal`.
  result = newSeq[uint16](D)
  for c in 0 ..< D:
    result[c] = fp32ToFp16(gammaVal(c, seed))

# ═════════════════════════════════════════════════════════════════════════
#  The view-addressing verification (host-side, the view contract
#  check): every kernel view against the flat-buffer formulas
# ═════════════════════════════════════════════════════════════════════════

func flatElem(o0, o1, o2, o3, s0, s1, s2, s3, R, C, r, c: int): int =
  ## Mirror of gd + local_tile_dyn (workspace/ceramic/src/tile_algebra/
  ## tiles.nim): the flat index of element (r, c) of an R×C tile at origin (o0..o3)
  ## on a view with strides (s0..s3). Origin components are in tile-extent units:
  ## o2 advances the tile R rows at stride s2, o3 advances the tile C cols at stride s3.
  o0 * s0 + o1 * s1 + o2 * s2 * R + o3 * s3 * C + r * s2 + c * s3

proc verifyViewAddressing() =
  ## Checks the kernel's four gd views against the flat-buffer formulas
  ## under the local_tile_dyn addressing above:
  ## - glX (stride (xTokenStride, 0, 128, 1), origin
  ##   (gidY, 0, gidZ + xColBase div 1024, 0)): element (r, c) must be
  ##   gidY·xTokenStride + gidZ·1024 + xColBase + r·128 + c (the kernel's xBase formula),
  ##   for xColBase a 1024-multiple.
  ## - glG (stride (0, 0, 0, 1), origin (0, 0, 0, 0)): element (r, c)
  ##   must be G[c] (rows broadcast).
  ## - glCos/glSin (stride (cosTokenStride·64, 0, 64, 1), origin
  ##   (gidY, 0, gidZ, 0)): element (r, c) must be
  ##   (gidY·cosTokenStride + gidZ·8 + r)·64 + c.
  ## - glOut (stride (8·128, 0, 128, 1), origins (tile, 0, 0, 0) and (tile, 0, 0, 1)):
  ##   element (r, c) must be tile·1024 + r·128 + c and tile·1024 + r·128 + 64 + c
  ##   (the two 8×64 column halves).
  let xTokenStride = 40 * 128          # composed nQkv = 40
  for gidY in 0 ..< 2:
    for gidZ in 0 ..< 2:
      for xColBase in [0, 16 * 128]:   # q / k (H = 16)
        let o2 = gidZ + xColBase div 1024
        for r in 0 ..< 8:
          for c in 0 ..< 128:
            let viaView = flatElem(gidY, 0, o2, 0,
                                   xTokenStride, 0, 128, 1, 8, 128, r, c)
            let target = gidY * xTokenStride + gidZ * 1024 + xColBase +
                         r * 128 + c
            doAssert viaView == target,
              "glX addressing drifted at " & $gidY & "," & $gidZ & "," &
              $xColBase & "," & $r & "," & $c
  for r in 0 ..< 8:
    for c in 0 ..< 128:
      doAssert flatElem(0, 0, 0, 0, 0, 0, 0, 1, 8, 128, r, c) == c,
        "glG addressing drifted (rows must broadcast)"
  for cosTokenStride in [8, 16]:
    for gidY in 0 ..< 2:
      for gidZ in 0 ..< 2:
        for r in 0 ..< 8:
          for c in 0 ..< 64:
            let viaView = flatElem(gidY, 0, gidZ, 0,
                                   cosTokenStride * 64, 0, 64, 1, 8, 64, r, c)
            let target = (gidY * cosTokenStride + gidZ * 8 + r) * 64 + c
            doAssert viaView == target,
              "glCos/glSin addressing drifted at " & $gidY & "," & $gidZ &
              "," & $r & "," & $c
  for tile in 0 ..< 4:
    for r in 0 ..< 8:
      for c in 0 ..< 64:
        doAssert flatElem(tile, 0, 0, 0, 8 * 128, 0, 128, 1, 8, 64, r, c) ==
          tile * 1024 + r * 128 + c, "glOut half 0 addressing drifted"
        doAssert flatElem(tile, 0, 0, 1, 8 * 128, 0, 128, 1, 8, 64, r, c) ==
          tile * 1024 + r * 128 + 64 + c, "glOut half 1 addressing drifted"

# ═════════════════════════════════════════════════════════════════════════
#  Anchors
# ═════════════════════════════════════════════════════════════════════════

proc checkConversionAnchors() =
  ## Checks the shared fp16 conversion helpers this test's three sides
  ## (buffer build, reference, output readback) all consume: a silent
  ## rounding-mode drift must fail loudly here, not shift every case.
  doAssert fp32ToFp16(1.0005'f32) == 0x3C01'u16      # RNE: 1.0005 -> 1.0009765625
  doAssert fp32ToFp16(-1.0005'f32) == 0xBC01'u16
  doAssert fp32ToFp16(0.1'f32) == 0x2E66'u16        # -> 0.0999755859375
  doAssert fp32ToFp16(1.00048828125'f32) == 0x3C00'u16  # RNE tie -> even mantissa
  doAssert fp16ToFp32(0x3C01'u16) == 1.0009765625'f32
  doAssert fp16ToFp32(0x2E66'u16) == 0.0999755859375'f32

proc checkQkNormRopeAnchors() =
  ## Checks the "qk_norm_rope" case stream's first hash bytes (the composed cases' token count)
  ## and the value generators' first outputs: a silent drift must fail loudly here,
  ## not silently change every case's coverage. Plus the view-addressing verification.
  var cases = initShapeDraws("qk_norm_rope")
  let b0 = cases.nextBytes()
  doAssert b0[0] == 0xF5'u8 and b0[1] == 0xD1'u8 and b0[2] == 0x6B'u8 and
    b0[3] == 0xE6'u8, "qk_norm_rope:0 hash bytes changed"
  doAssert drawInRange(b0, 0, 1, 4) == 2, "qk_norm_rope:0 token count changed"
  doAssert drawInRange(b0, 1, 0, 2047) == 209, "qk_norm_rope:0 position changed"
  let b1 = cases.nextBytes()
  doAssert b1[0] == 0x58'u8 and b1[1] == 0xFA'u8 and b1[2] == 0x53'u8 and
    b1[3] == 0x9A'u8, "qk_norm_rope:1 hash bytes changed"
  # value generators (the EXL3 family anchors)
  doAssert exl3Val(0, 0, 3, 1.0'f32) == -1.5'f32
  doAssert exl3Val(1, 0, 3, 1.0'f32) == -1.125'f32
  doAssert exl3Val(0, 1, 3, 1.0'f32) == -0.625'f32
  doAssert exl3Val(0, 0, 5, 1.0'f32) == 1.5'f32
  # the odd-mantissa generator: the same k-mix scaled by 0.3 (not a power of two),
  # so the fp16-rounded values are not all multiples of 2^-6 and the fp32 row sums
  # genuinely round. Anchors: the exact fp32 values (the fp32 product of the grid point
  # and 0.3'f32) and their fp16 roundings.
  doAssert exl3ValOdd(0, 0, 3) == -0.45000001788139343'f32
  doAssert exl3ValOdd(1, 0, 3) == -0.3375000059604645'f32
  doAssert exl3ValOdd(0, 1, 3) == -0.1875'f32
  doAssert fp32ToFp16(exl3ValOdd(0, 0, 3)) == 0xB733'u16
  doAssert fp32ToFp16(exl3ValOdd(1, 0, 3)) == 0xB566'u16
  doAssert (fp32ToFp16(exl3ValOdd(0, 0, 3)) and 0x000F'u16) != 0'u16,
    "the odd-mantissa case must contain values off the 2^-6 grid"
  # distinct rows: the windows every composed case reads (the q rows
  # 0..15, the k rows 16..23 and the combined-origin k2 rows 16..31)
  # must be pairwise distinct (the q-vs-k difference anchors and the k2
  # combined-origin anchor depend on it). The check spans rows 0..31:
  # the (r div 32) row-block term aliases the fill-only rows 32..39 back
  # into 0..31, but no case reads those.
  var nAlias = 0
  for r in 0 ..< 32:
    for r2 in 0 ..< 32:
      if r != r2:
        for c in 0 ..< 128:
          if exl3Val(r, c, 3, 1.0'f32) == exl3Val(r2, c, 3, 1.0'f32):
            inc nAlias
  doAssert nAlias == 0, "composed qkv rows must be pairwise distinct"
  # gamma: in [0.5, 1.5), fp16-exact (a multiple of 2^-4), distinct
  # per (c div 32) block
  for c in 0 ..< 128:
    let g = gammaVal(c, 5)
    doAssert g >= 0.5'f32 and g < 1.5'f32, "gamma out of [0.5, 1.5)"
    doAssert fp16ToFp32(fp32ToFp16(g)) == g,
      "gamma must be fp16-exact"
  doAssert gammaVal(0, 5) == 0.9375'f32     # (0 + 55) mod 16 = 7
  doAssert gammaVal(33, 5) == 1.1875'f32    # (231 + 55 + 13) mod 16 = 11
  verifyViewAddressing()

# ═════════════════════════════════════════════════════════════════════════
#  The per-case runner: kernel vs reference, checks, ids
# ═════════════════════════════════════════════════════════════════════════

type CaseStats = object
  nElem: int
  nBitExact: int
  nBad: int
  worst: float32

func fnv1a(acc: var uint32; data: seq[uint16]) =
  ## FNV-1a over an fp16 output stream (the test's id).
  for i in 0 ..< data.len:
    let b0 = uint32(data[i] and 0xFF'u16)
    let b1 = uint32(data[i] shr 8)
    acc = (acc xor b0) * 0x01000193'u32
    acc = (acc xor b1) * 0x01000193'u32

type
  QkNormRopeCase = object
    ## One qk-norm+rope case: the X buffer (tokens·xTokenStride fp16
    ## rows of 128), the composed-view parameters, the reference's row
    ## window (rowBase = xColBase div 128 rows into each token,
    ## rowsPerToken rows per token), the shared cos/sin tables and the positions.
    x*: seq[uint16]
    gamma*: seq[uint16]
    cosT*, sinT*: seq[float32]
    positions*: seq[int32]
    tokens*, xTokenStride*, rowBase*, rowsPerToken*: int
    cosTokenStride*, headBlocks*, xColBase*: int32
    eps*: float32
    outRows*: int            # the checked output rows (outRows·128 elements)
    ropeRowsPerToken*: int   # the reference rope's rows per token (1 for the
                             # flat per-row positions, H/slotK composed)

proc runCase(engine: var auto; dr: QkNormRopeCase; tag: string;
             st: var CaseStats; hAll: var uint32):
             tuple[line: string, outBuf, yRef: seq[uint16]] =
  ## One qk-norm+rope case: the real Metal kernel vs the pure-Nim
  ## reference (hostNorm then hostRope on the case's row window).
  ## Checks: worst |Δ| ≤ 1e-2, NaN = failure, per-case bit-exact
  ## floor ≥ 90%. Returns the case's summary line (id included), the kernel's output buffer
  ## and the reference output (the addressing anchors' and the pad-row checks' readback).
  let D = 128
  # the reference's row window: rows rowBase..rowBase+rowsPerToken of each token's xTokenStride rows, extracted flat
  var rows = newSeq[uint16](dr.tokens * dr.rowsPerToken * D)
  for t in 0 ..< dr.tokens:
    for h in 0 ..< dr.rowsPerToken:
      let srcBase = t * dr.xTokenStride + (dr.rowBase + h) * D
      let dstBase = (t * dr.rowsPerToken + h) * D
      for c in 0 ..< D:
        rows[dstBase + c] = dr.x[srcBase + c]
  let normed = hostNorm(rows, dr.tokens * dr.rowsPerToken, D, dr.gamma,
                        dr.eps)
  let yRef = hostRope(normed, dr.ropeRowsPerToken, dr.positions, D,
                      dr.cosT, dr.sinT)
  var outBuf = newSeq[uint16](dr.outRows * D)
  engine.run << (grid: (1, dr.tokens, int(dr.headBlocks)), blk: (32, 1)) >> (
    "qkNormRope", outBuf,
    (dr.x, dr.gamma, dr.cosT, dr.sinT, dr.xTokenStride, dr.cosTokenStride,
     dr.headBlocks, dr.xColBase, dr.eps))
  var nBit = 0
  var nBad = 0
  var gotF = newSeq[float32](outBuf.len)
  var refF = newSeq[float32](outBuf.len)
  # note: fp32 == is sign-agnostic for ±0, so zero-valued outputs are
  # compared by value. The only zero outputs in these cases are the zero-padded rows,
  # whose sign pattern is separately counted below
  # (the fp16 sign of +0·negative cos is −0 by IEEE arithmetic)
  for i in 0 ..< outBuf.len:
    gotF[i] = fp16ToFp32(outBuf[i])
    refF[i] = fp16ToFp32(yRef[i])
    if gotF[i] == refF[i]:
      inc nBit
    else:
      let d = abs(gotF[i] - refF[i])
      if d != d or d > 1e-2'f32:      # NaN is a failure, not a pass
        inc nBad
  let worst = worstAbsDiff(gotF, refF)
  st.nElem += outBuf.len
  st.nBitExact += nBit
  st.nBad += nBad
  if worst > st.worst: st.worst = worst
  fnv1a(hAll, outBuf)
  var h = 0x811C9DC5'u32
  fnv1a(h, outBuf)
  let pct = 100 * nBit div outBuf.len
  result.line = &"  [{tag}] rows={dr.outRows} eps={dr.eps:.0e}: " &
           &"bit-exact {nBit}/{outBuf.len} ({pct}%), worst |Δ| = " &
           &"{worst:.3e}, id 0x{h:08X}"
  result.outBuf = outBuf
  result.yRef = yRef
  if nBad != 0:
    echo result.line
    echo &"  FAIL: {nBad}/{outBuf.len} elements beyond the 1e-2 check or NaN"
    doAssert false, &"{nBad}/{outBuf.len} elements beyond the 1e-2 check or NaN"
  if pct < 90:
    echo result.line
    echo &"  FAIL: bit-exact floor breached ({pct}% < 90%)"
    doAssert false, &"bit-exact floor breached ({pct}% < 90%)"

# ═════════════════════════════════════════════════════════════════════════
#  Section 2: the flat-contract cases (the special case xTokenStride =
#  8·128, cosTokenStride = 8, headBlocks = 1, xColBase = 0)
# ═════════════════════════════════════════════════════════════════════════

proc runFlatSection(engine: var auto; st: var CaseStats; hAll: var uint32) =
  echo "\n── flat-contract cases: Mp ∈ {8, 16, 100} × eps ∈ {1e-6, 1e-2}, " &
       "plus the odd-mantissa case ──"
  let D = 128
  let scaleF = 1.0'f32
  # the 100-row case pads to the 8-row tile grid (Mp = 104): the pad
  # rows are exact zeros and must come out zero-valued (a zero row
  # norms to 0·rsqrt(eps) and ropes to 0. The sign bit follows IEEE
  # signed-zero arithmetic, see the check below)
  let cases = [(rows: 8, eps: 1e-6'f32, fp: "0xF0151BE8"),
               (rows: 16, eps: 1e-2'f32, fp: "0x3557DACF"),
               (rows: 100, eps: 1e-6'f32, fp: "0xB66FDA08"),
               (rows: 100, eps: 1e-2'f32, fp: "0x3849C2CE")]
  for d in cases:
    let Mp = ((d.rows + 7) shr 3) shl 3
    var xPad = newSeq[uint16](Mp * D)
    for r in 0 ..< d.rows:
      for c in 0 ..< D:
        xPad[r * D + c] = fp32ToFp16(exl3Val(r, c, 3, scaleF))
    var positions = newSeq[int32](Mp)
    for m in 0 ..< Mp:
      positions[m] = int32(m)
    let (cosT, sinT) = buildRopeCosSin(Mp, 1, D, positions, 1000000.0'f32)
    let dr = QkNormRopeCase(
      x: xPad, gamma: buildGamma(5, D), cosT: cosT, sinT: sinT,
      positions: positions,
      tokens: Mp div 8, xTokenStride: 8 * D, rowBase: 0, rowsPerToken: 8,
      cosTokenStride: 8, headBlocks: 1, xColBase: 0,
      eps: d.eps, outRows: Mp,
      ropeRowsPerToken: 1)
    let (line, outBuf, yRef) = runCase(engine, dr, &"flat Mp={Mp} ({d.rows} rows)",
                                      st, hAll)
    echo line
    doAssert line.contains("id " & d.fp),
      "flat Mp=" & $Mp & " (" & $d.rows & " rows) id drifted"
    # the padded rows of the 100 case must be exactly zero-valued: a zero row norms to 0·rsqrt(eps)
    # and ropes to 0, but the fp16 sign bit follows IEEE signed-zero arithmetic
    # (+0·negative cos = −0), so the bit pattern is 0x0000 or 0x8000, never a nonzero magnitude
    if d.rows < Mp:
      var nPadBad = 0
      var nNeg = 0
      for r in d.rows ..< Mp:
        for c in 0 ..< D:
          let v = outBuf[r * D + c]
          if v != 0'u16 and v != 0x8000'u16:
            inc nPadBad
          elif v == 0x8000'u16:
            inc nNeg
      if nPadBad != 0:
        var shown = 0
        for r in d.rows ..< Mp:
          for c in 0 ..< D:
            let v = outBuf[r * D + c]
            if v != 0'u16 and v != 0x8000'u16 and shown < 8:
              echo &"    pad row {r} col {c}: out 0x{v:04X} ref 0x{yRef[r * D + c]:04X}"
              inc shown
        echo &"  FAIL: {nPadBad} padded-row outputs are not zero-valued"
        doAssert false, &"{nPadBad} padded-row outputs are not zero-valued"
      echo &"    rows {d.rows}..{Mp - 1} (zero-padded) zero-valued " &
           &"({nNeg} of them signed −0, the IEEE sign of +0·negative cos)"
  # the odd-mantissa flat case (16 rows, eps 1e-6): full-mantissa fp16
  # values off the 1/8 grid, so the fp32 row sums genuinely round (the kernel's shuffle-tree sum
  # and the reference's sequential sum differ on every row,
  # unlike the 1/8-grid cases, whose partial sums are exact
  # in any order). The ≥ 90% floor is an honest order check here. On this data
  # the kernel-vs-reference output delta stays sub-ulp (100% bit-exact), and the case's id
  # anchors the stream against a materially reordered row_sum.
  let MpOdd = 16
  var xOdd = newSeq[uint16](MpOdd * D)
  for r in 0 ..< MpOdd:
    for c in 0 ..< D:
      xOdd[r * D + c] = fp32ToFp16(exl3ValOdd(r, c, 3))
  var positionsOdd = newSeq[int32](MpOdd)
  for m in 0 ..< MpOdd:
    positionsOdd[m] = int32(m)
  let (cosOdd, sinOdd) = buildRopeCosSin(MpOdd, 1, D, positionsOdd,
                                         1000000.0'f32)
  let drOdd = QkNormRopeCase(
    x: xOdd, gamma: buildGamma(5, D), cosT: cosOdd, sinT: sinOdd,
    positions: positionsOdd,
    tokens: MpOdd div 8, xTokenStride: 8 * D, rowBase: 0, rowsPerToken: 8,
    cosTokenStride: 8, headBlocks: 1, xColBase: 0,
    eps: 1e-6'f32, outRows: MpOdd,
    ropeRowsPerToken: 1)
  let (lineOdd, _, _) = runCase(engine, drOdd, "flat odd-mantissa (16 rows)",
                                st, hAll)
  echo lineOdd
  doAssert lineOdd.contains("id 0x6980195A"),
    "flat odd-mantissa id drifted"

# ═════════════════════════════════════════════════════════════════════════
#  Section 3: the composed-view cases (H = 16, Nkv = 12, nQkv = 40)
# ═════════════════════════════════════════════════════════════════════════

proc runComposedSection(engine: var auto; cases: var ShapeDraws;
                        st: var CaseStats; hAll: var uint32) =
  echo "\n── composed-view cases: H=16 Nkv=12 (nQkv=40): q, k, k2 views ──"
  let D = 128
  let H = 16
  let Nkv = 12
  let nQkv = H + 2 * Nkv
  let xTokenStride = nQkv * D
  let b = cases.nextBytes()
  let tokens = drawInRange(b, 0, 1, 4)
  var positions = newSeq[int32](tokens)
  for t in 0 ..< tokens:
    positions[t] = int32(drawInRange(b, 1 + t, 0, 2047))
  # the qkv buffer: 40 rows of 128 per token, all rows distinct. The
  # combined-origin k2 case's gidZ = 1 block reads rows 24..31
  # (xRowBlock = 3), so the buffer must carry nQkv = 40 rows: rows
  # 24..31 are valid memory only with the extended width.
  var qkv = newSeq[uint16](tokens * nQkv * D)
  for t in 0 ..< tokens:
    for h in 0 ..< nQkv:
      for c in 0 ..< D:
        qkv[(t * nQkv + h) * D + c] = fp32ToFp16(exl3Val(h, c, 3, 1.0'f32))
  let gamma = buildGamma(5, D)
  # the q view: slotQ = H cos rows per token, 2 head blocks, xColBase 0
  let slotQ = H
  let (cosQ, sinQ) = buildRopeCosSin(tokens, slotQ, D, positions,
                                     1000000.0'f32)
  let drQ = QkNormRopeCase(
    x: qkv, gamma: gamma, cosT: cosQ, sinT: sinQ, positions: positions,
    tokens: tokens, xTokenStride: xTokenStride, rowBase: 0,
    rowsPerToken: H, cosTokenStride: int32(slotQ), headBlocks: 2,
    xColBase: 0, eps: 1e-6'f32,
    outRows: tokens * H, ropeRowsPerToken: H)
  let (lineQ, qOut, _) = runCase(engine, drQ,
                           &"q view tokens={tokens} hb=2 xColBase=0", st, hAll)
  echo lineQ
  doAssert lineQ.contains("id 0x7E09CC42"), "composed q view id drifted"
  doAssert lineQ.contains("bit-exact 4096/4096"), "composed q view not fully bit-exact"
  echo &"    headBlocks=2 placement: tile (t, b) rows (t·2+b)·8..+7 check " &
       &"against the reference for heads b·8..+7 (the full-output compare above)"
  # the single-block k view: slotK = 8 cos rows per token (the first 8
  # key rows), 1 head block, xColBase = H·D = 2048 → xRowBlock = 2
  let slotK1 = 8
  let (cosK, sinK) = buildRopeCosSin(tokens, slotK1, D, positions,
                                     1000000.0'f32)
  let drK = QkNormRopeCase(
    x: qkv, gamma: gamma, cosT: cosK, sinT: sinK, positions: positions,
    tokens: tokens, xTokenStride: xTokenStride, rowBase: H,
    rowsPerToken: slotK1, cosTokenStride: int32(slotK1), headBlocks: 1,
    xColBase: int32(H * D), eps: 1e-2'f32,
    outRows: tokens * slotK1, ropeRowsPerToken: slotK1)
  let (lineK, kOut, _) = runCase(engine, drK,
                         &"k view tokens={tokens} hb=1 xColBase={H * D}", st, hAll)
  echo lineK
  doAssert lineK.contains("id 0xDC54C7B7"), "composed k view id drifted"
  # the q-vs-k addressing anchor: the k case reads the xColBase-shifted
  # rows (H..H+7 of each token). If xColBase were ignored, the k case
  # would read rows 0..7 and its output would equal the q case's
  # block-0 rows. Every k row must instead differ from q block 0 on the same token.
  var nSameRows = 0
  for t in 0 ..< tokens:
    for h in 0 ..< slotK1:
      var same = true
      for c in 0 ..< D:
        if kOut[(t * slotK1 + h) * D + c] != qOut[(t * H + h) * D + c]:
          same = false
          break
      if same: inc nSameRows
  echo &"    k-view rows vs q block-0 rows on the same token: " &
       &"{tokens * slotK1 - nSameRows}/{tokens * slotK1} differ " &
       &"(the xColBase shift is essential)"
  if nSameRows != 0:
    echo "  FAIL: the k view must read the xColBase-shifted rows, not q block 0"
    doAssert false, "the k view must read the xColBase-shifted rows, not q block 0"
  # the combined-origin k view: slotK2 = ceil8(Nkv) = 16 cos rows per
  # token (Nkv = 12 real key rows + 4 pad slots), 2 head blocks,
  # xColBase = H·D = 2048 → xRowBlock = gidZ + xColBase div 1024 =
  # gidZ + 2 ∈ {2, 3}: the gidZ = 1 block reads rows 24..31, the first case
  # with both origin terms nonzero together
  let slotK2 = ((Nkv + 7) shr 3) shl 3
  let (cosK2, sinK2) = buildRopeCosSin(tokens, slotK2, D, positions,
                                       1000000.0'f32)
  let drK2 = QkNormRopeCase(
    x: qkv, gamma: gamma, cosT: cosK2, sinT: sinK2, positions: positions,
    tokens: tokens, xTokenStride: xTokenStride, rowBase: H,
    rowsPerToken: slotK2, cosTokenStride: int32(slotK2), headBlocks: 2,
    xColBase: int32(H * D), eps: 1e-2'f32,
    outRows: tokens * slotK2, ropeRowsPerToken: slotK2)
  let (lineK2, k2Out, _) = runCase(engine, drK2,
    &"k2 view tokens={tokens} hb=2 xColBase={H * D} (xRowBlock = gidZ + 2)",
    st, hAll)
  echo lineK2
  doAssert lineK2.contains("id 0x52FE06F2"),
    "composed k2 view id drifted"
  # the combined-origin anchor: the gidZ = 1 block's rows must differ from the gidZ = 0 block's rows
  # on the same token: a kernel that collapsed xRowBlock to max/· would read rows
  # 16..23 for both blocks
  # and produce identical blocks
  var nSameBlocks = 0
  for t in 0 ..< tokens:
    for h in 0 ..< 8:
      var same = true
      for c in 0 ..< D:
        if k2Out[(t * slotK2 + 8 + h) * D + c] != k2Out[(t * slotK2 + h) * D + c]:
          same = false
          break
      if same: inc nSameBlocks
  echo &"    k2 gidZ=1 block rows vs gidZ=0 block rows on the same token: " &
       &"{tokens * 8 - nSameBlocks}/{tokens * 8} differ " &
       &"(the combined xRowBlock = gidZ + xColBase div 1024 origin is essential)"
  if nSameBlocks != 0:
    echo "  FAIL: the k2 gidZ=1 block must read rows 24..31, not the gidZ=0 rows"
    doAssert false, "the k2 gidZ=1 block must read rows 24..31, not the gidZ=0 rows"
  # the k2-vs-q addressing anchor: the k2 case reads the xColBase-shifted
  # rows 16..31 of each token. If xColBase were ignored it would read
  # rows 0..15 and equal the q case's rows on the same token
  var nSameRows2 = 0
  for t in 0 ..< tokens:
    for h in 0 ..< slotK2:
      var same = true
      for c in 0 ..< D:
        if k2Out[(t * slotK2 + h) * D + c] != qOut[(t * H + h) * D + c]:
          same = false
          break
      if same: inc nSameRows2
  echo &"    k2 rows vs q rows on the same token: " &
       &"{tokens * slotK2 - nSameRows2}/{tokens * slotK2} differ"
  if nSameRows2 != 0:
    echo "  FAIL: the k2 view must read the xColBase-shifted rows, not q rows"
    doAssert false, "the k2 view must read the xColBase-shifted rows, not q rows"

# ═════════════════════════════════════════════════════════════════════════
#  checkQkNormRope
# ═════════════════════════════════════════════════════════════════════════

proc checkQkNormRope(): bool =
  checkConversionAnchors()
  checkQkNormRopeAnchors()
  var engine = bkMetal.init()
  engine.ingest(qkNormRopeMsl)
  echo qkNormRopeMsl            # keep the generated MSL inspectable
  doAssert qkNormRopeMsl.contains("fma("),
    "the rotation must codegen as fused multiply-adds (MSL fma)"
  var cases = initShapeDraws("qk_norm_rope")
  var st: CaseStats
  var hAll = 0x811C9DC5'u32
  runFlatSection(engine, st, hAll)
  runComposedSection(engine, cases, st, hAll)
  echo "\n── summary ──"
  echo &"  flat + composed cases: {st.nElem} elements, " &
       &"bit-exact {st.nBitExact} " &
       &"({100.0 * float(st.nBitExact) / float(st.nElem):.2f}%), " &
       &"worst |Δ| = {st.worst:.3e}, combined id 0x{hAll:08X}"
  doAssert st.nElem == 41984, "aggregate element count drifted"
  doAssert hAll == 0xC545CD7D'u32, "aggregate id drifted"
  echo "\nqk-norm+rope check: conversion + value + view-addressing anchors " &
       "PASS, the flat-contract cases PASS on Mp ∈ {8, 16, 100} × eps " &
       "∈ {1e-6, 1e-2} plus the odd-mantissa case (the 1/8-grid sums are " &
       "exact in any order; the odd-mantissa case's row sums genuinely " &
       "round, 100% bit-exact observed, inside the 1e-2 check and the " &
       "≥ 90% floor), the " &
       "composed q/k/k2 view cases PASS (the headBlocks=2 tile " &
       "placement, the xColBase-shifted k rows, the combined xRowBlock = " &
       "gidZ + xColBase div 1024 origin, the k-vs-q difference anchors), " &
       "and the rotation is bit-exact (kernel `fma` ≡ host `fmaf`, the " &
       "MSL codegen anchored)"
  result = true

when isMainModule:
  runCppTest("qk_norm_rope vs the reference implementation", checkQkNormRope)
