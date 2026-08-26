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
##   workspace/positron/tests/manual_exl3_linear_fp16.nim

import std/[strformat]
import workspace/crucible
import workspace/ceramic
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/exl3_ops
import ../src/kernels/ceramic/exl3_linear_fwd

proc worstAbsDiff(actual, expected: seq[float32]): float32 =
  ## Largest |actual[i] - expected[i]| over all elements.
  result = 0.0'f32
  for i in 0 ..< actual.len:
    let d = abs(actual[i] - expected[i])
    if d > result: result = d

# ═════════════════════════════════════════════════════════════════════════
#  The thin {.global.} launchers. First param = the output buffer Out:
#  engine.run binds the separate outBuf argument to it and the args
#  tuple to the rest. One linear launcher per static bits (3, 5, 8),
#  D = 128 baked in.
# ═════════════════════════════════════════════════════════════════════════

proc dequantTileWrite(
    outBuf: ptr UncheckedArray[float16],
    trellis: ptr UncheckedArray[int16],
    kk, tilesN, tgx, nt: int32,
    bits: static int, cb: static int, useShuffle: static bool) {.device.} =
  ## Decodes one 16×32 weight tile (k-block kk, n-tile nt, grid x block tgx)
  ## and writes it row-major to `outBuf` (32 columns), the test-side mirror
  ## of the dequantTrellis fill. The lane→element mapping is the RtRight loadTile convention,
  ## so the write uses the same (row, col) the decode used.
  const A = getTileConfig(float32, float16)
  const M = A.getM()
  const N = A.getN()
  const rowTiles = 16 div M
  const colTiles = 32 div N
  const vpt = A.getVpt()
  var b_reg: RtRight[float16, 16, 32, A]
  dequantTrellis(b_reg, trellis, kk, tilesN, tgx, nt, bits, cb, useShuffle)
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  for m in 0 ..< colTiles:
    for n in 0 ..< rowTiles:
      for v in 0 ..< vpt:
        let r = row + n * M
        let c = col + m * N + v
        outBuf[r * 32 + c] = b_reg.frags[m][n].frag[v]

const exl3Msl = metal:
  proc exl3LinearB3(
      Out: ptr UncheckedArray[float16],
      x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16],
      suh, svh: ptr UncheckedArray[float16],
      M, K, N: int32) {.global.} =
    exl3_linear_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 128)
  proc exl3LinearB5(
      Out: ptr UncheckedArray[float16],
      x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16],
      suh, svh: ptr UncheckedArray[float16],
      M, K, N: int32) {.global.} =
    exl3_linear_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 128)
  proc exl3LinearB8(
      Out: ptr UncheckedArray[float16],
      x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16],
      suh, svh: ptr UncheckedArray[float16],
      M, K, N: int32) {.global.} =
    exl3_linear_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 128)
  proc dequantKernel(
      outBuf: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16],
      kk, tilesN, tgx, nt, bits, cb: int32) {.global.} =
    if bits == 1 and cb == 0:
      dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 1, 0, true)
    else:
        if bits == 1 and cb == 1:
          dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 1, 1, true)
        else:
            if bits == 1 and cb == 2:
              dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 1, 2, true)
            else:
                if bits == 2 and cb == 0:
                  dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 2, 0, true)
                else:
                    if bits == 2 and cb == 1:
                      dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 2, 1, true)
                    else:
                        if bits == 2 and cb == 2:
                          dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 2, 2, true)
                        else:
                            if bits == 3 and cb == 0:
                              dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 3, 0, true)
                            else:
                                if bits == 3 and cb == 1:
                                  dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 3, 1, true)
                                else:
                                    if bits == 3 and cb == 2:
                                      dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 3, 2, true)
                                    else:
                                        if bits == 4 and cb == 0:
                                          dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 4, 0, true)
                                        else:
                                            if bits == 4 and cb == 1:
                                              dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 4, 1, true)
                                            else:
                                                if bits == 4 and cb == 2:
                                                  dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 4, 2, true)
                                                else:
                                                    if bits == 5 and cb == 0:
                                                      dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 5, 0, true)
                                                    else:
                                                        if bits == 5 and cb == 1:
                                                          dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 5, 1, true)
                                                        else:
                                                            if bits == 5 and cb == 2:
                                                              dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 5, 2, true)
                                                            else:
                                                                if bits == 6 and cb == 0:
                                                                  dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 6, 0, true)
                                                                else:
                                                                    if bits == 6 and cb == 1:
                                                                      dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 6, 1, true)
                                                                    else:
                                                                        if bits == 6 and cb == 2:
                                                                          dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 6, 2, true)
                                                                        else:
                                                                            if bits == 7 and cb == 0:
                                                                              dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 7, 0, true)
                                                                            else:
                                                                                if bits == 7 and cb == 1:
                                                                                  dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 7, 1, true)
                                                                                else:
                                                                                    if bits == 7 and cb == 2:
                                                                                      dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 7, 2, true)
                                                                                    else:
                                                                                        if bits == 8 and cb == 0:
                                                                                          dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 8, 0, true)
                                                                                        else:
                                                                                            if bits == 8 and cb == 1:
                                                                                              dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 8, 1, true)
                                                                                            else:
                                                                                                dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 8, 2, true)
  proc dequantKernelNat(
      outBuf: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16],
      kk, tilesN, tgx, nt, bits, cb: int32) {.global.} =
    if bits == 3 and cb == 0:
      dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 3, 0, false)
    else:
        if bits == 3 and cb == 1:
          dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 3, 1, false)
        else:
            if bits == 3 and cb == 2:
              dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 3, 2, false)
            else:
                if bits == 5 and cb == 0:
                  dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 5, 0, false)
                else:
                    if bits == 5 and cb == 1:
                      dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 5, 1, false)
                    else:
                        if bits == 5 and cb == 2:
                          dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 5, 2, false)
                        else:
                            if bits == 8 and cb == 0:
                              dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 8, 0, false)
                            else:
                                if bits == 8 and cb == 1:
                                  dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 8, 1, false)
                                else:
                                    dequantTileWrite(outBuf, trellis, kk, tilesN, tgx, nt, 8, 2, false)

# ═════════════════════════════════════════════════════════════════════════
#  Deterministic value generators
# ═════════════════════════════════════════════════════════════════════════

func exl3Val(r, c, seed: int; scaleF: float32): float32 =
  ## Deterministic fp16-exact value in [-2, 2)·scaleF: an fp16 grid
  ## point scaled by a power of two (scaleF = 2^k keeps the grid point
  ## exactly representable). The mix carries a (c div 32) column-block
  ## term and a (r div 32) row-block term, so the pattern never
  ## aliases the 32-col tile width, the 128-col FWHT block, or the 32-row grid.y tile
  ## (rows r and r+32 differ: their k shifts by 17 mod 32,
  ## never 0).
  let k = (seed * (r + 1) + 7 * c + 11 * seed + 13 * (c div 32) +
           17 * (r div 32)) mod 32
  (float32(k) / 8.0'f32 - 2.0'f32) * scaleF

func trellisWord(i, seed: int): int16 =
  ## Deterministic packed-trellis word: a per-index hash with a seed
  ## term and no short period.
  let h = uint32(i) * 2654435761'u32 + uint32(seed) * 40503'u32
  cast[int16]((h shr 16) xor (h and 0xFFFF'u32))

func genTrellis(tilesK, tilesN, bits, seed: int): seq[int16] =
  ## Deterministic packed trellis buffer of tilesK·tilesN·(16·bits)
  ## int16 words.
  let n = tilesK * tilesN * (256 * bits div 16)
  result = newSeq[int16](n)
  for i in 0 ..< n:
    result[i] = trellisWord(i, seed)

# ═════════════════════════════════════════════════════════════════════════
#  Host fp16 arithmetic (single-rounding semantics, the CUDA __hadd /
#  __hfma the codebook decode uses)
# ═════════════════════════════════════════════════════════════════════════

func f64ToF16(x: float64): uint16 =
  ## IEEE-754 binary64 → binary16 bit pattern, round-to-nearest-even
  ## (normal, subnormal, zero, Inf, NaN quieted like fp32ToFp16).
  let u = cast[uint64](x)
  let sign = uint16((u shr 48) and 0x8000'u64)
  let aexp = int((u shr 52) and 0x7FF)
  let mant = u and 0xFFFFFFFFFFFFF'u64
  if aexp == 0x7FF:
    if mant == 0: return uint16(sign or 0x7C00'u16)
    return uint16(sign or 0x7C00'u16 or 0x0200'u16)
  var exp = aexp - 1023 + 15
  if exp >= 31:
    return uint16(sign or 0x7C00'u16)
  if exp <= 0:
    if exp < -10:
      return sign
    let m2 = mant or 0x10000000000000'u64
    let shift = 1051 - aexp
    let half = 1'u64 shl (shift - 1)
    var r = m2 shr shift
    let rem = m2 and ((1'u64 shl shift) - 1)
    if rem > half or (rem == half and (r and 1) == 1):
      r += 1
    return uint16(sign or uint16(r))
  let half = 1'u64 shl 41
  var r = mant shr 42
  let rem = mant and ((1'u64 shl 42) - 1)
  if rem > half or (rem == half and (r and 1) == 1):
    r += 1
    if r == 0x400:
      exp += 1
      r = 0
      if exp >= 31:
        return uint16(sign or 0x7C00'u16)
  uint16(sign or (uint16(exp) shl 10) or uint16(r))

func f16Add(a, b: uint16): uint16 =
  ## fp16 add with a single rounding: the exact sum of two fp16 values
  ## is exact in fp64, rounded once to fp16.
  f64ToF16(fp16ToFp32(a).float64 + fp16ToFp32(b).float64)

func f16Fma(a, b, c: uint16): uint16 =
  ## fp16 fused multiply-add with a single rounding: the exact a·b + c
  ## is exact in fp64 for the codebook's value ranges, rounded once to fp16.
  f64ToF16(fp16ToFp32(a).float64 * fp16ToFp32(b).float64 + fp16ToFp32(c).float64)

# ═════════════════════════════════════════════════════════════════════════
#  The trellis dequant reference (pure Nim, no torch): the gemm test's funnelWord + decodeCb +
#  tileShufflePerm + dequantTiles, ported verbatim
# ═════════════════════════════════════════════════════════════════════════

func funnelWord(packed: seq[uint32]; pw: int; t, K: int): uint32 =
  ## Extracts the 16-bit window holding trellis word t of a 256-word tile
  ## from the packed uint32 words (little-endian pairs of the int16 words),
  ## per exllamav3's funnel shift: the window starts at bit b0 = t·K + K − 16 + 256·K.
  let b0 = t * K + K - 16 + 256 * K
  let b1 = b0 + 16
  let i0 = (b0 div 32) mod pw
  let i1 = ((b1 - 1) div 32) mod pw
  let s0 = (((b1 - 1) div 32) + 1) * 32 - b1
  let merged = (uint64(packed[i0]) shl 32) or uint64(packed[i1])
  result = uint32((merged shr s0) and 0xFFFF)

func decodeCb(word: uint32; cb: int): uint16 =
  ## Procedural codebook decode (codebook.cuh). cb0/cb1: LCG (uint32 wrap)
  ## + LOP3 with the SASS 0x6a truth-table semantics (the function is (x and 0x8fff8fff)
  ## xor 0x3b603b60), then the fp16 add of the low and high halves.
  ## cb2: LCG + the byte-sum, the sum's low 16 bits reinterpreted as fp16,
  ## then a fused half multiply-add with the codebook constants
  ## (0x1eee/0xc931).
  if cb == 0:
    var x = word * 89226354'u32 + 64248484'u32
    x = (x and 0x8fff8fff'u32) xor 0x3b603b60'u32
    return f16Add(uint16(x and 0xFFFF), uint16((x shr 16) and 0xFFFF))
  elif cb == 1:
    var x = word * 0xCBAC1FED'u32
    x = (x and 0x8fff8fff'u32) xor 0x3b603b60'u32
    return f16Add(uint16(x and 0xFFFF), uint16((x shr 16) and 0xFFFF))
  else:
    var x = word * 0x83DCD12D'u32
    let sum = (x and 0xFF'u32) + ((x shr 8) and 0xFF'u32) +
              ((x shr 16) and 0xFF'u32) + ((x shr 24) and 0xFF'u32) +
              0x6400'u32
    return f16Fma(uint16(sum), 0x1EEE'u16, 0xC931'u16)

func decodeCbNumeric(word: uint32): uint16 =
  ## The kernel's frozen cb2 decode (exl3_ops' module-doc gap note):
  ## the byte sum's low 16 bits reinterpreted as fp16 bits, then two
  ## single-rounding fp16 ops with the RNE constant images: one fp16
  ## round of the fp32 product with 0x1EEF, one fp16 round of the fp32
  ## sum with 0xC932. This is the decode `dequantTrellis` instantiates
  ## at cb = 2 (the CUDA-faithful single-rounding fma with 0x1EEE/0xC931
  ## deviates by a few fp16 ulps).
  var x = word * 0x83DCD12D'u32
  let sum = (x and 0xFF'u32) + ((x shr 8) and 0xFF'u32) +
            ((x shr 16) and 0xFF'u32) + ((x shr 24) and 0xFF'u32) +
            0x6400'u32
  let sumF = fp16ToFp32(uint16(sum))
  let t1 = fp32ToFp16(sumF * fp16ToFp32(0x1EEF'u16))
  fp32ToFp16(fp16ToFp32(t1) + fp16ToFp32(0xC932'u16))

func tileShufflePerm(): array[256, int] =
  ## The 256-entry word-index permutation of a decoded 16×16 tile in
  ## exllamav3's reconstruct.cu (the tensor-core fragment layout →
  ## row-major output order), simulated from the kernel source. The
  ## anchored row 0 below is the independent hand-derived check.
  for R in 0 ..< 16:
    for C in 0 ..< 16:
      let h = 4 * (C div 8) + ((C mod 8) div 2)
      let g = C mod 2
      let j = 2 * (R div 8) + (R mod 2) + 4 * (h div 4)
      let r0 = (R mod 8) - (R mod 2)
      let lane = 8 * (h mod 4) + (r0 div 2)
      result[R * 16 + C] = 8 * lane + j + 32 * g
  doAssert result[0 .. 15] ==
    [0, 32, 64, 96, 128, 160, 192, 224, 4, 36, 68, 100, 132, 164, 196, 228],
    "tile shuffle row 0 must match the reconstruct.cu derivation"

proc dequantTiles(trellis: seq[int16]; tilesK, tilesN, K, cb: int;
                  useShuffle: bool): seq[uint16] =
  ## Reconstructs the fp16 weight matrix [tilesK·16, tilesN·16] from the packed
  ## trellis words, mirroring exllamav3's reconstruct: per tile,
  ## pair the int16 words into uint32 (little-endian), extract each of the 256 K-bit windows
  ## via the funnel shift, decode via the procedural codebook, and place each decoded value
  ## through the tensor-core shuffle (useShuffle) or at the natural row-major position
  ## (the negative check's natural placement).
  let tileWords = 256 * K div 16
  let outF = tilesN * 16
  result = newSeq[uint16](tilesK * 16 * outF)
  let perm = tileShufflePerm()
  for tk in 0 ..< tilesK:
    for tn in 0 ..< tilesN:
      let base = (tk * tilesN + tn) * tileWords
      var packed = newSeq[uint32](tileWords div 2)
      for i in 0 ..< tileWords div 2:
        packed[i] = uint32(cast[uint16](trellis[base + 2 * i])) or
                    (uint32(cast[uint16](trellis[base + 2 * i + 1])) shl 16)
      var dec = newSeq[uint16](256)
      for t in 0 ..< 256:
        dec[t] = decodeCb(funnelWord(packed, tileWords div 2, t, K), cb)
      for i in 0 ..< 256:
        let R = i div 16
        let C = i mod 16
        let wordIdx = if useShuffle: perm[i] else: i
        result[(tk * 16 + R) * outF + tn * 16 + C] = dec[wordIdx]

# ═════════════════════════════════════════════════════════════════════════
#  The FWHT arithmetic-order check: host mirrors of the sequential and the fragment-layout FWHT-128
# ═════════════════════════════════════════════════════════════════════════

func fwhtSeq128(x: var array[128, float32]) =
  ## The sequential 7-stage FWHT-128 (the reference's fwht_128): each
  ## stage over groups of 2·step with the elementwise butterfly
  ## (a+b, a−b), in fp32.
  var step = 1
  while step < 128:
    let half = step
    let groups = 128 div (step * 2)
    for p in 0 ..< groups:
      let offset = p * step * 2
      for i in 0 ..< half:
        let a = x[offset + i]
        let b = x[offset + half + i]
        x[offset + i] = a + b
        x[offset + half + i] = a - b
    step *= 2

func fwhtFrag128New(x: var array[128, float32]; row: int) =
  ## The fused kernel's fragment-layout FWHT-128 under the measured
  ## atom layout, host mirror: the row's 128 elements are held by the 4 lanes
  ## of the row's lane group (lane = base + {0, 1, 8, 9},
  ## base = 2·(row and 1) + 4·((row div 2) and 1) + 16·((row div 4) and 1))
  ## each lane holding 32 slots, slot (2·kk + m)·2 + v ↔ element 16·kk + 8·m + col + v
  ## with col = 2·(lane and 1) + 4·((lane shr 3) and 1) (the same slot map the kernel's
  ## hadamard128 uses over the 8 k-block tiles). Stage 1 pairs the (v0, v1) slots locally,
  ## stage 2 exchanges slot values between lanes `lane xor 1` and stage 4 between lanes `lane xor 8`
  ## (the sign-flip trick: the lane with the stage bit set computes −own + partner), and stages
  ## 8/16/32/64 pair the lane's own subtiles. The stage-2/4 exchanges read the pre-stage values
  ## (the kernel's simd_shuffle is warp-collective. The host mirrors it with a per-stage snapshot).
  let base = 2 * (row and 1) + 4 * ((row div 2) and 1) + 16 * ((row div 4) and 1)
  let lanes = [base, base + 1, base + 8, base + 9]
  var rv: array[4, array[32, float32]]
  for i in 0 ..< 4:
    let colI = 2 * (lanes[i] and 1) + 4 * ((lanes[i] shr 3) and 1)
    for kk in 0 ..< 8:
      for m in 0 ..< 2:
        for v in 0 ..< 2:
          rv[i][(kk * 2 + m) * 2 + v] = x[16 * kk + 8 * m + colI + v]
  # stage 1: local (v0, v1) pairs
  for i in 0 ..< 4:
    for kk in 0 ..< 8:
      for m in 0 ..< 2:
        let s0 = rv[i][(kk * 2 + m) * 2 + 0] + rv[i][(kk * 2 + m) * 2 + 1]
        let d0 = rv[i][(kk * 2 + m) * 2 + 0] - rv[i][(kk * 2 + m) * 2 + 1]
        rv[i][(kk * 2 + m) * 2 + 0] = s0
        rv[i][(kk * 2 + m) * 2 + 1] = d0
  # stage 2: lane^1 exchanges (col ± 2)
  block:
    var rvPrev = rv
    for i in 0 ..< 4:
      let partner = i xor 1
      let flip = (lanes[i] and 1) != 0
      for kk in 0 ..< 8:
        for m in 0 ..< 2:
          for v in 0 ..< 2:
            let slot = (kk * 2 + m) * 2 + v
            rv[i][slot] = (if flip: -rvPrev[i][slot] else: rvPrev[i][slot]) +
                          rvPrev[partner][slot]
  # stage 4: lane^8 exchanges (col ± 4)
  block:
    var rvPrev = rv
    for i in 0 ..< 4:
      let partner = i xor 2
      let flip = (lanes[i] and 8) != 0
      for kk in 0 ..< 8:
        for m in 0 ..< 2:
          for v in 0 ..< 2:
            let slot = (kk * 2 + m) * 2 + v
            rv[i][slot] = (if flip: -rvPrev[i][slot] else: rvPrev[i][slot]) +
                          rvPrev[partner][slot]
  # stage 8: (m=0, m=1) per (kk, v), local
  for i in 0 ..< 4:
    for kk in 0 ..< 8:
      for v in 0 ..< 2:
        let a8 = rv[i][(kk * 2 + 0) * 2 + v]
        let b8 = rv[i][(kk * 2 + 1) * 2 + v]
        rv[i][(kk * 2 + 0) * 2 + v] = a8 + b8
        rv[i][(kk * 2 + 1) * 2 + v] = a8 - b8
  # stage 16: (kk even, kk+1) per (m, v), local
  for i in 0 ..< 4:
    for kk in 0 ..< 8:
      for m in 0 ..< 2:
        for v in 0 ..< 2:
          if (kk and 1) == 0:
            let a16 = rv[i][(kk * 2 + m) * 2 + v]
            let b16 = rv[i][((kk + 1) * 2 + m) * 2 + v]
            rv[i][(kk * 2 + m) * 2 + v] = a16 + b16
            rv[i][((kk + 1) * 2 + m) * 2 + v] = a16 - b16
  # stage 32: (kk mod 4 in {0,1}, kk+2) per (m, v), local
  for i in 0 ..< 4:
    for kk in 0 ..< 8:
      for m in 0 ..< 2:
        for v in 0 ..< 2:
          if (kk and 2) == 0:
            let a32 = rv[i][(kk * 2 + m) * 2 + v]
            let b32 = rv[i][((kk + 2) * 2 + m) * 2 + v]
            rv[i][(kk * 2 + m) * 2 + v] = a32 + b32
            rv[i][((kk + 2) * 2 + m) * 2 + v] = a32 - b32
  # stage 64: (kk 0..3, kk+4) per (m, v), local
  for i in 0 ..< 4:
    for kk in 0 ..< 4:
      for m in 0 ..< 2:
        for v in 0 ..< 2:
          let a64 = rv[i][(kk * 2 + m) * 2 + v]
          let b64 = rv[i][((kk + 4) * 2 + m) * 2 + v]
          rv[i][(kk * 2 + m) * 2 + v] = a64 + b64
          rv[i][((kk + 4) * 2 + m) * 2 + v] = a64 - b64
  # write back through the slot map
  for i in 0 ..< 4:
    let colI = 2 * (lanes[i] and 1) + 4 * ((lanes[i] shr 3) and 1)
    for kk in 0 ..< 8:
      for m in 0 ..< 2:
        for v in 0 ..< 2:
          x[16 * kk + 8 * m + colI + v] = rv[i][(kk * 2 + m) * 2 + v]

# ═════════════════════════════════════════════════════════════════════════
#  Anchors
# ═════════════════════════════════════════════════════════════════════════

proc checkConversionAnchors() =
  ## Checks the shared fp16 conversion helpers that all three sides
  ## (buffer build, reference, output readback) consume: a silent
  ## rounding-mode drift must fail loudly here, not shift every case.
  doAssert fp32ToFp16(1.0005'f32) == 0x3C01'u16      # RNE: 1.0005 -> 1.0009765625
  doAssert fp32ToFp16(-1.0005'f32) == 0xBC01'u16
  doAssert fp32ToFp16(0.1'f32) == 0x2E66'u16        # -> 0.0999755859375
  doAssert fp32ToFp16(1.00048828125'f32) == 0x3C00'u16  # RNE tie -> even mantissa
  doAssert fp16ToFp32(0x3C01'u16) == 1.0009765625'f32
  doAssert fp16ToFp32(0x2E66'u16) == 0.0999755859375'f32

proc checkExl3Anchors() =
  ## Checks the "exl3_linear" case stream's first hash bytes (the dequant builder's shape source)
  ## and the value generators' first outputs: a silent drift must fail loudly here, not silently
  ## change every case's coverage.
  var cases = initShapeDraws("exl3_linear")
  let b0 = cases.nextBytes()
  doAssert b0[0] == 0xEB'u8 and b0[1] == 0x2E'u8 and b0[2] == 0x51'u8 and
    b0[3] == 0x5F'u8, "exl3_linear:0 hash bytes changed"
  doAssert drawInRange(b0, 0, 1, 2) == 2, "exl3_linear:0 tilesK changed"
  doAssert drawInRange(b0, 1, 2, 3) == 2, "exl3_linear:0 tilesN changed"
  let b1 = cases.nextBytes()
  doAssert b1 != b0, "exl3_linear cases must differ"
  # value generators
  doAssert exl3Val(0, 0, 3, 1.0'f32) == -1.5'f32
  doAssert exl3Val(1, 0, 3, 1.0'f32) == -1.125'f32
  doAssert exl3Val(0, 1, 3, 1.0'f32) == -0.625'f32
  doAssert exl3Val(4, 4, 3, 1.0'f32 / 128.0'f32) ==
                  exl3Val(4, 4, 3, 1.0'f32) / 128.0'f32
  doAssert exl3Val(0, 0, 5, 1.0'f32) == 1.5'f32
  # the row-block term: row 32 must differ from row 0, and rows r and r+32
  # must never alias on any seed (17·(r div 32) shifts k by 17)
  doAssert exl3Val(32, 0, 3, 1.0'f32) == 0.625'f32   # (99 + 33 + 17) mod 32 = 21
  var nAlias = 0
  for seed in [3, 5, 7]:
    for r in 0 ..< 32:
      for c in 0 ..< 256:
        if exl3Val(r, c, seed, 1.0'f32) == exl3Val(r + 32, c, seed, 1.0'f32):
          inc nAlias
  doAssert nAlias == 0, "rows r and r + 32 must never alias"
  doAssert exl3Val(0, 0, 7, 1.0'f32) == 0.5'f32   # the svh seed, every linear case
  # trellis word generator (first words of the first fused-linear
  # case's seed-100 buffer and the first dequant cell's seed-10 buffer)
  doAssert trellisWord(0, 100) == int16(-12991)
  doAssert trellisWord(1, 100) == int16(-9896)
  doAssert trellisWord(2, 100) == int16(-910)
  doAssert trellisWord(0, 10) == int16(11808)
  doAssert trellisWord(1, 10) == int16(14826)

# ═════════════════════════════════════════════════════════════════════════
#  Sections
# ═════════════════════════════════════════════════════════════════════════

proc checkDequantCell(engine: var auto; bits, cb: int; tilesK, tilesN: int;
                      trellis: seq[int16]; host: seq[uint16];
                      kernelName: string): tuple[nBitExact, nDiff: int;
                                                 worst: float32] =
  ## Launches the dequant wrapper for the (bits, cb) cell and compares
  ## the 16×32 row-major tile against the reference's first 16×32 tile
  ## (rows 0..15, cols 0..31). Returns the bit-exact count, the differing count
  ## and the worst |Δ|.
  var outBuf = newSeq[uint16](512)
  engine.run << (grid: (1, 1, 1), blk: (32, 1)) >> (
    kernelName, outBuf,
    (trellis, int32(0), int32(tilesN), int32(0), int32(0),
     int32(bits), int32(cb)))
  let outF = tilesN * 16
  var nBitExact = 0
  var worst = 0.0'f32
  for i in 0 ..< 512:
    let want = host[(i div 32) * outF + (i mod 32)]
    let got = outBuf[i]
    if got == want:
      inc nBitExact
    else:
      let d = abs(fp16ToFp32(got) - fp16ToFp32(want))
      if d > worst: worst = d
  result = (nBitExact, 512 - nBitExact, worst)

proc runDequantSection(engine: var auto; cases: var ShapeDraws) =
  echo "\n── trellis dequant check (kernel decode vs the pure-Nim dequantTiles reference) ──"
  # the CUDA-faithful cb2 expected anchors (the gemm test's anchors)
  let g0 = decodeCb(0x0000'u32, 2)
  let g1 = decodeCb(0x0001'u32, 2)
  doAssert g0 == 0xC2E8'u16, "cb2 expected anchor 0x0000 broken"
  doAssert g1 == 0x3921'u16, "cb2 expected anchor 0x0001 broken"
  echo &"  decodeCb(0x0000, cb=2) = 0x{g0:04X} (anchored 0xC2E8)"
  echo &"  decodeCb(0x0001, cb=2) = 0x{g1:04X} (anchored 0x3921)"
  # the numeric-vs-faithful family deviation (the gemm test's demo,
  # host-side): the kernel's two-rounding numeric form vs the CUDA-faithful single-rounding fma,
  # bounded by a few fp16 ulps
  var devN = 0
  var devWorst = 0.0'f32
  for w in 0 ..< 1024:
    let a = decodeCb(uint32(w), 2)
    let b = decodeCbNumeric(uint32(w))
    if a != b:
      inc devN
      let d = abs(fp16ToFp32(a) - fp16ToFp32(b))
      if d > devWorst: devWorst = d
  doAssert devN != 0,
    "the numeric cb2 family must differ from the CUDA-faithful decode"
  doAssert devWorst <= 2e-2'f32,
    "the numeric-vs-faithful cb2 deviation must stay within a few fp16 ulps"
  echo &"  numeric-sum cb2 vs CUDA-faithful cb2: {devN}/1024 differ, " &
       &"worst |Δ| = {devWorst:.3e} (within a few fp16 ulps)"
  var nCells = 0
  var nBitExactCells = 0
  for bits in 1 .. 8:
    for cb in 0 .. 2:
      let b = cases.nextBytes()
      let tilesK = drawInRange(b, 0, 1, 2)
      let tilesN = drawInRange(b, 1, 2, 3)
      let trellis = genTrellis(tilesK, tilesN, bits, seed = bits * 10 + cb)
      let host = dequantTiles(trellis, tilesK, tilesN, bits, cb, useShuffle = true)
      # host-side negative check: the natural placement must fail the reference
      let natural = dequantTiles(trellis, tilesK, tilesN, bits, cb, useShuffle = false)
      var nNatDiff = 0
      var natWorst = 0.0'f32
      let outF = tilesN * 16
      for i in 0 ..< 512:
        let want = host[(i div 32) * outF + (i mod 32)]
        let got = natural[(i div 32) * outF + (i mod 32)]
        if got != want:
          inc nNatDiff
          let d = abs(fp16ToFp32(got) - fp16ToFp32(want))
          if d > natWorst: natWorst = d
      if nNatDiff == 0:
        echo &"  negative check failed: natural placement matches the reference on cell bits={bits} cb={cb}"
        doAssert false, "natural placement must differ from the reference on cell bits={bits} cb={cb}"
      # kernel check (the shuffled decode)
      let g = checkDequantCell(engine, bits, cb, tilesK, tilesN, trellis, host,
                               "dequantKernel")
      if cb <= 1:
        if g.nDiff != 0:
          echo &"  DEQUANT FAIL: cell bits={bits} cb={cb} tiles={tilesK}x{tilesN}: " &
               &"{g.nDiff}/512 bad vs the reference, worst |Δ| = {g.worst}"
          doAssert false, &"dequant cell bits={bits} cb={cb} tiles={tilesK}x{tilesN}: " &
               &"{g.nDiff}/512 bad vs the reference, worst |Δ| = {g.worst}"
        inc nBitExactCells
        echo &"  cell bits={bits} cb={cb} tiles={tilesK}x{tilesN}: bit-exact " &
             &"{g.nBitExact}/512 (worst |Δ| = {g.worst}), negative check: natural placement " &
             &"{nNatDiff}/512 differ, worst |Δ| = {natWorst}"
      else:
        # cb2: the two-rounding numeric decode with the RNE constants
        # deviates from the CUDA-faithful single-rounding fma by a few
        # fp16 ulps (module-doc gap). Check that deviation.
        if g.worst > 2e-2'f32:
          echo &"  cb2 DEVIATION OUT OF BOUNDS: cell bits={bits} cb=2 tiles=" &
               &"{tilesK}x{tilesN}: {g.nDiff}/512 differ, worst |Δ| = {g.worst} " &
               &"(check 2e-2)"
          doAssert false, &"cb2 deviation out of bounds: cell bits={bits} cb=2 tiles=" &
               &"{tilesK}x{tilesN}: {g.nDiff}/512 differ, worst |Δ| = {g.worst} " &
               &"(check 2e-2)"
        echo &"  cell bits={bits} cb=2 tiles={tilesK}x{tilesN}: within the cb2 " &
             &"deviation ({g.nDiff}/512 differ, worst |Δ| = {g.worst}, check 2e-2) " &
             &", negative check: natural placement {nNatDiff}/512 differ, worst |Δ| = {natWorst}"
      inc nCells
  # kernel-side negative check on the bits {3,5,8} × cb {0,1,2} subset:
  # the natural-placement instantiation must fail the reference on device
  for bits in [3, 5, 8]:
    for cb in 0 .. 2:
      let b = cases.nextBytes()
      let tilesK = drawInRange(b, 0, 1, 2)
      let tilesN = drawInRange(b, 1, 2, 3)
      let trellis = genTrellis(tilesK, tilesN, bits, seed = bits * 10 + cb)
      let host = dequantTiles(trellis, tilesK, tilesN, bits, cb, useShuffle = true)
      let g = checkDequantCell(engine, bits, cb, tilesK, tilesN, trellis, host,
                               "dequantKernelNat")
      if g.nBitExact == 512:
        echo &"  KERNEL negative check failed: natural decode matches the reference on cell bits={bits} cb={cb}"
        doAssert false, "kernel negative check: the natural decode must differ from the reference on cell bits={bits} cb={cb}"
      echo &"  kernel negative check cell bits={bits} cb={cb} tiles={tilesK}x{tilesN}: natural " &
           &"decode {g.nDiff}/512 differ from the reference, worst |Δ| = {g.worst}"
  echo &"  dequant check: {nCells} cells PASS ({nBitExactCells} bit-exact cb0/cb1 cells, " &
       &"8 cb2 cells within a few fp16 ulps), natural placement differs on " &
       &"every cell, host-side and kernel-side"

proc checkFwhtOrder() =
  echo "\n── FWHT arithmetic-order check: fragment-layout butterfly ≡ sequential fwht_128 ──"
  var nDiff = 0
  var worst = 0.0'f32
  for trial in 0 ..< 16:
    for row in 0 ..< 8:
      var xSeq: array[128, float32]
      var xFrag: array[128, float32]
      for i in 0 ..< 128:
        # the (i div 32) block term de-periodizes the input: a period-32 input
        # keeps every stage-32/64 partner equal, and on equal partners
        # a flipped difference-output sign is
        # bit-invisible (0.0 == -0.0)
        let v = float32((i * 7 + trial * 13 + row * 5 +
                         11 * (i div 32)) mod 32) / 8.0'f32 - 2.0'f32
        xSeq[i] = v
        xFrag[i] = v
      fwhtSeq128(xSeq)
      fwhtFrag128New(xFrag, row)
      for i in 0 ..< 128:
        if xSeq[i] != xFrag[i]:
          inc nDiff
          let d = abs(xSeq[i] - xFrag[i])
          if d > worst: worst = d
  echo &"  fragment-layout ≡ sequential: {nDiff} bit-differences over " &
       &"16×8×128 elements, worst |Δ| = {worst}"
  doAssert nDiff == 0,
    "fragment-layout FWHT must be bit-identical to the sequential fwht_128"

# ═════════════════════════════════════════════════════════════════════════
#  The fused linear forward
# ═════════════════════════════════════════════════════════════════════════

proc referenceLinear(xBits, suh, svh: seq[uint16]; M, K, N: int;
                  wBits: seq[uint16]): seq[float32] =
  ## The pure-Nim fused-linear reference, the kernel's arithmetic order:
  ## per output row, fp16(x·suh) (one round), the sequential input
  ## FWHT with the 1/sqrt(128) scale and fp16 round, the matmul over
  ## all k accumulated in fp32 with one fp16 round, the sequential
  ## output FWHT with the same scale and round, then fp16(y·svh).
  ## The matmul accumulation order is the only expected delta vs the kernel
  ## (the per-case bit-exact floor in runLinearSection anchors the fused-EXL3 epilogue order (F-EX5)
  ## and the fp16 conversion semantics).
  result = newSeq[float32](M * N)
  for r in 0 ..< M:
    var C: array[256, float32]
    for blk in 0 ..< K div 128:
      var xh: array[128, float32]
      for k in 0 ..< 128:
        let xv = fp16ToFp32(xBits[r * K + blk * 128 + k])
        let sv = fp16ToFp32(suh[blk * 128 + k])
        xh[k] = fp16ToFp32(fp32ToFp16(xv * sv))
      fwhtSeq128(xh)
      for k in 0 ..< 128:
        xh[k] = fp16ToFp32(fp32ToFp16(xh[k] * 0.088388347648'f32))
      for c in 0 ..< N:
        var acc = 0.0'f32
        for k in 0 ..< 128:
          acc += xh[k] * fp16ToFp32(wBits[(blk * 128 + k) * N + c])
        C[c] += acc
    for blk in 0 ..< N div 128:
      var y: array[128, float32]
      for c in 0 ..< 128:
        y[c] = fp16ToFp32(fp32ToFp16(C[blk * 128 + c]))
      fwhtSeq128(y)
      for c in 0 ..< 128:
        let yv = fp16ToFp32(fp32ToFp16(y[c] * 0.088388347648'f32))
        result[r * N + blk * 128 + c] =
          fp16ToFp32(fp32ToFp16(yv * fp16ToFp32(svh[blk * 128 + c])))

proc runLinearSection(engine: var auto) =
  echo "\n── fused linear check: kernel vs the pure-Nim reference (M=1..32 decode + M>32 boundary) ──"
  # the case table sweeps the decode shape: M 1..32 (plus 64 = the M>32 boundary,
  # grid.y = 2 row tiles), K/N ∈ {128, 256}, bits ∈ {3, 5, 8}
  # (the funnel family, 5 = production). Inputs and suh/svh scales are fp16 grid points
  # scaled by 0.125, so the values sit in
  # [-0.25, 0.25) and the output magnitude stays ~1e-2..1e-1.
  let specs = [
    (1, 128, 128, 5), (32, 128, 128, 5), (8, 128, 256, 5),
    (16, 256, 256, 5), (2, 128, 128, 3), (5, 256, 128, 8),
    (31, 128, 256, 5), (64, 128, 128, 5), (3, 256, 128, 5),
  ]
  var worstAll = 0.0'f32
  for d in 0 ..< specs.len:
    let (M, K, N, bits) = specs[d]
    let kernelName = "exl3LinearB" & $bits
    let scaleF = 0.125'f32
    let trellis = genTrellis(K div 16, N div 16, bits, seed = 100 + d)
    let wBits = dequantTiles(trellis, K div 16, N div 16, bits, 0, useShuffle = true)
    var xBits = newSeq[uint16](M * K)
    var suh = newSeq[uint16](K)
    var svh = newSeq[uint16](N)
    for r in 0 ..< M:
      for c in 0 ..< K:
        xBits[r * K + c] = fp32ToFp16(exl3Val(r, c, 3, scaleF))
    for c in 0 ..< K:
      suh[c] = fp32ToFp16(exl3Val(0, c, 5, scaleF))
    for c in 0 ..< N:
      svh[c] = fp32ToFp16(exl3Val(0, c, 7, scaleF))
    let yRef = referenceLinear(xBits, suh, svh, M, K, N, wBits)
    var outBuf = newSeq[uint16](M * N)
    engine.run << (grid: (N div 128, (M + 31) div 32, 1), blk: (32, 1)) >> (
      kernelName, outBuf,
      (xBits, trellis, suh, svh, int32(M), int32(K), int32(N)))
    var nBitExact = 0
    var nBad = 0
    var gotF = newSeq[float32](outBuf.len)
    for i in 0 ..< outBuf.len:
      gotF[i] = fp16ToFp32(outBuf[i])
      if gotF[i] == yRef[i]:
        inc nBitExact
      else:
        let d = abs(gotF[i] - yRef[i])
        if d != d or d > 1e-2'f32:      # NaN is a failure, not a pass
          inc nBad
    let worst = worstAbsDiff(gotF, yRef)
    if nBad != 0:
      echo &"  case {d}: M={M} K={K} N={N} bits={bits}, FAIL: {nBad}/{outBuf.len} " &
           &"bad vs the reference, worst |Δ| = {worst}"
      doAssert false, &"case {d} M={M} K={K} N={N} bits={bits}: {nBad}/{outBuf.len} " &
           &"bad vs the reference, worst |Δ| = {worst}"
    if nBitExact * 100 div outBuf.len < 90:
      echo &"  case {d}: M={M} K={K} N={N} bits={bits}, bit-exact floor " &
           &"breached ({nBitExact}/{outBuf.len})"
      doAssert false, &"case {d} M={M} K={K} N={N} bits={bits}: bit-exact floor " &
           &"breached ({nBitExact}/{outBuf.len})"
    if worst > worstAll: worstAll = worst
    var h = 0x811C9DC5'u32
    for i in 0 ..< outBuf.len:
      h = (h xor uint32(outBuf[i])) * 0x01000193'u32
    if d == 0:
      doAssert h == 0x99ADF850'u32, "case 0 (M=1, r < 32) id drifted"
    echo &"  case {d}: M={M} K={K} N={N} bits={bits}, bit-exact " &
         &"{nBitExact}/{outBuf.len} ({(100 * nBitExact div outBuf.len)}%), " &
         &"worst |Δ| = {worst:.3e}, id 0x{h:08X}"
  # marker case: an M < 32 case into a 32-row output buffer
  # pre-filled with 0x7C00 (+inf). The kernel's row guard must not
  # write rows M..31, and rows 0..<M must match the reference.
  block:
    let (M, K, N, bits) = (5, 128, 128, 5)
    let kernelName = "exl3LinearB" & $bits
    let scaleF = 0.125'f32
    let trellis = genTrellis(K div 16, N div 16, bits, seed = 100 + 9)
    let wBits = dequantTiles(trellis, K div 16, N div 16, bits, 0, useShuffle = true)
    var xBits = newSeq[uint16](M * K)
    var suh = newSeq[uint16](K)
    var svh = newSeq[uint16](N)
    for r in 0 ..< M:
      for c in 0 ..< K:
        xBits[r * K + c] = fp32ToFp16(exl3Val(r, c, 3, scaleF))
    for c in 0 ..< K:
      suh[c] = fp32ToFp16(exl3Val(0, c, 5, scaleF))
    for c in 0 ..< N:
      svh[c] = fp32ToFp16(exl3Val(0, c, 7, scaleF))
    let yRef = referenceLinear(xBits, suh, svh, M, K, N, wBits)
    var outBuf = newSeq[uint16](32 * N)
    for i in 0 ..< outBuf.len:
      outBuf[i] = 0x7C00'u16
    engine.run << (grid: (N div 128, 1, 1), blk: (32, 1)) >> (
      kernelName, outBuf,
      (xBits, trellis, suh, svh, int32(M), int32(K), int32(N)))
    var nMarkerBad = 0
    for i in M * N ..< outBuf.len:
      if outBuf[i] != 0x7C00'u16:
        inc nMarkerBad
    if nMarkerBad != 0:
      echo &"  marker case: M={M} K={K} N={N} bits={bits}, FAIL: the row " &
           &"guard wrote {nMarkerBad}/{outBuf.len - M * N} marker elements"
      doAssert false, &"marker case M={M} K={K} N={N} bits={bits}: the row guard wrote " &
           &"{nMarkerBad}/{outBuf.len - M * N} marker elements"
    var nBitExact = 0
    var nBad = 0
    var gotF = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      gotF[i] = fp16ToFp32(outBuf[i])
      if gotF[i] == yRef[i]:
        inc nBitExact
      else:
        let d = abs(gotF[i] - yRef[i])
        if d != d or d > 1e-2'f32:
          inc nBad
    let worst = worstAbsDiff(gotF, yRef)
    if worst > worstAll: worstAll = worst
    if nBad != 0 or nBitExact * 100 div (M * N) < 90:
      echo &"  marker case: M={M} K={K} N={N} bits={bits}, FAIL: " &
           &"{nBad}/{M * N} bad vs the reference, bit-exact " &
           &"{nBitExact}/{M * N}, worst |Δ| = {worst}"
      doAssert false, &"marker case M={M} K={K} N={N} bits={bits}: {nBad}/{M * N} bad " &
           &"vs the reference, bit-exact {nBitExact}/{M * N}, worst |Δ| = {worst}"
    echo &"  marker case: M={M} K={K} N={N} bits={bits}, 32-row buffer, " &
         &"rows {M}..31 untouched ({nMarkerBad} guard violations), rows " &
         &"0..<M bit-exact {nBitExact}/{M * N}, worst |Δ| = {worst:.3e}"
  doAssert worstAll <= 1.25e-4'f32,
    "fused-linear worst |Δ| = " & $worstAll &
    " exceeds the 1.25e-4 bound (~2 fp16 ulps at output magnitude ~1e-1)"
  echo &"  fused-linear check: PASS, all {specs.len} cases + the marker " &
       &"case within the 1e-2 tolerance and the ≥ 90% per-case bit-exact " &
       &"floor (worst |Δ| across cases = {worstAll}), the M=64 case proves " &
       &"the M>32 boundary generalizes (grid.y row tiles)"

# ═════════════════════════════════════════════════════════════════════════
#  checkExl3Linear
#  ═════════════════════════════════════════════════════════════════════════

proc checkExl3Linear(): bool =
  checkConversionAnchors()
  checkExl3Anchors()
  var engine = bkMetal.init()
  engine.ingest(exl3Msl)
  echo exl3Msl            # keep the generated MSL inspectable
  var cases = initShapeDraws("exl3_linear")
  runDequantSection(engine, cases)
  checkFwhtOrder()
  runLinearSection(engine)
  echo "\nexl3-linear check: trellis dequant PASS vs the pure-Nim " &
       "dequantTiles reference on the full bits 1..8 × cb 0..2 matrix " &
       "(cb0/cb1 bit-exact, cb2 within a few fp16 ulps; natural placement " &
       "differs on every cell, host-side and kernel-side), FWHT " &
       "arithmetic-order anchor bit-exact, fused kernel PASS vs the " &
       "pure-Nim reference on M=1..32 + the M>32 boundary incl. the " &
       "marker row-guard case and the per-case bit-exact floor"
  result = true

when isMainModule:
  runCppTest("exl3_linear_fwd vs the reference implementation", checkExl3Linear)
