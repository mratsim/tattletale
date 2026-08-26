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
##   workspace/positron/tests/manual_exl3_gemv_fp16.nim

import std/[strformat, strutils]
import workspace/crucible
import workspace/ceramic
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/exl3_ops
import ../src/kernels/ceramic/exl3_gemv
import ../src/kernels/ceramic/exl3_gemm_fwd

proc worstAbsDiff(actual, expected: seq[float32]): float32 =
  ## Largest |actual[i] - expected[i]| over all elements.
  result = 0.0'f32
  for i in 0 ..< actual.len:
    let d = abs(actual[i] - expected[i])
    if d > result: result = d

# ═════════════════════════════════════════════════════════════════════════
#  The metal blocks. One launcher per MMODE dispatches over runtime
#  bits/cb int32 args to the 24 static instantiations, plus the 3 gemm
#  launchers for the cross-check. First param = the output buffer Out:
#  engine.run binds the separate outBuf argument to it and the args
#  tuple to the rest. D = 128 baked into every call.
# ═════════════════════════════════════════════════════════════════════════

const exl3GemvM0Msl = metal:
  proc exl3GemvM0(
      Out: ptr UncheckedArray[float16],
      x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16],
      suh, svh: ptr UncheckedArray[float16],
      M, K, N, bits, cb: int32) {.global.} =
    if bits == 1 and cb == 0:
      exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 1, 0, 0, 128)
    else:
      if bits == 1 and cb == 1:
        exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 1, 1, 0, 128)
      else:
        if bits == 1 and cb == 2:
          exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 1, 2, 0, 128)
        else:
          if bits == 2 and cb == 0:
            exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 2, 0, 0, 128)
          else:
            if bits == 2 and cb == 1:
              exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 2, 1, 0, 128)
            else:
              if bits == 2 and cb == 2:
                exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 2, 2, 0, 128)
              else:
                if bits == 3 and cb == 0:
                  exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 0, 0, 128)
                else:
                  if bits == 3 and cb == 1:
                    exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 1, 0, 128)
                  else:
                    if bits == 3 and cb == 2:
                      exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 2, 0, 128)
                    else:
                      if bits == 4 and cb == 0:
                        exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 4, 0, 0, 128)
                      else:
                        if bits == 4 and cb == 1:
                          exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 4, 1, 0, 128)
                        else:
                          if bits == 4 and cb == 2:
                            exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 4, 2, 0, 128)
                          else:
                            if bits == 5 and cb == 0:
                              exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 0, 0, 128)
                            else:
                              if bits == 5 and cb == 1:
                                exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 1, 0, 128)
                              else:
                                if bits == 5 and cb == 2:
                                  exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 2, 0, 128)
                                else:
                                  if bits == 6 and cb == 0:
                                    exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 6, 0, 0, 128)
                                  else:
                                    if bits == 6 and cb == 1:
                                      exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 6, 1, 0, 128)
                                    else:
                                      if bits == 6 and cb == 2:
                                        exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 6, 2, 0, 128)
                                      else:
                                        if bits == 7 and cb == 0:
                                          exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 7, 0, 0, 128)
                                        else:
                                          if bits == 7 and cb == 1:
                                            exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 7, 1, 0, 128)
                                          else:
                                            if bits == 7 and cb == 2:
                                              exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 7, 2, 0, 128)
                                            else:
                                              if bits == 8 and cb == 0:
                                                exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 0, 0, 128)
                                              else:
                                                if bits == 8 and cb == 1:
                                                  exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 1, 0, 128)
                                                else:
                                                  if bits == 8 and cb == 2:
                                                    exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 2, 0, 128)

const exl3GemvM1Msl = metal:
  proc exl3GemvM1(
      Out: ptr UncheckedArray[float16],
      x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16],
      suh, svh: ptr UncheckedArray[float16],
      M, K, N, bits, cb: int32) {.global.} =
    if bits == 1 and cb == 0:
      exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 1, 0, 1, 128)
    else:
      if bits == 1 and cb == 1:
        exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 1, 1, 1, 128)
      else:
        if bits == 1 and cb == 2:
          exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 1, 2, 1, 128)
        else:
          if bits == 2 and cb == 0:
            exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 2, 0, 1, 128)
          else:
            if bits == 2 and cb == 1:
              exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 2, 1, 1, 128)
            else:
              if bits == 2 and cb == 2:
                exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 2, 2, 1, 128)
              else:
                if bits == 3 and cb == 0:
                  exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 0, 1, 128)
                else:
                  if bits == 3 and cb == 1:
                    exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 1, 1, 128)
                  else:
                    if bits == 3 and cb == 2:
                      exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 2, 1, 128)
                    else:
                      if bits == 4 and cb == 0:
                        exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 4, 0, 1, 128)
                      else:
                        if bits == 4 and cb == 1:
                          exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 4, 1, 1, 128)
                        else:
                          if bits == 4 and cb == 2:
                            exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 4, 2, 1, 128)
                          else:
                            if bits == 5 and cb == 0:
                              exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 0, 1, 128)
                            else:
                              if bits == 5 and cb == 1:
                                exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 1, 1, 128)
                              else:
                                if bits == 5 and cb == 2:
                                  exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 2, 1, 128)
                                else:
                                  if bits == 6 and cb == 0:
                                    exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 6, 0, 1, 128)
                                  else:
                                    if bits == 6 and cb == 1:
                                      exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 6, 1, 1, 128)
                                    else:
                                      if bits == 6 and cb == 2:
                                        exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 6, 2, 1, 128)
                                      else:
                                        if bits == 7 and cb == 0:
                                          exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 7, 0, 1, 128)
                                        else:
                                          if bits == 7 and cb == 1:
                                            exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 7, 1, 1, 128)
                                          else:
                                            if bits == 7 and cb == 2:
                                              exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 7, 2, 1, 128)
                                            else:
                                              if bits == 8 and cb == 0:
                                                exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 0, 1, 128)
                                              else:
                                                if bits == 8 and cb == 1:
                                                  exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 1, 1, 128)
                                                else:
                                                  if bits == 8 and cb == 2:
                                                    exl3_gemv_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 2, 1, 128)

const exl3GemmMsl = metal:
  proc exl3GemmB3(
      Out: ptr UncheckedArray[float16],
      x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16],
      suh, svh: ptr UncheckedArray[float16],
      M, K, N: int32) {.global.} =
    exl3_gemm_fwd(Out, x, trellis, suh, svh, M, K, N, 3, 0, 128)
  proc exl3GemmB5(
      Out: ptr UncheckedArray[float16],
      x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16],
      suh, svh: ptr UncheckedArray[float16],
      M, K, N: int32) {.global.} =
    exl3_gemm_fwd(Out, x, trellis, suh, svh, M, K, N, 5, 0, 128)
  proc exl3GemmB8(
      Out: ptr UncheckedArray[float16],
      x: ptr UncheckedArray[float16],
      trellis: ptr UncheckedArray[int16],
      suh, svh: ptr UncheckedArray[float16],
      M, K, N: int32) {.global.} =
    exl3_gemm_fwd(Out, x, trellis, suh, svh, M, K, N, 8, 0, 128)

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
#  The trellis dequant reference implementation (pure Nim, no torch):
#  funnelWord + decodeCb + tileShufflePerm + dequantTiles
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
  ## Procedural codebook decode (codebook.cuh), the CUDA-faithful form.
  ## cb0/cb1: LCG (uint32 wrap) + LOP3 with the SASS 0x6a truth-table semantics
  ## (the function is (x and 0x8fff8fff) xor 0x3b603b60), then the fp16 add
  ## of the low and high halves. cb2: LCG + the byte-sum, the sum's low 16 bits
  ## reinterpreted as fp16, then a fused half multiply-add with the codebook constants
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
  ## at cb = 2, so the test's cb2 cells check bit-exact against it
  ## (the CUDA-faithful single-rounding fma with 0x1EEE/0xC931
  ## deviates by a few fp16 ulps, demonstrated in runReferenceAnchors).
  var x = word * 0x83DCD12D'u32
  let sum = (x and 0xFF'u32) + ((x shr 8) and 0xFF'u32) +
            ((x shr 16) and 0xFF'u32) + ((x shr 24) and 0xFF'u32) +
            0x6400'u32
  let sumF = fp16ToFp32(uint16(sum))
  let t1 = fp32ToFp16(sumF * fp16ToFp32(0x1EEF'u16))
  fp32ToFp16(fp16ToFp32(t1) + fp16ToFp32(0xC932'u16))

func buildDecodeTable(cb: int): array[65536, uint16] =
  ## One-time decode table for a codebook: `decodeCb` for cb 0/1, the kernel-faithful `decodeCbNumeric`
  ## for cb 2. The dequant reference's per-word decode is a pure function
  ## of the 16-bit funneled word, so the table turns the lm_head case's 155M decodes
  ## into lookups.
  for w in 0 ..< 65536:
    result[w] = if cb == 2: decodeCbNumeric(uint32(w))
                else: decodeCb(uint32(w), cb)

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
  ## via the funnel shift, decode via the codebook table (numeric cb2,
  ## the kernel's frozen semantics), and place each decoded value through the tensor-core shuffle
  ## (useShuffle) or at the natural row-major position (the placement the shuffle negative check
  ## must differ from).
  let tileWords = 256 * K div 16
  let outF = tilesN * 16
  result = newSeq[uint16](tilesK * 16 * outF)
  let table = buildDecodeTable(cb)
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
        dec[t] = table[funnelWord(packed, tileWords div 2, t, K)]
      for i in 0 ..< 256:
        let R = i div 16
        let C = i mod 16
        let wordIdx = if useShuffle: perm[i] else: i
        result[(tk * 16 + R) * outF + tn * 16 + C] = dec[wordIdx]

# ═════════════════════════════════════════════════════════════════════════
#  The FWHT arithmetic-order reference: the sequential 7-stage transform
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

# ═════════════════════════════════════════════════════════════════════════
#  The fused decode-GEMV reference
# ═════════════════════════════════════════════════════════════════════════

proc referenceGemm(xBits, suh, svh: seq[uint16]; M, K, N: int;
                wBits: seq[uint16]): seq[float32] =
  ## The pure-Nim fused decode-GEMV reference, the kernel's arithmetic
  ## order: per output row, fp16(x·suh) (one round), the sequential
  ## input FWHT with the 1/sqrt(128) scale and fp16 round, the matmul
  ## over all k accumulated in fp32 with one fp16 round, the sequential
  ## output FWHT with the same scale and round, then fp16(y·svh).
  ## The matmul accumulation order is the only expected delta vs the kernel
  ## (the per-case bit-exact floor anchors the fused-EXL3 epilogue order (F-EX5)
  ## and the fp16 conversion semantics). The k-outer loop streams the weight matrix
  ## contiguously (the lm_head case's 1.6e8 FMAs at M=1).
  result = newSeq[float32](M * N)
  let wPtr = cast[ptr UncheckedArray[uint16]](wBits[0].unsafeAddr)
  for r in 0 ..< M:
    var C = newSeq[float32](N)
    for blk in 0 ..< K div 128:
      var xh: array[128, float32]
      for k in 0 ..< 128:
        let xv = fp16ToFp32(xBits[r * K + blk * 128 + k])
        let sv = fp16ToFp32(suh[blk * 128 + k])
        xh[k] = fp16ToFp32(fp32ToFp16(xv * sv))
      fwhtSeq128(xh)
      for k in 0 ..< 128:
        xh[k] = fp16ToFp32(fp32ToFp16(xh[k] * 0.088388347648'f32))
      let wBase = blk * 128 * N
      for k in 0 ..< 128:
        let xk = xh[k]
        let wRow = wBase + k * N
        for c in 0 ..< N:
          C[c] += xk * fp16ToFp32(wPtr[wRow + c])
    for blk in 0 ..< N div 128:
      var y: array[128, float32]
      for c in 0 ..< 128:
        y[c] = fp16ToFp32(fp32ToFp16(C[blk * 128 + c]))
      fwhtSeq128(y)
      for c in 0 ..< 128:
        let yv = fp16ToFp32(fp32ToFp16(y[c] * 0.088388347648'f32))
        result[r * N + blk * 128 + c] =
          fp16ToFp32(fp32ToFp16(yv * fp16ToFp32(svh[blk * 128 + c])))

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

proc checkGemvAnchors() =
  ## Checks the "exl3_gemv" case stream's first hash bytes (the M=1
  ## matrix's per-cell K/N source) and the value generators' first
  ## outputs: a silent drift must fail loudly here, not silently
  ## change every case's coverage.
  var cases = initShapeDraws("exl3_gemv")
  let b0 = cases.nextBytes()
  doAssert b0[0] == 0xA9'u8 and b0[1] == 0x05'u8 and b0[2] == 0x9E'u8 and
    b0[3] == 0x49'u8, "exl3_gemv:0 hash bytes changed"
  doAssert drawInRange(b0, 0, 1, 2) == 2, "exl3_gemv:0 K block changed"
  doAssert drawInRange(b0, 1, 1, 2) == 2, "exl3_gemv:0 N block changed"
  let b1 = cases.nextBytes()
  doAssert b1 != b0, "exl3_gemv cases must differ"
  # value generators (the gemm test's anchors, same generators)
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
  doAssert exl3Val(0, 0, 7, 1.0'f32) == 0.5'f32   # the svh seed, every case
  # trellis word generator (the first words of the M=1 matrix's first
  # cell, seed 100, and the reference-anchor cells' seed-10 buffer)
  doAssert trellisWord(0, 100) == int16(-12991)
  doAssert trellisWord(1, 100) == int16(-9896)
  doAssert trellisWord(2, 100) == int16(-910)
  doAssert trellisWord(0, 10) == int16(11808)
  doAssert trellisWord(1, 10) == int16(14826)

# ═════════════════════════════════════════════════════════════════════════
#  Sections
# ═════════════════════════════════════════════════════════════════════════

proc runReferenceAnchors() =
  echo "\n── reference anchors: cb2 anchors, the numeric-vs-faithful deviation, the shuffle negative check ──"
  # the CUDA-faithful cb2 expected anchors (the gemm test's anchors)
  let g0 = decodeCb(0x0000'u32, 2)
  let g1 = decodeCb(0x0001'u32, 2)
  doAssert g0 == 0xC2E8'u16, "cb2 expected anchor 0x0000 broken"
  doAssert g1 == 0x3921'u16, "cb2 expected anchor 0x0001 broken"
  echo &"  decodeCb(0x0000, cb=2) = 0x{g0:04X} (anchored 0xC2E8)"
  echo &"  decodeCb(0x0001, cb=2) = 0x{g1:04X} (anchored 0x3921)"
  # the kernel-faithful numeric decode's own expected values
  let n0 = decodeCbNumeric(0x0000'u32)
  let n1 = decodeCbNumeric(0x0001'u32)
  doAssert n0 == 0xC2EA'u16, "numeric cb2 anchor 0x0000 broken"
  doAssert n1 == 0x3920'u16, "numeric cb2 anchor 0x0001 broken"
  echo &"  decodeCbNumeric(0x0000) = 0x{n0:04X}, decodeCbNumeric(0x0001) = 0x{n1:04X} (anchored)"
  # the family deviation: the kernel's two-rounding numeric form vs
  # the CUDA-faithful single-rounding fma, bounded by a few fp16 ulps
  # (the gemm test's cb2 deviation, demonstrated once)
  var nDiff = 0
  var devWorst = 0.0'f32
  for w in 0 ..< 65536:
    let a = decodeCb(uint32(w), 2)
    let b = decodeCbNumeric(uint32(w))
    if a != b:
      inc nDiff
      let d = abs(fp16ToFp32(a) - fp16ToFp32(b))
      if d > devWorst: devWorst = d
  doAssert nDiff != 0,
    "the numeric cb2 family must differ from the CUDA-faithful decode"
  doAssert devWorst <= 2e-2'f32,
    "the numeric-vs-faithful cb2 deviation must stay within a few fp16 ulps"
  echo &"  numeric-sum cb2 vs CUDA-faithful cb2: {nDiff}/65536 differ, " &
       &"worst |Δ| = {devWorst:.3e} (within a few fp16 ulps)"
  # host-side negative check: the natural (unshuffled) placement must fail the shuffled decode
  # on every cell of the bits × cb matrix
  for bits in 1 .. 8:
    for cb in 0 .. 2:
      let trellis = genTrellis(8, 8, bits, seed = bits * 10 + cb)
      let shuffled = dequantTiles(trellis, 8, 8, bits, cb, useShuffle = true)
      let natural = dequantTiles(trellis, 8, 8, bits, cb, useShuffle = false)
      var nNatDiff = 0
      for i in 0 ..< 128 * 128:
        if shuffled[i] != natural[i]:
          inc nNatDiff
      if nNatDiff == 0:
        echo &"  negative check failed: natural placement matches the reference on cell " &
             &"bits={bits} cb={cb}"
        doAssert false, "natural placement must differ from the reference on cell " &
             &"bits={bits} cb={cb}"
  echo "  host-side negative check: natural placement differs from the shuffled decode on " &
       "all 24 cells"

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

proc runGemvCase(engine: var auto; M, K, N, bits, cb, mmode, trellisSeed: int;
                 scaleF: float32; tag: string;
                 st: var CaseStats; hAll: var uint32): string =
  ## One decode-GEMV case: the real Metal kernel (the mmode/bit/cb
  ## dispatch) vs the pure-Nim reference. Checks: worst |Δ| ≤ 1e-2, NaN
  ## = failure, per-case bit-exact floor ≥ 90%. Returns the case's
  ## summary line (id included).
  let trellis = genTrellis(K div 16, N div 16, bits, seed = trellisSeed)
  let wBits = dequantTiles(trellis, K div 16, N div 16, bits, cb,
                           useShuffle = true)
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
  let yRef = referenceGemm(xBits, suh, svh, M, K, N, wBits)
  var outBuf = newSeq[uint16](M * N)
  let launcher = if mmode == 0: "exl3GemvM0" else: "exl3GemvM1"
  engine.run << (grid: (N div 128, (M + 31) div 32, 1), blk: (32, 1)) >> (
    launcher, outBuf,
    (xBits, trellis, suh, svh, int32(M), int32(K), int32(N),
     int32(bits), int32(cb)))
  var nBit = 0
  var nBad = 0
  var gotF = newSeq[float32](outBuf.len)
  for i in 0 ..< outBuf.len:
    gotF[i] = fp16ToFp32(outBuf[i])
    if gotF[i] == yRef[i]:
      inc nBit
    else:
      let d = abs(gotF[i] - yRef[i])
      if d != d or d > 1e-2'f32:      # NaN is a failure, not a pass
        inc nBad
  let worst = worstAbsDiff(gotF, yRef)
  st.nElem += outBuf.len
  st.nBitExact += nBit
  st.nBad += nBad
  if worst > st.worst: st.worst = worst
  fnv1a(hAll, outBuf)
  var h = 0x811C9DC5'u32
  fnv1a(h, outBuf)
  let pct = 100 * nBit div outBuf.len
  result = &"  [{tag}] mmode={mmode} M={M} K={K} N={N} bits={bits} cb={cb}: " &
           &"bit-exact {nBit}/{outBuf.len} ({pct}%), worst |Δ| = " &
           &"{worst:.3e}, id 0x{h:08X}"
  if nBad != 0:
    echo result
    echo &"  FAIL: {nBad}/{outBuf.len} elements beyond the 1e-2 check or NaN"
    doAssert false, &"{nBad}/{outBuf.len} elements beyond the 1e-2 check or NaN"
  if pct < 90:
    echo result
    echo &"  FAIL: bit-exact floor breached ({pct}% < 90%)"
    doAssert false, &"bit-exact floor breached ({pct}% < 90%)"

# ═════════════════════════════════════════════════════════════════════════
#  Section 3: the MMODE-0 matrix, M=1 for every (bits, cb) cell on a case-driven K/N grid
# ═════════════════════════════════════════════════════════════════════════

proc runMmode0Matrix(engine: var auto; cases: var ShapeDraws;
                     st: var CaseStats; hAll: var uint32) =
  echo "\n── MMODE-0 matrix: M=1 for the bits 1..8 × cb 0..2 cells on " &
       "K/N ∈ {128, 256} ──"
  var cellIdx = 0
  for bits in 1 .. 8:
    for cb in 0 .. 2:
      let b = cases.nextBytes()
      let K = 128 * drawInRange(b, 0, 1, 2)
      let N = 128 * drawInRange(b, 1, 1, 2)
      let line = runGemvCase(engine, 1, K, N, bits, cb, 0,
                             trellisSeed = 100 + cellIdx,
                             scaleF = 0.125'f32,
                             tag = &"m0 bits={bits} cb={cb}", st, hAll)
      echo line
      if cellIdx == 0:
        doAssert line.contains("id 0xAA69F3A9"), "cell 0 id drifted"
      inc cellIdx

# ═════════════════════════════════════════════════════════════════════════
#  Section 3b: the lm_head case, M=1, bits=6, cb=0, K=1024, N=151936
#  (the real model's only non-5 tensor at decode M, grid.x = 1187)
# ═════════════════════════════════════════════════════════════════════════

proc runLmHeadCase(engine: var auto; st: var CaseStats; hAll: var uint32) =
  echo "\n── the lm_head case: M=1, bits=6, cb=0, K=1024, N=151936 ──"
  let line = runGemvCase(engine, 1, 1024, 151936, 6, 0, 0,
                         trellisSeed = 1000, scaleF = 0.125'f32,
                         tag = "lm_head", st, hAll)
  echo line
  doAssert line.contains("id 0xF6DFD3CB"), "lm_head id drifted"

# ═════════════════════════════════════════════════════════════════════════
#  Section 3c: the MMODE-0 fold-proof case, M=8 through the m=1 fast
#  path (rows 1..7 must be exactly ±0)
# ═════════════════════════════════════════════════════════════════════════

proc runMmode0FoldCase(engine: var auto; st: var CaseStats; hAll: var uint32) =
  ## The fold proof: an M=8 case through the MMODE-0 m=1 fast path.
  ## The compile-time row limit 1 folds rows 1..31 of the x tile to statically-zero fragments,
  ## so the stored rows 1..7 must be exactly ±0 while row 0 must match the reference.
  ## A leaky fold (rowLimit = M) computes the real gemv rows 1..7 and fails the zero assert.
  ## fp16 ±0: mask the sign bit (0 * negative svh/suh = -0.0 = 0x8000).
  ## The buffer is pre-filled with +inf so a skipped store also fails the zero assert.
  ## (the fold must write ±0, not leave rows untouched).
  echo "\n── the MMODE-0 fold-proof case: M=8 through the m=1 fast path ──"
  let (M, K, N, bits, cb) = (8, 128, 128, 5, 0)
  let trellis = genTrellis(K div 16, N div 16, bits, seed = 203)
  let wBits = dequantTiles(trellis, K div 16, N div 16, bits, cb,
                           useShuffle = true)
  var xBits = newSeq[uint16](M * K)
  var suh = newSeq[uint16](K)
  var svh = newSeq[uint16](N)
  let scaleF = 0.125'f32
  for r in 0 ..< M:
    for c in 0 ..< K:
      xBits[r * K + c] = fp32ToFp16(exl3Val(r, c, 3, scaleF))
  for c in 0 ..< K:
    suh[c] = fp32ToFp16(exl3Val(0, c, 5, scaleF))
  for c in 0 ..< N:
    svh[c] = fp32ToFp16(exl3Val(0, c, 7, scaleF))
  let yRef = referenceGemm(xBits, suh, svh, M, K, N, wBits)
  var outBuf = newSeq[uint16](M * N)
  for i in 0 ..< outBuf.len:
    outBuf[i] = 0x7C00'u16        # +inf marker: unwritten rows fail the zero assert
  engine.run << (grid: (N div 128, 1, 1), blk: (32, 1)) >> (
    "exl3GemvM0", outBuf,
    (xBits, trellis, suh, svh, int32(M), int32(K), int32(N),
     int32(bits), int32(cb)))
  var nRow0Bit = 0
  var nRow0Bad = 0
  var nZeroBad = 0
  var worst = 0.0'f32
  for r in 0 ..< M:
    for c in 0 ..< N:
      let i = r * N + c
      if r == 0:
        let got = fp16ToFp32(outBuf[i])
        if got == yRef[i]:
          inc nRow0Bit
        else:
          let d = abs(got - yRef[i])
          if d > worst: worst = d
          if d != d or d > 1e-2'f32:      # NaN is a failure, not a pass
            inc nRow0Bad
      elif (outBuf[i] and 0x7FFF'u16) != 0:
        inc nZeroBad        # rows 1..M-1 must be exactly ±0 (the fold)
  st.nElem += N
  st.nBitExact += nRow0Bit
  st.nBad += nRow0Bad
  if worst > st.worst: st.worst = worst
  fnv1a(hAll, outBuf[0 ..< N])
  if nRow0Bad != 0 or nZeroBad != 0 or worst > 1e-2'f32 or
     nRow0Bit * 100 div N < 90:
    echo &"  MMODE-0 fold case M={M}: FAIL row 0 {nRow0Bad}/{N} beyond the " &
         &"1e-2 check, bit-exact {nRow0Bit}/{N}, rows 1..{M - 1} {nZeroBad} " &
         &"non-zero (leaky fold?)"
    doAssert false, &"MMODE-0 fold case M={M}: row 0 {nRow0Bad}/{N} beyond the " &
         &"1e-2 check, bit-exact {nRow0Bit}/{N}, rows 1..{M - 1} {nZeroBad} " &
         &"non-zero (leaky fold?)"
  echo &"  MMODE-0 fold case: M={M} K={K} N={N} bits={bits} cb={cb}, row 0 " &
       &"bit-exact {nRow0Bit}/{N} (worst |Δ| = {worst:.3e}), rows 1..{M - 1} " &
       &"all ±0 (the m=1 fast-path fold)"

# ═════════════════════════════════════════════════════════════════════════
#  Section 4: the MMODE-1 cases, m ∈ {2,3,5,8} on the production projection geometry,
#  the M=8 cap, the M=16 proof, the marker row guard
# ═════════════════════════════════════════════════════════════════════════

proc runMmode1Cases(engine: var auto; st: var CaseStats; hAll: var uint32) =
  echo "\n── MMODE-1 cases: m ∈ {2,3,5,8} on the production projection " &
       "geometry, the M=8 cap, the M=16 proof, the marker, the " &
       "M=64 grid.y=2 proof ──"
  let specs = [
    (2, 5, 0, 1024, 2048),   # q
    (3, 5, 1, 1024, 1024),   # k/v + mcg
    (5, 6, 0, 1024, 3072),   # the projection pair at the lm_head bitrate
    (8, 4, 1, 3072, 1024),   # down, the 3072→1024-shaped case
  ]
  for d in 0 ..< specs.len:
    let (m, bits, cb, K, N) = specs[d]
    let line = runGemvCase(engine, m, K, N, bits, cb, 1,
                           trellisSeed = 300 + m,
                           scaleF = 0.125'f32,
                           tag = &"m1 m={m}", st, hAll)
    echo line
    if m == 5:
      doAssert line.contains("id 0x1386764E"), "MMODE-1 m=5 case id drifted"
  # the M=8 cap boundary (the MMODE-1 contract cap)
  echo runGemvCase(engine, 8, 1024, 1024, 5, 0, 1,
                   trellisSeed = 330, scaleF = 0.125'f32,
                   tag = "M=8 cap", st, hAll)
  # the M>8 proof: M=16 on the same MMODE-1 instantiation, the 32-row
  # tile parameterization generalizes beyond the 8-row contract cap
  echo runGemvCase(engine, 16, 1024, 1024, 5, 0, 1,
                   trellisSeed = 331, scaleF = 0.125'f32,
                   tag = "M=16 proof", st, hAll)
  # marker case: an M=8 case into a 32-row output buffer
  # pre-filled with 0x7C00 (+inf). The kernel's row guard must not
  # write rows 8..31, and rows 0..<8 must match the reference.
  block:
    let (M, K, N, bits, cb) = (8, 128, 128, 5, 0)
    let trellis = genTrellis(K div 16, N div 16, bits, seed = 202)
    let wBits = dequantTiles(trellis, K div 16, N div 16, bits, cb,
                             useShuffle = true)
    var xBits = newSeq[uint16](M * K)
    var suh = newSeq[uint16](K)
    var svh = newSeq[uint16](N)
    let scaleF = 0.125'f32
    for r in 0 ..< M:
      for c in 0 ..< K:
        xBits[r * K + c] = fp32ToFp16(exl3Val(r, c, 3, scaleF))
    for c in 0 ..< K:
      suh[c] = fp32ToFp16(exl3Val(0, c, 5, scaleF))
    for c in 0 ..< N:
      svh[c] = fp32ToFp16(exl3Val(0, c, 7, scaleF))
    let yRef = referenceGemm(xBits, suh, svh, M, K, N, wBits)
    var outBuf = newSeq[uint16](32 * N)
    for i in 0 ..< outBuf.len:
      outBuf[i] = 0x7C00'u16
    engine.run << (grid: (N div 128, 1, 1), blk: (32, 1)) >> (
      "exl3GemvM1", outBuf,
      (xBits, trellis, suh, svh, int32(M), int32(K), int32(N),
       int32(bits), int32(cb)))
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
    st.nElem += M * N
    st.nBitExact += nBitExact
    st.nBad += nBad
    if worst > st.worst: st.worst = worst
    if nBad != 0 or nBitExact * 100 div (M * N) < 90:
      echo &"  marker case: M={M} K={K} N={N} bits={bits}, FAIL: " &
           &"{nBad}/{M * N} bad vs the reference, bit-exact " &
           &"{nBitExact}/{M * N}, worst |Δ| = {worst}"
      doAssert false, &"marker case M={M} K={K} N={N} bits={bits}: {nBad}/{M * N} bad " &
           &"vs the reference, bit-exact {nBitExact}/{M * N}, worst |Δ| = {worst}"
    echo &"  marker case: M={M} K={K} N={N} bits={bits}, 32-row buffer, " &
         &"rows {M}..31 untouched ({nMarkerBad} guard violations), rows " &
         &"0..<M bit-exact {nBitExact}/{M * N}, worst |Δ| = {worst:.3e}"
  # the grid.y proof: M=64 (grid.y = 2) makes the (r div 32) row term
  # live in a kernel case (rows 32..63 differ from rows 0..31 by construction)
  # and tests the gemv kernel's own row-tile origin at grid.y >= 2
  # (the module doc's row-tile extension claim)
  echo runGemvCase(engine, 64, 128, 128, 5, 0, 1,
                   trellisSeed = 332, scaleF = 0.125'f32,
                   tag = "M=64 grid.y=2", st, hAll)

# ═════════════════════════════════════════════════════════════════════════
#  Section 5: the cross-check, gemv (M=1, MMODE 1) ≡ gemm kernel on identical inputs, one case per bits value
# ═════════════════════════════════════════════════════════════════════════

proc runCrossCheck(engineM1: var auto; engineGemm: var auto) =
  echo "\n── cross-check: gemv (M=1, MMODE 1) ≡ gemm kernel (bits 3/5/8) ──"
  let scaleF = 0.125'f32
  for bits in [3, 5, 8]:
    let M = 1
    let K = 128
    let N = 1024
    let trellis = genTrellis(K div 16, N div 16, bits, seed = 300 + bits)
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
    var outGemv = newSeq[uint16](M * N)
    var outGemm = newSeq[uint16](M * N)
    engineM1.run << (grid: (N div 128, 1, 1), blk: (32, 1)) >> (
      "exl3GemvM1", outGemv,
      (xBits, trellis, suh, svh, int32(M), int32(K), int32(N),
       int32(bits), int32(0)))
    engineGemm.run << (grid: (N div 128, 1, 1), blk: (32, 1)) >> (
      "exl3GemmB" & $bits, outGemm,
      (xBits, trellis, suh, svh, int32(M), int32(K), int32(N)))
    var nDiff = 0
    for i in 0 ..< outGemv.len:
      if outGemv[i] != outGemm[i]:
        inc nDiff
    echo &"  bits={bits}: gemv (MMODE 1) vs gemm at M=1 K={K} N={N}: " &
         &"{outGemv.len - nDiff}/{outGemv.len} bit-exact"
    if nDiff != 0:
      echo "  FAIL: the gemv must be bit-identical to the gemm kernel " &
           "(same body, the mmode row guard must not change the M=1 result)"
      doAssert false, "the gemv must be bit-identical to the gemm kernel " &
           "(same body, the mmode row guard must not change the M=1 result)"

# ═════════════════════════════════════════════════════════════════════════
#  checkExl3Gemv
#  ═════════════════════════════════════════════════════════════════════════

proc checkExl3Gemv(): bool =
  checkConversionAnchors()
  checkGemvAnchors()
  var engineM0 = bkMetal.init()
  engineM0.ingest(exl3GemvM0Msl)
  var engineM1 = bkMetal.init()
  engineM1.ingest(exl3GemvM1Msl)
  var engineGemm = bkMetal.init()
  engineGemm.ingest(exl3GemmMsl)
  echo exl3GemvM0Msl            # keep the generated MSL inspectable
  echo exl3GemvM1Msl
  echo exl3GemmMsl
  var cases = initShapeDraws("exl3_gemv")
  runReferenceAnchors()
  var stM0: CaseStats
  var stM1: CaseStats
  var hM0 = 0x811C9DC5'u32
  var hM1 = 0x811C9DC5'u32
  runMmode0Matrix(engineM0, cases, stM0, hM0)
  runLmHeadCase(engineM0, stM0, hM0)
  runMmode0FoldCase(engineM0, stM0, hM0)
  runMmode1Cases(engineM1, stM1, hM1)
  doAssert hM0 == 0xB3D07784'u32, "MMODE-0 aggregate id drifted"
  doAssert hM1 == 0x13A8D426'u32, "MMODE-1 aggregate id drifted"
  runCrossCheck(engineM1, engineGemm)
  echo "\n── summary ──"
  echo &"  MMODE-0 section (M=1 matrix + lm_head + M=8 fold case): " &
       &"{stM0.nElem} elements, bit-exact {stM0.nBitExact} " &
       &"({100.0 * float(stM0.nBitExact) / float(stM0.nElem):.2f}%), " &
       &"worst |Δ| = {stM0.worst:.3e}, id 0x{hM0:08X}"
  echo &"  MMODE-1 section (m=2/3/5/8 + M=8 cap + M=16 proof + M=64 " &
       &"grid.y=2 + marker): {stM1.nElem} elements, bit-exact " &
       &"{stM1.nBitExact} " &
       &"({100.0 * float(stM1.nBitExact) / float(stM1.nElem):.2f}%), " &
       &"worst |Δ| = {stM1.worst:.3e}, id 0x{hM1:08X}"
  echo "\nexl3-gemv check: reference anchors PASS (cb2 expected values, " &
       "numeric-vs-faithful deviation, shuffle negative check), the MMODE-0 " &
       "M=1 matrix PASS on all 24 bits × cb cells vs the pure-Nim " &
       "reference (1e-2 + ≥ 90% bit-exact floor), the lm_head case PASS at " &
       "M=1 on the production geometry, the M=8 fold-proof case PASS " &
       "(row 0 reference-checked, rows 1..7 exactly ±0), the MMODE-1 " &
       "m ∈ {2,3,5,8} production-geometry cases, the M=8 cap, the M=16 " &
       "generalization, the M=64 grid.y=2 case and the marker row-guard " &
       "case all PASS, and the gemv ≡ gemm cross-check bit-exact on bits 3/5/8"
  result = true

when isMainModule:
  runCppTest("exl3_gemv vs the reference implementation", checkExl3Gemv)
