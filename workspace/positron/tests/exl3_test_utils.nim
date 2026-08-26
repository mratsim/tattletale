## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared EXL3 test support: the deterministic fp16 input generators, the
## trellis dequant reference (funnel window, codebook decode table,
## tensor-core shuffle) that rebuilds the [K, N] fp16 weight matrix the
## fused kernels dequantize on the fly, and the torch FWHT-128 plus the
## worst-difference metric the manual_exl3_* tests share.

import std/bitops
import workspace/libtorch
import workspace/libtorch as F
import ../../ceramic/tests/tile_test_utils

func exl3Val*(r, c, seed: int; scaleF: float32): float32 =
  ## Deterministic fp16-exact value in [-2, 2)·scaleF. The (c div 32)
  ## and (r div 32) terms keep the pattern off the 32-col tile, the
  ## 128-col FWHT block and the 32-row tile (rows r and r + 32 differ).
  let k = (seed * (r + 1) + 7 * c + 11 * seed + 13 * (c div 32) +
           17 * (r div 32)) mod 32
  (float32(k) / 8.0'f32 - 2.0'f32) * scaleF

func trellisWord*(i, seed: int): int16 =
  ## Deterministic packed-trellis word: a per-index mix with a seed term.
  let h = uint32(i) * 2654435761'u32 + uint32(seed) * 40503'u32
  cast[int16]((h shr 16) xor (h and 0xFFFF'u32))

func genTrellis*(tilesK, tilesN, bits, seed: int): seq[int16] =
  ## Deterministic packed trellis buffer, tilesK·tilesN·(256·bits div 16)
  ## int16 words (the kernel's trellis layout).
  result = newSeq[int16](tilesK * tilesN * (256 * bits div 16))
  for i in 0 ..< result.len:
    result[i] = trellisWord(i, seed)

func buildExl3Inputs*(M, K, N, seed: int): tuple[xBits, suh, svh: seq[uint16]] =
  ## Deterministic fp16 inputs for one fused-linear case at the 0.125
  ## scale: x (M, K), suh (K), svh (N). The kernel and the reference
  ## consume the same buffers.
  const scaleF = 0.125'f32
  result.xBits = newSeq[uint16](M * K)
  for r in 0 ..< M:
    for c in 0 ..< K:
      result.xBits[r * K + c] = fp32ToFp16(exl3Val(r, c, 3, scaleF))
  result.suh = newSeq[uint16](K)
  for c in 0 ..< K:
    result.suh[c] = fp32ToFp16(exl3Val(0, c, 5, scaleF))
  result.svh = newSeq[uint16](N)
  for c in 0 ..< N:
    result.svh[c] = fp32ToFp16(exl3Val(0, c, 7, scaleF))

func decodeWord*(word: uint32; cb: int): uint16 =
  ## The codebook decode the kernels instantiate (exllamav3 codebook.cuh):
  ## cb0/cb1 the LCG or MCG plus the LOP3 mask and the single-rounding
  ## fp16 add of the halves, cb2 the byte-sum scale/bias in the kernel's
  ## two-rounding numeric form (0x1EEF then 0xC932).
  var x =
    case cb
    of 0: word * 89226354'u32 + 64248484'u32
    of 1: word * 0xCBAC1FED'u32
    else: word * 0x83DCD12D'u32
  if cb <= 1:
    x = (x and 0x8fff8fff'u32) xor 0x3b603b60'u32
    return fp32ToFp16(fp16ToFp32(uint16(x and 0xFFFF)) +
                      fp16ToFp32(uint16((x shr 16) and 0xFFFF)))
  let sum = (x and 0xFF'u32) + ((x shr 8) and 0xFF'u32) +
            ((x shr 16) and 0xFF'u32) + ((x shr 24) and 0xFF'u32) +
            0x6400'u32
  let t1 = fp32ToFp16(fp16ToFp32(uint16(sum)) * fp16ToFp32(0x1EEF'u16))
  fp32ToFp16(fp16ToFp32(t1) + fp16ToFp32(0xC932'u16))

func buildDecodeTable*(cb: int): array[65536, uint16] =
  ## One decode per possible 16-bit trellis word: the per-tile decode
  ## becomes a table gather.
  for w in 0 ..< 65536:
    result[w] = decodeWord(uint32(w), cb)

func funnelWord(packed: seq[uint32]; pw: int; t, bits: int): uint32 =
  ## The 16-bit funnel window holding trellis word t of a 256-word tile,
  ## the exllamav3 shift from the packed uint32 words (little-endian
  ## int16 pairs): the window starts at bit b0 = t·bits + bits − 16 + 256·bits.
  let b0 = t * bits + bits - 16 + 256 * bits
  let b1 = b0 + 16
  let i0 = (b0 div 32) mod pw
  let i1 = ((b1 - 1) div 32) mod pw
  let s0 = (((b1 - 1) div 32) + 1) * 32 - b1
  let merged = (uint64(packed[i0]) shl 32) or uint64(packed[i1])
  result = uint32((merged shr s0) and 0xFFFF)

func tileShufflePerm(): array[256, int] =
  ## The word-index permutation of a decoded 16×16 tile in exllamav3
  ## reconstruct.cu (tensor-core fragment layout to row-major). Row 0
  ## anchors the closed form.
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

proc dequantWeights*(trellis: seq[int16]; tilesK, tilesN, bits, cb: int): seq[uint16] =
  ## Rebuilds the fp16 weight matrix [tilesK·16, tilesN·16] row-major:
  ## per tile, pack the int16 words into uint32, extract the 256 windows
  ## via the funnel shift, gather the decode table, shuffle.
  let table = buildDecodeTable(cb)
  let tileWords = 256 * bits div 16
  let perm = tileShufflePerm()
  let outF = tilesN * 16
  result = newSeq[uint16](tilesK * 16 * outF)
  for tk in 0 ..< tilesK:
    for tn in 0 ..< tilesN:
      let base = (tk * tilesN + tn) * tileWords
      var packed = newSeq[uint32](tileWords div 2)
      for i in 0 ..< tileWords div 2:
        packed[i] = uint32(cast[uint16](trellis[base + 2 * i])) or
                    (uint32(cast[uint16](trellis[base + 2 * i + 1])) shl 16)
      var dec = newSeq[uint16](256)
      for t in 0 ..< 256:
        dec[t] = table[funnelWord(packed, tileWords div 2, t, bits)]
      for i in 0 ..< 256:
        let R = i div 16
        let C = i mod 16
        result[(tk * 16 + R) * outF + tn * 16 + C] = dec[perm[i]]

proc fp16sToF32*(hs: seq[uint16]): seq[float32] =
  ## Widens an fp16 bit-pattern buffer to fp32 values.
  result = newSeq[float32](hs.len)
  for i in 0 ..< hs.len:
    result[i] = fp16ToFp32(hs[i])

func hadamardScaledSeq(): array[128 * 128, float32] =
  ## Sylvester Hadamard-128 scaled by 1/sqrt(128) (0.088388347648):
  ## H[i][j] = ±0.088388347648 with the countSetBits(i and j) parity.
  for i in 0 ..< 128:
    for j in 0 ..< 128:
      result[i * 128 + j] =
        if (countSetBits(i and j) and 1) == 0: 0.088388347648'f32
        else: -0.088388347648'f32

proc fwhtBlocks*(x: F.Tensor, K: int): F.Tensor =
  ## FWHT-128 over each 128-block of the last dim of the [M, K] fp32
  ## tensor x via the scaled Hadamard matmul (H @ H = 128·I, scale
  ## 1/sqrt(128)). Returns [M, K] fp32.
  const hs = hadamardScaledSeq()
  let M = x.size(0)
  let hsT = toTensor(hs).reshape(128, 128)
  result = F.matmul(x.reshape(M, K div 128, 128), hsT).reshape(M, K)

proc worstAbsDiff*(a, b: F.Tensor): float32 =
  ## Largest |a[i] - b[i]| over all elements.
  let ap = a.contiguous().data_ptr(float32)
  let bp = b.contiguous().data_ptr(float32)
  result = 0.0'f32
  for i in 0 ..< a.numel():
    let d = abs(ap[i] - bp[i])
    if d > result: result = d
