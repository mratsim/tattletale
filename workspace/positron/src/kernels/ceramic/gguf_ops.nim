## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     GGUF dequant tile ops: Q8_0 / Q4_K / IQ4_XS
#
# ############################################################

## GGUF dequant tile ops: the Q8_0, Q4_K and IQ4_XS fp16 weight-tile
## decoders for the quantized linear forward. Each op decodes one 16×32
## mma-B b-tile from the packed GGUF byte stream: tile element (k, c) =
## file element (row = (tgx·4 + nt)·32 + c, col = kk·16 + k), file
## row-major with row = the N axis and col = the K axis. Every scheme
## runs its fp32 decode chain in the fixed op order with one fp16 RNE
## per element at the tile write. Known contract: K and N are
## 128-multiples.

import workspace/crucible
import workspace/ceramic

# ═════════════════════════════════════════════════════════════════════
#  dequantGGUF_Q8_0
#  ═════════════════════════════════════════════════════════════════════

proc dequantGGUF_Q8_0*[A: static MmaAtom](
    bReg: var RtRight[float16, 16, 32, A],
    w: ptr UncheckedArray[uint8],
    kk, rowBytes: int32,
    tgx, nt: int32) {.device.} =
  ## Decodes the 16×32 Q8_0 weight tile at K-block `kk` (file cols
  ## kk·16 .. kk·16+15), N-tile `nt` of the 128-col output block `tgx`
  ## (file rows (tgx·4 + nt)·32 .. +31), into the mma-B arrangement.
  ## The tile spans one 32-col block, constant per call: block =
  ## kk div 2, within-block col offset = 16·(kk and 1).
  ##
  ## File layout per 34-byte block: fp16 scale `d` at bytes 0..1, the
  ## 32 int8 `q` at bytes 2..33. Decode chain per element, the fixed
  ## op order: t1 = fp32(d)·q with `q` sign-extended arithmetically,
  ## one fp16 RNE at the store. `rowBytes` is the byte span of one
  ## file row.
  const M = A.getM()
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  let baseRow = (tgx * 4 + nt) * 32
  let baseCol = kk * 16
  let blockByte = (baseCol div 32) * 34
  let withinCol = baseCol mod 32
  for m in 0'i32 ..< 4:
    for v in 0'i32 ..< vpt:
      let rowByte = (baseRow + col + 8 * m + v) * rowBytes + blockByte
      let d = uint16(uint16(w[rowByte]) or (uint16(w[rowByte + 1]) shl 8)).asFp16().to(float32)
      for n in 0'i32 ..< 2:
        let k = row + 8 * n
        let qv = int32(w[rowByte + 2 + withinCol + k])
        let qF = float32(qv) - 256.0'f32 * float32(qv shr 7)
        bReg.frags[m][n].frag[v] = (d * qF).to(float16)

# ═════════════════════════════════════════════════════════════════════
#  dequantGGUF_Q4_K
#  ═════════════════════════════════════════════════════════════════════

proc dequantGGUF_Q4_K*[A: static MmaAtom](
    bReg: var RtRight[float16, 16, 32, A],
    w: ptr UncheckedArray[uint8],
    kk, rowBytes: int32,
    tgx, nt: int32) {.device.} =
  ## Decodes the 16×32 Q4_K weight tile at K-block `kk`, N-tile `nt`
  ## of `tgx`, into the mma-B arrangement. The tile spans one 32-col
  ## sub-block, constant per call: super-block = kk div 16, sub-block
  ## = (kk div 2) and 7, within-sub-block col offset = 16·(kk and 1).
  ##
  ## File layout per 144-byte super-block: fp16 `d` at bytes 0..1,
  ## fp16 `dmin` at bytes 2..3, the 12 scale bytes at bytes 4..15, the
  ## 128 nibble bytes at bytes 16..143. For sub-block nt (0..7), the
  ## 6-bit scale and min unpack (llama.cpp get_scale_min_k4):
  ##
  ## - nt < 4:  sc = scales[nt] & 0x3F, m = scales[nt+4] & 0x3F
  ## - nt ≥ 4:  sc = (scales[nt+4] & 0xF) | ((scales[nt-4] shr 6) shl 4)
  ##            m  = (scales[nt+4] shr 4) | ((scales[nt] shr 6) shl 4)
  ##
  ## Nibble at within-sub-block col c: (qs[32·(nt shr 1) + c] shr
  ## (4·(nt and 1))) and 0xF. Decode chain per element: t1 = fp32(d)·sc,
  ## t2 = t1·nibble, t3 = t2 − fp32(dmin)·m, one fp16 RNE at the store.
  const M = A.getM()
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  let baseRow = (tgx * 4 + nt) * 32
  let baseCol = kk * 16
  let blockByte = (baseCol div 256) * 144
  let sb = (baseCol div 32) and 7
  let withinCol = baseCol mod 32
  for m in 0'i32 ..< 4:
    for v in 0'i32 ..< vpt:
      let rowByte = (baseRow + col + 8 * m + v) * rowBytes + blockByte
      let d = uint16(uint16(w[rowByte]) or (uint16(w[rowByte + 1]) shl 8)).asFp16().to(float32)
      let dmin = uint16(uint16(w[rowByte + 2]) or (uint16(w[rowByte + 3]) shl 8)).asFp16().to(float32)
      var scv: int32
      var mv: int32
      if sb < 4:
        scv = int32(w[rowByte + 4 + sb]) and 63
        mv = int32(w[rowByte + 8 + sb]) and 63
      else:
        scv = (int32(w[rowByte + 8 + sb]) and 0xF) or
              ((int32(w[rowByte + sb]) shr 6) shl 4)
        mv = (int32(w[rowByte + 8 + sb]) shr 4) or
             ((int32(w[rowByte + 4 + sb]) shr 6) shl 4)
      let dl = d * float32(scv)
      let ml = dmin * float32(mv)
      for n in 0'i32 ..< 2:
        let k = row + 8 * n
        let nib = (int32(w[rowByte + 16 + (sb shr 1) * 32 + withinCol + k]) shr
                   (4 * (sb and 1))) and 0xF
        let prod = dl * float32(nib)
        let wv = prod - ml
        bReg.frags[m][n].frag[v] = wv.to(float16)

# ═════════════════════════════════════════════════════════════════════
#  dequantGGUF_IQ4_XS
#  ═════════════════════════════════════════════════════════════════════

proc dequantGGUF_IQ4_XS*[A: static MmaAtom](
    bReg: var RtRight[float16, 16, 32, A],
    w: ptr UncheckedArray[uint8],
    kk, rowBytes: int32,
    tgx, nt: int32) {.device.} =
  ## Decodes the 16×32 IQ4_XS weight tile at K-block `kk`, N-tile `nt`
  ## of `tgx`, into the mma-B arrangement. The tile spans one 32-col
  ## sub-block, constant per call: super-block = kk div 16, sub-block
  ## = (kk div 2) and 7, within-sub-block col offset = 16·(kk and 1).
  ##
  ## File layout per 136-byte super-block: fp16 `d` at bytes 0..1, the
  ## uint16 `sc_h` at bytes 2..3, the 4 scale-low bytes at bytes 4..7,
  ## the 128 nibble bytes at bytes 8..135. For sub-block j (0..7):
  ## scale6' = ((scales_l[j shr 1] shr (4·(j and 1))) and 0xF) or
  ## (((sc_h shr (2·j)) and 0x3) shl 4) − 32. Nibble at within-sub-block
  ## col p: (qs[16·j + (p mod 16)] shr (4·(p div 16))) and 0xF, with
  ## the 16-entry kvaluesIQ4NL codebook. Decode chain per element:
  ## t1 = fp32(d)·scale6', t2 = t1·kvaluesIQ4NL[nibble], one fp16 RNE
  ## at the store.
  const M = A.getM()
  const vpt = A.getVpt()
  const kvaluesIQ4NL = [int32(-127), -104, -83, -65, -49, -35, -22, -10,
                         1, 13, 25, 38, 53, 69, 89, 113]
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  let baseRow = (tgx * 4 + nt) * 32
  let baseCol = kk * 16
  let blockByte = (baseCol div 256) * 136
  let j = (baseCol div 32) and 7
  let withinCol = baseCol mod 32
  for m in 0'i32 ..< 4:
    for v in 0'i32 ..< vpt:
      let rowByte = (baseRow + col + 8 * m + v) * rowBytes + blockByte
      let d = uint16(uint16(w[rowByte]) or (uint16(w[rowByte + 1]) shl 8)).asFp16().to(float32)
      let scH = uint16(uint16(w[rowByte + 2]) or (uint16(w[rowByte + 3]) shl 8))
      let low4 = (int32(w[rowByte + 4 + (j shr 1)]) shr (4 * (j and 1))) and 0xF
      let high2 = (int32(scH) shr (2 * j)) and 0x3
      let scale6v = (low4 or (high2 shl 4)) - 32
      let dl = d * float32(scale6v)
      for n in 0'i32 ..< 2:
        let p = withinCol + row + 8 * n
        let nib = (int32(w[rowByte + 8 + 16 * j + (p mod 16)]) shr
                   (4 * (p div 16))) and 0xF
        let prod = dl * float32(kvaluesIQ4NL[nib])
        bReg.frags[m][n].frag[v] = prod.to(float16)
