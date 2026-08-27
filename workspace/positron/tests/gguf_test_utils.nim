## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared GGUF test support: the deterministic packed block-stream
## generators (one per scheme), the decode-table fp16 weight
## reconstruction that rebuilds the weight matrix from the packed bytes
## in file order, and the fp16 value helpers the linear and attn tests
## share. The single home of the reconstruction, imported by later
## GGUF tests instead of re-derived. The reconstruction is an
## independent reference and never imports the kernel ops.

import ../../ceramic/tests/tile_test_utils

# ═════════════════════════════════════════════════════════════════════
#  Deterministic values
#  ═════════════════════════════════════════════════════════════════════

func ggufMix(r, c, seed: int): int =
  ## Deterministic 0..31 mix of the seed and the (row, col) index. The
  ## (c div 32) and (r div 32) terms keep the pattern off the 32-col
  ## block and the 32-row tile (the exl3Val pattern).
  (seed * (r + 1) + 7 * c + 11 * seed + 13 * (c div 32) +
   17 * (r div 32)) mod 32

func ggufFp16(r, c, seed: int): uint16 =
  ## Deterministic fp16-exact value: an fp16 grid point in [-4.0, 3.75]
  ## at 0.25 steps (0.25·k is exactly representable in fp16, so
  ## fp32ToFp16 is exact).
  fp32ToFp16(0.25'f32 * float32(ggufMix(r, c, seed) - 16))

func buildX*(M, K, seed: int, scale: float32): seq[uint16] =
  ## Deterministic fp16-exact x buffer, row-major M×K: 0.25-grid points scaled by `scale`.
  for r in 0 ..< M:
    for c in 0 ..< K:
      result.add fp32ToFp16(scale * 0.25'f32 * float32(ggufMix(r, c, seed) - 16))

func fp16sToF32*(hs: seq[uint16]): seq[float32] =
  ## Widens an fp16 bit buffer to fp32 values, one fp16ToFp32 per element.
  for h in hs: result.add fp16ToFp32(h)

# ═════════════════════════════════════════════════════════════════════
#  Packed stream generators (file order: N rows × rowBytes)
#  ═════════════════════════════════════════════════════════════════════

proc genQ8_0*(K, N, seed: int): seq[uint8] =
  ## Deterministic packed Q8_0 stream: N rows × (K div 32) blocks of
  ## 34 bytes. d = fp16-exact per (row, block), q = int8 spanning the
  ## signed range.
  let blocks = K div 32
  let rowBytes = blocks * 34
  result = newSeq[uint8](N * rowBytes)
  for r in 0 ..< N:
    for b in 0 ..< blocks:
      let base = r * rowBytes + b * 34
      let d16 = ggufFp16(r, b * 32, seed)
      result[base] = uint8(d16 and 0xFF)
      result[base + 1] = uint8(d16 shr 8)
      for i in 0 ..< 32:
        let c = b * 32 + i
        let g = r * K + c
        if g mod 7 == 0:
          # every 7th element takes one of the int8 boundaries so the
          # 0x80 / 0x7F / 0x00 byte handling cannot silently drift
          let e = (g div 7) mod 3
          result[base + 2 + i] = if e == 0: 0x80'u8
                                 elif e == 1: 0x7F'u8
                                 else: 0x00'u8
        else:
          let qv = ggufMix(r, c, seed + 1) * 8 - 127
          result[base + 2 + i] = uint8(qv and 0xFF)

proc genQ4_K*(K, N, seed: int): seq[uint8] =
  ## Deterministic packed Q4_K stream: N rows × (K div 256)
  ## super-blocks of 144 bytes. d, dmin, the 12 scale bytes and the
  ## nibble bytes are per-index mixes. The scale bytes exercise both
  ## the nt < 4 and the nt ≥ 4 unpack branches.
  let sbs = K div 256
  let rowBytes = sbs * 144
  result = newSeq[uint8](N * rowBytes)
  for r in 0 ..< N:
    for sb in 0 ..< sbs:
      let base = r * rowBytes + sb * 144
      let d16 = ggufFp16(r, sb * 256, seed)
      let dm16 = ggufFp16(r, sb * 256 + 8, seed + 2)
      result[base] = uint8(d16 and 0xFF)
      result[base + 1] = uint8(d16 shr 8)
      result[base + 2] = uint8(dm16 and 0xFF)
      result[base + 3] = uint8(dm16 shr 8)
      # the 12 scale bytes: 0..3 sc low-6 of sub-blocks 0..3, 4..7 m
      # low-6 of sub-blocks 0..3, 8..11 the sc low-4 and m high-4 of
      # sub-blocks 4..7. Bits 6..7 of bytes 0..3 and 4..7 also carry
      # the cross-byte high-2 sc and m terms of the nt ≥ 4 unpack
      # (scales[nt-4] shr 6, scales[nt] shr 6). The low-3 bits of
      # bytes 8..11 vary the sc low-4 field beyond {0, 8}.
      for t in 0 ..< 4:
        let mixSc = ggufMix(r, sb * 256 + t, seed + 3)
        let mixM = ggufMix(r, sb * 256 + 16 + t, seed + 4)
        let mixSc4 = ggufMix(r, sb * 256 + 32 + t, seed + 5)
        result[base + 4 + t] = uint8(mixSc * 2 + 64 * (mixSc mod 4))
        result[base + 8 + t] = uint8(mixM * 2 + 1 + 64 * (mixM mod 4))
        result[base + 12 + t] = uint8(mixSc4 * 8 + (mixSc4 mod 8))
      # the 128 nibble bytes: byte 32·j + c holds sub-block 2j col c
      # in the low nibble and sub-block 2j+1 col c in the high nibble
      for j in 0 ..< 4:
        for c in 0 ..< 32:
          let lo = ggufMix(r, sb * 256 + j * 64 + c, seed + 6) mod 16
          let hi = ggufMix(r, sb * 256 + j * 64 + 32 + c, seed + 7) mod 16
          result[base + 16 + 32 * j + c] = uint8(lo or (hi shl 4))

proc genIQ4_XS*(K, N, seed: int): seq[uint8] =
  ## Deterministic packed IQ4_XS stream: N rows × (K div 256)
  ## super-blocks of 136 bytes. d, sc_h, the 4 scale-low bytes and the
  ## nibble bytes are per-index mixes.
  let sbs = K div 256
  let rowBytes = sbs * 136
  result = newSeq[uint8](N * rowBytes)
  for r in 0 ..< N:
    for sb in 0 ..< sbs:
      let base = r * rowBytes + sb * 136
      let d16 = ggufFp16(r, sb * 256, seed)
      result[base] = uint8(d16 and 0xFF)
      result[base + 1] = uint8(d16 shr 8)
      # sc_h: 8 two-bit fields, field j is the high-2 scale bits of
      # sub-block j
      var scH: uint16 = 0
      for j in 0 ..< 8:
        scH = scH or uint16((ggufMix(r, sb * 256 + 32 + j, seed + 7) mod 4) shl (2 * j))
      result[base + 2] = uint8(scH and 0xFF)
      result[base + 3] = uint8(scH shr 8)
      # the 4 scale-low bytes: byte t holds the low nibble of sub-block
      # 2t and the high nibble of sub-block 2t+1
      for t in 0 ..< 4:
        let loN = ggufMix(r, sb * 256 + 8 + t, seed + 5) mod 16
        let hiN = ggufMix(r, sb * 256 + 16 + t, seed + 6) mod 16
        result[base + 4 + t] = uint8(loN or (hiN shl 4))
      # the 128 nibble bytes: byte 16·j + (p mod 16) holds sub-block j
      # col p's nibble at shift 4·(p div 16)
      var qs = newSeq[uint8](128)
      for j in 0 ..< 8:
        for p in 0 ..< 32:
          let nib = ggufMix(r, sb * 256 + 64 + j * 32 + p, seed + 8) mod 16
          qs[16 * j + (p mod 16)] =
            uint8(int(qs[16 * j + (p mod 16)]) or (nib shl (4 * (p div 16))))
      for i in 0 ..< 128:
        result[base + 8 + i] = qs[i]

# ═════════════════════════════════════════════════════════════════════
#  The decode-table reconstruction (file order, one fp16 RNE per element)
#  ═════════════════════════════════════════════════════════════════════

const kvaluesIQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10,
                       1, 13, 25, 38, 53, 69, 89, 113]
  ## 16-entry IQ4_NL codebook: the fp32 multiply factor of nibble v.

func decodeWeightsQ8_0*(packed: seq[uint8], K, N: int): seq[uint16] =
  ## Rebuilds the fp16 weight matrix from a packed Q8_0 stream in file
  ## order: element (row r over N, col c over K) at index r·K + c. Per
  ## (row, block) a 256-entry table holds fp32(d)·q for every int8 q,
  ## the gather rounds once with fp32ToFp16 (RNE), the kernel's exact
  ## chain.
  let blocks = K div 32
  let rowBytes = blocks * 34
  result = newSeq[uint16](N * K)
  for r in 0 ..< N:
    for b in 0 ..< blocks:
      let base = r * rowBytes + b * 34
      let d = fp16ToFp32(uint16(packed[base]) or (uint16(packed[base + 1]) shl 8))
      var table: array[256, float32]
      for qi in 0 ..< 256:
        let qv = int32(qi)
        table[qi] = d * (float32(qv) - 256.0'f32 * float32(qv shr 7))
      for i in 0 ..< 32:
        result[r * K + b * 32 + i] = fp32ToFp16(table[int(packed[base + 2 + i])])

func decodeWeightsQ4_K*(packed: seq[uint8], K, N: int): seq[uint16] =
  ## Rebuilds the fp16 weight matrix from a packed Q4_K stream in file
  ## order: element (row r over N, col c over K) at index r·K + c. Per
  ## (row, super-block, sub-block) a 16-entry table holds
  ## t2 = (fp32(d)·sc)·nibble − fp32(dmin)·m for every nibble, the
  ## gather rounds once with fp32ToFp16 (RNE), the kernel's exact
  ## chain order.
  let sbs = K div 256
  let rowBytes = sbs * 144
  result = newSeq[uint16](N * K)
  for r in 0 ..< N:
    for sb in 0 ..< sbs:
      let base = r * rowBytes + sb * 144
      let d = fp16ToFp32(uint16(packed[base]) or (uint16(packed[base + 1]) shl 8))
      let dmin = fp16ToFp32(uint16(packed[base + 2]) or (uint16(packed[base + 3]) shl 8))
      for nt in 0 ..< 8:
        var scv: int
        var mv: int
        if nt < 4:
          scv = int(packed[base + 4 + nt]) and 63
          mv = int(packed[base + 8 + nt]) and 63
        else:
          scv = (int(packed[base + 8 + nt]) and 0xF) or
                ((int(packed[base + nt]) shr 6) shl 4)
          mv = (int(packed[base + 8 + nt]) shr 4) or
               ((int(packed[base + 4 + nt]) shr 6) shl 4)
        let dl = d * float32(scv)
        let ml = dmin * float32(mv)
        var table: array[16, float32]
        for nib in 0 ..< 16:
          let prod = dl * float32(nib)
          table[nib] = prod - ml
        for c in 0 ..< 32:
          let nib = (int(packed[base + 16 + (nt shr 1) * 32 + c]) shr
                     (4 * (nt and 1))) and 0xF
          result[r * K + sb * 256 + nt * 32 + c] = fp32ToFp16(table[nib])

func decodeWeightsIQ4_XS*(packed: seq[uint8], K, N: int): seq[uint16] =
  ## Rebuilds the fp16 weight matrix from a packed IQ4_XS stream in
  ## file order: element (row r over N, col c over K) at index r·K + c.
  ## Per (row, super-block, sub-block) a 16-entry table holds
  ## fp32(d)·scale6'·kvaluesIQ4NL[nibble], the gather rounds once with
  ## fp32ToFp16 (RNE), the kernel's exact chain order.
  let sbs = K div 256
  let rowBytes = sbs * 136
  result = newSeq[uint16](N * K)
  for r in 0 ..< N:
    for sb in 0 ..< sbs:
      let base = r * rowBytes + sb * 136
      let d = fp16ToFp32(uint16(packed[base]) or (uint16(packed[base + 1]) shl 8))
      let scH = uint16(packed[base + 2]) or (uint16(packed[base + 3]) shl 8)
      for j in 0 ..< 8:
        let low4 = (int(packed[base + 4 + (j shr 1)]) shr (4 * (j and 1))) and 0xF
        let high2 = (int(scH) shr (2 * j)) and 0x3
        let scale6v = (low4 or (high2 shl 4)) - 32
        let dl = d * float32(scale6v)
        var table: array[16, float32]
        for nib in 0 ..< 16:
          table[nib] = dl * float32(kvaluesIQ4NL[nib])
        for p in 0 ..< 32:
          let nib = (int(packed[base + 8 + 16 * j + (p mod 16)]) shr
                     (4 * (p div 16))) and 0xF
          result[r * K + sb * 256 + j * 32 + p] = fp32ToFp16(table[nib])
