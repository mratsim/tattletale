# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/libtorch as F

# ─── Decode constants ──────────────────────────────────────────────

type
  Exl3DecodeConstants = object
    ## Precomputed index/shift tables for a given bitrate K.
    K: int
    tile_size: int            # always 256
    packed_words: int         # 256*K//32
    i0: F.Tensor              # [256] int64 — uint32 word indices
    i1: F.Tensor              # [256] int64
    s0: F.Tensor              # [256] int64 — shift amounts


proc compute_decode_constants(K: int): Exl3DecodeConstants =
  ## Precompute word indices and shift amounts for bitrate K.
  ## Uses scalar Nim arithmetic (not tensor division) to avoid
  ## libtorch floor_divide issues.
  const tile_size = 256
  let pw = tile_size * K div 32

  var i0_arr{.noInit.}: array[tile_size, int64]
  var i1_arr{.noInit.}: array[tile_size, int64]
  var s0_arr{.noInit.}: array[tile_size, int64]

  for t in 0 ..< tile_size:
    let b0 = t * K + K - 16 + tile_size * K
    let b1 = b0 + 16
    let ii0 = b0 div 32
    let ii1 = (b1 - 1) div 32
    i0_arr[t] = (ii0 mod pw).int64
    i1_arr[t] = (ii1 mod pw).int64
    s0_arr[t] = ((ii1 + 1) * 32 - b1).int64

  result = Exl3DecodeConstants(
    K: K,
    tile_size: tile_size,
    packed_words: pw,
    i0: F.toTensor(i0_arr),
    i1: F.toTensor(i1_arr),
    s0: F.toTensor(s0_arr)
  )


# ─── Funnel shift ─────────────────────────────────────────────────

proc funnel_shift_batch(b, a, shift: F.Tensor): F.Tensor =
  let a64 = a.to(kInt64) and F.toTensor([0xFFFFFFFF'i64])  # mask to avoid sign-ext
  let b64 = b.to(kInt64) and F.toTensor([0xFFFFFFFF'i64])  # mask to avoid sign-ext
  let shifted = a64 shl 32
  let merged = shifted or b64
  let shifted_out = merged shr shift
  let mask16 = F.toTensor([0xFFFF'i64])
  result = (shifted_out and mask16).to(kInt16)

# ─── Codebook decode ───────────────────────────────────────────────

proc decode_codebook(words: F.Tensor, cb: int): F.Tensor =
  ## Decode uint16 words to float16 values.
  let device = words.deviceType()
  var x = words.to(kInt64) and F.toTensor([0xFFFF'i64]).to(device)  # mask to avoid sign-ext

  if cb == 0:
    x = x * 89226354 + 64248484
  elif cb == 1:
    x = x * 0xCBAC1FED'i64
  elif cb == 2:
    x = x * 0x83DCD12D'i64
    let mask8 = F.toTensor([0xFF'i64]).to(device)
    let b0 = x and mask8
    let b1 = (x shr 8  and mask8)
    let b2 = (x shr 16 and mask8)
    let b3 = (x shr 24 and mask8)
    x = b0 + b1 + b2 + b3 + F.toTensor([0x6400'i64]).to(device)
    return x.to(kFloat16) * 0.00677'f32 + (-10.39'f32)
  # Mask to 32 bits (simulate uint32 overflow)
  let mask32 = F.toTensor([0xFFFFFFFF'i64]).to(device)
  x = x and mask32

  # LOP3 with truth table 0x6a, SASS index reversal (inline asm):
  # PTX spec says 0x6a = a^(b&c), but inline asm passes LUT to SASS
  # which reverses the index, so 0x6a = c^(a&b) = (x & m1) ^ m2
  let m1 = F.toTensor([0x8fff8fff'i64]).to(device)
  let m2 = F.toTensor([0x3b603b60'i64]).to(device)
  x = (x and m1) xor m2

  # Reinterpret lower/upper 16 bits as float16 and sum
  let mask16 = F.toTensor([0xFFFF'i64]).to(device)
  let lo = (x and mask16).to(kInt16)
  let hi = ((x shr 16) and mask16).to(kInt16)
  result = lo.view(kFloat16) + hi.view(kFloat16)

# ─── Tile shuffle (CUDA tensor-core → row-major) ──────────────────
## Inverse permutation from CUDA kernel: maps row-major output positions
## back to tensor-core input positions. 256-element index table.
const tileShuffle: array[256, int64] = [
     0,  32,  64,  96, 128, 160, 192, 224,   4,  36,  68, 100, 132, 164, 196, 228,
     1,  33,  65,  97, 129, 161, 193, 225,   5,  37,  69, 101, 133, 165, 197, 229,
     8,  40,  72, 104, 136, 168, 200, 232,  12,  44,  76, 108, 140, 172, 204, 236,
     9,  41,  73, 105, 137, 169, 201, 233,  13,  45,  77, 109, 141, 173, 205, 237,
    16,  48,  80, 112, 144, 176, 208, 240,  20,  52,  84, 116, 148, 180, 212, 244,
    17,  49,  81, 113, 145, 177, 209, 241,  21,  53,  85, 117, 149, 181, 213, 245,
    24,  56,  88, 120, 152, 184, 216, 248,  28,  60,  92, 124, 156, 188, 220, 252,
    25,  57,  89, 121, 153, 185, 217, 249,  29,  61,  93, 125, 157, 189, 221, 253,
     2,  34,  66,  98, 130, 162, 194, 226,   6,  38,  70, 102, 134, 166, 198, 230,
     3,  35,  67,  99, 131, 163, 195, 227,   7,  39,  71, 103, 135, 167, 199, 231,
    10,  42,  74, 106, 138, 170, 202, 234,  14,  46,  78, 110, 142, 174, 206, 238,
    11,  43,  75, 107, 139, 171, 203, 235,  15,  47,  79, 111, 143, 175, 207, 239,
    18,  50,  82, 114, 146, 178, 210, 242,  22,  54,  86, 118, 150, 182, 214, 246,
    19,  51,  83, 115, 147, 179, 211, 243,  23,  55,  87, 119, 151, 183, 215, 247,
    26,  58,  90, 122, 154, 186, 218, 250,  30,  62,  94, 126, 158, 190, 222, 254,
    27,  59,  91, 123, 155, 187, 219, 251,  31,  63,  95, 127, 159, 191, 223, 255,
]

proc shuffleTile(decoded: F.Tensor, idx: F.Tensor): F.Tensor =
  ## Apply tensor-core → row-major shuffle to a [256] decoded tile.
  F.index_select(decoded, 0, idx)


# ─── Full weight reconstruction ────────────────────────────────────

proc exl3_reconstruct*(trellis: F.Tensor, K: int, cb: int,
                  in_features, out_features: int): F.Tensor =
  ## Decode packed trellis to [in_features, out_features] float16 weight matrix.
  let device = trellis.deviceType()

  let consts = compute_decode_constants(K)
  let tiles_k = trellis.size(0)
  let tiles_n = trellis.size(1)
  const tile_size = 256

  # Pair uint16 values into uint32, exactly as CUDA does:
  # packed[i] = trellis[2*i] | (trellis[2*i+1] << 16)
  let u16 = trellis.to(kUInt16)
  let packed_lo = u16[_, _, |2].to(kInt64)      # even indices [tk, tn, pw]
  let packed_hi = u16[_, _, 1.._|2].to(kInt64)  # odd indices [tk, tn, pw]
  let packed = (packed_hi shl 16) or packed_lo  # [tk, tn, pw]

  # Expand indices: [256] -> [1, 1, 256] -> [tk, tn, 256]
  let i0_exp = consts.i0.unsqueeze(0).unsqueeze(0).expand([tiles_k, tiles_n, tile_size]).to(device)
  let i1_exp = consts.i1.unsqueeze(0).unsqueeze(0).expand([tiles_k, tiles_n, tile_size]).to(device)
  let s0_exp = consts.s0.unsqueeze(0).unsqueeze(0).expand([tiles_k, tiles_n, tile_size]).to(device)

  # Batch gather a and b: [tk,tn,pw] gather [tk,tn,256] -> [tk,tn,256]
  let mask32 = F.toTensor([0xFFFFFFFF'i64]).to(device)
  let a = packed.gather(2, i0_exp.to(kInt64)) and mask32
  let b = packed.gather(2, i1_exp.to(kInt64)) and mask32

  # Funnel shift (batched): merge (a<<32|b) then shift
  let a64 = a.to(kInt64) and mask32
  let b64 = b.to(kInt64) and mask32
  let merged = (a64 shl 32) or b64
  let words64 = merged shr s0_exp.to(kInt64)
  let words = (words64 and F.toTensor([0xFFFF'i64]).to(device)).to(kInt16)  # [tk, tn, 256]

  # Codebook decode (element-wise, flatten to 1D then reshape)
  let decoded = decode_codebook(words.reshape(-1), cb).reshape(tiles_k, tiles_n, tile_size)

  # Tile shuffle (batched: index_select on last dim)
  let sIdx = F.toTensor(tileShuffle).to(device)
  let shuffled = F.index_select(decoded, 2, sIdx)  # [tk, tn, 256]

  # Reshape to weight matrix
  result = shuffled
    .view(tiles_k, tiles_n, 16, 16)
    .permute(0, 2, 1, 3)
    .reshape(in_features, out_features)
