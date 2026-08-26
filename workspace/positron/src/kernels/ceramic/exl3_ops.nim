## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     EXL3 device-op layer: FWHT-128 + trellis dequant + row-bounded IO
#
# ############################################################

## Shared device-op layer of the EXL3 kernel family on the ceramic Tile API.
## Provides the FWHT-128 op `hadamard128` and the trellis dequant
## weight-tile op `dequantTrellis`. Row-bounded fp16 tile load/store
## (`loadTileRows`/`storeTileRows`) come from `tile_io_rows`.
##
## Fragment layout convention. All ops follow the loadTile mapping:
##   - lane `l` owns the 8×8 fragment cell
##     `cell = crd2idx(A.getLayoutA(), (l, 0))`, `row = cell mod 8`,
##     `col = cell div 8`
##   - element (r, c) sits in `frags[n][m].frag[v]` at
##     `r = row + n·8`, `c = col + m·8 + v`
##   - the atom's two per-lane values are the horizontal pair
##     (row, col) and (row, col+1)
##   - the span `col + v` = 0..7 is the trellis word-index range
##
## FWHT-128 (both forms): the 7-stage butterfly over each lane's 32
## register slots of a tile row, in fp32, scaled by 1/sqrt(128)
## (0.088388347648) and rounded to fp16 on the scatter.
## The stages, lane exchanges and slot maps are documented on `butterflyCore` and the two `hadamard128` overloads.
##
## Dequant dataflow (`dequantTrellis`):
##
##     packed int16 words
##       ─► funnel window (i0, i1, s0) on the word index
##       ─► 16-bit window w
##       ─► codebook decode (cb 0/1/2)
##       ─► fp16 halves (lo, hi) ── fp16 add ──► 16×32 fp16 W tile
##
## The window extraction is a compile-time (i0, i1, s0) table over the runtime word index.
## The word index is the closed-form tensor-core-shuffle placement (the fragment-to-row-major permutation).
## The codebook is procedural at static `cb`.
## The word-index formula and the decode arithmetic are documented on the op.
##
## Known gaps:
## - cb2 uses a two-rounding numeric form (fp16(sum·k_inv) then fp16 add of k_bias, RNE constants 0x1EEF/0xC932),
##   a few fp16 ulps from the single-rounding reference half fma (constants 0x1EEE/0xC931).
##   cb0/cb1 decode with the single-rounding fp16 add.
## - No fp32 path: the ops are fp16 in, fp16 out.
## - Row-bounded IO guards rows only. A partial last column block is out of contract.

import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

# ═════════════════════════════════════════════════════════════════════
#  The fragment-layout FWHT-128
#  ═════════════════════════════════════════════════════════════════════

proc butterflyCore(rv: var array[32, float32]; lane: uint32) {.device.} =
  ## The 7 FWHT-128 stages over one lane's 32 register slots of a tile
  ## row, in place. Stage 1 pairs the (2s, 2s+1) slots (the lane's two
  ## vpt values, adjacent columns). Stages 2/4 exchange every slot
  ## with lanes `lane xor 1` / `lane xor 8`, the stage-bit lane
  ## negating its own value (exact fp32 sign flip) before the add.
  ## Stages 8/16/32/64 pair the lane's own slots at offsets 2/4/8/16.
  ## The stage-2/4 exchanges read the partner's pre-stage value:
  ## SIMT lockstep runs every lane's shuffle instruction before any
  ## lane's write.
  for s in 0'i32 ..< 16:
    let a = rv[2 * s]
    let b = rv[2 * s + 1]
    rv[2 * s] = a + b
    rv[2 * s + 1] = a - b
  let sgn1 = 1.0'f32 - 2.0'f32 * float32(lane and 1'u32)
  for s in 0'i32 ..< 32:
    let p = simdShuffle(rv[s], lane xor 1'u32)
    rv[s] = rv[s] * sgn1 + p
  let sgn8 = 1.0'f32 - 2.0'f32 * float32((lane and 8'u32) shr 3)
  for s in 0'i32 ..< 32:
    let p = simdShuffle(rv[s], lane xor 8'u32)
    rv[s] = rv[s] * sgn8 + p
  for p in 0'i32 ..< 16:
    let s = 4 * (p shr 1) + (p and 1)
    let a = rv[s]
    let b = rv[s + 2]
    rv[s] = a + b
    rv[s + 2] = a - b
  for p in 0'i32 ..< 16:
    let s = 8 * (p shr 2) + (p and 3)
    let a = rv[s]
    let b = rv[s + 4]
    rv[s] = a + b
    rv[s + 4] = a - b
  for p in 0'i32 ..< 16:
    let s = 16 * (p shr 3) + (p and 7)
    let a = rv[s]
    let b = rv[s + 8]
    rv[s] = a + b
    rv[s + 8] = a - b
  for p in 0'i32 ..< 16:
    let a = rv[p]
    let b = rv[p + 16]
    rv[p] = a + b
    rv[p + 16] = a - b

proc hadamard128*[A: static MmaAtom](
    tiles: var array[8, RtLeft[float16, 32, 16, A]]) {.device.} =
  ## In-place FWHT-128 over the 8 16-column k-block tiles of one 128-column block
  ## (the fused linear kernel's input pass), per the module doc's slot map:
  ## slot s = (2·kk + m)·2 + v of tile-row 8·n + row holds element
  ## (8·n + row, 16·kk + 8·m + col + v) at `tiles[kk].frags[n][m].frag[v]`.
  ## The fp16 values are promoted to fp32 (exact), transformed, scaled by 1/sqrt(128), rounded back to fp16.
  const M = A.getM()
  const rowTiles = 32 div M
  let lane = uint32(thread_index_in_threadgroup)
  for n in 0'i32 ..< rowTiles:
    var rv: array[32, float32]
    for s in 0'i32 ..< 32:
      rv[s] = tiles[s shr 2].frags[n][(s shr 1) and 1].frag[s and 1].to(float32)
    butterflyCore(rv, lane)
    for s in 0'i32 ..< 32:
      tiles[s shr 2].frags[n][(s shr 1) and 1].frag[s and 1] =
        (rv[s] * 0.088388347648'f32).to(float16)

proc hadamard128*[A: static MmaAtom](
    tiles: var array[4, RtLeft[float16, 32, 32, A]]) {.device.} =
  ## In-place FWHT-128 over the 4 32-column tiles (the fused linear kernel's output pass),
  ## per the module doc's slot map: slot s = (4·t + m)·2 + v of tile-row 8·n + row holds element
  ## (8·n + row, 32·t + 8·m + col + v) at `tiles[t].frags[n][m].frag[v]`.
  ## Same butterfly, scale and rounding as the 8-tile form. The caller quantizes the matmul accumulator
  ## to fp16 first, so the fp32 working registers hold the exact fp16 images.
  const M = A.getM()
  const rowTiles = 32 div M
  let lane = uint32(thread_index_in_threadgroup)
  for n in 0'i32 ..< rowTiles:
    var rv: array[32, float32]
    for s in 0'i32 ..< 32:
      rv[s] = tiles[s shr 3].frags[n][(s shr 1) and 3].frag[s and 1].to(float32)
    butterflyCore(rv, lane)
    for s in 0'i32 ..< 32:
      tiles[s shr 3].frags[n][(s shr 1) and 3].frag[s and 1] =
        (rv[s] * 0.088388347648'f32).to(float16)

# ═════════════════════════════════════════════════════════════════════
#  dequantTrellis: the on-the-fly fp16 weight-tile decode
#  ═════════════════════════════════════════════════════════════════════

func funnelEntry(bits, t: int): uint32 =
  ## Packs the funnel window of trellis word t for the given bits into one uint32:
  ## i0 | (i1 shl 6) | (s0 shl 12). The fields are the word indices of the 16-bit window's first and last halves
  ## and the shift within the high word, from the exl3_dq.cuh formulas
  ## (host-side only: the compile-time table source).
  let b0 = (t + 257) * bits - 16
  let b1 = b0 + 16
  let pw = 8 * bits
  let i0 = (b0 div 32) mod pw
  let i1u = (b1 - 1) div 32
  let i1 = i1u mod pw
  let s0 = (i1u + 1) * 32 - b1
  uint32(i0) or (uint32(i1) shl 6) or (uint32(s0) shl 12)

func funnelTable(bits: static int): array[256, uint32] =
  ## The compile-time funnel table: element word t (0..255, the tensor-core-shuffle index range)
  ## maps to its packed window (i0, i1, s0). The kernel's per-element funnel math is one table
  ## lookup, no runtime div/mod.
  for t in 0 ..< 256:
    result[t] = funnelEntry(bits, t)

proc dequantTrellis*[A: static MmaAtom](
    bReg: var RtRight[float16, 16, 32, A],
    trellis: ptr UncheckedArray[int16],
    kk, tilesN, tgx, nt: int32,
    bits: static int,
    cb: static int = 0,
    useShuffle: static bool = true) {.device.} =
  ## Decodes the 16×32 fp16 weight tile at k-block `kk` (the 16-wide K
  ## index), n-tile `nt` (the 32-wide column tile), from the packed
  ## trellis codes: 2 trellis tiles of 16 columns each, selected by `tn = 8·tgx + 2·nt + (m div 2)`
  ## with `m` the tile's 8-column block.
  ## `bReg.frags[m][n].frag[v]` receives weight
  ## W(row + 8·n, col + 8·m + v) (the RtRight convention,
  ## frags[colBlock][rowBlock]).
  ##
  ## The word index of element (k, c) with k = row + 8·n,
  ## c = col + 8·m + v is the tensor-core-shuffle closed form
  ## `t = 32·(col + v) + 8·(row div 2) + (row mod 2) + 2·n +
  ## 4·(m mod 2)` (reconstruct.cu's fragment-to-row-major permutation,
  ## the same mixed-radix word as the fm → row, fn → col, kSub → n,
  ## nSub → m mapping). `useShuffle = false` substitutes the natural
  ## row-major word `t = k·16 + (c mod 16)` (no tensor-core shuffle).
  ## The natural placement does not match the fragment order of the
  ## tensor-core decode.
  ##
  ## The decode arithmetic is exllamav3's: the 16-bit funnel window at bit b0 = t·bits + bits − 16 + 256·bits
  ## of the packed tile, the procedural codebook and the closed-form word placement. The codebook:
  ## - cb0: an LCG (linear congruential generator) plus the LOP3 (the NVIDIA three-input logic-op)
  ##   with the SASS (NVIDIA shader assembly) 0x6a truth-table semantics,
  ##   then the single-rounding fp16 add of the halves
  ## - cb1: an MCG (multiplicative congruential generator) plus the LOP3, same rounding
  ## - cb2: the mul1 byte-sum with the two-rounding numeric form
  ##   (see the module doc's gap note)
  ## `bits` is static (1..8) and `cb` is static (0..2), so the funnel
  ## table and the codebook branch fold to one instantiation.
  static: doAssert bits in 1 .. 8, "the funnel table covers bits 1..8"
  const funnel = funnelTable(bits)
  const M = A.getM()
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  for n in 0'i32 ..< 2:
    for m in 0'i32 ..< 4:
      for v in 0'i32 ..< vpt:
        let tn = 8 * tgx + 2 * nt + (m div 2)
        let t =
          when useShuffle:
            32 * (col + v) + 8 * (row div 2) + (row mod 2) +
            2 * n + 4 * (m mod 2)
          else:
            (row + 8 * n) * 16 + col + 8 * (m mod 2) + v
        let packed = funnel[t]
        let i0 = int32(packed and 63'u32)
        let i1 = int32((packed shr 6) and 63'u32)
        let s0 = int32((packed shr 12) and 31'u32)
        let base = (kk * tilesN + tn) * (16 * bits)
        let p0 = uint32(uint16(trellis[base + 2 * i0])) or
                 (uint32(uint16(trellis[base + 2 * i0 + 1])) shl 16)
        let p1 = uint32(uint16(trellis[base + 2 * i1])) or
                 (uint32(uint16(trellis[base + 2 * i1 + 1])) shl 16)
        # the 16-bit window: the 64-bit merged shift, defined for every s0
        # including 0 (a (p0 shl (31 - s0)) shl 1 form is a uint32 shift-by-32 at s0 = 0, undefined in the emitted C/MSL)
        let w = uint32((uint64(p0) shl 32 or uint64(p1)) shr s0) and 0xFFFF'u32
        if cb == 0:
          var xd = w * 89226354'u32 + 64248484'u32
          xd = (xd and 0x8fff8fff'u32) xor 0x3b603b60'u32
          let lo = uint16(xd and 0xFFFF'u32)
          let hi = uint16((xd shr 16) and 0xFFFF'u32)
          bReg.frags[m][n].frag[v] =
            (lo.asFp16().to(float32) + hi.asFp16().to(float32)).to(float16)
        elif cb == 1:
          var xd = w * 0xCBAC1FED'u32
          xd = (xd and 0x8fff8fff'u32) xor 0x3b603b60'u32
          let lo = uint16(xd and 0xFFFF'u32)
          let hi = uint16((xd shr 16) and 0xFFFF'u32)
          bReg.frags[m][n].frag[v] =
            (lo.asFp16().to(float32) + hi.asFp16().to(float32)).to(float16)
        else:
          var xd = w * 0x83DCD12D'u32
          let sum = (xd and 0xFF'u32) + ((xd shr 8) and 0xFF'u32) +
                    ((xd shr 16) and 0xFF'u32) +
                    ((xd shr 24) and 0xFF'u32) + 0x6400'u32
          let sum16 = uint16(sum).asFp16()
          let kInv = 0x1EEF'u16.asFp16()
          let kBias = 0xC932'u16.asFp16()
          let t1 = (sum16.to(float32) * kInv.to(float32)).to(float16)
          bReg.frags[m][n].frag[v] =
            (t1.to(float32) + kBias.to(float32)).to(float16)
