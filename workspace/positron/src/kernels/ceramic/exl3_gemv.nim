## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Fused EXL3 decode-GEMV forward (exl3_gemv_fwd): Tile API port
#
# ############################################################

## Fused EXL3 decode-GEMV forward on the ceramic Tile API.
##
## Contract:
##
##     out = FWHT-128( svh ⊙ ( FWHT-128( suh ⊙ x ) @ W_dequant ) )
##
## Buffers:
##   - Out: (M, N) fp16 output
##   - x: (M, K) fp16 input
##   - trellis: (tiles_k, tiles_n, 256·bits div 16) packed int16 codes
##   - suh: (K) fp16 input scales
##   - svh: (N) fp16 output scales
##
## The weight matrix is not stored. Each 16×32 fp16 weight tile
## is reconstructed on the fly by `dequantTrellis` (exl3_ops).
## D is the static FWHT block (128).
## `bits`, `cb` and `mmode` are the static instantiation family
## (bits 1..8 × cb 0..2, MMODE 0/1). cb0 is the production default codebook.
##
## MMODE:
##   - MMODE 0: the m = 1 fast path. The x row limit is the compile-time 1.
##     Rows 1..31 fold to statically-zero fragments.
##   - MMODE 1: m ≤ 8 with the runtime M guard.
## The store guard is M in both modes.
##
## Shapes: K and N must be 128-multiples. Rows ≥ the mode's x row
## limit are zero-filled on load. Rows ≥ M are skipped on store.
##
## Dataflow per 128-column K-block:
##
##     x ──► suh ⊙ ──► FWHT-128 ────┐
##                                  ▼
##     trellis ──► dequantTrellis ──► mma_AB ──► fp32 accum
##                                                 │
##                                                 ▼
##     Out ◄── svh ⊙ ◄── FWHT-128 ◄── fp16 round ◄─┘
##
## Known gaps:
## - K and N must be 128-multiples. Partial shapes are out of contract.
## - The cb2 decode is the two-rounding numeric form, a few fp16 ulps
##   from the single-rounding reference decode (see the exl3_ops module doc).
## - No fp32 path.

import workspace/crucible
import workspace/ceramic
import ./exl3_ops
import ./tile_io_rows

# ═════════════════════════════════════════════════════════════════════
#  Local device extensions: the per-element arithmetic the fusion needs
#  ═════════════════════════════════════════════════════════════════════

proc mulF16[R, C: static int; A: static MmaAtom](
    dst: var RtLeft[float16, R, C, A],
    a, b: RtLeft[float16, R, C, A]) {.device.} =
  ## Per-element fp16 multiply with one rounding (the `mul` op
  ## semantics): the fp32 product of the promoted operands, one RNE
  ## fp16 round. The frag walk covers exactly the elements the loads
  ## produced (same lane→element mapping as `loadTileRows`), so a, b
  ## and dst agree elementwise. Aliasing dst with a or b is safe: the element reads complete before the element write.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst.frags[n][m].frag[v] =
          (a.frags[n][m].frag[v].to(float32) *
           b.frags[n][m].frag[v].to(float32)).to(float16)

proc quantizeF16[R, C: static int; A: static MmaAtom](
    dst: var RtLeft[float16, R, C, A],
    src: RtLeft[float32, R, C, A]) {.device.} =
  ## Per-element fp32 → fp16 quantization (RNE) over one register
  ## tile: the accumulator's one fp16 round before the output FWHT.
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst.frags[n][m].frag[v] = src.frags[n][m].frag[v].to(float16)

# ═════════════════════════════════════════════════════════════════════
#  The kernel
#  ═════════════════════════════════════════════════════════════════════

proc exl3_gemv_fwd*(
    Out: ptr UncheckedArray[float16],    # (M, N) fp16 output
    x: ptr UncheckedArray[float16],      # (M, K) fp16 input
    trellis: ptr UncheckedArray[int16],  # (tiles_k, tiles_n, 256*bits div 16) packed
    suh: ptr UncheckedArray[float16],    # (K) fp16 input scale
    svh: ptr UncheckedArray[float16],    # (N) fp16 output scale
    M, K, N: int32,
    bits: static int,
    cb: static int,
    mmode: static int,
    D: static int) {.device.} =
  ## Computes the module doc's contract for one 32-row × 128-column
  ## output tile. D is the static FWHT block (128).
  ## `bits`, `cb` and `mmode` are the static instantiation family:
  ##   - bits 1..8 × cb 0..2
  ##   - MMODE 0: the m = 1 fast path, compile-time x row limit 1
  ##   - MMODE 1: m ≤ 8, runtime M x row limit
  ## M is the runtime row count.
  static: doAssert D == 128
  static: doAssert bits in {1, 2, 3, 4, 5, 6, 7, 8},
    "the dequantTrellis funnel Layout is instantiated for bits 1..8"
  static: doAssert cb in {0, 1, 2},
    "the dequantTrellis codebook is instantiated for cb 0..2"
  static: doAssert mmode in {0, 1},
    "MMODE 0 = the m=1 fast path, MMODE 1 = m <= 8"
  let tgx = int32(threadgroup_position_in_grid.x)
  let tgy = int32(threadgroup_position_in_grid.y)
  let tilesN = N div 16

  # x/Out carry the natural row strides. suh/svh are stride-0-row
  # column-broadcast views (the rmsnorm γ pattern)
  let glX = gd(x, shape = (-1, -1, -1, -1), stride = (32 * K, 0, K, 1))
  let glSuh = gd(suh, shape = (-1, -1, -1, -1), stride = (0, 0, 0, 1))
  let glSvh = gd(svh, shape = (-1, -1, -1, -1), stride = (0, 0, 0, 1))
  let glOut = gd(Out, shape = (-1, -1, -1, -1), stride = (32 * N, 0, N, 1))

  # the output accumulators, array-resident and zeroed before the K loop
  # (declared on the fp16 atom so the mma's three operands share one
  # fragment layout)
  var d: array[4, rt_l(float32, 32, 32, getTileConfig(float32, float16))]
  for i in 0 ..< 4:
    d[i].zero()

  # the FWHT'd x rows: 8 fp16 k-block tiles (one 128-block's worth)
  var aStore: array[8, rt_l(float16, 32, 16)]
  var a_reg: rt_l(float16, 32, 16)
  var suhReg: rt_l(float16, 32, 16)
  var b_reg: rt_r(float16, 16, 32)

  # the x row guard: MMODE 0's limit is the compile-time 1 (the m=1 fast path:
  # rows 1..31 are statically zero). MMODE 1's is the runtime M
  # (m ≤ 8 by contract, and the 32-row tile holds up to 32)
  when mmode == 0:
    let rowLimit = 1'i32
  else:
    let rowLimit = M

  # ── input pass: per 128-block, predicated load + suh pre-scale,
  #    tile-level FWHT-128 (1/sqrt(128) norm + fp16 round inside the op)
  for blk in 0'i32 ..< K div 128:
    for kk in 0'i32 ..< 8:
      loadTileRows(a_reg, glX, (0, 0, tgy, blk * 8 + kk), rowLimit)
      loadTile(suhReg, glSuh, (0, 0, 0, blk * 8 + kk))
      mulF16(a_reg, a_reg, suhReg)
      aStore[kk] = a_reg
    hadamard128(aStore)

    # ── the GEMM over the block's 8 k-blocks ──
    for kk in 0'i32 ..< 8:
      a_reg = aStore[kk]
      for nt in 0'i32 ..< 4:
        dequantTrellis(b_reg, trellis, blk * 8 + kk, tilesN, tgx, nt, bits, cb)
        mma_AB(d[nt], a_reg, b_reg)

  # ── output pass: quantize the accumulator to fp16 first, tile-level
  #    FWHT-128, svh post-scale, predicated store ──
  var y: array[4, rt_l(float16, 32, 32)]
  for nt in 0'i32 ..< 4:
    quantizeF16(y[nt], d[nt])
  hadamard128(y)
  var svhReg: rt_l(float16, 32, 32)
  for nt in 0'i32 ..< 4:
    loadTile(svhReg, glSvh, (0, 0, 0, tgx * 4 + nt))
    mulF16(y[nt], y[nt], svhReg)
    storeTileRows(glOut, y[nt], (0, 0, tgy, tgx * 4 + nt), M)
