## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Fused EXL3 linear forward, shared core, Tile API
#
# ############################################################

## Fused EXL3 linear forward on the ceramic Tile API, the one kernel body
## behind the three entry procs of the linear_exl3 module family.
##
## Contract:
##
##     out = FWHT-128( svh ⊙ ( FWHT-128( suh ⊙ x ) @ W_dequant ) )
##
## - linear_exl3, linear_exl3_prefill and linear_exl3_decode_single forward
##   into this body
## - Out is the (M, N) fp16 output, x the (M, K) fp16 input, trellis holds
##   the (tiles_k, tiles_n, 256·bits div 16) packed int16 code stream, suh
##   (K) and svh (N) are the fp16 input and output scales
##
## - the weight matrix is not stored, each 16×32 fp16 weight tile is
##   reconstructed on the fly by `dequantTrellis` (quant_exl3_ops)
## - D is the static FWHT block (128), bits and cb are the static
##   instantiation family (bits 1..8 × cb 0..2), cb0 is the production
##   default codebook instance
## - m1FastPath selects the compile-time x row limit 1, the m = 1 fast
##   path where rows 1..31 fold to statically-zero fragments, otherwise
##   the x row limit is the runtime M, the store guard is M in both cases
##
## - rows ≥ the x row limit zero-fill on load, rows ≥ M are never stored,
##   K and N must be 128-multiples, partial shapes stay out of contract
##
## Dataflow per 128-column K-block:
##
##     x --> suh ⊙ --> FWHT-128 --> mma_AB --> fp32 accum --> quantizeF16
##     trellis --> dequantTrellis --> mma_AB --> fp32 accum --> quantizeF16
##     quantizeF16 --> FWHT-128 --> svh ⊙ --> Out fp16 store
##
## Known gaps:
## - the cb2 decode is the two-rounding numeric form, a few fp16 ulps off
##   the single-rounding reference decode, the quant_exl3_ops module doc
##   carries the reference decode detail
## - no fp32 path

import workspace/crucible
import workspace/ceramic
import ./quant_exl3_ops
import ./tile_io_rows

const
  fwhtBlock = 128       # the static D, the FWHT block width
  kTiles = 8            # the 16-wide k tiles walked per 128-column block
  tileRows = 32         # the 32-row output tile
  weightTileRows = 16   # the dequant weight tile's 16 output columns
  accumBlocks = 4       # the 32-column weight blocks per output tile

# ═════════════════════════════════════════════════════════════════════
#  The kernel
#  ═════════════════════════════════════════════════════════════════════

proc exl3_fwd_core*(
    Out: ptr UncheckedArray[float16],
    x: ptr UncheckedArray[float16],
    trellis: ptr UncheckedArray[int16],
    suh: ptr UncheckedArray[float16],
    svh: ptr UncheckedArray[float16],
    M, K, N: int32,
    bits: static int,
    cb: static int = 0,
    m1FastPath: static bool = false,
    D: static int) {.device.} =
  ## Computes the module doc's contract for one 32-row × 128-column output
  ## tile per threadgroup, M the runtime row count.
  ##
  ## Expected input:
  ##   - x rows [tgy*32, tgy*32 + rowLimit) over the K columns, suh and svh
  ##     the stride-0-row column-broadcast scale views, the rmsnorm γ pattern
  ##   - trellis k-blocks walked 8 per 128-column block, the output tile over
  ##     the weight columns [tgx*128, tgx*128 + 128)
  ##   - D the static FWHT block (128), bits and cb the static instantiation
  ##     family (bits 1..8 × cb 0..2), m1FastPath the compile-time x row
  ##     limit 1 over the runtime M
  ##
  ## per threadgroup, suh ⊙ x → FWHT-128 → mma_AB over dequantTrellis → fp32
  ## accum → quantizeF16 → FWHT-128 → svh ⊙ → fp16 store
  ##
  ## Output:
  ##   - Out rows [tgy*32, tgy*32 + M) over the tile columns, the store
  ##     guard M, rows ≥ M skipped
  static: doAssert D == fwhtBlock
  static: doAssert bits in {1, 2, 3, 4, 5, 6, 7, 8},
    "the dequantTrellis funnel Layout is instantiated for bits 1..8"
  static: doAssert cb in {0, 1, 2},
    "the dequantTrellis codebook is instantiated for cb 0..2"
  let tgx = int32(threadgroup_position_in_grid.x)
  let tgy = int32(threadgroup_position_in_grid.y)
  let tilesN = N div weightTileRows

  # x/Out carry the natural row strides. suh/svh are stride-0-row
  # column-broadcast views (the rmsnorm γ pattern)
  let gdX = x.gd(shape = (-1, -1, -1, -1), stride = (tileRows * K, 0, K, 1))
  let gdSuh = suh.gd(shape = (-1, -1, -1, -1), stride = (0, 0, 0, 1))
  let gdSvh = svh.gd(shape = (-1, -1, -1, -1), stride = (0, 0, 0, 1))
  let gdOut = Out.gd(shape = (-1, -1, -1, -1), stride = (tileRows * N, 0, N, 1))

  # the output accumulators, array-resident and zeroed before the K loop,
  # declared on the fp16 atom so the mma's three operands share one
  # fragment layout across the whole accumulator
  var d: array[accumBlocks, rt_l(float32, tileRows, tileRows, getTileConfig(float32, float16))]
  for i in 0 ..< accumBlocks:
    d[i].zero()

  # the FWHT'd x rows, 8 fp16 k-block tiles (one 128-block's worth)
  var aStore: array[kTiles, rt_l(float16, tileRows, weightTileRows)]
  var a_reg: rt_l(float16, tileRows, weightTileRows)
  var suhReg: rt_l(float16, tileRows, weightTileRows)
  var b_reg: rt_r(float16, weightTileRows, tileRows)

  # the x row guard, the m=1 fast path's limit is the compile-time 1
  # (rows 1..31 are statically zero), otherwise the runtime M
  when m1FastPath:
    let rowLimit = 1'i32
  else:
    let rowLimit = M

  # Input pass, per 128-block, predicated load + suh pre-scale,
  # tile-level FWHT-128 (1/sqrt(128) norm + fp16 round inside the op)
  for blk in 0'i32 ..< K div fwhtBlock:
    for kk in 0'i32 ..< kTiles:
      a_reg.loadTileRows(gdX, (0, 0, tgy, blk * kTiles + kk), rowLimit)
      suhReg.loadTile(gdSuh, (0, 0, 0, blk * kTiles + kk))
      a_reg.mulF16(a_reg, suhReg)
      aStore[kk] = a_reg
    aStore.hadamard128()

    # the GEMM over the block's 8 k-blocks
    for kk in 0'i32 ..< kTiles:
      a_reg = aStore[kk]
      for nt in 0'i32 ..< accumBlocks:
        b_reg.dequantTrellis(trellis, blk * kTiles + kk, tilesN, tgx, nt, bits, cb)
        d[nt].mma_AB(a_reg, b_reg)

  # Output pass, quantize the accumulator to fp16 first, tile-level
  # FWHT-128, svh post-scale, predicated store
  var y: array[accumBlocks, rt_l(float16, tileRows, tileRows)]
  for nt in 0'i32 ..< accumBlocks:
    y[nt].quantizeF16(d[nt])
  y.hadamard128()
  var svhReg: rt_l(float16, tileRows, tileRows)
  for nt in 0'i32 ..< accumBlocks:
    svhReg.loadTile(gdSvh, (0, 0, 0, tgx * accumBlocks + nt))
    y[nt].mulF16(y[nt], svhReg)
    gdOut.storeTileRows(y[nt], (0, 0, tgy, tgx * accumBlocks + nt), M)
