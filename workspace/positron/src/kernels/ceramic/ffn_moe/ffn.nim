## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Gated MLP forward (ffn), Tile API megakernel
#
# ############################################################

## Fused GatedMLP forward on the ceramic Tile API, the GatedDenseFFN.forward
## contract (`workspace/transformers/src/layers/ffn.nim`):
## `out = down_proj(silu(gate_proj(x)) · up_proj(x))`.
##
## One TileC×TileC Out tile per threadgroup (TileC = 32 at the entry),
## each threadgroup walks the NIntm div TileC intermediate tiles and per
## tile re-derives the gate/up tiles through the 16-wide hidden K-loop:
##
##   x (TileC, 16) ──┬─► [ gate = x·W_g (TileC, TileC) ] ─┐  fp32 mma,
##                   └─► [ up   = x·W_u (TileC, TileC) ] ─┴─► fp16 RNE convert
##                                                              │
##                              [ act = silu(gate16)·up16 ] ◄──┘  silu_and_mul
##                                     │                           semantics, no clamp
##   W_d (TileC, TileC) ─► [ O += act·W_d (fp32) ] ◄────────┘
##
## O = the Out tile, fp16 RNE at the store. The weights load through the
## transposed B-operand views of the row-major (K, NIntm) / (NIntm, NOut)
## buffers. Partial-M batches need no host padding: the X tile loads
## row-bounded (rows >= M zero-filled, tile_io_rows) and the Out store
## skips them.
##
## Contract (not enforced, a violated shape under-covers the grid):
##   - K a multiple of 16 (the gate/up K-chunk), NIntm and NOut
##     multiples of TileC (the tile width, 32)
##
## Known production gaps (documented, not fixed):
##   - each threadgroup re-derives its gate/up tiles once per NOut tile,
##     the fusion stages no intermediate to gmem
##   - no actLimit clamp (the reference GatedMLP has none)

import workspace/crucible
import workspace/ceramic
import ./ffn_silu
import ../tile_io_rows

proc gated_mlp_silu_fwd*(
    Out: ptr UncheckedArray[float16],          # (M, NOut): the MLP output
    X: ptr UncheckedArray[float16],            # (M, K): the input rows
    WGate, WUp: ptr UncheckedArray[float16],   # (K, NIntm): gate/up weights
    WDown: ptr UncheckedArray[float16],        # (NIntm, NOut): the down weight
    M, K, NIntm, NOut: int32,
    TileC: static int) {.device.} =
  ## One TileC×TileC Out tile per threadgroup, grid (NOut div TileC,
  ## ceil(M div TileC)), x = the Out col tile, y = the M row tile.
  ## Per intermediate block ic, the shared x tiles feed both GEMMs:
  ##
  ##   x (TileC, 16) ──┬─► [ gate = x·W_g (TileC, TileC) ] ─┐
  ##                   └─► [ up   = x·W_u (TileC, TileC) ] ─┴─► [ fp16 RNE ]
  ##                                                              │
  ##                              [ act = silu(gate16)·up16 ] ◄──┘
  ##                                     │
  ##   W_d (TileC, TileC) ─► [ O += act·W_d (fp32) ] ◄────────┘
  ##
  ## O = the Out tile, fp16 at the store. Rows >= M load
  ## zero-filled, the store skips them. TileC must divide NIntm
  ## and NOut.
  let tx = int32(threadgroup_position_in_grid.x)
  let ty = int32(threadgroup_position_in_grid.y)
  let gdX = X.gd(shape = (-1, -1, -1, -1), stride = (K, 0, K, 1))
  let gdWg = WGate.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1, NIntm))
  let gdWu = WUp.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1, NIntm))
  let gdWd = WDown.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1, NOut))
  let gdOut = Out.gd(shape = (-1, -1, -1, -1), stride = (NOut, 0, NOut, 1))
  var d_rtl: rt_l(float32, TileC, TileC, getTileConfig(float32, float16))
  d_rtl.zero()
  for ic in 0'i32 ..< NIntm div int32(TileC):
    var gate: rt_l(float32, TileC, TileC, getTileConfig(float32, float16))
    var up: rt_l(float32, TileC, TileC, getTileConfig(float32, float16))
    gate.zero()
    up.zero()
    for k in 0'i32 ..< K div 16:
      var x_t: rt_l(float16, TileC, 16)
      x_t.loadTileRows(gdX, (0, 0, ty, k), M)
      var wg_t: rt_r(float16, 16, TileC)
      wg_t.loadTile(gdWg, (0, 0, ic, k))
      var wu_t: rt_r(float16, 16, TileC)
      wu_t.loadTile(gdWu, (0, 0, ic, k))
      gate.mma_AB(x_t, wg_t)
      up.mma_AB(x_t, wu_t)
    var gate16: rt_l(float16, TileC, TileC)
    var up16: rt_l(float16, TileC, TileC)
    gate16.convert(gate)
    up16.convert(up)
    var act: rt_l(float16, TileC, TileC)
    siluAndMulElem(act, gate16, up16, 0.0'f32)
    var wd_t: rt_r(float16, TileC, TileC)
    wd_t.loadTile(gdWd, (0, 0, tx, ic))
    d_rtl.mma_AB(act, wd_t)
  gdOut.storeTileRows(d_rtl, (0, 0, ty, tx), M)
