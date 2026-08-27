## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Gated MLP forward (gated_mlp_silu): Tile API megakernel
#
# ############################################################

## Fused GatedMLP inference on the ceramic Tile API: the gate and up
## gemms, the silu·mul activation and the down gemm in one kernel,
## the GatedMLP.forward contract (`workspace/transformers/src/layers/mlp.nim`):
## `out = down_proj(silu(gate_proj(x)) · up_proj(x))`.
## Experimental: not a production kernel, known gaps below, not fixed.
##
## Per-element semantics:
##   - gate/up = the fp16 weight gemms with fp32 accumulation. The result
##     rounds to fp16 (RNE) at the convert, matching the materialized fp16
##     gate_out/up_out of the reference linear layers
##   - act = silu(gate16)·up16 with the silu_and_mul semantics (fp32
##     silu from the fp16 gate, fp16-rounded silu, one fp16-rounded
##     product), no actLimit clamp
##   - out = the fp16 act gemm with fp32 accumulation, rounded to fp16
##     at the store through the `to` chokepoint
##
## Tile geometry:
##   - one TileC×TileC output tile per threadgroup (the launcher
##     instantiates TileC = 32), grid (NOut div TileC, ceil(M/TileC)),
##     the atom's 32 lanes
##   - each threadgroup loops the NIntm div TileC intermediate tiles
##     and per tile re-derives the TileC×TileC gate/up tiles through
##     the hidden K-loop (16-col chunks), converts them to fp16, fuses
##     the silu·mul into a TileC×TileC act tile, then accumulates
##     the act·WDown product into the output tile
##   - the weights load through the transposed B-operand views of the row-major
##     (K, NIntm) / (NIntm, NOut) buffers
##   - partial M batches need no host padding: the X tile loads
##     row-bounded (rows ≥ M zero-filled, tile_io_rows) and the Out
##     store skips them
##
## Contract (not enforced, a violated shape under-covers the grid):
##   - K multiple of 16 (the gate/up K-chunk), NIntm and NOut
##     multiples of TileC (the tile width, 32)
##
## Known production gaps (documented, not fixed):
##   - each threadgroup re-derives its gate/up tiles once per NOut
##     tile (the fusion stages no intermediate to gmem). Production
##     would stage the act tile once per (batch, intermediate) block
##   - no actLimit clamp (the reference GatedMLP has none)

import workspace/crucible
import workspace/ceramic
import ./silu_and_mul
import ./tile_io_rows

proc gated_mlp_silu_fwd*(
    Out: ptr UncheckedArray[float16],          # (M, NOut): the MLP output
    X: ptr UncheckedArray[float16],            # (M, K): the input rows
    WGate, WUp: ptr UncheckedArray[float16],   # (K, NIntm): gate/up weights
    WDown: ptr UncheckedArray[float16],        # (NIntm, NOut): the down weight
    M, K, NIntm, NOut: int32,
    TileC: static int) {.device.} =
  ## One TileC×TileC output tile per threadgroup (grid x = the NOut tile, y = the M tile),
  ## the module doc's contract applied per intermediate tile:
  ##   - the gate/up gemms
  ##   - the fused silu·mul act tile
  ##   - the down gemm accumulation
  ##   - the row-bounded X load and Out store
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
