## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     GGUF quantized linear forward (gguf_linear_fwd)
#
# ############################################################

## GGUF quantized linear forward: out = x @ W_dequant over the packed
## GGUF stream. The fp16 weight tiles dequantize in registers via the
## gguf_ops decoders into the mma-B arrangement (file row = the N axis,
## file col = the K axis), accumulate in fp32 and round once to fp16 in
## the epilogue. Static scheme dispatch: 0 = Q8_0, 1 = Q4_K, 2 = IQ4_XS.
## Grid x = N div 128, grid y = (M + 31) div 32, threadgroup = 32 lanes,
## M row-bounded. Known contract: K a multiple of 128 (Q8_0) or 256
## (Q4_K, IQ4_XS), N a multiple of 128.

import workspace/crucible
import workspace/ceramic
import ./gguf_ops
import ./exl3_ops
import ./tile_io_rows

proc gguf_linear_fwd*(
    Out: ptr UncheckedArray[float16],    # (M, N) fp16 output
    x: ptr UncheckedArray[float16],      # (M, K) fp16 input
    w: ptr UncheckedArray[uint8],        # the packed GGUF stream, N rows × rowBytes
    M, K, N, rowBytes: int32,
    scheme: static int) {.device.} =
  ## Runs the GGUF quantized linear forward for one output block: the
  ## 32×16 x-tile loads row-bounded at K-block `kk`, each 32-col n-tile
  ## dequantizes its 16×32 b-tile from the packed stream and
  ## accumulates into the 4 fp32 accumulators, the epilogue rounds each
  ## accumulator once to fp16 and stores row-bounded. Grid x = N div
  ## 128, grid y = (M + 31) div 32, threadgroup = 32 lanes.
  let tgx = int32(threadgroup_position_in_grid.x)
  let tgy = int32(threadgroup_position_in_grid.y)

  # x and Out carry the natural row strides (the w stream is addressed
  # by the decoders through the raw pointer and rowBytes)
  let gdX = x.gd(shape = (-1, -1, -1, -1), stride = (32 * K, 0, K, 1))
  let gdOut = Out.gd(shape = (-1, -1, -1, -1), stride = (32 * N, 0, N, 1))

  # the output accumulators, array-resident and zeroed before the K
  # loop (declared on the fp16 atom so the mma's three operands share
  # one fragment layout)
  var d: array[4, rt_l(float32, 32, 32, getTileConfig(float32, float16))]
  for i in 0 ..< 4:
    d[i].zero()

  var a_reg: rt_l(float16, 32, 16)
  var b_reg: rt_r(float16, 16, 32)

  for kk in 0'i32 ..< K div 16:
    a_reg.loadTileRows(gdX, (0, 0, tgy, kk), M)
    for nt in 0'i32 ..< 4:
      when scheme == 0:
        dequantGGUF_Q8_0(b_reg, w, kk, rowBytes, tgx, nt)
      elif scheme == 1:
        dequantGGUF_Q4_K(b_reg, w, kk, rowBytes, tgx, nt)
      else:
        dequantGGUF_IQ4_XS(b_reg, w, kk, rowBytes, tgx, nt)
      d[nt].mma_AB(a_reg, b_reg)

  var y: array[4, rt_l(float16, 32, 32)]
  for nt in 0'i32 ..< 4:
    y[nt].convert(d[nt])
    gdOut.storeTileRows(y[nt], (0, 0, tgy, tgx * 4 + nt), M)
