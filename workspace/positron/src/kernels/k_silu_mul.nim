# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Fused gated-activation kernel.
##
## - `k_silu_mul`: GatedDenseFFN activation, out = silu(g) * v over paired (tokens, cols) views
## - one threadgroup per token, grid-stride over tokens, lanes striding the columns
## - the silu input and the multiplier share one element dtype
##   (bfloat16 at the FFN shapes), values upcast to float32 on the load
## - silu and the product run in float32, one rounding at the store
##   through the output's element type (bfloat16 round-to-nearest-even)

import workspace/crucible
import workspace/math
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/tile_algebra
import workspace/tensorcery/src/device_view

proc k_silu_mul*(o, gate, value: distinct DeviceView) =
  ## Elementwise gated activation, fully overwriting o:
  ## o[t, c] = silu(g) * v with g = `gate[t, c]` and v = `value[t, c]`.
  ##
  ## Expected input:
  ## - `gate`, `value`: (tokens, cols) views of one element type, any strides, paired by coordinate
  ##
  ## Output:
  ## - o: (tokens, cols) at the inputs' element type, fully written
  ##
  ## silu(g) = g / (1 + exp(-g)): the SiLU (Swish) formula of the reference
  ## `F.silu` (x / (1 + exp(-x))).
  ## exp(-g) runs as exp2(-g * Log2e) through the crucible exp2 builtin.
  ## Quotient and product stay float32, one rounding at the store.
  ##
  ## Saturation at the float32 boundaries:
  ## - g large positive: -g * Log2e rounds to -inf, exp2(-inf) = +0, silu(g) = g exact
  ## - g large negative: -g * Log2e rounds to +inf, exp2(+inf) = +inf, silu(g) = g / +inf = -0
  ## - g moderately negative, silu(g) subnormal in bfloat16: the quotient
  ##   underflows to -0 at the store, matching the f64 reference rounding
  let
    ov = make_view(o.data +% o.offset, o.layout)
    gv = make_view(gate.data +% gate.offset, gate.layout)
    vv = make_view(value.data +% value.offset, value.layout)
  let
    rows = int(ov.shape[0])
    cols = int(ov.shape[1])
    lane = getLocalThreadID()
    lanes = threads_per_threadgroup.x
    group0 = threadgroup_position_in_grid.x
    groupStride = getGridSize()
  for token in countup(group0, rows - 1, groupStride):
    for i in countup(lane, cols - 1, lanes):
      let g = gv[token, i].to(float32)
      # silu(g) = g / (1 + exp(-g)), exp(-g) = exp2(-g * Log2e)
      let s = g / (1.0'f32 + exp2(-g * Log2e))
      ov[token, i] = (s * vv[token, i].to(float32)).to(ov.T)
