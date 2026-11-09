# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## RoPE apply kernel.
##
## - `k_rope`: q*cos + rotate_half(q)*sin over the last axis
## - one threadgroup per token, grid-stride over tokens, lanes striding the pair columns
## - cos/sin (tokens, heads, dim div 2) reach the kernel pre-broadcast:
##   their head axis strides 0
## - values upcast to float32 on the load, the rotation runs in float32,
##   one rounding at the store through the output's element type
##   (bfloat16 round-to-nearest-even)

import workspace/crucible
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/tile_algebra
import workspace/tensorcery/src/device_view

proc k_rope*(o, q, cos, sin: distinct DeviceView) =
  ## NEOX rotate_half rotation of the last axis, fully overwriting o.
  ##
  ## Expected input:
  ## - q: (tokens, heads, dim), dim even
  ## - cos, sin: (tokens, heads, dim div 2) float32 tables broadcast over
  ##   the head axis, entry (t, i) pairs with the columns
  ##   (i, i + dim div 2) of token t
  ##
  ## Output:
  ## - o fully written:
  ##   o[t, h, i]       = q[t, h, i]*cos[t, i] - q[t, h, i + half]*sin[t, i] for the even column
  ##   o[t, h, i + half] = q[t, h, i + half]*cos[t, i] + q[t, h, i]*sin[t, i] for the odd column
  ## The index i runs over 0 ..< dim div 2, t over the tokens and h over the heads.
  ##
  ## This produces q*cos + rotate_half(q)*sin. rotate_half is the pairwise
  ## split [-x2, x1] over the last dimension: the NEOX duplicated half table
  ## folds into the pair indexing.
  let
    ov = make_view(o.data +% o.offset, o.layout)
    qv = make_view(q.data +% q.offset, q.layout)
    cv = make_view(cos.data +% cos.offset, cos.layout)
    sv = make_view(sin.data +% sin.offset, sin.layout)
  let
    numTokens = int(qv.shape[0])
    heads = int(qv.shape[1])
    half = int(qv.shape[2]) div 2
  let
    lane = getLocalThreadID()
    lanes = threads_per_threadgroup.x
    group0 = threadgroup_position_in_grid.x
    groupStride = getGridSize()
  for token in countup(group0, numTokens - 1, groupStride):
    for h in 0 ..< heads:
      for i in countup(lane, half - 1, lanes):
        let
          c = cv[token, h, i].to(float32)
          s = sv[token, h, i].to(float32)
          qEven = qv[token, h, i].to(float32)
          qOdd = qv[token, h, i + half].to(float32)
        ov[token, h, i] = (qEven * c - qOdd * s).to(ov.T)
        ov[token, h, i + half] = (qOdd * c + qEven * s).to(ov.T)
