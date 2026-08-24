## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#        Tile GEMM kernel: one 32×32 output tile per threadgroup
#
# ############################################################

import workspace/crucible
import ../int_tuples
import ../layouts
import ../layout_constructors
import ../layout_indexing
import ../tensors
import ../ptr_arithmetic
import ../atoms
import ../tile_algebra/tile_config
import ../tile_algebra/tile_views
import ../tile_algebra/tile_io
import ../tile_algebra/tile_ops
import ../tile_algebra/tile_mma

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_config, tile_views, tile_io, tile_ops, tile_mma

proc gemm*[TIn, TOut](
    D: ptr UncheckedArray[TOut];
    M, N, K: int32;
    A: ptr UncheckedArray[TIn]; rsa, csa: int32;
    B: ptr UncheckedArray[TIn]; rsb, csb: int32;
    Lengths: ptr UncheckedArray[uint16];
    branching: static bool) {.device.} =
  ## D = A·B, one 32×32 output tile per threadgroup, the K-loop over K/16 k-blocks.
  ## Grid: x = output column tiles, y = output row tiles.
  ## Expected input: A and B with explicit row/col strides (BLIS order),
  ## so row-major, col-major, negative and strided layouts address directly.
  ## D is row-major M×N.
  ## When `branching`, row m contributes only k < Lengths[m].
  ## With `branching`, the K-loop stops at the tile's effective k-block count.
  ## The A rows are host-padded to `rsa`, so the mma never guards.
  let gl_a = makeGlStrided(A, 0, 0, rsa, csa)
  let gl_b = makeGlStrided(B, 0, 0, csb, rsb)
  let gl_d = makeGl(D, 0, 0, M, N)

  const TileDim = 32
  const TileK = 16
  var a_rtl: rt_l(TIn, TileDim, TileK)
  var b_rtr: rt_r(TIn, TileK, TileDim)
  var d_rtl: rt_l(float32, TileDim, TileDim)

  d_rtl.zero()

  let OUTPUT_Y = threadgroup_position_in_grid.y
  let OUTPUT_X = threadgroup_position_in_grid.x

  var kLimit = K div int32(TileK)
  when branching:
    kLimit = min(kLimit, int32(tileKMax(Lengths, OUTPUT_Y)))

  for k in 0'i32 ..< kLimit:
    a_rtl.load(gl_a, (0, 0, OUTPUT_Y, k))
    b_rtr.load(gl_b, (0, 0, OUTPUT_X, k))
    d_rtl.mma_AB(a_rtl, b_rtr)

  d_rtl.store(gl_d, (0, 0, OUTPUT_Y, OUTPUT_X))
