## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#          Tile mma: the accumulator seed and the in-place mma
#
# ############################################################
#
# The math is the register-surface `mma` from tile_config.
# An fp32 A operand (the attention P·V step) feeds through at full
# precision, no intermediate cast.
# The 8×8×8 atom's A and C fragments share one layout (C-role == A-role),
# so an accumulator feeds the next mma's A operand with zero movement:
# no cross-thread redistribution exists in this layer.

import ../int_tuples
import ../atoms
import ./tiles
import ./tile_config
import ./tile_fma_partition
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  The accumulator seed
# ═════════════════════════════════════════════════════════════════════════

proc zero*[R, C: static int; A: static MmaAtom; TL: static ThreadLayout](
    tile: var RtLeft[float32, R, C, A, TL]) =
  ## `tile.zero()`: zeroes the accumulator's owned cells.
  const rowTiles = R div A.mnk.m
  const colTiles = C div A.mnk.n
  let thr = fmaSlice[A, TL]()
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in countup(thr.tn, colTiles - 1, TL.thrN):
      zero(tile.frags[n][m].frag)

# ═════════════════════════════════════════════════════════════════════════
#  The in-place mma
# ═════════════════════════════════════════════════════════════════════════

proc mma_AB*[TIn, TB; R, C, K: static int; AD, AA, AB: static MmaAtom; TL: static ThreadLayout](
    dst: var RtLeft[float32, R, C, AD, TL];
    a: RtLeft[TIn, R, K, AA, TL];
    b: RtRight[TB, K, C, AB, TL]) =
  ## `dst.mma_AB(a, b)`: dst += a·b over the K subtiles, accumulated in fp32.
  ## The storage elements widen at the multiply.
  ## The fp16 gemm and the fp32-A attention P·V forms share one body.
  ## The K-loop order is the host reference's accumulation order.
  const rowTiles = R div AD.mnk.m
  const colTiles = C div AD.mnk.n
  const rK = K div AA.mnk.k
  let thr = fmaSlice[AD, TL]()
  for n in countup(thr.tm, rowTiles - 1, TL.thrM):
    for m in countup(thr.tn, colTiles - 1, TL.thrN):
      for k in 0 ..< rK:
        mma[AD, float32, TIn, TB](dst.frags[n][m].frag, a.frags[n][k].frag, b.frags[m][k].frag)
