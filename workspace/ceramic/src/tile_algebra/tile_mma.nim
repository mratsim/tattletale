## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../hardware/h_mma_dispatch
import ./tiles
import ./tile_config

# ═════════════════════════════════════════════════════════════════════════
#  In-place mma
# ═════════════════════════════════════════════════════════════════════════

proc mma_AB*[TIn; R, C, K: static int; A: static MmaAtom](
    dst: var RtLeft[float32, R, C, A],
    a: RtLeft[TIn, R, K, A],
    b: RtRight[TIn, K, C, A]) =
  ## `dst.mma_AB(a, b)`: dst += a·b over the K subtiles, accumulated in fp32.
  # TODO: tensor cores
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const rK = K div A.getK()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for k in 0 ..< rK:
        universalMma8x8x8(dst.frags[n][m].frag, a.frags[n][k].frag, b.frags[m][k].frag)
