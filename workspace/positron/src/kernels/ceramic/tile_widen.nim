# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ───────────────  tile_widen (exact 16-bit → f32 register-tile widening)  ───────────────

## Exact widening of a 16-bit register tile to f32, shared by the ceramic
## tile kernels over the 16-bit element dtypes.
##
## Contract:
## - widening is exact, every lane reads only its own fragments
## - the f32 tile's values are the exact widenings of the source tile's values
## - the walk follows the dst atom's lane→element mapping, both atoms must
##   share one 8×8×8 geometry class, asserted statically
##
## The tile API's mma epilogues keep the accumulator f32, an element-dtype
## operand tile re-enters the f32 arithmetic only through this widen.

import workspace/crucible
import workspace/ceramic

# ─── Module-local device helpers ─────────────────────────────────────

proc widen*[A, B: static MmaAtom; T; R, C: static int](
    dst: var RtLeft[float32, R, C, A],
    src: RtLeft[T, R, C, B]) {.device.} =
  ## Exact 16-bit-dtype → f32 widening, walking each tile's own atom
  ## lane→element mapping, the source element type a compile-time parameter.
  ##
  ## Contract:
  ## - every dst fragment holds the float32 widening of the matching src fragment
  ## - `T` is unconstrained, the float32 widening exact for the 16-bit
  ##   element dtypes today's callers instantiate
  ##
  ## Example, widening a (8, Dk) 16-bit-dtype key tile into the f32 arithmetic tile:
  ##
  ##   k32.widen(kT)   # widens the key tile in place, T from kT's element type
  ##
  ## For every (n, m, v), each f32 fragment satisfies:
  ##
  ##   k32.frags[n][m].frag[v] == kT.frags[n][m].frag[v].float32
  static:
    doAssert A.getM() == B.getM() and A.getN() == B.getN() and
      A.getVpt() == B.getVpt(),
      "widen: both atoms must share the lane→fragment cell mapping"
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst.frags[n][m].frag[v] = src.frags[n][m].frag[v].float32

# ─── RNE narrowing ──────────────────────────────────────────────────

proc roundToRne*[T](x: float32): T {.device.} =
  ## One round-to-nearest-even of an f32 value into the dtype `T`,
  ## the scalar counterpart of the mma epilogue's single-round contract.
  ##
  ## - `T` is unconstrained
  ## - the body keeps one variant per element dtype, a further dtype
  ##   adds its own variant
  when T is bfloat16:
    x.bfloat16
  else:
    x.to(float16)
