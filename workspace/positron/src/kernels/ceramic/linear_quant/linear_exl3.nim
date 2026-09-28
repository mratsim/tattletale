## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.


# ############################################################
#
#     Fused EXL3 linear forward (linear_exl3), Tile API port
#
# ############################################################

## Fused EXL3 linear forward on the ceramic Tile API.
## cb0-only entry over the shared core (linear_exl3_core).
##
## Contract:
##
##     out = FWHT-128( svh ⊙ ( FWHT-128( suh ⊙ x ) @ W_dequant ) )
##
## Buffers:
##   - Out: (M, N) fp16 output
##   - x: (M, K) fp16 input
##   - trellis: (tiles_k, tiles_n, 256·bits div 16) packed int16 codes
##   - suh: (K) fp16 input scales
##   - svh: (N) fp16 output scales
##
## The weight matrix is not stored. Each 16×32 fp16 weight tile
## is reconstructed on the fly by `dequantTrellis` (quant_exl3_ops):
##   - the funnel shift
##   - the procedural cb0 codebook
##   - the tensor-core-shuffle word placement
## Only the cb0 codebook is instantiated.
## D is the static FWHT block (128). `bits` is static, restricted to {3, 5, 8}.
##
## Shapes: K and N must be 128-multiples. Rows ≥ M are zero-filled on load and skipped on store.
##
## Dataflow per 128-column K-block:
##
##     x --> suh ⊙ --> FWHT-128 ----+
##                                  v
##     trellis --> dequantTrellis --> mma_AB --> fp32 accum
##                                                 |
##                                                 v
##     Out <-- svh ⊙ <-- FWHT-128 <-- fp16 round <-+
##
## Known gaps:
## - K and N must be 128-multiples. Partial shapes are out of contract.
## - cb0 codebook only. cb1/cb2 are not instantiated.
## - No fp32 path.

import workspace/crucible
import workspace/ceramic
import ./linear_exl3_core

# ═════════════════════════════════════════════════════════════════════
#  The kernel
#  ═════════════════════════════════════════════════════════════════════

proc exl3_linear_fwd*(
    Out: ptr UncheckedArray[float16],    # (M, N) fp16 output
    x: ptr UncheckedArray[float16],      # (M, K) fp16 input
    trellis: ptr UncheckedArray[int16],  # (tiles_k, tiles_n, 256*bits div 16) packed
    suh: ptr UncheckedArray[float16],    # (K) fp16 input scale
    svh: ptr UncheckedArray[float16],    # (N) fp16 output scale
    M, K, N: int32,
    bits: static int,
    D: static int) {.device.} =
  ## cb0-only binding over the shared core (linear_exl3_core)
  ## - bits restricted to {3, 5, 8}
  ## - the runtime M as the x row limit
  static: doAssert bits in {3, 5, 8},
    "the dequantTrellis funnel Layout is instantiated for bits 3/5/8 only"
  exl3_fwd_core(Out, x, trellis, suh, svh, M, K, N, bits, cb = 0, D = D)
