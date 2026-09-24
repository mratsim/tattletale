## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.


# ############################################################
#
#     Fused EXL3 decode-GEMV forward (linear_exl3_decode_single), Tile API port
#
# ############################################################

## Fused EXL3 decode-GEMV forward on the ceramic Tile API.
## MMODE entry over the shared core (linear_exl3_core).
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
## is reconstructed on the fly by `dequantTrellis` (quant_exl3_ops).
## D is the static FWHT block (128).
## `bits`, `cb` and `mmode` are the static instantiation family
## (bits 1..8 × cb 0..2, MMODE 0/1). cb0 is the production default codebook.
##
## MMODE:
##   - MMODE 0: the m = 1 fast path. The x row limit is the compile-time 1.
##     Rows 1..31 fold to statically-zero fragments.
##   - MMODE 1: m ≤ 8 with the runtime M guard.
## The store guard is M in both modes.
##
## Shapes: K and N must be 128-multiples. Rows ≥ the mode's x row
## limit are zero-filled on load. Rows ≥ M are skipped on store.
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
## - The cb2 decode is the two-rounding numeric form, a few fp16 ulps
##   from the single-rounding reference decode (see the quant_exl3_ops module doc).
## - No fp32 path.

import workspace/crucible
import workspace/ceramic
import ./linear_exl3_core

# ═════════════════════════════════════════════════════════════════════
#  The kernel
#  ═════════════════════════════════════════════════════════════════════

proc exl3_gemv_fwd*(
    Out: ptr UncheckedArray[float16],    # (M, N) fp16 output
    x: ptr UncheckedArray[float16],      # (M, K) fp16 input
    trellis: ptr UncheckedArray[int16],  # (tiles_k, tiles_n, 256*bits div 16) packed
    suh: ptr UncheckedArray[float16],    # (K) fp16 input scale
    svh: ptr UncheckedArray[float16],    # (N) fp16 output scale
    M, K, N: int32,
    bits: static int,
    cb: static int,
    mmode: static int,
    D: static int) {.device.} =
  ## MMODE binding over the shared core (linear_exl3_core), bits 1..8 × cb 0..2
  ##   - MMODE 0 runs the m=1 fast path with the compile-time x row limit 1
  ##   - MMODE 1 takes m ≤ 8 with the runtime M x row limit
  ## M is the runtime row count.
  static: doAssert bits in {1, 2, 3, 4, 5, 6, 7, 8},
    "the dequantTrellis funnel Layout is instantiated for bits 1..8"
  static: doAssert cb in {0, 1, 2},
    "the dequantTrellis codebook is instantiated for cb 0..2"
  static: doAssert mmode in {0, 1},
    "MMODE 0 = the m=1 fast path, MMODE 1 = m <= 8"
  exl3_fwd_core(Out, x, trellis, suh, svh, M, K, N, bits,
    cb = cb, m1FastPath = mmode == 0, D = D)
