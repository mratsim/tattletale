# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ──────────────  KDA decode_single (one per-channel gated-delta-rule step per threadgroup)  ──────────────

## One decode step (T = 1) of the Kimi Delta Attention recurrence
## (arXiv:2510.26692) on the ceramic Tile API:
##
##   S ← S·Diag(exp(g)) + k ⊗ (β·(v − (S·Diag(exp(g)))·k))    y ← S'·(q/√Dk)
##
## - g is a (B·Hk, Dk) matrix, one log-decay per KEY channel, decayed elementwise BEFORE the kv read
## - the kv read contracts the decayed state kᵀ·Diag(exp(g))·S, a post-contraction decay kᵀ·S·exp(g) is the GDN op
##
## | contract      | value                                                                                                |
## | ------------- | ---------------------------------------------------------------------------------------------------- |
## | state math    | all fp32 and never rounds, one 8-row state tile per threadgroup, no inter-threadgroup sync           |
## | q, k, g, β    | (B·Hk, Dk) f32 q/k/g post-l2norm, (B·Hv,) f32 beta, never rounded to the element dtype               |
## | v, y          | (B·Hv, Dv) element dtype each, y gets one round-to-nearest-even                                      |
## | element dtype | one compile-time element type (`kdaDecodeStepTile`'s `T` generic)                                    |
## | head mapping  | value head bh reads key head `(bh mod Hv) div hkRatio + (bh div Hv)·Hk`, hkRatio = Hv div Hk         |
## | batch         | the head axis, one launch at grid (Dv div TileR, B·Hv) over per-sequence stacked inputs              |
## | decay         | exp2(g·log2e) per channel, log2e is the shared `math_consts.Log2e` (Metal has no exp device builtin) |
## | q̃            | divides q per element by the runtime f32 `qScale`, the host's f64 √Dk cast to f32                    |
##
## | contract       | value                                                                                                                            |
## | -------------- | -------------------------------------------------------------------------------------------------------------------------------- |
## | g precondition | finite and ≤ 0 per key channel (−exp(A_log)·softplus ≤ 0 by construction), no kernel clamp, a violating g explodes the f32 state |

#
## Register-tile naming and the state ABI contract:
## gated_delta_net_decode_single.nim's header, shared by the gdn and kda kernels.
from ../math_consts import Log2e
import workspace/crucible
import workspace/ceramic
import gated_delta_net_decode_single

export layout_algebra, tensors, tile_algebra, ptr_arithmetic

# ─── Core tile procs (inline-tile property) ──────────────────────────

proc kdaDecodeStepTile*[T](
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[T],           # (B·Hv, Dv) element dtype core output
    k: ptr UncheckedArray[float32],       # (B·Hk, Dk) f32, post-l2norm
    q: ptr UncheckedArray[float32],       # (B·Hk, Dk) f32, post-l2norm
    v: ptr UncheckedArray[T],           # (B·Hv, Dv) element dtype
    g: ptr UncheckedArray[float32],       # (B·Hk, Dk) f32 log decay
    beta: ptr UncheckedArray[float32],    # (B·Hv,) f32 beta, one per value head
    qScale: float32,                      # √Dk, the host's f64 sqrt cast to f32
    Hv, Hk, hkRatio: int32,
    Dk, Dv, TileR: static int) {.device.} =
  ## Launch-driven kda binding of the shared `gatedDeltaDecodeStepTileAt`
  ## (per-channel decay, f32 k/q/beta, divide-by-qScale q̃), the caller's
  ## `metal:` entry wraps this proc.
  ##
## Launch dims (Dv div TileR, B·Hv), 32 lanes, x = the Dv/TileR row block,
## y = the (sequence, value head) flat head index bh, TileR = 8.
##
## Dataflow, one threadgroup:
##
##   k, q, g (hk row) f32 post-l2norm ──┐
##   v (bh row) El, beta (bh) f32 ──────┼─► [ gatedDeltaDecodeStepTileAt, kda form ]
##   qScale (f32 √Dk) ──────────────────┤        │
##   state (bh, dvBlock) f32 ───────────┘        ▼
##        y (bh, dvBlock) El ◄── one round-to-nearest-even per element
##        state (bh, dvBlock) f32 ◄── in-place update, never rounds
  ##
  ## - the recorded contract keeps q/k/g/beta f32, this spelling's element-dtype
  ##   axis covers v and y only
  ## - the element dtype is the unconstrained compile-time generic `T`
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  gatedDeltaDecodeStepTileAt(state, y, k, q, v, g, beta, qScale, Hv, Hk, hkRatio,
    dvBlock, bh, true, true, Dk, Dv, TileR)
