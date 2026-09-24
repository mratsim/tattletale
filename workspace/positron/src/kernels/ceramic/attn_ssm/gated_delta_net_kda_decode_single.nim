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
## Register-tile naming convention, shared by the gdn and kda kernels:
##
## - `<x>T`, the element-dtype register tile of operand x, loaded from memory
## - `<x>32`, the fp32 register tile of the same operand, an fp32-storage
##   operand loads straight into its `32` form, an element-dtype operand
##   widens its `T` form into the `32` form
##
## - the recorded contract keeps q/k/g/beta f32, this spelling's element-dtype axis covers v and y only
## - the bf16 spelling is the recorded Kimi spelling, fp16 follows the element dtype verdict
##
## - Entries are consumer-side, a `metal` block wraps the grid-driven proc with concrete
##   static (Dk, Dv, TileR), one call-site line per static binding set
## - The engine's monomorphization key erases static bindings, calls sharing a call-site line collapse into one body
## - The tile core is the shared `gatedDeltaDecodeStepTileAt` in the gdn module, the kda
##   grid-driven entry forwards into it with decayChannel = true and qDivQScale = true
##
## Binding and state ABI:
## - hosts binding through the Metal engine's no-copy path get in-place state
##   updates and visible y writes from one run
##
## - any other binding copies and the y writes are lost
## - `state` is the engine's output buffer, `y` is written by the kernel
##
## - the state's ABI is (B·Hv, Dv, Dk) f32, dense row-major, head-major over
##   (sequence, value head), one unrounded fp32 tile per (bh, Dv-row-block)
## - the f32 state buffer persists across steps and launches with no in-kernel reset,
##   the host owns the layout and the lifetime
## - rebinding the state to a 16-bit dtype or a strided view silently corrupts the recurrence
from ../math_consts import Log2e
import workspace/crucible
import workspace/ceramic
import gated_delta_net_decode_single

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

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
  ## Grid-driven form of the shared `gatedDeltaDecodeStepTileAt` in its kda binding
  ## (per-channel decay, f32 k/q/beta, divide-by-qScale q̃), the caller's `metal:` entry wraps this proc.
  ## Grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8.
  ##
  ## Parameters, pointers naming their dtypes, shapes bound at the call, grid coordinates arriving from the grid:
  ##
  ## | parameter     | shape, dtype, layout                                                                                                                                                           | producer                                                                                                                                                         | unit                |
  ## | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
  ## | state         | (B·Hv, Dv, Dk) f32, dense row-major, head-major over (sequence, value head), the persistent recurrence state, written back in place                                            | the previous step's launch writes it, this kernel reads and rewrites it in place, the buffer persists with no in-kernel reset, the host owns layout and lifetime | f32, never rounds   |
  ## | y             | (B·Hv, Dv) El, row-major, the step's output, one round-to-nearest-even per element                                                                                             | this kernel                                                                                                                                                      | El                  |
  ## | k             | (B·Hk, Dk) f32, row-major, the key vector of the key head this tile serves, post-l2norm (the l2norm stays host-side)                                                           | host-computed                                                                                                                                                    | f32                 |
  ## | q             | (B·Hk, Dk) f32, row-major, the query vector of the same key head, post-l2norm, host-computed, q̃ divides by the device-side runtime qScale, never rounded to the element dtype | host-computed (the divide device-side)                                                                                                                           | f32                 |
  ## | v             | (B·Hv, Dv) El, row-major, the value vector of the value head this tile serves                                                                                                  | host-computed                                                                                                                                                    | El                  |
  ## | g             | (B·Hk, Dk) f32 log decay, one log-decay per KEY channel, finite and ≤ 0 per key channel, no kernel clamp, a violating g explodes the f32 state                                 | host-computed (the elementwise prefix)                                                                                                                           | log2-decay exponent |
  ## | beta          | (B·Hv,) f32, one per value head, the delta weighting                                                                                                                           | host-computed                                                                                                                                                    | dimensionless       |
  ## | qScale        | √Dk, the host's f64 sqrt cast to f32                                                                                                                                           | host-computed (the divide device-side)                                                                                                                           | dimensionless       |
  ## | Hv, Hk        | value and key head counts, host-derived from the model config                                                                                                                  | host-computed                                                                                                                                                    | heads               |
  ## | hkRatio       | Hv div Hk, the GQA head ratio                                                                                                                                                  | host-computed                                                                                                                                                    | dimensionless       |
  ## | Dk, Dv, TileR | static tile geometry (head dim, value dim, the row block height)                                                                                                               | compile-time                                                                                                                                                     | elements            |
  ## | dvBlock, bh   | the Dv div TileR row-block index and the (sequence, value head) flat head index                                                                                                | device-computed grid coordinates                                                                                                                                 | elements            |
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  gatedDeltaDecodeStepTileAt(state, y, k, q, v, g, beta, qScale, Hv, Hk, hkRatio,
    dvBlock, bh, true, true, Dk, Dv, TileR)
