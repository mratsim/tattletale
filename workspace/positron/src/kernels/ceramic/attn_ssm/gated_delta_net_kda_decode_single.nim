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

##
## - the recorded contract keeps q/k/g/beta f32, this spelling's element-dtype axis covers v and y only
## - the bf16 spelling is the recorded Kimi spelling, fp16 follows the element dtype verdict
##
## - Entries are consumer-side, a `metal` block wraps the grid-driven proc with concrete
##   static (Dk, Dv, TileR), one call-site line per static binding set
## - The engine's monomorphization key erases static bindings, calls sharing a call-site line collapse into one body
## - The decode mega kernel composes the tile core `kdaDecodeStepTileAt` inline instead
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
from ../tile_widen import roundToRne

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Core tile procs (inline-tile property) ──────────────────────────

proc kdaDecodeStepTileAt*[T](
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[T],           # (B·Hv, Dv) element dtype core output
    k: ptr UncheckedArray[float32],       # (B·Hk, Dk) f32, post-l2norm
    q: ptr UncheckedArray[float32],       # (B·Hk, Dk) f32, post-l2norm
    v: ptr UncheckedArray[T],           # (B·Hv, Dv) element dtype
    g: ptr UncheckedArray[float32],       # (B·Hk, Dk) f32 log decay
    beta: ptr UncheckedArray[float32],    # (B·Hv,) f32 beta, one per value head
    qScale: float32,                      # √Dk, the host's f64 sqrt cast to f32
    Hv, Hk, hkRatio: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR: static int) {.device.} =
  ## One (bh, TileR-row) state tile of the KDA decode step at the caller's coordinates
  ## (the element dtype an unconstrained compile-time generic):
  ##
  ##   S ← S·Diag(exp(g)) + k ⊗ (β·(v − (S·Diag(exp(g)))·k))    y ← S'·(q/√Dk)
  ##
  ## Contract:
  ## - all state arithmetic is fp32, the state never rounds
  ## - the per-channel decay applies BEFORE the kv read, the recurrence's step order
  ##   decayed[dkc] = exp2(g[dkc]·log2e)·S[dkc], the kv read contracts the decayed state
  ## - the state stores in place, f32, no rounding
  ##
  ## - the element-dtype v load and the one element-dtype y rounding are the only
  ##   dtype-dependent steps, the tile walk is dtype-mechanical
  ##
  ## - precondition, Hk > 0, Hv an exact multiple of Hk and hkRatio = Hv div Hk
  ##
  ##   kv_mem[row] = Σ_dkc decayed[row][dkc]·k[dkc]
  ##   delta[row] = β·(v[row] − kv_mem[row])
  ##   y[row] = Σ_dkc S'[row][dkc]·(q[dkc]/qScale), one element-dtype round
  ##
  ## Y write goes to the lanes whose fragment column is 0, one lane per state row.
  ##
  ## - `bh` is the (sequence, value head) row block, `dvBlock` the Dv/TileR row block
  ## - grid-driven wrapper, receiving the threadgroup coordinates from the grid
  ## - generic only over the element dtype and the static shape, every (Dk, Dv, TileR)
  ##   binding needs its own call-site line
  const atom = getTileConfig(float32, float32)
  static:
    doAssert TileR == 8, "the y store covers one atom row block per column block"
    doAssert Dv mod TileR == 0, "the column grid covers Dv in whole row blocks"
    doAssert TileR mod atom.getM() == 0 and Dk mod atom.getN() == 0
  let hk = ((bh mod Hv) div hkRatio) + ((bh div Hv) * Hk)
  let headLin = bh * Dv * Dk
  let yLin = bh * Dv
  let kLin = hk * Dk

  let glState = state.gd(shape = (-1, -1, -1, -1), stride = (1, 0, Dk, 1))
  let glK = k.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glQ = q.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glG = g.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 0, 1))
  let glV = v.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1, 0))

  var s: rt_l(float32, TileR, Dk)
  var k32: rt_l(float32, TileR, Dk)
  var q32: rt_l(float32, TileR, Dk)
  var gT: rt_l(float32, TileR, Dk)
  var vT: rt_l(T, TileR, 8)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))
  k32.loadTile(glK, (kLin, 0, 0, 0))
  q32.loadTile(glQ, (kLin, 0, 0, 0))
  gT.loadTile(glG, (kLin, 0, 0, 0))
  vT.loadTile(glV, (yLin, 0, dvBlock, 0))

  # Per-channel decay, BEFORE the kv read (the recurrence's step order).
  # The g tile broadcasts one key head's log-decay row over the tile rows,
  # the exp2 form (see the module doc), one tile mul into the state.
  # gT is dead past the decay, the output walk uses its own oProd tile.
  gT.mul(gT, Log2e)
  exp2(gT, gT)
  s.mul(s, gT)

  # kv_mem[row] = Σ_dkc decayed[row][dkc]·k[dkc] over the decayed state, the k
  # tile broadcasts one key vector over the tile rows, one row sum per lane
  var prod: rt_l(float32, TileR, Dk)
  prod.mul(s, k32)
  var kvVec: rv(float32, TileR, Dk)
  kvVec.row_sum(prod)
  let kvMem = kvVec.data[0]

  let v32 = vT.frags[0][0].frag[0].float32
  let delta = beta[bh] * (v32 - kvMem)

  const rowTiles = TileR div atom.getM()
  const colTiles = Dk div atom.getN()
  const vpt = atom.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        s.frags[n][m].frag[v] =
          s.frags[n][m].frag[v] + k32.frags[n][m].frag[v] * delta

  var oProd: rt_l(float32, TileR, Dk)
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        oProd.frags[n][m].frag[v] =
          s.frags[n][m].frag[v] * (q32.frags[n][m].frag[v] / qScale)
  var oVec: rv(float32, TileR, Dk)
  oVec.row_sum(oProd)
  let oVal = oVec.data[0]

  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (lane, 0)).toIntVal()
  let rowIn = cell mod 8
  let colIn = cell div 8
  if colIn == 0:
      y[yLin + dvBlock * 8 + int32(rowIn)] = roundToRne[T](oVal)
  glState.storeTile(s, (headLin, 0, dvBlock, 0))


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
  ## Grid-driven form of `kdaDecodeStepTileAt`, the caller's `metal:` entry wraps this proc.
  ## - grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  kdaDecodeStepTileAt(state, y, k, q, v, g, beta, qScale, Hv, Hk, hkRatio,
    dvBlock, bh, Dk, Dv, TileR)
