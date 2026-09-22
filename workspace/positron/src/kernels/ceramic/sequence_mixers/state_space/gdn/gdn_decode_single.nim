# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ──────────────────────────────  GDN decode_single (one gated-delta-rule step per threadgroup)  ───────────────────────────────

## One decode step (T = 1) of the gated delta-rule recurrence (arXiv:2412.06464):
##
##   S ← S·exp2(g·log2e) + k ⊗ (β·(v − (S·exp2(g·log2e))·k))    y ← S'·(q·Dk^-0.5)
##
## | contract       | value                                                                                                                                   |
## | -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
## | state math     | all fp32 and never rounds, one 8-row state tile per threadgroup, no inter-threadgroup sync                                              |
## | q, k           | (B·Hk, Dk) family dtype, already l2-normalized (l2norm stays host-side)                                                                 |
## | v, beta        | (B·Hv, Dv) and (B·Hv,) family dtype, g is (B·Hv,) f32 log-decay                                                                         |
## | y              | (B·Hv, Dv) family dtype, one round-to-nearest-even                                                                                      |
## | family dtype   | fp16 primary (`gdnDecodeStepTileF16`), bf16 the range-robust fallback (`gdnDecodeStepTileBf16`)                                         |
## | head mapping   | value head bh reads key head `(bh mod Hv) div hkRatio + (bh div Hv)·Hk`, hkRatio = Hv div Hk                                            |
## | batch          | the head axis, one launch at grid (Dv div TileR, B·Hv) over per-sequence stacked inputs is the batched decode step                      |
## | decay / q̃     | exp2(g·log2e), log2e = 1.4426950408889634'f32, Dk^-0.5 folded into q in f32 (rsqrt-multiply form, Metal has no exp device builtin)      |
## | lanes          | 32 per threadgroup, the lane→element walk is a 32-lane contract (launch_contract.assertLanes32 at the launch site)                      |
## | g precondition | finite and ≤ 0 by construction, no kernel clamp, a violating g explodes the persistent f32 state (assertDecayFinite at the launch site) |
##
## - design provenance, WIP spelling in the 20260912-positron-taxonomy worktree,
##   kernel design mined, test shapes not carried over
##
## - Entries are consumer-side, a `metal:` block wraps the grid-driven proc with concrete
##   static (Dk, Dv, TileR), one call-site line per static binding set
## - The engine's monomorphization key erases static bindings, calls sharing a call-site line collapse into one body
## - The decode mega kernel composes the tile core `gdnDecodeStepTileBf16At` and `gdnDecodeStepTileF16At` inline instead
##
## Binding and state ABI:
## - hosts binding through the Metal engine's no-copy path get in-place state
##   updates and visible y writes from one run
## - the path needs a page-aligned pointer and a page-multiple byte length, launch_contract.assertNocopyBinding asserts it
##
## - any other binding copies and the y writes are lost
## - `state` is the engine's output buffer, `y` is written by the kernel
##
## - the state's ABI is (B·Hv, Dv, Dk) f32, dense row-major, head-major over
##   (sequence, value head), one unrounded fp32 tile per (bh, Dv-row-block)
## - the f32 state buffer persists across steps and launches with no in-kernel reset,
##   the host owns the layout and the lifetime
## - rebinding the state to a 16-bit dtype or a strided view silently
##   corrupts the recurrence

import workspace/crucible
import workspace/ceramic
import ../../../tile_widen

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Module-local device helpers ─────────────────────────────────────

# ─── Core tile procs (inline-tile property) ──────────────────────────

proc gdnDecodeStepTileBf16At*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[bfloat16],      # (B·Hv, Dv) bf16 core output
    k: ptr UncheckedArray[bfloat16],      # (B·Hk, Dk) bf16, post-l2norm
    q: ptr UncheckedArray[bfloat16],      # (B·Hk, Dk) bf16, post-l2norm
    v: ptr UncheckedArray[bfloat16],      # (B·Hv, Dv) bf16
    g: ptr UncheckedArray[float32],       # (B·Hv,) f32 log decay
    beta: ptr UncheckedArray[bfloat16],   # (B·Hv,) bf16, one per value head
    Hv, Hk, hkRatio: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR: static int) {.device.} =
  ## One (bh, TileR-row) state tile of the gated delta-rule decode step at the caller's
  ## coordinates:
  ##
  ##   S ← S·exp2(g·log2e) + k ⊗ (β·(v − (S·exp2(g·log2e))·k))    y ← S'·(q·Dk^-0.5)
  ##
  ## Contract:
  ## - all state arithmetic is fp32, the state never rounds
  ## - precondition, Hk > 0, Hv an exact multiple of Hk and hkRatio = Hv div Hk
  ##   (launch_contract.assertHeadMapping at the launch site)
  ##
  ## - the decay applies before the kv read (the recurrence's step order)
  ## - the state stores in place, f32, no rounding
  ##
  ##   kv_mem[row] = Σ_dk decayed[row][dk]·k[dk]
  ##   delta[row] = β·(v[row] − kv_mem[row])
  ##   y[row] = Σ_dk S'[row][dk]·(q[dk]·Dk^-0.5), one bf16 round
  ##
  ## The y write goes to the lanes whose fragment column is 0, one lane per state row.
  ##
  ## `bh` is the (sequence, value head) row block, `dvBlock` the Dv/TileR row block.
  ## The grid-driven wrapper passes the threadgroup coordinates.
  ## Generic only over the static shape, every (Dk, Dv, TileR) binding needs its own call-site line.
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
  let glV = v.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1, 0))

  var s: rt_l(float32, TileR, Dk)
  var kT: rt_l(bfloat16, TileR, Dk)
  var qT: rt_l(bfloat16, TileR, Dk)
  var vT: rt_l(bfloat16, TileR, 8)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))
  kT.loadTile(glK, (kLin, 0, 0, 0))
  qT.loadTile(glQ, (kLin, 0, 0, 0))
  vT.loadTile(glV, (yLin, 0, dvBlock, 0))

  let dec = exp2(g[bh] * 1.4426950408889634'f32)
  s.mul(s, dec)

  # kv_mem[row] = Σ_dk decayed[row][dk]·k[dk] over the decayed state, the k
  # tile broadcasts one key vector over the tile rows, one row sum per lane
  var k32: rt_l(float32, TileR, Dk)
  k32.widenBf16(kT)
  var prod: rt_l(float32, TileR, Dk)
  prod.mul(s, k32)
  var kvVec: rv(float32, TileR, Dk)
  kvVec.row_sum(prod)
  let kvMem = kvVec.data[0]

  let v32 = vT.frags[0][0].frag[0].float32
  let delta = beta[bh].float32 * (v32 - kvMem)

  const rowTiles = TileR div atom.getM()
  const colTiles = Dk div atom.getN()
  const vpt = atom.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        s.frags[n][m].frag[v] =
          s.frags[n][m].frag[v] + k32.frags[n][m].frag[v] * delta

  let scale = rsqrt(float32(Dk))
  var q32: rt_l(float32, TileR, Dk)
  q32.widenBf16(qT)
  var oProd: rt_l(float32, TileR, Dk)
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        oProd.frags[n][m].frag[v] =
          s.frags[n][m].frag[v] * (q32.frags[n][m].frag[v] * scale)
  var oVec: rv(float32, TileR, Dk)
  oVec.row_sum(oProd)
  let oVal = oVec.data[0]

  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (lane, 0)).toIntVal()
  let rowIn = cell mod 8
  let colIn = cell div 8
  if colIn == 0:
    y[yLin + dvBlock * 8 + int32(rowIn)] = oVal.bfloat16
  glState.storeTile(s, (headLin, 0, dvBlock, 0))

proc gdnDecodeStepTileBf16*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[bfloat16],      # (B·Hv, Dv) bf16 core output
    k: ptr UncheckedArray[bfloat16],      # (B·Hk, Dk) bf16, post-l2norm
    q: ptr UncheckedArray[bfloat16],      # (B·Hk, Dk) bf16, post-l2norm
    v: ptr UncheckedArray[bfloat16],      # (B·Hv, Dv) bf16
    g: ptr UncheckedArray[float32],       # (B·Hv,) f32 log decay
    beta: ptr UncheckedArray[bfloat16],   # (B·Hv,) bf16, one per value head
    Hv, Hk, hkRatio: int32,
    Dk, Dv, TileR: static int) {.device.} =
  ## Grid-driven form of `gdnDecodeStepTileBf16At`, the caller's `metal:` entry wraps this proc.
  ## - grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  gdnDecodeStepTileBf16At(state, y, k, q, v, g, beta, Hv, Hk, hkRatio,
    dvBlock, bh, Dk, Dv, TileR)

proc gdnDecodeStepTileF16At*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[float16],       # (B·Hv, Dv) fp16 core output
    k: ptr UncheckedArray[float16],       # (B·Hk, Dk) fp16, post-l2norm
    q: ptr UncheckedArray[float16],       # (B·Hk, Dk) fp16, post-l2norm
    v: ptr UncheckedArray[float16],       # (B·Hv, Dv) fp16
    g: ptr UncheckedArray[float32],       # (B·Hv,) f32 log decay
    beta: ptr UncheckedArray[float16],    # (B·Hv,) fp16, one per value head
    Hv, Hk, hkRatio: int32,
    dvBlock, bh: int32,
    Dk, Dv, TileR: static int) {.device.} =
  ## `gdnDecodeStepTileBf16At` with the fp16 family dtype, the same fp32 state arithmetic,
  ## fp16 loads and one fp16 y rounding.
  ##
  ## The 8×8×8 fp16 atom shares the bf16 lane→element geometry, tile walk, geometry contract and static asserts are identical.
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
  let glV = v.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 1, 0))

  var s: rt_l(float32, TileR, Dk)
  var kT: rt_l(float16, TileR, Dk)
  var qT: rt_l(float16, TileR, Dk)
  var vT: rt_l(float16, TileR, 8)
  s.loadTile(glState, (headLin, 0, dvBlock, 0))
  kT.loadTile(glK, (kLin, 0, 0, 0))
  qT.loadTile(glQ, (kLin, 0, 0, 0))
  vT.loadTile(glV, (yLin, 0, dvBlock, 0))

  let dec = exp2(g[bh] * 1.4426950408889634'f32)
  s.mul(s, dec)

  var k32: rt_l(float32, TileR, Dk)
  k32.widenF16(kT)
  var prod: rt_l(float32, TileR, Dk)
  prod.mul(s, k32)
  var kvVec: rv(float32, TileR, Dk)
  kvVec.row_sum(prod)
  let kvMem = kvVec.data[0]

  let v32 = vT.frags[0][0].frag[0].float32
  let delta = beta[bh].float32 * (v32 - kvMem)

  const rowTiles = TileR div atom.getM()
  const colTiles = Dk div atom.getN()
  const vpt = atom.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        s.frags[n][m].frag[v] =
          s.frags[n][m].frag[v] + k32.frags[n][m].frag[v] * delta

  let scale = rsqrt(float32(Dk))
  var q32: rt_l(float32, TileR, Dk)
  q32.widenF16(qT)
  var oProd: rt_l(float32, TileR, Dk)
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        oProd.frags[n][m].frag[v] =
          s.frags[n][m].frag[v] * (q32.frags[n][m].frag[v] * scale)
  var oVec: rv(float32, TileR, Dk)
  oVec.row_sum(oProd)
  let oVal = oVec.data[0]

  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(APPLE_8x8x8_F32.getLayoutA(), (lane, 0)).toIntVal()
  let rowIn = cell mod 8
  let colIn = cell div 8
  if colIn == 0:
    y[yLin + dvBlock * 8 + int32(rowIn)] = oVal.float16
  glState.storeTile(s, (headLin, 0, dvBlock, 0))

proc gdnDecodeStepTileF16*(
    state: ptr UncheckedArray[float32],   # (B·Hv, Dv, Dk) f32, in place
    y: ptr UncheckedArray[float16],       # (B·Hv, Dv) fp16 core output
    k: ptr UncheckedArray[float16],       # (B·Hk, Dk) fp16, post-l2norm
    q: ptr UncheckedArray[float16],       # (B·Hk, Dk) fp16, post-l2norm
    v: ptr UncheckedArray[float16],       # (B·Hv, Dv) fp16
    g: ptr UncheckedArray[float32],       # (B·Hv,) f32 log decay
    beta: ptr UncheckedArray[float16],    # (B·Hv,) fp16, one per value head
    Hv, Hk, hkRatio: int32,
    Dk, Dv, TileR: static int) {.device.} =
  ## Grid-driven form of `gdnDecodeStepTileF16At`, the caller's `metal:` entry wraps this proc.
  ## - grid (Dv div TileR, B·Hv), one (bh, TileR-row) state tile per threadgroup, TileR = 8
  let dvBlock = int32(threadgroup_position_in_grid.x)
  let bh = int32(threadgroup_position_in_grid.y)
  gdnDecodeStepTileF16At(state, y, k, q, v, g, beta, Hv, Hk, hkRatio,
    dvBlock, bh, Dk, Dv, TileR)
