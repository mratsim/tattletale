# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ─────────────────────────────────────────────────────────────────────
# ─── MoE decode slot-group walk (moe_fwd_decode_at) ──────────────────
# ─────────────────────────────────────────────────────────────────────

## Decode-regime routed expert-body walk on the ceramic Tile API,
## one (token, slot) threadgroup recomputing the router in-group,
## the expert's gate_up rows into `h_scratch`, the down projection into the fp32 partial row.
##
## Slot groups plus the merge launch (in `moe_router`) compose
## the full MoE decode pass, the megakernel composes this core inline.
##
  ## | contract   | value                                                                                              |
  ## | ---------- | -------------------------------------------------------------------------------------------------- |
  ## | router     | the `moeRoute` softmax form only, logits round to El, softmax + top-K in fp32, weights round to El |
  ## | storage    | bfloat16, the decode composition's production dtype                                                |
  ## | partials   | row t·(K+1)+slot holds w[slot]·down(t, slot), slot < K, row t·(K+1)+K holds gateVal·shared_down    |
  ## | partials 2 | the merge launch applies the single El round to the shared contribution                            |
  ## | buffers    | no-copy page-aligned host memory with page-multiple byte lengths                                   |

import math_consts
import workspace/crucible
import workspace/ceramic
import ./moe_router

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# ─── Module-local bf16 row-bounded tile load/store ────────────────────
# tile_io_rows ships fp16 variants only. The bf16 guards live
# module-local (the silu_and_mul and paged_attn precedent).

proc loadTileRowsBf16[R, C: static int; A: static MmaAtom](
    tile: var RtLeft[bfloat16, R, C, A],
    gl: GlView[bfloat16],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded loadTile for bf16 tiles. Tile plane rows origin[2]·R + r at or above `rowLimit`
  ## are zero-filled, not read.
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  let o = (int(origin[0]), int(origin[1]), int(origin[2]), int(origin[3]))
  let src = local_tile_dyn(gl, R, C, o)
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        if int32(origin[2]) * int32(R) + int32(n * M + row) < rowLimit:
          tile.frags[n][m].frag[v] = src[row + n * M, col + m * N + v]
        else:
          tile.frags[n][m].frag[v] = (0.0'f32).bfloat16

proc storeTileRowsBf16[R, C: static int; A: static MmaAtom](
    gl: GlView[bfloat16],
    tile: RtLeft[bfloat16, R, C, A],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded storeTile for bf16 tiles. Tile plane rows origin[2]·R + r at or above `rowLimit`
  ## are not written.
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let col = cell div M
  let o = (int(origin[0]), int(origin[1]), int(origin[2]), int(origin[3]))
  var dst = local_tile_dyn(gl, R, C, o)
  for n in 0 ..< rowTiles:
    if int32(origin[2]) * int32(R) + int32(n * M + row) < rowLimit:
      for m in 0 ..< colTiles:
        for v in 0 ..< vpt:
          dst[row + n * M, col + m * N + v] = tile.frags[n][m].frag[v]

proc loadRowsE[R, C: static int; A: static MmaAtom](
    tile: var RtLeft[bfloat16, R, C, A],
    gl: GlView[bfloat16],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded loadTile for the bfloat16 storage element.
  loadTileRowsBf16(tile, gl, origin, rowLimit)

proc storeRowsE[R, C: static int; A: static MmaAtom](
    gl: GlView[bfloat16],
    tile: RtLeft[bfloat16, R, C, A],
    origin: tuple,
    rowLimit: int32) {.device.} =
  ## Row-bounded storeTile for the bfloat16 storage element.
  storeTileRowsBf16(gl, tile, origin, rowLimit)

# ─── Local device extensions: the activation and partial arithmetic ──

proc siluMulElemEager[R, C: static int; A: static MmaAtom](
    dst: var RtLeft[bfloat16, R, C, A],
    gHalf, uHalf: RtLeft[float32, R, C, A]) {.device.} =
  ## Expert activation, `dst[r][c] = bf16(silu(gHalf[r][c]) · uHalf[r][c])` over
  ## the fp32 g/u accumulator operands. The frag walk follows the loadTile
  ## lane→element mapping, the operands agreeing elementwise.
  ##
  ## Eager name separates the two `siluMulElem` contracts.
  ##
  ## | proc                               | silu operand at the multiply |
  ## | ---------------------------------- | ---------------------------- |
  ## | `siluMulElemEager` (this module)   | the bf16-rounded silu        |
  ## | `o_norm_gated.nim`'s `siluMulElem` | the unrounded f32 silu       |
  ##
  ## Rounding, per storage element:
  ## - the silu result rounds to bf16 (RNE)
  ## - the bf16-rounded silu times the fp32 up operand rounds once at the store
  ##   (the eager chain)
  const rowTiles = R div A.getM()
  const colTiles = C div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let g = gHalf.frags[n][m].frag[v]
        let s = g / (1.0'f32 + exp2(-g * Log2e))
        dst.frags[n][m].frag[v] =
          (s.bfloat16.float32 * uHalf.frags[n][m].frag[v]).bfloat16

proc storeRowsScaledF32[R, C: static int; RT: static int; A: static MmaAtom](
    dst: ptr UncheckedArray[float32],
    tile: RtLeft[float32, R, C, A],
    rowIdx: array[RT, int32],
    rowStride: int32,
    rowS: array[RT, float32],
    colTile: int32) {.device.} =
  ## Per-row scaled fp32 store, each fp32 partial row at one uniform scale.
  ##
  ## Callers load the operand rows with `rowLimit = 1`, so accumulator
  ## row 0 carries the projection's value and the rows above it are exact zeros.
  ##
  ## - the store guard requires `row == 0`, exactly one lane per stored element
  ## - the GDN y store keeps the same single-writer spelling
  ##
  ## | element (c, v) of tile row n                       | written when                    |
  ## | -------------------------------------------------- | ------------------------------- |
  ## | dst[rowIdx[n]·rowStride + colTile·C + m·N + c + v] | `row == 0` and `rowIdx[n] >= 0` |
  ## | stored value                                       | rowS[n]·tile value              |
  ## | rows with rowIdx[n] < 0                            | not written                     |
  static:
    doAssert RT == R div A.getM()
  const M = A.getM()
  const N = A.getN()
  const rowTiles = R div M
  const colTiles = C div N
  const vpt = A.getVpt()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let r = cell mod M
  let c = cell div M
  for n in 0 ..< rowTiles:
    if rowIdx[n] >= 0 and r == 0:
      for m in 0 ..< colTiles:
        for v in 0 ..< vpt:
          dst[int(rowIdx[n]) * int(rowStride) + int(colTile) * C +
              m * N + c + v] = rowS[n] * tile.frags[n][m].frag[v]

# ─── The decode slot-group walk ───────────────────────────────────────

proc moe_fwd_decode_at*[H, E, K, I: static int; Scale: static float32;
    SharedGate: static bool](
    partial: ptr UncheckedArray[float32],  # (num_tokens, K+1, H) fp32 partials
    x, router_w, gate_up_w, down_w: ptr UncheckedArray[bfloat16],
    shared_gate_w, shared_up_w, shared_down_w: ptr UncheckedArray[bfloat16],
    shared_gate_vec_w: ptr UncheckedArray[bfloat16] = nil,
        # (1, H), read only when SharedGate
        # non-null is the caller's obligation whenever SharedGate is true
    h_scratch: ptr UncheckedArray[bfloat16],     # (num_tokens, K, I) working buffer
    hs_scratch: ptr UncheckedArray[bfloat16],    # (num_tokens, I) working buffer
    t, y: int32) {.device.} =
  ## One (token, slot) pair's decode walk, `t` the token, `y` the slot group, routed y < K, the shared group y = K.
  ## `moe_fwd_decode` is the grid-driven wrapper, the megakernel composes this core inline.
  ##
  ## | stage        | behavior                                                                                                                                                   |
  ## | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
  ## | routed y < K | recompute the `moeRoute` router in-group, walk expert ids[y]'s gate_up rows into h_scratch[t, y] and the down projection, store the fp32 partial w[y]·down |
  ## | shared y = K | recompute the shared-expert sigmoid scalar, walk the shared expert into hs_scratch[t], store gateVal·shared_down                                           |
  ## | partials     | the module header's partial-buffer contract, the merge launch applies the single El round                                                                  |
  ##
  ## Instantiation contract:
  ## - each static binding set of this core needs its own call-site line
  ##
  ## shape preconditions, each one a static assert below:
  ##
  ## | precondition  | a violated shape's failure                                             |
  ## | ------------- | ---------------------------------------------------------------------- |
  ## | H mod 16 == 0 | columns silently dropped from every mma dot                            |
  ## | H mod 32 == 0 | the down walk's 32-wide column tiles silently truncate H               |
  ## | I mod 32 == 0 | h_scratch rows left unwritten, stale values re-read on the next launch |
  ## | E mod 64 == 0 | the 64-expert router chunk mis-tiles                                   |
  static:
    doAssert H mod 16 == 0,
      "moe_fwd_decode_at: H must be a multiple of the 16-wide K step"
    doAssert H mod 32 == 0,
      "moe_fwd_decode_at: H must be a multiple of the 32-wide down walk"
    doAssert I mod 32 == 0,
      "moe_fwd_decode_at: I must be a multiple of the 32-wide output tile"
    doAssert E mod 64 == 0,
      "moe_fwd_decode_at: E must be a multiple of the 64-expert router chunk"

  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (H, 0, H, 1))
  let glGu = gate_up_w.gd(shape = (-1, -1, -1, -1), stride = (2 * I * H, 0, H, 1))
  let glDown = down_w.gd(shape = (-1, -1, -1, -1), stride = (H * I, 0, I, 1))
  let glSg = shared_gate_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, H, 1))
  let glSu = shared_up_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, H, 1))
  let glSd = shared_down_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, I, 1))
  let glH = h_scratch.gd(shape = (-1, -1, -1, -1), stride = (I, I, I, 1))
  let glHs = hs_scratch.gd(shape = (-1, -1, -1, -1), stride = (I, 0, I, 1))

  var gHalf: rt_l(float32, 32, 32, getTileConfig(float32, bfloat16))
  var uHalf: rt_l(float32, 32, 32, getTileConfig(float32, bfloat16))
  var h16: rt_l(bfloat16, 32, 32)
  var d: rt_l(float32, 32, 32, getTileConfig(float32, bfloat16))
  var a: rt_l(bfloat16, 32, 16)
  var b16: rt_r(bfloat16, 16, 32)

  if y < K:
    var ids: array[K, int32]
    var w: array[K, float32]
    moeRoute[bfloat16, H, E, K, Scale](x, router_w, t, ids, w)
    # ── gate/up walk for ids[y] -> h_scratch[t, y] ──
    for nt in 0'i32 ..< I div 32:
      gHalf.zero()
      uHalf.zero()
      for kk in 0'i32 ..< H div 16:
        a.loadRowsE(glX, (t, 0, 0, kk), 1)
        b16.loadTile(glGu, (ids[y], 0, nt, kk))
        gHalf.mma_AB(a, b16)
        b16.loadTile(glGu, (ids[y], 0, nt + I div 32, kk))
        uHalf.mma_AB(a, b16)
      h16.siluMulElemEager(gHalf, uHalf)
      glH.storeRowsE(h16, (t * K + y, 0, 0, nt), 1)
    # ── threadgroup barrier ──
    # the down walk re-reads the whole threadgroup's stored scratch rows
    # from device memory, the barrier ordering that cross-lane read
    # after the stores (mem_device, the scratch rows in device memory)
    {.emit: """
    threadgroup_barrier(mem_flags::mem_device);
    """.}
    # ── down walk -> partial[t, y] = w[y]·down ──
    for nt in 0'i32 ..< H div 32:
      d.zero()
      for kk in 0'i32 ..< I div 16:
        a.loadRowsE(glH, (t * K + y, 0, 0, kk), 1)
        b16.loadTile(glDown, (ids[y], 0, nt, kk))
        d.mma_AB(a, b16)
      var rowIdx = [int32(t * (K + 1) + y), -1'i32, -1'i32, -1'i32]
      var rowS = [w[y], 0.0'f32, 0.0'f32, 0.0'f32]
      storeRowsScaledF32(partial, d, rowIdx, int32(H), rowS, nt)
  else:
    # ── shared gate scalar (the moe_fwd chain's rounding form) ──
    var gateVal = 1.0'f32
    when SharedGate:
      let l32 = sharedGateLogit[bfloat16, H](x, shared_gate_vec_w, t)
      gateVal = (1.0'f32 / (1.0'f32 +
        exp2(-l32 * Log2e))).bfloat16.float32
    # ── shared expert activation -> hs_scratch[t] ──
    for nt in 0'i32 ..< I div 32:
      gHalf.zero()
      uHalf.zero()
      for kk in 0'i32 ..< H div 16:
        a.loadRowsE(glX, (t, 0, 0, kk), 1)
        b16.loadTile(glSg, (0, 0, nt, kk))
        gHalf.mma_AB(a, b16)
        b16.loadTile(glSu, (0, 0, nt, kk))
        uHalf.mma_AB(a, b16)
      h16.siluMulElemEager(gHalf, uHalf)
      glHs.storeRowsE(h16, (t, 0, 0, nt), 1)
    # ── threadgroup barrier ──
    # the down walk re-reads the whole threadgroup's stored scratch rows
    # from device memory, the barrier ordering that cross-lane read
    # after the stores (mem_device, the scratch rows in device memory)
    {.emit: """
    threadgroup_barrier(mem_flags::mem_device);
    """.}
    # ── shared down walk -> partial[t, K] = gateVal·shared_down ──
    for nt in 0'i32 ..< H div 32:
      d.zero()
      for kk in 0'i32 ..< I div 16:
        a.loadRowsE(glHs, (t, 0, 0, kk), 1)
        b16.loadTile(glSd, (0, 0, nt, kk))
        d.mma_AB(a, b16)
      var rowIdx = [int32(t * (K + 1) + K), -1'i32, -1'i32, -1'i32]
      var rowS = [gateVal, 0.0'f32, 0.0'f32, 0.0'f32]
      storeRowsScaledF32(partial, d, rowIdx, int32(H), rowS, nt)

proc moe_fwd_decode*[H, E, K, I: static int; Scale: static float32;
    SharedGate: static bool](
    partial: ptr UncheckedArray[float32],  # (num_tokens, K+1, H) fp32 partials
    x, router_w, gate_up_w, down_w: ptr UncheckedArray[bfloat16],
    shared_gate_w, shared_up_w, shared_down_w: ptr UncheckedArray[bfloat16],
    shared_gate_vec_w: ptr UncheckedArray[bfloat16] = nil,
        # (1, H), read only when SharedGate
        # non-null is the caller's obligation whenever SharedGate is true
    h_scratch: ptr UncheckedArray[bfloat16],     # (num_tokens, K, I) working buffer
    hs_scratch: ptr UncheckedArray[bfloat16]) {.device.} =  # (num_tokens, I) buffer
  ##  Grid-driven form of `moe_fwd_decode_at`:
  ##    grid (num_tokens, K+1, 1)
  ## at 32 lanes, one (token, slot) pair per threadgroup.
  let t = int32(threadgroup_position_in_grid.x)
  let y = int32(threadgroup_position_in_grid.y)
  moe_fwd_decode_at[H, E, K, I, Scale, SharedGate](
    partial, x, router_w, gate_up_w, down_w,
    shared_gate_w, shared_up_w, shared_down_w, shared_gate_vec_w,
    h_scratch, hs_scratch, t, y)
