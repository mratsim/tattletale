## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     MLA latent projection forward (mla_latent_fwd): Tile API port
#
# ############################################################

## MLA latent projection forward on the ceramic Tile API: the
## Glm4MoeLiteAttention low-rank chain rebuilds the per-head q/k/v
## states from the fp16 hidden rows and writes them for the SDPA core.
## Experimental: not a production kernel, known gaps below, not fixed.
##
## The chain per token t (fp16 inputs, fp32 accumulation, one fp16
## RNE round at every state store. The model's RMSNorm does ONE
## rounding, unlike qk_norm_rope's two-rounding form):
##
##   qa  = qa_w @ x                     (768)
##   qn  = rms_norm(qa, 768, qa_g)      -> fp16
##   q   = qb_w @ qn                    (num_heads · 256)
##   kv  = kva_w @ x                    (576 = 512 k_pass + 64 k_rot)
##   kp  = rms_norm(kv[0:512], kva_g)   -> fp16
##   kb  = kvb_w @ kp                   (num_heads · 448)
##   per head: q_nope = q[0:192], q_rot = q[192:256]
##             k_nope = kb[0:192], v = kb[192:448]
##   rope (interleaved, below) on q_rot and the shared k_rot
##   store qo = cat(q_nope, q_rot), ko = cat(k_nope, k_rot), vo = v,
##   each (num_tokens, num_heads, 256) fp16.
##
## Interleaved rope (the model's apply_rotary_pos_emb_interleave,
## rope_interleave = true): the 64-dim rot part rotates in ADJACENT
## pairs (2i, 2i+1), i in [0, 32), by pos · theta^(−2i/64) with theta
## = 1e6. The cos/sin tables are (num_tokens, 64) fp32 with entry c
## equal to cos/sin of freq (c mod 32) (the model's emb =
## cat((freqs, freqs))). The kernel reads table col i for pair i.
## k_rot is computed once per token and replicated across heads (the
## model's `k_rot.expand`).
##
## Grid (ceil(num_tokens/8), num_heads div HBLK, 1), 32 lanes, one
## 8-token × HBLK-head block per threadgroup. HBLK = 2: the kv_b
## accumulator alone is 8·(2·448) fp32 = 224 registers per lane. Do
## not raise without re-checking the register budget. The shared
## projections (qa, qn, kv, krot, kp) compute once per threadgroup,
## the per-head parts for the block's HBLK heads.
##
## GEMM structure: the exl3 K-loop with mma_AB into fp32 accumulators.
## The qa/kv chains step 16-wide x tiles over 2048 (128 steps) against
## 16×32 weight tiles. The q/kv_b chains step the 8×32 activation
## tiles over 768/512 (24/16 steps) against 32×32 weight tiles. Every
## output width is an exact 32-multiple (768 = 24×32, 576 = 18×32,
## 448 = 14×32), so no tail tiles.
##
## Local device procs: normRound16 (the one-rounding RMSNorm
## epilogue) and ropeTile32 (the interleaved adjacent-pair rotation)
## live in this module. The row-bounded load/store come from
## tile_io_rows (positron local extension).
##
## Known production gaps (documented, not fixed):
##   - the states feed the SDPA core with scale 0.0625
##     (= qk_head_dim^−0.5, a consumer concern, not applied here)
##   - prefill contract: num_tokens = the kv length, no cache or
##     position offsets. num_tokens must be a multiple of 8 for the
##     k/v store (rows beyond it are zero-filled and skipped)
##   - x and the weights re-read from global per projection (no
##     threadgroup staging)

import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# The real GLM-4.7-Flash dims, baked as module constants. The kernel is
# non-generic: its tile types cannot take the rt_l/rv default atoms
# (those call getTileConfig, which asserts a metal:/cuda: block context,
# and a non-generic proc body is typechecked on the host import). The
# explicit universal-atom enum members below are exactly what the
# defaults resolve to on the Metal and CUDA backends.

const
  HiddenDim = 2048      # the model hidden size
  QLoRARank = 768       # the q_a projection width
  KvLoRARank = 512      # the kv_a k_pass width
  QkRopeDim = 64        # the rotary head dim
  QkNopeDim = 192       # the non-rotary q/k head dim
  QkHeadDim = 256       # qk_nope + qk_rope, the stored q/k head width
  VHeadDim = 256        # the stored v head width
  KvAOut = 576          # KvLoRARank + QkRopeDim, the kv_a width
  KvBPerHead = 448      # QkNopeDim + VHeadDim, the per-head kv_b width
  RmsEps = 1e-5'f32     # rms_norm_eps

# ═════════════════════════════════════════════════════════════════════
#  Local device extensions: the fusion's per-element arithmetic
#  ═════════════════════════════════════════════════════════════════════

proc normRound16[A: static MmaAtom](
    dst: var RtLeft[float16, 8, 32, A],
    src: RtLeft[float32, 8, 32, A],
    gamma: RtLeft[float16, 8, 32, A],
    rowVals: Tensor[float32, (Int[1], Int[2]), (Int[2], Int[1])]) {.device.} =
  ## `dst[r][c] = fp16(src[r][c] · rowVals[r] · gamma[r][c])`: the MLA
  ## RMSNorm epilogue with ONE fp16 RNE round at the end (the model's
  ## rounding, not the two-round qk_norm_rope form). `rowVals` is the
  ## row-rsqrt'ed variance col-vec (fp32 variance mean(x²) + eps). The
  ## frag walk follows the loadTile lane→element mapping, so src,
  ## gamma and dst agree elementwise.
  const rowTiles = 8 div A.getM()
  const colTiles = 32 div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        let x = src.frags[n][m].frag[v] * rowVals.data[n * vpt + v]
        dst.frags[n][m].frag[v] =
          (x * gamma.frags[n][m].frag[v].to(float32)).to(float16)

proc ropeTile32[A: static MmaAtom](
    tile: var RtLeft[float32, 8, 32, A],
    cosT, sinT: ptr UncheckedArray[float32],
    t0, pairOff: int32) {.device.} =
  ## In-place interleaved rope over one (8, 32) fp32 rot half-tile:
  ## each lane owns one adjacent pair (the AC layout col pairs), so
  ## the rotation needs no cross-lane data. The cos/sin for pair index
  ## `pairOff + 4m + (cell div 16)` come from the (num_tokens, 64)
  ## tables' col i (entry c = freq (c mod 32)). `t0` is the tile's
  ## first token row. `pairOff` is 0 for the low half (pairs 0..15)
  ## and 16 for the high half (pairs 16..31).
  const M = A.getM()
  const colTiles = 32 div A.getN()
  let lane = int(thread_index_in_threadgroup)
  let cell = crd2idx(A.getLayoutA(), (lane, 0)).toIntVal()
  let row = cell mod M
  let pairInFrag = cell div 16
  for m in 0 ..< colTiles:
    let i = pairOff + int32(4 * m + pairInFrag)
    let c = cosT[(t0 + int32(row)) * 64 + i]
    let s = sinT[(t0 + int32(row)) * 64 + i]
    let v0 = tile.frags[0][m].frag[0]
    let v1 = tile.frags[0][m].frag[1]
    tile.frags[0][m].frag[0] = v0 * c - v1 * s
    tile.frags[0][m].frag[1] = v1 * c + v0 * s

# ═════════════════════════════════════════════════════════════════════
#  The kernel
#  ═════════════════════════════════════════════════════════════════════

proc mla_latent_fwd*(
    qo: ptr UncheckedArray[float16],   # (num_tokens, num_heads, 256) fp16
    ko: ptr UncheckedArray[float16],   # (num_tokens, num_heads, 256) fp16
    vo: ptr UncheckedArray[float16],   # (num_tokens, num_heads, 256) fp16
    x: ptr UncheckedArray[float16],    # (num_tokens, 2048) fp16
    qa_w: ptr UncheckedArray[float16], # (768, 2048) fp16 row-major (out, in)
    qa_g: ptr UncheckedArray[float16], # (768) fp16 RMSNorm weight
    qb_w: ptr UncheckedArray[float16], # (num_heads·256, 768) fp16
    kva_w: ptr UncheckedArray[float16],# (576, 2048) fp16
    kva_g: ptr UncheckedArray[float16],# (512) fp16 RMSNorm weight
    kvb_w: ptr UncheckedArray[float16],# (num_heads·448, 512) fp16
    cos_t, sin_t: ptr UncheckedArray[float32],  # (num_tokens, 64) fp32
    num_tokens, num_heads: int32) {.device.} =
  ## Computes the module doc's contract for one 8-token × 2-head
  ## block: the qa/qn/q and kv/krot/kp/kb chains, the interleaved rope
  ## on the in-register rot tiles, and the q/k/v stores (fp16 RNE at
  ## every state store). The q/k/v views carry the (token, head, dim)
  ## strides. The qb_w/kvb_w views carry the head block in the batch
  ## component (head0 · 256 or head0 · 448 rows), the rest are the
  ## transposed B-operand views of the row-major (out, in) buffers.
  let tBlock = int32(threadgroup_position_in_grid.x)
  let hBlock = int32(threadgroup_position_in_grid.y)
  let head0 = hBlock * 2

  let glQo = qo.gd(shape = (-1, -1, -1, -1), stride = (num_heads * 256, 256, num_heads * 256, 1))
  let glKo = ko.gd(shape = (-1, -1, -1, -1), stride = (num_heads * 256, 256, num_heads * 256, 1))
  let glVo = vo.gd(shape = (-1, -1, -1, -1), stride = (num_heads * 256, 256, num_heads * 256, 1))
  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (2048, 0, 2048, 1))
  let glQaW = qa_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 2048, 1))
  let glQaG = qa_g.gd(shape = (-1, -1, -1, -1), stride = (0, 0, 0, 1))
  let glQbW = qb_w.gd(shape = (-1, -1, -1, -1), stride = (256 * 768, 0, 768, 1))
  let glKvaW = kva_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 2048, 1))
  let glKvaG = kva_g.gd(shape = (-1, -1, -1, -1), stride = (0, 0, 0, 1))
  let glKvbW = kvb_w.gd(shape = (-1, -1, -1, -1), stride = (448 * 512, 0, 512, 1))

  # The accumulator arrays: 32-wide N-tiles over the threadgroup's
  # 8 tokens × 2 heads. Register budget per lane (fp32 units): qa 192,
  # q 128, kv 144, kb 224, plus the qn/kp fp16 state and the B tiles.
  var qaAcc: array[24, rt_l(float32, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]
  var qnT: array[24, rt_l(float16, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]
  var qAcc: array[16, rt_l(float32, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]
  var kvAcc: array[18, rt_l(float32, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]
  var kpT: array[16, rt_l(float16, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]
  var kbAcc: array[28, rt_l(float32, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]

  var a: rt_l(float16, 8, 16, UNIVERSAL_8x8x8_F32F16F16F32)
  var b: rt_r(float16, 16, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var b32: rt_r(float16, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var sq: rt_l(float32, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var ss: rv(float32, 8, 8, UNIVERSAL_8x8x8_F32F16F16F32)
  var gammaT: rt_l(float16, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)

  # ── qa = qa_w @ x, then the one-rounding RMSNorm over 768 ──
  for nt in 0 ..< 24:
    qaAcc[nt].zero()
  for kk in 0'i32 ..< HiddenDim div 16:
    a.loadTileRows(glX, (0, 0, tBlock, kk), num_tokens)
    for nt in 0'i32 ..< 24:
      b.loadTile(glQaW, (0, 0, nt, kk))
      qaAcc[nt].mma_AB(a, b)
  ss.zero()
  for nt in 0 ..< 24:
    sq.mul(qaAcc[nt], qaAcc[nt])
    ss.row_sum(sq, ss)
  ss.mul(ss, 1.0'f32 / 768.0'f32)
  ss.add(ss, RmsEps)
  ss.rsqrt(ss)
  for nt in 0 ..< 24:
    gammaT.loadTile(glQaG, (0, 0, 0, nt))
    qnT[nt].normRound16(qaAcc[nt], gammaT, ss)

  # ── q = qb_w @ qn (768-wide K, 16 N-tiles = 2 heads × 256) ──
  for nt in 0 ..< 16:
    qAcc[nt].zero()
  for kk in 0'i32 ..< QLoRARank div 32:
    for nt in 0'i32 ..< 16:
      b32.loadTile(glQbW, (head0, 0, nt, kk))
      qAcc[nt].mma_AB(qnT[kk], b32)

  # ── kv = kva_w @ x (576 = 512 k_pass + 64 k_rot) ──
  for nt in 0 ..< 18:
    kvAcc[nt].zero()
  for kk in 0'i32 ..< HiddenDim div 16:
    a.loadTileRows(glX, (0, 0, tBlock, kk), num_tokens)
    for nt in 0'i32 ..< 18:
      b.loadTile(glKvaW, (0, 0, nt, kk))
      kvAcc[nt].mma_AB(a, b)

  # ── kp = rms_norm(kv[0:512], kva_g), one rounding ──
  ss.zero()
  for nt in 0 ..< 16:
    sq.mul(kvAcc[nt], kvAcc[nt])
    ss.row_sum(sq, ss)
  ss.mul(ss, 1.0'f32 / 512.0'f32)
  ss.add(ss, RmsEps)
  ss.rsqrt(ss)
  for nt in 0 ..< 16:
    gammaT.loadTile(glKvaG, (0, 0, 0, nt))
    kpT[nt].normRound16(kvAcc[nt], gammaT, ss)

  # ── kb = kvb_w @ kp (512-wide K, 28 N-tiles = 2 heads × 448) ──
  for nt in 0 ..< 28:
    kbAcc[nt].zero()
  for kk in 0'i32 ..< KvLoRARank div 32:
    for nt in 0'i32 ..< 28:
      b32.loadTile(glKvbW, (head0, 0, nt, kk))
      kbAcc[nt].mma_AB(kpT[kk], b32)

  # ── interleaved rope, in place on the rot accumulator tiles:
  #    kvAcc[16..17] = the shared k_rot (pairs 0..15, 16..31),
  #    qAcc[hh·8 + 6..7] = head hh's q_rot halves ──
  kvAcc[16].ropeTile32(cos_t, sin_t, tBlock * 8, 0)
  kvAcc[17].ropeTile32(cos_t, sin_t, tBlock * 8, 16)
  for hh in 0 ..< 2:
    qAcc[hh * 8 + 6].ropeTile32(cos_t, sin_t, tBlock * 8, 0)
    qAcc[hh * 8 + 7].ropeTile32(cos_t, sin_t, tBlock * 8, 16)

  # ── stores: the nope parts straight from the accumulators, the
  #    roped rot tiles at the head's cols [192, 256), the shared
  #    k_rot at both heads' rot cols ──
  for hh in 0'i32 ..< 2:
    let head = head0 + hh
    for nt in 0'i32 ..< 6:
      glQo.storeTileRows(qAcc[hh * 8 + nt], (0, head, tBlock, nt), num_tokens)
    glQo.storeTileRows(qAcc[hh * 8 + 6], (0, head, tBlock, 6), num_tokens)
    glQo.storeTileRows(qAcc[hh * 8 + 7], (0, head, tBlock, 7), num_tokens)
    for nt in 0'i32 ..< 6:
      glKo.storeTileRows(kbAcc[hh * 14 + nt], (0, head, tBlock, nt), num_tokens)
    glKo.storeTileRows(kvAcc[16], (0, head, tBlock, 6), num_tokens)
    glKo.storeTileRows(kvAcc[17], (0, head, tBlock, 7), num_tokens)
    for nt in 0'i32 ..< 8:
      glVo.storeTileRows(kbAcc[hh * 14 + 6 + nt], (0, head, tBlock, nt), num_tokens)
