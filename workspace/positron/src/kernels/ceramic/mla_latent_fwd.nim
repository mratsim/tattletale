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

## MLA latent projection forward: the Glm4MoeLiteAttention low-rank chain rebuilds per-head q/k/v states for the SDPA core.
## Experimental: not a production kernel, known gaps below, not fixed.
##
## Dataflow per token (fp16 inputs, fp32 accumulation, fp16 RNE round at every state store):
##
##   q chain:  x --> qa_w @ --> qa --> rms_norm(qa, 768, qa_g) --> qn --> qb_w @ --> q
##   kv chain: x --> kva_w @ --> kv = (kp_part 512, k_rot 64)
##             kv[0:512] --> rms_norm(kv, 512, kva_g) --> kp --> kvb_w @ --> kb
##   per head: q = (q_nope 192, q_rot 64), kb = (k_nope 192, v 256)
##             q_rot, k_rot --> interleaved rope
##   stores:   qo = (q_nope, q_rot), ko = (k_nope, k_rot), vo = v
##             each (num_tokens, num_heads, 256) fp16
##
## Buffers:
##   - qo, ko, vo: (num_tokens, num_heads, 256) fp16 output states
##   - x: (num_tokens, 2048) fp16 hidden rows
##   - qa_w: (768, 2048) fp16, qa_g: (768,) fp16
##   - qb_w: (num_heads·256, 768) fp16
##   - kva_w: (576, 2048) fp16, kva_g: (512,) fp16
##   - kvb_w: (num_heads·448, 512) fp16
##   - cos_t, sin_t: (num_tokens, 64) fp32 rope tables
##
## Semantics:
##   - the interleaved rope rotates the 64-dim rot parts in adjacent pairs
##     (2i, 2i+1), i in [0, 32), by pos·theta^(−2i/64)
##     (theta = 1e6). Table entry c = cos/sin of freq (c mod 32).
##     k_rot computes once per token and replicates across heads.
##   - the RMSNorm rounds once (the model's `weight * x.to(dtype)`)
##
## Known production gaps (documented, not fixed):
##   - the states feed the SDPA core with scale 0.0625
##     (qk_head_dim^−0.5, a consumer concern, not applied here)
##   - prefill contract: num_tokens = the kv length, no cache
##     or position offsets. num_tokens must be a multiple of 8.
##     The k/v store zero-fills and skips rows beyond it.
##   - x and the weights re-read from global per projection
##     (no threadgroup staging)

import workspace/crucible
import workspace/ceramic
import ./tile_io_rows

export int_tuples, layouts, layout_constructors, layout_indexing, tensors,
       ptr_arithmetic, tile_algebra

# The real GLM-4.7-Flash dims, baked as module constants.
# The kernel is non-generic: its tile types cannot take the rt_l/rv
# default atoms, which call getTileConfig and assert a metal:/cuda:
# block context. A non-generic proc body is typechecked on the host
# import.
# The explicit universal-atom enum members below are exactly what the defaults resolve to on the Metal and CUDA backends.

const
  HiddenDim = 2048      # the model hidden size
  QLoRARank = 768       # the q_a projection width
  KvLoRARank = 512      # the kv_a k_pass width
  QkNopeDim = 192       # the non-rotary q/k head dim
  QkHeadDim = 256       # qk_nope + qk_rope, the stored q/k head width
  VHeadDim = 256        # the stored v head width
  KvAOut = 576          # KvLoRARank + the 64-dim rotary, the kv_a width
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
  ## RMSNorm epilogue with one fp16 RNE round at the end (the model's
  ## `weight * x.to(dtype)` rounding). `rowVals` is the row-rsqrt'ed
  ## variance col-vec (fp32 variance mean(x²) + eps). The frag walk
  ## follows the loadTile lane→element mapping, so src, gamma and dst
  ## agree elementwise.
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
  ## each lane owns one adjacent pair (the AC layout col pairs).
  ## The rotation needs no cross-lane data. The cos/sin for the pair
  ## index `pairOff + 4m + (cell div 16)` come from the tables' col i
  ## (entry c = freq (c mod 32)).
  ## `t0` is the tile's first token row. `pairOff` is 0 for the low
  ## half (pairs 0..15) and 16 for the high half (pairs 16..31).
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
  ## on the in-register rot tiles, and the q/k/v stores.
  ##
  ## Grid (ceil(num_tokens/8), num_heads div 2, 1), 32 lanes.
  ## One 8-token × 2-head block per threadgroup. HBLK = 2 keeps the kv_b
  ## accumulator at 224 fp32 registers per lane, the largest single accumulator.
  ## Raising HBLK requires re-checking the register budget.
  ## The full breakdown is inline at the accumulator declarations.
  ## The shared projections (qa, qn, kv, k_rot, kp) compute once per threadgroup.
  ## The per-head parts follow for the block's 2 heads.
  ##
  ## GEMM structure: the qa/kv chains step 16-wide x tiles over 2048
  ## (128 steps) against 16×32 weight tiles. The q/kv_b chains step
  ## the 8×32 activation tiles over 768/512 (24/16 steps) against
  ## 32×32 weight tiles. Every output width is an exact 32-multiple
  ## (768 = 24×32, 576 = 18×32, 448 = 14×32), so no tail tiles.
  ##
  ## Views: the q/k/v output views carry the (token, head, dim)
  ## strides. The qb_w/kvb_w views carry the head block in the batch
  ## component (head0 · 256 or head0 · 448 rows), the other weights
  ## are transposed B-operand views of the row-major (out, in)
  ## buffers.
  let tBlock = int32(threadgroup_position_in_grid.x)
  let hBlock = int32(threadgroup_position_in_grid.y)
  let head0 = hBlock * 2

  let glQo = qo.gd(shape = (-1, -1, -1, -1), stride = (num_heads * QkHeadDim, QkHeadDim, num_heads * QkHeadDim, 1))
  let glKo = ko.gd(shape = (-1, -1, -1, -1), stride = (num_heads * QkHeadDim, QkHeadDim, num_heads * QkHeadDim, 1))
  let glVo = vo.gd(shape = (-1, -1, -1, -1), stride = (num_heads * VHeadDim, VHeadDim, num_heads * VHeadDim, 1))
  let glX = x.gd(shape = (-1, -1, -1, -1), stride = (2048, 0, 2048, 1))
  let glQaW = qa_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 2048, 1))
  let glQaG = qa_g.gd(shape = (-1, -1, -1, -1), stride = (0, 0, 0, 1))
  let glQbW = qb_w.gd(shape = (-1, -1, -1, -1), stride = (QkHeadDim * QLoRARank, 0, QLoRARank, 1))
  let glKvaW = kva_w.gd(shape = (-1, -1, -1, -1), stride = (1, 0, 2048, 1))
  let glKvaG = kva_g.gd(shape = (-1, -1, -1, -1), stride = (0, 0, 0, 1))
  let glKvbW = kvb_w.gd(shape = (-1, -1, -1, -1), stride = (KvBPerHead * KvLoRARank, 0, KvLoRARank, 1))

  # The accumulator arrays: 32-wide N-tiles over the threadgroup's
  # 8 tokens × 2 heads. Register budget per lane (fp32 units): qa 192,
  # q 128, kv 144, kb 224, plus the qn/kp fp16 state and the B tiles.
  var qaAcc: array[24, rt_l(float32, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]
  var qnT: array[24, rt_l(float16, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]
  var qAcc: array[2 * QkHeadDim div 32, rt_l(float32, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]
  var kvAcc: array[KvAOut div 32, rt_l(float32, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]
  var kpT: array[KvLoRARank div 32, rt_l(float16, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]
  var kbAcc: array[2 * KvBPerHead div 32, rt_l(float32, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)]

  var a: rt_l(float16, 8, 16, UNIVERSAL_8x8x8_F32F16F16F32)
  var b: rt_r(float16, 16, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var b32: rt_r(float16, 32, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var sq: rt_l(float32, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)
  var ss: rv(float32, 8, 8, UNIVERSAL_8x8x8_F32F16F16F32)
  var gammaT: rt_l(float16, 8, 32, UNIVERSAL_8x8x8_F32F16F16F32)

  # ── qa = qa_w @ x, then the one-rounding RMSNorm over 768 ──
  for nt in 0 ..< QLoRARank div 32:
    qaAcc[nt].zero()
  for kk in 0'i32 ..< HiddenDim div 16:
    a.loadTileRows(glX, (0, 0, tBlock, kk), num_tokens)
    for nt in 0'i32 ..< QLoRARank div 32:
      b.loadTile(glQaW, (0, 0, nt, kk))
      qaAcc[nt].mma_AB(a, b)
  ss.zero()
  for nt in 0 ..< QLoRARank div 32:
    sq.mul(qaAcc[nt], qaAcc[nt])
    ss.row_sum(sq, ss)
  ss.mul(ss, 1.0'f32 / float32(QLoRARank))
  ss.add(ss, RmsEps)
  ss.rsqrt(ss)
  for nt in 0 ..< QLoRARank div 32:
    gammaT.loadTile(glQaG, (0, 0, 0, nt))
    qnT[nt].normRound16(qaAcc[nt], gammaT, ss)

  # ── q = qb_w @ qn (768-wide K, 16 N-tiles = 2 heads × 256) ──
  for nt in 0 ..< 2 * QkHeadDim div 32:
    qAcc[nt].zero()
  for kk in 0'i32 ..< QLoRARank div 32:
    for nt in 0'i32 ..< 16:
      b32.loadTile(glQbW, (head0, 0, nt, kk))
      qAcc[nt].mma_AB(qnT[kk], b32)

  # ── kv = kva_w @ x (576 = 512 k_pass + 64 k_rot) ──
  for nt in 0 ..< KvAOut div 32:
    kvAcc[nt].zero()
  for kk in 0'i32 ..< HiddenDim div 16:
    a.loadTileRows(glX, (0, 0, tBlock, kk), num_tokens)
    for nt in 0'i32 ..< KvAOut div 32:
      b.loadTile(glKvaW, (0, 0, nt, kk))
      kvAcc[nt].mma_AB(a, b)

  # ── kp = rms_norm(kv[0:512], kva_g), one rounding ──
  ss.zero()
  for nt in 0 ..< KvLoRARank div 32:
    sq.mul(kvAcc[nt], kvAcc[nt])
    ss.row_sum(sq, ss)
  ss.mul(ss, 1.0'f32 / float32(KvLoRARank))
  ss.add(ss, RmsEps)
  ss.rsqrt(ss)
  for nt in 0 ..< KvLoRARank div 32:
    gammaT.loadTile(glKvaG, (0, 0, 0, nt))
    kpT[nt].normRound16(kvAcc[nt], gammaT, ss)

  # ── kb = kvb_w @ kp (512-wide K, 28 N-tiles = 2 heads × 448) ──
  for nt in 0 ..< 2 * KvBPerHead div 32:
    kbAcc[nt].zero()
  for kk in 0'i32 ..< KvLoRARank div 32:
    for nt in 0'i32 ..< 2 * KvBPerHead div 32:
      b32.loadTile(glKvbW, (head0, 0, nt, kk))
      kbAcc[nt].mma_AB(kpT[kk], b32)

  # ── interleaved rope, in place on the rot accumulator tiles:
  #    kvAcc[16..17] = the shared k_rot (pairs 0..15, 16..31),
  #    qAcc[hh·8 + 6..7] = head hh's q_rot halves ──
  kvAcc[KvLoRARank div 32].ropeTile32(cos_t, sin_t, tBlock * 8, 0)
  kvAcc[KvLoRARank div 32 + 1].ropeTile32(cos_t, sin_t, tBlock * 8, 16)
  for hh in 0 ..< 2:
    qAcc[hh * (QkHeadDim div 32) + QkNopeDim div 32].ropeTile32(cos_t, sin_t, tBlock * 8, 0)
    qAcc[hh * (QkHeadDim div 32) + QkNopeDim div 32 + 1].ropeTile32(cos_t, sin_t, tBlock * 8, 16)

  # ── stores: nope parts from the accumulators.
  #    Roped rot tiles land in the head's cols [192, 256).
  #    The shared k_rot lands in both heads' rot cols ──
  for hh in 0'i32 ..< 2:
    let head = head0 + hh
    let headTiles = QkHeadDim div 32
    let nopeTiles = QkNopeDim div 32
    for nt in 0'i32 ..< nopeTiles:
      glQo.storeTileRows(qAcc[hh * headTiles + nt], (0, head, tBlock, nt), num_tokens)
    glQo.storeTileRows(qAcc[hh * headTiles + nopeTiles], (0, head, tBlock, nopeTiles), num_tokens)
    glQo.storeTileRows(qAcc[hh * headTiles + nopeTiles + 1], (0, head, tBlock, nopeTiles + 1), num_tokens)
    for nt in 0'i32 ..< nopeTiles:
      glKo.storeTileRows(kbAcc[hh * (KvBPerHead div 32) + nt], (0, head, tBlock, nt), num_tokens)
    glKo.storeTileRows(kvAcc[KvLoRARank div 32], (0, head, tBlock, nopeTiles), num_tokens)
    glKo.storeTileRows(kvAcc[KvLoRARank div 32 + 1], (0, head, tBlock, nopeTiles + 1), num_tokens)
    for nt in 0'i32 ..< VHeadDim div 32:
      glVo.storeTileRows(kbAcc[hh * (KvBPerHead div 32) + nopeTiles + nt], (0, head, tBlock, nt), num_tokens)
