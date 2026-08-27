## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     GGUF attention-forward composition driver (gguf_attn)
#
# ############################################################

## GGUF attention-forward composition driver: one Metal block hosting
## the quantized-linear, qk-norm+rope and paged-attention launchers,
## plus the host `ggufAttnForward` forward, one `engine.run` per
## kernel: q/k/v projections, qk-norm+rope, the host cache write
## (write-before staging, decode rows at cache_seqlen - 1, prefill rows
## at cache_seqlen + j), paged attention, o_proj. Requires H % 8 == 0,
## Nkv % 8 == 0 and a power-of-two pageSize (the write's shift/mask).

import std/[math, bitops]
import workspace/crucible
import ./gguf_linear_fwd
import ./qk_norm_rope
import ./paged_attn

# ═════════════════════════════════════════════════════════════════════
#  The thin {.global.} launchers, one per GGufScheme for the quantized
#  linear plus the qk-norm+rope and paged attention forwards at D =
#  128. Each launcher's first param is the output buffer: engine.run
#  binds the separate outBuf argument to it and the args tuple to the
#  rest.
#  ═════════════════════════════════════════════════════════════════════

const ggufAttnMsl* = metal:
  proc ggufLinearQ8(
      Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      w: ptr UncheckedArray[uint8], M, K, N, rowBytes: int32) {.global.} =
    gguf_linear_fwd(Out, x, w, M, K, N, rowBytes, 0)

  proc ggufLinearQ4K(
      Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      w: ptr UncheckedArray[uint8], M, K, N, rowBytes: int32) {.global.} =
    gguf_linear_fwd(Out, x, w, M, K, N, rowBytes, 1)

  proc ggufLinearIQ4XS(
      Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      w: ptr UncheckedArray[uint8], M, K, N, rowBytes: int32) {.global.} =
    gguf_linear_fwd(Out, x, w, M, K, N, rowBytes, 2)

  proc ggufQkNormRopeD128(
      Out: ptr UncheckedArray[float16], X, G: ptr UncheckedArray[float16],
      Cos, Sin: ptr UncheckedArray[float32],
      xTokenStride, cosTokenStride, headBlocks, xColBase: int32,
      eps: float32) {.global.} =
    qk_norm_rope_fwd(Out, X, G, Cos, Sin, xTokenStride, cosTokenStride,
                     headBlocks, xColBase, eps)

  proc ggufPagedD128(
      o, q: ptr UncheckedArray[float16],
      k_cache, v_cache: ptr UncheckedArray[float16],
      block_table, cache_seqlens, cu_seqlens_q: ptr UncheckedArray[int32],
      num_seqs, H, Nkv, max_pages, page_size: int32) {.global.} =
    paged_attn_fwd(o, q, k_cache, v_cache, block_table, cache_seqlens,
                   cu_seqlens_q, num_seqs, H, Nkv, max_pages, page_size, 128)

# ═════════════════════════════════════════════════════════════════════
#  The attention-forward parameters and result
#  ═════════════════════════════════════════════════════════════════════

type
  GGufScheme* = enum
    ## The GGUF block-quantization schemes the four projections accept.
    gsQ8_0, gsQ4_K, gsIQ4_XS

  GGufAttnParams* = object
    ## Geometry, tables and weights of one attention-layer forward.
    ## `x`, the token rows, is passed separately to `ggufAttnForward`.
    ## The K/V slabs are in-out: the forward appends the batch's new
    ## rows before attention runs.
    hidden*: int              # x row width = the projections' K
    H*, Nkv*: int             # query / kv head counts (GQA ratio H div Nkv)
    D*: int                   # head dim (128 in this composition)
    pageSize*, maxPages*: int
    eps*, theta*: float32     # the qk-norm epsilon and the RoPE base
    cacheSeqlens*: seq[int32]     # (num_seqs), see the module doc's staging contract
    cuSeqlensQ*: seq[int32]       # (num_seqs+1), the cumulative q-token ranges
    blockTable*: seq[int32]       # (num_seqs, max_pages) dense, -1 padding
    positions*: seq[int32]        # (num_qo_tokens), the per-token positions
    qw*, kw*, vw*, ow*: seq[byte] # the raw GGUF block streams
    qRowBytes*, kRowBytes*, vRowBytes*, oRowBytes*: int32
    qScheme*, kScheme*, vScheme*, oScheme*: GGufScheme
    normGammaQ*, normGammaK*: seq[uint16]  # (D,) fp16 per-dim weights

  GGufAttnResult* = object
    ## Per-launch outputs of the driver, one buffer per kernel stage.
    ## `outBuf`, the final buffer, is the layer output. Sizes are
    ## token-based: num_qo_tokens = cuSeqlensQ[^1].
    outBuf*: seq[uint16]      # (num_qo_tokens, hidden) fp16, the layer output
    qBuf*, kBuf*, vBuf*: seq[uint16]  # the pre-norm projections
    qRope*, kRope*: seq[uint16]       # the qk-norm+rope outputs
    attnOut*: seq[uint16]     # (num_qo_tokens, H, D) fp16, the paged output

# ═════════════════════════════════════════════════════════════════════
#  The shared NEOX cos/sin table builder (see the module doc, the
#  qk_norm_rope kernel's table contract)
#  ═════════════════════════════════════════════════════════════════════

proc buildRopeCosSin*(nTokens, rowsPerToken, D: int; positions: seq[int32];
                      theta: float32): tuple[cosT, sinT: seq[float32]] =
  ## Precomputes the NEOX fp32 cos/sin tables: one (rows, half)
  ## row-major table per signal, rows = nTokens·rowsPerToken, half =
  ## D div 2. Row r of token m (r = m·rowsPerToken + h) carries the
  ## cos/sin of inv_freq[t]·pos with inv_freq[t] = theta^(−2t/D),
  ## the pow in float64 rounded to fp32, the sin/cos in fp32.
  ## The dims pair t and t + half share the value, so the table stores
  ## the half once and the kernel's two half-tiles both read it.
  let half = D shr 1
  var invFreq = newSeq[float32](half)
  for t in 0 ..< half:
    invFreq[t] = float32(pow(float64(theta), -float64(2 * t) / float64(D)))
  let rows = nTokens * rowsPerToken
  result.cosT = newSeq[float32](rows * half)
  result.sinT = newSeq[float32](rows * half)
  for m in 0 ..< nTokens:
    let pos = float32(positions[m])
    for h in 0 ..< rowsPerToken:
      let row = m * rowsPerToken + h
      for t in 0 ..< half:
        let ang = invFreq[t] * pos
        result.cosT[row * half + t] = cos(ang)
        result.sinT[row * half + t] = sin(ang)

func ggufLinearLauncher(scheme: GGufScheme): string =
  ## Metal launcher name for the packed-stream scheme.
  case scheme
  of gsQ8_0: "ggufLinearQ8"
  of gsQ4_K: "ggufLinearQ4K"
  of gsIQ4_XS: "ggufLinearIQ4XS"

func slabOffset(pageId, inPage, h, d: int; pageSize, Nkv, D: int): int =
  ## K/V slab offset of (pageId, inPage, head h, dim d) in the flat
  ## (num_pages, page_size, Nkv, D) layout, head-dim contiguous:
  ## (pageId·pageSize + inPage)·(Nkv·D) + h·D + d.
  (pageId * pageSize + inPage) * (Nkv * D) + h * D + d

# ═════════════════════════════════════════════════════════════════════
#  The layer driver
#  ═════════════════════════════════════════════════════════════════════

proc ggufAttnForward*(engine: var auto, p: GGufAttnParams, x: seq[uint16],
                      kSlab, vSlab: var seq[uint16]): GGufAttnResult =
  ## Runs one attention-layer forward, one `engine.run` per kernel in
  ## the module doc's order: the q/k/v projections over the shared x,
  ## the fused qk-norm+rope on the separate q and k buffers, the cache
  ## write (the module doc's staging contract), the paged attention
  ## over the slabs and the o_proj. `x` is (num_qo_tokens, hidden)
  ## fp16. The slabs are mutated by the append: each seq's roped k and
  ## plain v land at the rows the staging contract fixes. Decode
  ## (q_len == 1) and prefill (q_len >= 2) seqs mix under one
  ## `cu_seqlens_q`. `positions` are per token. Requires H % 8 == 0,
  ## Nkv % 8 == 0 and a power-of-two pageSize (asserted).
  doAssert (p.H and 7) == 0 and (p.Nkv and 7) == 0,
    "the composed q/k views require H % 8 == 0 and Nkv % 8 == 0"
  doAssert p.pageSize >= 1 and (p.pageSize and (p.pageSize - 1)) == 0,
    "the cache write requires a power-of-two pageSize (shift/mask)"
  doAssert p.D == 128, "the qk-norm+rope and paged attention kernels run 128-wide"
  doAssert p.hidden mod 128 == 0,
    "the o_proj grid tiles hidden in 128-column blocks"
  let numSeqs = p.cuSeqlensQ.len - 1
  let nTokens = p.cuSeqlensQ[numSeqs].int
  doAssert nTokens >= 1, "the forward needs at least one query token"
  let nQ = p.H * p.D
  let nKv = p.Nkv * p.D

  # the q/k/v projections (K = hidden, N = H·D / Nkv·D)
  result.qBuf = newSeq[uint16](nTokens * nQ)
  engine.run << (grid: (nQ div 128, (nTokens + 31) div 32, 1),
                 blk: (32, 1)) >> (
    ggufLinearLauncher(p.qScheme), result.qBuf,
    (x, p.qw, int32(nTokens), int32(p.hidden), int32(nQ), p.qRowBytes))
  result.kBuf = newSeq[uint16](nTokens * nKv)
  engine.run << (grid: (nKv div 128, (nTokens + 31) div 32, 1),
                 blk: (32, 1)) >> (
    ggufLinearLauncher(p.kScheme), result.kBuf,
    (x, p.kw, int32(nTokens), int32(p.hidden), int32(nKv), p.kRowBytes))
  result.vBuf = newSeq[uint16](nTokens * nKv)
  engine.run << (grid: (nKv div 128, (nTokens + 31) div 32, 1),
                 blk: (32, 1)) >> (
    ggufLinearLauncher(p.vScheme), result.vBuf,
    (x, p.vw, int32(nTokens), int32(p.hidden), int32(nKv), p.vRowBytes))

  # the fused qk-norm+rope over the separate q/k buffers: the composed
  # (token, head-block) view, q's head-blocks = H div 8, k's = Nkv
  # div 8. The x view's token stride is the buffer row width and the
  # head-column offset is 0 (each buffer holds only its own heads)
  let headBlocksQ = p.H div 8
  let headBlocksK = p.Nkv div 8
  let (cosQ, sinQ) = buildRopeCosSin(nTokens, p.H, p.D, p.positions, p.theta)
  let (cosK, sinK) = buildRopeCosSin(nTokens, p.Nkv, p.D, p.positions, p.theta)
  result.qRope = newSeq[uint16](nTokens * nQ)
  engine.run << (grid: (1, nTokens, headBlocksQ), blk: (32, 1)) >> (
    "ggufQkNormRopeD128", result.qRope,
    (result.qBuf, p.normGammaQ, cosQ, sinQ,
     int32(nQ), int32(p.H), int32(headBlocksQ), int32(0), p.eps))
  result.kRope = newSeq[uint16](nTokens * nKv)
  engine.run << (grid: (1, nTokens, headBlocksK), blk: (32, 1)) >> (
    "ggufQkNormRopeD128", result.kRope,
    (result.kBuf, p.normGammaK, cosK, sinK,
     int32(nKv), int32(p.Nkv), int32(headBlocksK), int32(0), p.eps))

  # the cache write (the write-before staging): each seq's roped k and
  # plain v land at the rows the module doc's staging contract fixes.
  # The page decomposition is shift/mask (pageSize is a power of two).
  let lgPageSize = countTrailingZeroBits(p.pageSize)
  let pageMask = p.pageSize - 1
  # the slabs hold whole pages of (page_size, Nkv, D); the per-token
  # write moves one (Nkv, D) row into both slabs at the same offset,
  # so the k and v slabs must share an equal whole-page capacity
  let slabPageElems = p.pageSize * p.Nkv * p.D
  let slabPageCount = kSlab.len div slabPageElems
  doAssert kSlab.len == vSlab.len,
    "the k and v slabs must hold an equal number of elements"
  doAssert kSlab.len mod slabPageElems == 0,
    "the k slab length must be a whole number of pages"
  for s in 0 ..< numSeqs:
    let qLen = (p.cuSeqlensQ[s + 1] - p.cuSeqlensQ[s]).int
    let q0 = p.cuSeqlensQ[s].int
    let writeStart = p.cacheSeqlens[s].int - (if qLen == 1: 1 else: 0)
    for j in 0 ..< qLen:
      let row = writeStart + j
      doAssert row >= 0, "the cache write row must be non-negative"
      let pageIdx = row shr lgPageSize
      doAssert pageIdx < p.maxPages, "the cache write row exceeds the block-table budget"
      let inPage = row and pageMask
      let pageId = p.blockTable[s * p.maxPages + pageIdx].int
      doAssert pageId >= 0, "cache write hit an unused block_table slot"
      doAssert pageId < slabPageCount,
        "cache write hit a pageId beyond the slab's page capacity"
      # each token's Nkv·D span is one contiguous move on both sides:
      # the slab row (pageId, inPage) and the kRope/vBuf token row are
      # both head-dim-contiguous (slabOffset's h·D + d layout)
      let t = q0 + j
      let dstBase = slabOffset(pageId, inPage, 0, 0, p.pageSize, p.Nkv, p.D)
      copyMem(addr kSlab[dstBase], addr result.kRope[t * nKv], nKv * 2)
      copyMem(addr vSlab[dstBase], addr result.vBuf[t * nKv], nKv * 2)

  # the paged attention's x extent: the batch's longest q_len in 8-row
  # q blocks (the kernel zero-fills the blocks beyond a seq's own q_len)
  var xBlocks = 1
  for s in 0 ..< numSeqs:
    let qLen = p.cuSeqlensQ[s + 1] - p.cuSeqlensQ[s]
    xBlocks = max(xBlocks, (qLen.int + 7) div 8)
  result.attnOut = newSeq[uint16](nTokens * nQ)
  engine.run << (grid: (xBlocks, p.H, numSeqs), blk: (32, 1)) >> (
    "ggufPagedD128", result.attnOut,
    (result.qRope, kSlab, vSlab, p.blockTable, p.cacheSeqlens, p.cuSeqlensQ,
     int32(numSeqs), int32(p.H), int32(p.Nkv),
     int32(p.maxPages), int32(p.pageSize)))

  # the o_proj over the attention output (num_qo_tokens, H·D) → hidden
  result.outBuf = newSeq[uint16](nTokens * p.hidden)
  engine.run << (grid: (p.hidden div 128, (nTokens + 31) div 32, 1),
                 blk: (32, 1)) >> (
    ggufLinearLauncher(p.oScheme), result.outBuf,
    (result.attnOut, p.ow, int32(nTokens), int32(nQ),
     int32(p.hidden), p.oRowBytes))
