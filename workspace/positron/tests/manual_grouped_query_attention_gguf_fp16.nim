## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run command, from the repo root:
## - nim cpp -r --hints:off --warnings:off \
##   --outdir:build/wip --nimcache:nimcache/wip workspace/positron/tests/manual_grouped_query_attention_gguf_fp16.nim

import std/[strformat, options, math, bitops]
import workspace/crucible
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/attn_ssm/grouped_query_attention_gguf
import ./gguf_test_utils

type
  AttnCase = object
    ## Geometry, tables and weights of the checked attention-layer
    ## forward. The host orchestration lives in this test, so every
    ## buffer the reference sees is staged here.
    hidden, H, Nkv, D: int
    pageSize, maxPages: int
    eps, theta: float32
    cacheSeqlens, cuSeqlensQ, blockTable, positions: seq[int32]
    qw, kw, vw, ow: seq[byte]
    qRowBytes, kRowBytes, vRowBytes, oRowBytes: int32
    qScheme, kScheme, vScheme, oScheme: GGufScheme
    normGammaQ, normGammaK: seq[uint16]

  AttnTensors = object
    ## Per-stage fp16 outputs widened to fp32 tensors.
    qBuf, kBuf, vBuf, qRope, kRope, attnOut, outBuf: F.Tensor

proc buildRopeCosSin(nTokens, rowsPerToken, D: int; positions: seq[int32];
                     theta: float32): tuple[cosT, sinT: seq[float32]] =
  ## Test-side NEOX fp32 cos/sin tables, one (rows, half) row-major
  ## table per signal, rows = nTokens·rowsPerToken, half = D div 2:
  ## - row r of token m (r = m·rowsPerToken + h) carries the cos/sin
  ##   of inv_freq[t]·pos with inv_freq[t] = theta^(−2t/D)
  ## - the pow runs in float64, rounded to fp32, the sin/cos in fp32
  ## - the dims pair t and t + half share the value, the table stores
  ##   the half once, the kernel's two half-tiles both read it
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

proc launcher(scheme: GGufScheme): string =
  ## Metal launcher name for the packed-stream scheme.
  case scheme
  of gsQ8_0: "ggufLinearQ8"
  of gsQ4_K: "ggufLinearQ4K"
  of gsIQ4_XS: "ggufLinearIQ4XS"

proc gammaVal(c, seed: int): float32 =
  ## Deterministic fp16-exact norm weight in [0.5, 1.5).
  let k = (7 * c + 11 * seed + 13 * (c div 32)) mod 16
  0.5'f32 + float32(k) / 16.0'f32

proc scaleQ4K(stream: seq[uint8], shift: int): seq[uint8] =
  ## Scales the d and dmin fp16 fields of every 144-byte Q4_K
  ## super-block by 2^-shift (exact in fp16). The test applies it to
  ## the o stream: the native Q4_K scale over the o_proj's K = 2048
  ## yields ~1e3 outputs, where the fp16 output ulp (1.0) alone
  ## exceeds the 5e-3 absolute tolerance. The 2^-12 shift brings the
  ## layer output to O(1). The reference decodes the same packed bytes.
  result = stream
  let sbs = stream.len div 144
  for sb in 0 ..< sbs:
    let base = sb * 144
    for off in [0, 2]:
      let d16 = uint16(result[base + off]) or (uint16(result[base + off + 1]) shl 8)
      let scaled = fp32ToFp16(fp16ToFp32(d16) / float32(1 shl shift))
      result[base + off] = uint8(scaled and 0xFF)
      result[base + off + 1] = uint8(scaled shr 8)

proc worstAbsDiff(a, b: F.Tensor): float32 =
  ## Largest |a[i] - b[i]| over all elements.
  doAssert a.numel() == b.numel(), "worstAbsDiff needs equal element counts"
  let ap = a.contiguous().data_ptr(float32)
  let bp = b.contiguous().data_ptr(float32)
  result = 0.0'f32
  for i in 0 ..< a.numel():
    let d = abs(ap[i] - bp[i])
    if d > result: result = d

proc decodeWeights(scheme: GGufScheme, packed: seq[uint8], K, N: int): seq[uint16] =
  ## (N, K) file-order fp16 reconstruction for the packed scheme.
  case scheme
  of gsQ8_0: decodeWeightsQ8_0(packed, K, N)
  of gsQ4_K: decodeWeightsQ4_K(packed, K, N)
  of gsIQ4_XS: decodeWeightsIQ4_XS(packed, K, N)

proc tensorFromFp16(hs: seq[uint16], rows, cols: int): F.Tensor =
  ## fp16 bit buffer widened to a (rows, cols) fp32 tensor.
  var f = newSeq[float32](hs.len)
  for i in 0 ..< hs.len: f[i] = fp16ToFp32(hs[i])
  toTensor(f).reshape(rows, cols)
proc kernelUnderTest(p: AttnCase, x: seq[uint16], kSlab, vSlab: var seq[uint16]): AttnTensors =
  ## Host orchestration, one `engine.run` per launcher from ggufAttnMsl:
  ## - the q/k/v projections
  ## - the fused qk-norm+rope
  ## - the host cache write, write-before staging, decode row
  ##   cache_seqlen - 1, prefill rows cache_seqlen + j
  ##
  ## Paged attention and the o_proj run after the cache write, each
  ## fp16 stage widened to fp32 tensors.
  doAssert (p.H and 7) == 0 and (p.Nkv and 7) == 0,
    "the composed q/k views require H % 8 == 0 and Nkv % 8 == 0"
  var engine = bkMetal.init()
  engine.ingest(ggufAttnMsl)
  let numSeqs = p.cuSeqlensQ.len - 1
  let nTokens = p.cuSeqlensQ[^1].int
  let nQ = p.H * p.D
  let nKv = p.Nkv * p.D
  # the q/k/v projections (K = hidden, N = H·D / Nkv·D)
  var qBufS = newSeq[uint16](nTokens * nQ)
  engine.run << (grid: (nQ div 128, (nTokens + 31) div 32, 1),
                 blk: (32, 1)) >> (
    launcher(p.qScheme), qBufS,
    (x, p.qw, int32(nTokens), int32(p.hidden), int32(nQ), p.qRowBytes))
  var kBufS = newSeq[uint16](nTokens * nKv)
  engine.run << (grid: (nKv div 128, (nTokens + 31) div 32, 1),
                 blk: (32, 1)) >> (
    launcher(p.kScheme), kBufS,
    (x, p.kw, int32(nTokens), int32(p.hidden), int32(nKv), p.kRowBytes))
  var vBufS = newSeq[uint16](nTokens * nKv)
  engine.run << (grid: (nKv div 128, (nTokens + 31) div 32, 1),
                 blk: (32, 1)) >> (
    launcher(p.vScheme), vBufS,
    (x, p.vw, int32(nTokens), int32(p.hidden), int32(nKv), p.vRowBytes))
  # the fused qk-norm+rope over the separate q/k buffers, the composed
  # (token, head-block) view. q's head-blocks = H div 8, k's = Nkv div 8
  let headBlocksQ = p.H div 8
  let headBlocksK = p.Nkv div 8
  let (cosQ, sinQ) = buildRopeCosSin(nTokens, p.H, p.D, p.positions, p.theta)
  let (cosK, sinK) = buildRopeCosSin(nTokens, p.Nkv, p.D, p.positions, p.theta)
  var qRopeS = newSeq[uint16](nTokens * nQ)
  engine.run << (grid: (1, nTokens, headBlocksQ), blk: (32, 1)) >> (
    "ggufQkNormRopeD128", qRopeS,
    (qBufS, p.normGammaQ, cosQ, sinQ,
     int32(nQ), int32(p.H), int32(headBlocksQ), int32(0), p.eps))
  var kRopeS = newSeq[uint16](nTokens * nKv)
  engine.run << (grid: (1, nTokens, headBlocksK), blk: (32, 1)) >> (
    "ggufQkNormRopeD128", kRopeS,
    (kBufS, p.normGammaK, cosK, sinK,
     int32(nKv), int32(p.Nkv), int32(headBlocksK), int32(0), p.eps))
  # the cache write. Each seq's roped k and plain v land at the rows
  # the staging contract fixes. The page decomposition is shift/mask
  # (pageSize is a power of two). A wrong table entry faults instead
  # of writing past the slab.
  let lgPageSize = countTrailingZeroBits(p.pageSize)
  let pageMask = p.pageSize - 1
  let slabPageElems = p.pageSize * p.Nkv * p.D
  let slabPageCount = kSlab.len div slabPageElems
  doAssert kSlab.len == vSlab.len, "the k and v slabs must hold an equal number of elements"
  doAssert kSlab.len mod slabPageElems == 0, "the k slab length must be a whole number of pages"
  for s in 0 ..< numSeqs:
    let qLen = (p.cuSeqlensQ[s + 1] - p.cuSeqlensQ[s]).int
    let q0 = p.cuSeqlensQ[s].int
    let writeStart = p.cacheSeqlens[s].int - (if qLen == 1: 1 else: 0)
    for j in 0 ..< qLen:
      let row = writeStart + j
      let pageIdx = row shr lgPageSize
      doAssert pageIdx < p.maxPages, "the cache write row exceeds the block-table budget"
      let inPage = row and pageMask
      let pageId = p.blockTable[s * p.maxPages + pageIdx].int
      doAssert pageId >= 0 and pageId < slabPageCount,
        "cache write hit an unused or out-of-slab block_table slot"
      # each token's Nkv·D span is one contiguous move on both sides:
      # the slab row (pageId, inPage) and the kRope/vBuf token row are
      # both head-dim-contiguous, slab layout (page, in_page, h, d)
      let t = q0 + j
      let dstBase = (pageId * p.pageSize + inPage) * (p.Nkv * p.D)
      copyMem(addr kSlab[dstBase], addr kRopeS[t * nKv], nKv * sizeof(uint16))
      copyMem(addr vSlab[dstBase], addr vBufS[t * nKv], nKv * sizeof(uint16))
  # the paged attention's x extent, the batch's longest q_len
  # in 8-row q blocks, the kernel zero-filling blocks beyond a seq's own q_len
  var xBlocks = 1
  for s in 0 ..< numSeqs:
    let qLen = p.cuSeqlensQ[s + 1] - p.cuSeqlensQ[s]
    xBlocks = max(xBlocks, (qLen.int + 7) div 8)
  var attnOutS = newSeq[uint16](nTokens * nQ)
  engine.run << (grid: (xBlocks, p.H, numSeqs), blk: (32, 1)) >> (
    "ggufPagedD128", attnOutS,
    (qRopeS, kSlab, vSlab, p.blockTable, p.cacheSeqlens, p.cuSeqlensQ,
     int32(numSeqs), int32(p.H), int32(p.Nkv),
     int32(p.maxPages), int32(p.pageSize)))
  # the o_proj over the attention output (num_qo_tokens, H·D) → hidden
  var outBufS = newSeq[uint16](nTokens * p.hidden)
  engine.run << (grid: (p.hidden div 128, (nTokens + 31) div 32, 1),
                 blk: (32, 1)) >> (
    launcher(p.oScheme), outBufS,
    (attnOutS, p.ow, int32(nTokens), int32(nQ),
     int32(p.hidden), p.oRowBytes))

  result.qBuf = tensorFromFp16(qBufS, nTokens, nQ)
  result.kBuf = tensorFromFp16(kBufS, nTokens, nKv)
  result.vBuf = tensorFromFp16(vBufS, nTokens, nKv)
  result.qRope = tensorFromFp16(qRopeS, nTokens, nQ)
  result.kRope = tensorFromFp16(kRopeS, nTokens, nKv)
  result.attnOut = tensorFromFp16(attnOutS, nTokens, nQ)
  result.outBuf = tensorFromFp16(outBufS, nTokens, p.hidden)

proc qkNormRopeRef(qT: F.Tensor, gamma: seq[uint16], cosT, sinT: seq[float32],
                   rowsPerToken, D: int, eps: float32): F.Tensor =
  ## Torch qk-norm+rope: rms_norm per (nTokens·rowsPerToken, D)
  ## row, the fp16 two-rounding, the fp32 NEOX rotation with the
  ## (nTokens·rowsPerToken, 64) tables, reshaped to the 3D layout.
  let rows = qT.size(0) * rowsPerToken
  let flat = qT.reshape(rows, D)
  let normed = rms_norm(flat, D, F.ones(D, kFloat32), float64(eps))
  let xr = normed.to(kFloat16)
  let gT = toTensor(fp16sToF32(gamma)).reshape(1, D).to(kFloat16)
  let xg = (xr.to(kFloat32) * gT.to(kFloat32)).to(kFloat16).to(kFloat32)
  let cosTns = toTensor(cosT).reshape(rows, D div 2)
  let sinTns = toTensor(sinT).reshape(rows, D div 2)
  let t1 = (xg.narrow(1, 0, D div 2) * cosTns -
            xg.narrow(1, D div 2, D div 2) * sinTns).to(kFloat16)
  let t2 = (xg.narrow(1, D div 2, D div 2) * cosTns +
            xg.narrow(1, 0, D div 2) * sinTns).to(kFloat16)
  F.cat([t1, t2], 1).reshape(qT.size(0), rowsPerToken, D).to(kFloat32)

proc writeRefSlab(dst: var seq[float32], src: F.Tensor, p: AttnCase, rowsPerToken: int) =
  ## Fills the flat (num_pages·page_size·Nkv·D) fp32 slab from the
  ## reference's own kRope/v with the kernel-side write-band addressing
  ## (rows outside the bands keep the seeded history).
  let lgPageSize = countTrailingZeroBits(p.pageSize)
  let pageMask = p.pageSize - 1
  let sp = src.contiguous().data_ptr(float32)
  for s in 0 ..< p.cuSeqlensQ.len - 1:
    let qLen = (p.cuSeqlensQ[s + 1] - p.cuSeqlensQ[s]).int
    let q0 = p.cuSeqlensQ[s].int
    let writeStart = p.cacheSeqlens[s].int - (if qLen == 1: 1 else: 0)
    for j in 0 ..< qLen:
      let row = writeStart + j
      let pageIdx = row shr lgPageSize
      let inPage = row and pageMask
      let pageId = p.blockTable[s * p.maxPages + pageIdx].int
      let t = q0 + j
      for h in 0 ..< p.Nkv:
        for d in 0 ..< p.D:
          let i = (pageId * p.pageSize + inPage) * (p.Nkv * p.D) + h * p.D + d
          dst[i] = sp[(t * rowsPerToken + h) * p.D + d]

proc gatherKv(src: F.Tensor, blockTable: seq[int32], maxPages, pageSize,
              covered, s: int): F.Tensor =
  ## Seq's K or V rows from the slab via the kernel's fetch math:
  ## block_table[s, t div page_size]·page_size + t mod page_size.
  var idx = newSeq[int64](covered)
  for t in 0 ..< covered:
    let pageIdx = t div pageSize
    let inPage = t mod pageSize
    idx[t] = int64(blockTable[s * maxPages + pageIdx] * pageSize + inPage)
  src.index_select(0, toTensor(idx))

proc sdpaDecode(qt, kt, vt: F.Tensor, H, Nkv: int): F.Tensor =
  ## q (1, H, 1, D), k/v (covered, Nkv, D), causal off, GQA on.
  let k2 = kt.reshape(1, kt.size(0), Nkv, kt.size(2)).transpose(1, 2)
  let v2 = vt.reshape(1, vt.size(0), Nkv, vt.size(2)).transpose(1, 2)
  scaled_dot_product_attention(qt, k2, v2, enable_gqa = H > Nkv)

proc sdpaPrefill(qt, kt, vt: F.Tensor, cacheSeqlen, qLen, H, Nkv: int): F.Tensor =
  ## q (1, H, qLen, D), k/v (covered, Nkv, D), banded causal mask over
  ## [0, cache_seqlen + q_len), GQA on.
  let covered = cacheSeqlen + qLen
  var maskF = newSeq[float32](qLen * covered)
  for j in 0 ..< qLen:
    for k in 0 ..< covered:
      maskF[j * covered + k] = if k <= cacheSeqlen + j: 0.0'f32 else: float32(NegInf)
  let k2 = kt.reshape(1, covered, Nkv, kt.size(2)).transpose(1, 2)
  let v2 = vt.reshape(1, covered, Nkv, vt.size(2)).transpose(1, 2)
  let mt = toTensor(maskF).reshape(1, qLen, covered)
  scaled_dot_product_attention(qt, k2, v2, attn_mask = some(mt), enable_gqa = H > Nkv)

proc reference(p: AttnCase, x: seq[uint16], numPages: int, kSeed, vSeed: int): AttnTensors =
  ## Torch composition: F.linear projections, rms_norm + rope with
  ## the test's cos/sin tables, the reference's own slab writes,
  ## per-seq SDPA with GQA, then the o_proj. Every fp16 round mirrors
  ## the kernel's fp16 buffers. The slabs never come from the kernel.
  ## The k/v history starts from the same fp16 buildX seeds as the
  ## kernel's slabs, so the fetched history rows are non-zero on both
  ## sides.
  let nTokens = p.cuSeqlensQ[^1].int
  let nQ = p.H * p.D
  let nKv = p.Nkv * p.D
  let xT = toTensor(fp16sToF32(x)).reshape(nTokens, p.hidden).to(kFloat16).to(kFloat32)
  let wq = toTensor(fp16sToF32(decodeWeights(p.qScheme, p.qw, p.hidden, nQ))).reshape(nQ, p.hidden).to(kFloat16).to(kFloat32)
  let wk = toTensor(fp16sToF32(decodeWeights(p.kScheme, p.kw, p.hidden, nKv))).reshape(nKv, p.hidden).to(kFloat16).to(kFloat32)
  let wv = toTensor(fp16sToF32(decodeWeights(p.vScheme, p.vw, p.hidden, nKv))).reshape(nKv, p.hidden).to(kFloat16).to(kFloat32)
  let wo = toTensor(fp16sToF32(decodeWeights(p.oScheme, p.ow, nQ, p.hidden))).reshape(p.hidden, nQ).to(kFloat16).to(kFloat32)
  result.qBuf = F.linear(xT, wq).to(kFloat16).to(kFloat32)
  result.kBuf = F.linear(xT, wk).to(kFloat16).to(kFloat32)
  result.vBuf = F.linear(xT, wv).to(kFloat16).to(kFloat32)
  let (cosQ, sinQ) = buildRopeCosSin(nTokens, p.H, p.D, p.positions, p.theta)
  let (cosK, sinK) = buildRopeCosSin(nTokens, p.Nkv, p.D, p.positions, p.theta)
  result.qRope = qkNormRopeRef(result.qBuf, p.normGammaQ, cosQ, sinQ, p.H, p.D, p.eps)
  result.kRope = qkNormRopeRef(result.kBuf, p.normGammaK, cosK, sinK, p.Nkv, p.D, p.eps)
  var kSlab = newSeq[float32](numPages * p.pageSize * p.Nkv * p.D)
  var vSlab = newSeq[float32](numPages * p.pageSize * p.Nkv * p.D)
  let kHist = buildX(numPages * p.pageSize * p.Nkv, p.D, kSeed, 1.0'f32)
  let vHist = buildX(numPages * p.pageSize * p.Nkv, p.D, vSeed, 1.0'f32)
  for i in 0 ..< kHist.len: kSlab[i] = fp16ToFp32(kHist[i])
  for i in 0 ..< vHist.len: vSlab[i] = fp16ToFp32(vHist[i])
  writeRefSlab(kSlab, result.kRope, p, p.Nkv)
  writeRefSlab(vSlab, result.vBuf, p, p.Nkv)
  let kT = toTensor(kSlab).reshape(numPages * p.pageSize, p.Nkv, p.D).to(kFloat16).to(kFloat32)
  let vT = toTensor(vSlab).reshape(numPages * p.pageSize, p.Nkv, p.D).to(kFloat16).to(kFloat32)
  var attnOutRef = newSeq[float32](nTokens * nQ)
  let qp = result.qRope.contiguous().data_ptr(float32)
  for s in 0 ..< p.cuSeqlensQ.len - 1:
    let qLen = (p.cuSeqlensQ[s + 1] - p.cuSeqlensQ[s]).int
    let q0 = p.cuSeqlensQ[s].int
    let covered = p.cacheSeqlens[s].int + qLen - (if qLen == 1: 1 else: 0)
    let kt = gatherKv(kT, p.blockTable, p.maxPages, p.pageSize, covered, s)
    let vt = gatherKv(vT, p.blockTable, p.maxPages, p.pageSize, covered, s)
    let qt = if qLen == 1:
               result.qRope.narrow(0, q0, 1).reshape(1, p.H, 1, p.D)
             else:
               var qf2 = newSeq[float32](qLen * p.H * p.D)
               for h in 0 ..< p.H:
                 for j in 0 ..< qLen:
                   for dd in 0 ..< p.D:
                     qf2[(h * qLen + j) * p.D + dd] = qp[((q0 + j) * p.H + h) * p.D + dd]
               toTensor(qf2).reshape(1, p.H, qLen, p.D)
    let ot = if qLen == 1: sdpaDecode(qt, kt, vt, p.H, p.Nkv)
             else: sdpaPrefill(qt, kt, vt, p.cacheSeqlens[s].int, qLen, p.H, p.Nkv)
    let pT = ot.contiguous().data_ptr(float32)
    for h in 0 ..< p.H:
      for j in 0 ..< qLen:
        for dd in 0 ..< p.D:
          attnOutRef[((q0 + j) * p.H + h) * p.D + dd] = pT[(h * qLen + j) * p.D + dd]
  result.attnOut = toTensor(attnOutRef).reshape(nTokens, nQ).to(kFloat16).to(kFloat32)
  result.outBuf = F.linear(result.attnOut, wo).to(kFloat16).to(kFloat32)

proc checkGGUFAttn(): bool =
  ## One mixed decode/prefill batch with all three schemes across the
  ## four projections: per-stage kernel output vs the torch reference.
  let hidden = 256
  let H = 16
  let Nkv = 8
  let D = 128
  let nQ = H * D
  let nKv = Nkv * D
  let numPages = 4
  let pageSize = 16
  let maxPages = 3
  let nTokens = 4
  let qw = genQ4_K(hidden, nQ, 1)
  let kw = genIQ4_XS(hidden, nKv, 2)
  let vw = genQ8_0(hidden, nKv, 3)
  let ow = scaleQ4K(genQ4_K(nQ, hidden, 4), 12)
  var gammaQ = newSeq[uint16](D)
  var gammaK = newSeq[uint16](D)
  for c in 0 ..< D:
    gammaQ[c] = fp32ToFp16(gammaVal(c, 5))
    gammaK[c] = fp32ToFp16(gammaVal(c, 7))
  var p = AttnCase(
    hidden: hidden, H: H, Nkv: Nkv, D: D,
    pageSize: pageSize, maxPages: maxPages,
    eps: 1e-6'f32, theta: 1e6'f32,
    cacheSeqlens: @[20'i32, 0],
    cuSeqlensQ: @[0'i32, 1, 4],
    blockTable: @[0'i32, 1, -1, 2, 3, -1],
    positions: @[19'i32, 0, 1, 2],
    qw: qw, kw: kw, vw: vw, ow: ow,
    qRowBytes: int32(qw.len div nQ),
    kRowBytes: int32(kw.len div nKv),
    vRowBytes: int32(vw.len div nKv),
    oRowBytes: int32(ow.len div hidden),
    qScheme: gsQ4_K, kScheme: gsIQ4_XS, vScheme: gsQ8_0, oScheme: gsQ4_K,
    normGammaQ: gammaQ, normGammaK: gammaK)
  let x = buildX(nTokens, hidden, 11, 1.0'f32 / 65536.0'f32)
  let kSeed = 13
  let vSeed = 17
  # the slabs start non-zero so the fetched history rows carry real
  # values on both sides (a wrong history-page fetch no longer
  # returns zeros and passes)
  var kSlab = buildX(numPages * pageSize * Nkv, D, kSeed, 1.0'f32)
  var vSlab = buildX(numPages * pageSize * Nkv, D, vSeed, 1.0'f32)
  let actual = kernelUnderTest(p, x, kSlab, vSlab)
  let expected = reference(p, x, numPages, kSeed, vSeed)
  var worstAll = 0.0'f32
  for stage in [("qBuf", actual.qBuf, expected.qBuf),
                ("kBuf", actual.kBuf, expected.kBuf),
                ("vBuf", actual.vBuf, expected.vBuf),
                ("qRope", actual.qRope, expected.qRope),
                ("kRope", actual.kRope, expected.kRope),
                ("attnOut", actual.attnOut, expected.attnOut),
                ("outBuf", actual.outBuf, expected.outBuf)]:
    let w = worstAbsDiff(stage[1], stage[2])
    worstAll = max(worstAll, w)
    echo &"  {stage[0]}: worst |Δ| = {w}, {stage[1].numel()} elements"
  echo &"  worst |Δ| across stages = {worstAll} (per-stage evidence, only outBuf is asserted at 6.25e-2)"
  # outBuf tolerance: the seeded k/v history lifts the layer output to
  # O(10-30), where the fp16 output ulp (0.03125 at |31.3|) alone
  # exceeds the 5e-3 bound that suits O(1) outputs. Measured worst
  # |Δ| = 0.03125, one ulp at the largest output. The 0.0625 bound is
  # two ulp at that magnitude, a 2x margin over the measurement.
  assertAllClose(actual.outBuf, expected.outBuf, rtol = 0.0'f64, abstol = 6.25e-2'f64)
  result = true

when isMainModule:
  runCppTest("GGUF attention launchers vs the torch SDPA reference", checkGGUFAttn)
