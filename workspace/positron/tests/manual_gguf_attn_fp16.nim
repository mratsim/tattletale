## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/wip \
##   --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_gguf_attn_fp16.nim

import std/[strformat, options, math, bitops]
import workspace/crucible
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/gguf_attn
import ./gguf_test_utils

type
  AttnTensors = object
    ## Per-stage fp16 outputs widened to fp32 tensors.
    qBuf, kBuf, vBuf, qRope, kRope, attnOut, outBuf: F.Tensor

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

proc decodeWeights(scheme: int, packed: seq[uint8], K, N: int): seq[uint16] =
  ## (N, K) file-order fp16 reconstruction for the packed scheme.
  case scheme
  of 0: decodeWeightsQ8_0(packed, K, N)
  of 1: decodeWeightsQ4_K(packed, K, N)
  of 2: decodeWeightsIQ4_XS(packed, K, N)
  else: raise newException(ValueError, "unknown scheme: " & $scheme)

proc tensorFromFp16(hs: seq[uint16], rows, cols: int): F.Tensor =
  ## fp16 bit buffer widened to a (rows, cols) fp32 tensor.
  var f = newSeq[float32](hs.len)
  for i in 0 ..< hs.len: f[i] = fp16ToFp32(hs[i])
  toTensor(f).reshape(rows, cols)

proc kernelUnderTest(p: GGufAttnParams, x: seq[uint16],
                     kSlab, vSlab: var seq[uint16]): AttnTensors =
  ## Runs `ggufAttnForward` and widens each fp16 stage to fp32 tensors.
  var engine = bkMetal.init()
  engine.ingest(ggufAttnMsl)
  let r = ggufAttnForward(engine, p, x, kSlab, vSlab)
  let nTokens = p.cuSeqlensQ[^1].int
  result.qBuf = tensorFromFp16(r.qBuf, nTokens, p.H * p.D)
  result.kBuf = tensorFromFp16(r.kBuf, nTokens, p.Nkv * p.D)
  result.vBuf = tensorFromFp16(r.vBuf, nTokens, p.Nkv * p.D)
  result.qRope = tensorFromFp16(r.qRope, nTokens, p.H * p.D)
  result.kRope = tensorFromFp16(r.kRope, nTokens, p.Nkv * p.D)
  result.attnOut = tensorFromFp16(r.attnOut, nTokens, p.H * p.D)
  result.outBuf = tensorFromFp16(r.outBuf, nTokens, p.hidden)

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

proc writeRefSlab(dst: var seq[float32], src: F.Tensor, p: GGufAttnParams,
                  rowsPerToken: int) =
  ## Fills the flat (num_pages·page_size·Nkv·D) fp32 slab from the
  ## reference's own kRope/v with the driver's write-band addressing
  ## (rows outside the bands stay zero).
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

proc reference(p: GGufAttnParams, x: seq[uint16], numPages: int): AttnTensors =
  ## Torch composition: F.linear projections, rms_norm + rope with
  ## the driver's cos/sin tables, the reference's own slab writes,
  ## per-seq SDPA with GQA, then the o_proj. Every fp16 round mirrors
  ## the kernel's fp16 buffers. The slabs never come from the kernel.
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
  var p = GGufAttnParams(
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
    qScheme: 1, kScheme: 2, vScheme: 0, oScheme: 1,
    normGammaQ: gammaQ, normGammaK: gammaK)
  let x = buildX(nTokens, hidden, 11, 1.0'f32 / 65536.0'f32)
  var kSlab = newSeq[uint16](numPages * pageSize * Nkv * D)
  var vSlab = newSeq[uint16](numPages * pageSize * Nkv * D)
  let actual = kernelUnderTest(p, x, kSlab, vSlab)
  let expected = reference(p, x, numPages)
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
  echo &"  worst |Δ| across stages = {worstAll} (per-stage evidence, only outBuf is asserted at 5e-3)"
  assertAllClose(actual.outBuf, expected.outBuf, rtol = 0.0'f64, abstol = 5e-3'f64)
  result = true

when isMainModule:
  runCppTest("GGUF attention composition vs the torch SDPA reference", checkGGUFAttn)
