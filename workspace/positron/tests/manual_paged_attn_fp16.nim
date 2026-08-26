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
##   workspace/positron/tests/manual_paged_attn_fp16.nim

import std/[strformat, options, math, sequtils]
import workspace/crucible
import workspace/libtorch
import workspace/libtorch as F
import workspace/libtorch_testutils
from workspace/libtorch/src/raw_libtorch import manual_seed
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/paged_attn

const pagedMsl = metal:
  proc pagedD128(
      o, q: ptr UncheckedArray[float16],
      k_cache, v_cache: ptr UncheckedArray[float16],
      block_table, cache_seqlens, cu_seqlens_q: ptr UncheckedArray[int32],
      num_seqs, H, Nkv, max_pages, page_size: int32) {.global.} =
    paged_attn_fwd(o, q, k_cache, v_cache, block_table, cache_seqlens,
                   cu_seqlens_q, num_seqs, H, Nkv, max_pages, page_size, 128)

proc scaledRand(rows, cols: int, scaleF: float32): seq[float32] =
  ## rand(rows, cols) shifted host-side to [-scaleF, scaleF). The
  ## kernel buffer and the reference tensors derive from the same seq.
  let t = rand(rows, cols)
  let p = t.contiguous().data_ptr(float32)
  result = newSeq[float32](rows * cols)
  for i in 0 ..< rows * cols:
    result[i] = (p[i] - 0.5'f32) * (2.0'f32 * scaleF)

proc worstAbsDiff(a, b: F.Tensor): float32 =
  ## Largest |a[i] - b[i]| over all elements.
  let ap = a.contiguous().data_ptr(float32)
  let bp = b.contiguous().data_ptr(float32)
  result = 0.0'f32
  for i in 0 ..< a.numel():
    let d = abs(ap[i] - bp[i])
    if d > result: result = d

proc gatherKv(src: F.Tensor, blockTable: seq[int32], maxPages, pageSize, covered, s: int): F.Tensor =
  ## The seq's contiguous K or V: the flat (num_pages·page_size, Nkv, D)
  ## slab indexed by block_table[s, t div page_size]·page_size +
  ## t mod page_size for t in [0, covered), the same address math the
  ## kernel's per-block fetch uses.
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

proc pagedAttnKernel(
    qf, kf, vf: seq[float32],
    blockTable, cacheSeqlens, cuSeqlensQ: seq[int32],
    numSeqs, H, Nkv, maxPages, pageSize, nQo, D: int): F.Tensor =
  ## Runs pagedD128 on the fp16-rounded inputs; returns the fp16
  ## output converted to fp32.
  var engine = bkMetal.init()
  engine.ingest(pagedMsl)
  var kSlab = newSeq[uint16](kf.len)
  var vSlab = newSeq[uint16](vf.len)
  var q = newSeq[uint16](qf.len)
  for i in 0 ..< kf.len:
    kSlab[i] = fp32ToFp16(kf[i])
    vSlab[i] = fp32ToFp16(vf[i])
  for i in 0 ..< qf.len:
    q[i] = fp32ToFp16(qf[i])
  var outO = newSeq[uint16](nQo * H * D)
  let args = (q, kSlab, vSlab, blockTable, cacheSeqlens, cuSeqlensQ,
              int32(numSeqs), int32(H), int32(Nkv), int32(maxPages),
              int32(pageSize))
  engine.run << (grid: (1, H, numSeqs), blk: (32, 1)) >> ("pagedD128", outO, args)
  var outF = newSeq[float32](nQo * H * D)
  for i in 0 ..< nQo * H * D:
    outF[i] = fp16ToFp32(outO[i])
  result = toTensor(outF).reshape(nQo, H, D)

proc pagedAttnReference(
    qf, kf, vf: seq[float32],
    blockTable, cacheSeqlens, cuSeqlensQ, qLens: seq[int32],
    numPages, maxPages, pageSize, numSeqs, H, Nkv, nQo, D: int): F.Tensor =
  ## torch SDPA over the same fp16-rounded values, per seq: decode
  ## (q_len = 1) causal off, prefill (q_len >= 2) banded causal over
  ## [0, cache_seqlen + q_len). SDPA wants (1, H, q_len, D), so the
  ## prefill q is transposed from the kernel's (q, H, D) order.
  let kT = toTensor(kf).reshape(numPages * pageSize, Nkv, D).to(kFloat16).to(kFloat32)
  let vT = toTensor(vf).reshape(numPages * pageSize, Nkv, D).to(kFloat16).to(kFloat32)
  let qT = toTensor(qf).reshape(nQo, H, D).to(kFloat16).to(kFloat32)
  var oRef = newSeq[float32](nQo * H * D)
  for s in 0 ..< numSeqs:
    let covered = cacheSeqlens[s].int + qLens[s].int - (if qLens[s] == 1: 1 else: 0)
    let kt = gatherKv(kT, blockTable, maxPages, pageSize, covered, s)
    let vt = gatherKv(vT, blockTable, maxPages, pageSize, covered, s)
    let q0 = cuSeqlensQ[s].int
    let qLen = qLens[s].int
    let qp = qT.contiguous().data_ptr(float32)
    let qt = if qLen == 1:
               qT.narrow(0, q0, 1).reshape(1, H, 1, D)
             else:
               var qf2 = newSeq[float32](qLen * H * D)
               for h in 0 ..< H:
                 for j in 0 ..< qLen:
                   for dd in 0 ..< D:
                     qf2[(h * qLen + j) * D + dd] = qp[((q0 + j) * H + h) * D + dd]
               toTensor(qf2).reshape(1, H, qLen, D)
    let ot = if qLen == 1: sdpaDecode(qt, kt, vt, H, Nkv)
             else: sdpaPrefill(qt, kt, vt, cacheSeqlens[s].int, qLen, H, Nkv)
    let p = ot.contiguous().data_ptr(float32)
    for h in 0 ..< H:
      for j in 0 ..< qLen:
        for dd in 0 ..< D:
          oRef[((q0 + j) * H + h) * D + dd] = p[(h * qLen + j) * D + dd]
  result = toTensor(oRef).reshape(nQo, H, D)

proc checkPagedAttn(): bool =
  ## One mixed batch (decode and prefill seqs under one cu_seqlens_q):
  ## kernel output vs torch SDPA per seq.
  Torch.manual_seed(0x5EED'u64)
  let numSeqs = 3
  let H = 4
  let Nkv = 2
  let D = 128
  let pageSize = 16
  let maxPages = 3
  let numPages = 7
  let nQo = 5
  let cacheSeqlens = [20'i32, 5, 33].toSeq()
  let qLens = [1'i32, 3, 1].toSeq()
  let blockTable = [0'i32, 1, -1, 2, 3, -1, 4, 5, 6].toSeq()
  let cuSeqlensQ = [0'i32, 1, 4, 5].toSeq()
  let kf = scaledRand(numPages * pageSize, Nkv * D, 2.0'f32)
  let vf = scaledRand(numPages * pageSize, Nkv * D, 2.0'f32)
  let qf = scaledRand(nQo, H * D, 2.0'f32)
  let actual = pagedAttnKernel(qf, kf, vf, blockTable, cacheSeqlens,
                               cuSeqlensQ, numSeqs, H, Nkv, maxPages,
                               pageSize, nQo, D)
  let expected = pagedAttnReference(qf, kf, vf, blockTable, cacheSeqlens,
                                    cuSeqlensQ, qLens, numPages, maxPages,
                                    pageSize, numSeqs, H, Nkv, nQo, D)
  echo &"  worst |Δ| = {worstAbsDiff(actual, expected)} (tolerance 5e-3)"
  assertAllClose(actual, expected, rtol = 0.0'f64, abstol = 5e-3'f64)
  result = true

when isMainModule:
  runCppTest("paged_attn vs torch SDPA (decode + prefill)", checkPagedAttn)
