## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Round-trip test for the paged KV quant/dequant kernels on a layer-major pool:
## `kv_quant_fwd` then `kv_dequant_fwd` at two fixed (bits, compander) combos,
## comparing the dequant rows to the fp16 slab they came from (lossy bound)
## and to the same page/token of layer 0 (the layer-major separation check).
## Layer 1 holds 100× layer 0's values: a missing layer term leaves layer 1's
## rows unwritten (zeros), which the rt ≤ 64 bound rejects (amplitude 100–200),
## and a wrong page stride reads layer 0's rows, which the cross > 64 bound
## rejects.
##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/wip \
##   --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_exl3_kvquant_fp16.nim

import std/[strformat, math, random, sequtils]
import workspace/crucible
import workspace/ceramic
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/exl3_kvquant

const kvquantMsl = metal:
  proc kvQuantB8L0(
      outBuf: ptr UncheckedArray[uint32],
      kIn, vIn: ptr UncheckedArray[uint16],
      block_table, cache_seqlens: ptr UncheckedArray[int32],
      num_seqs, max_pages, token_dim, groups_per_token, q_words_total,
      num_layers, layer: int32, compand_a: float32) {.global.} =
    kv_quant_fwd(outBuf, kIn, vIn, block_table, cache_seqlens,
                 num_seqs, max_pages, token_dim, groups_per_token,
                 q_words_total, num_layers, layer, compand_a, 8, 0, 256, 128)

  proc kvDequantB8L0(
      outBuf: ptr UncheckedArray[uint16],
      qK, qV: ptr UncheckedArray[uint32],
      sK, sV: ptr UncheckedArray[uint16],
      block_table, cache_seqlens: ptr UncheckedArray[int32],
      num_seqs, max_pages, token_dim, groups_per_token, out_total,
      num_layers, layer: int32, compand_a: float32) {.global.} =
    kv_dequant_fwd(outBuf, qK, qV, sK, sV, block_table, cache_seqlens,
                   num_seqs, max_pages, token_dim, groups_per_token,
                   out_total, num_layers, layer, compand_a, 8, 0, 256, 128)

  proc kvQuantB8L1(
      outBuf: ptr UncheckedArray[uint32],
      kIn, vIn: ptr UncheckedArray[uint16],
      block_table, cache_seqlens: ptr UncheckedArray[int32],
      num_seqs, max_pages, token_dim, groups_per_token, q_words_total,
      num_layers, layer: int32, compand_a: float32) {.global.} =
    kv_quant_fwd(outBuf, kIn, vIn, block_table, cache_seqlens,
                 num_seqs, max_pages, token_dim, groups_per_token,
                 q_words_total, num_layers, layer, compand_a, 8, 1, 256, 128)

  proc kvDequantB8L1(
      outBuf: ptr UncheckedArray[uint16],
      qK, qV: ptr UncheckedArray[uint32],
      sK, sV: ptr UncheckedArray[uint16],
      block_table, cache_seqlens: ptr UncheckedArray[int32],
      num_seqs, max_pages, token_dim, groups_per_token, out_total,
      num_layers, layer: int32, compand_a: float32) {.global.} =
    kv_dequant_fwd(outBuf, qK, qV, sK, sV, block_table, cache_seqlens,
                   num_seqs, max_pages, token_dim, groups_per_token,
                   out_total, num_layers, layer, compand_a, 8, 1, 256, 128)

proc kvRoundTrip(
    kf, vf: seq[float32], blockTable: seq[int32],
    numPages, numLayers, layer, numSeqs, maxPages, rows, kvHeads, D,
    bits, compander: int): tuple[rt, cross: float32] =
  ## Quant then dequant round trip at one fixed (bits, compander) combo
  ## on layer `layer` of a flat layer-major pool. Returns the worst
  ## |Δ| of the dequant rows vs the fp16 slab they came from (`rt`)
  ## and vs the same page/token of layer 0 (`cross`).
  let tokenDim = kvHeads * D
  let groups = tokenDim div 32
  let poolTokens = numPages * numLayers * 256
  let qWordsTotal = poolTokens * groups * bits
  let sTotal = qWordsTotal shr int(log2(float(bits)))
  let quantLen = 2 * qWordsTotal + 2 * sTotal
  let outTotal = poolTokens * tokenDim
  let compandA = (if compander == 0: 0.0'f32 else: 0.65'f32)
  var kSlab = newSeq[uint16](poolTokens * tokenDim)
  var vSlab = newSeq[uint16](poolTokens * tokenDim)
  for i in 0 ..< poolTokens * tokenDim:
    kSlab[i] = fp32ToFp16(kf[i])
    vSlab[i] = fp32ToFp16(vf[i])
  var engine = bkMetal.init()
  engine.ingest(kvquantMsl)
  var cacheZero = newSeq[int32](numSeqs)
  var cacheRows = newSeq[int32](numSeqs)
  for i in 0 ..< numSeqs:
    cacheRows[i] = int32(rows)
  var outBuf = newSeq[uint32](quantLen)
  let qArgs = (kSlab, vSlab, blockTable, cacheZero,
               int32(numSeqs), int32(maxPages), int32(tokenDim),
               int32(groups), int32(qWordsTotal),
               int32(numLayers), int32(layer), compandA)
  if bits == 8 and compander == 0:
    engine.run << (grid: (groups div 4, rows, numSeqs), blk: (32, 1)) >>
      ("kvQuantB8L0", outBuf, qArgs)
  else:
    engine.run << (grid: (groups div 4, rows, numSeqs), blk: (32, 1)) >>
      ("kvQuantB8L1", outBuf, qArgs)
  var qK = newSeq[uint32](qWordsTotal)
  var qV = newSeq[uint32](qWordsTotal)
  var sK = newSeq[uint16](sTotal)
  var sV = newSeq[uint16](sTotal)
  for i in 0 ..< qWordsTotal:
    qK[i] = outBuf[i]
    qV[i] = outBuf[qWordsTotal + i]
  for i in 0 ..< sTotal:
    sK[i] = uint16(outBuf[2 * qWordsTotal + i])
    sV[i] = uint16(outBuf[2 * qWordsTotal + sTotal + i])
  var kd = newSeq[uint16](2 * outTotal)
  let dArgs = (qK, qV, sK, sV, blockTable, cacheRows,
               int32(numSeqs), int32(maxPages), int32(tokenDim),
               int32(groups), int32(outTotal),
               int32(numLayers), int32(layer), compandA)
  if bits == 8 and compander == 0:
    engine.run << (grid: (groups div 4, rows, numSeqs), blk: (32, 1)) >>
      ("kvDequantB8L0", kd, dArgs)
  else:
    engine.run << (grid: (groups div 4, rows, numSeqs), blk: (32, 1)) >>
      ("kvDequantB8L1", kd, dArgs)
  var rt = 0.0'f32
  var cross = 0.0'f32
  for b in 0 ..< numSeqs:
    for y in 0 ..< rows:
      let pageIdx = y shr 8
      let pageId = blockTable[b * maxPages + pageIdx].int
      let tp = pageId * numLayers * 256 + layer * 256 + (y and 255)
      let pos = tp * tokenDim
      let posL0 = pos - layer * 256 * tokenDim
      for i in 0 ..< tokenDim:
        let a = fp16ToFp32(kd[pos + i])
        let aV = fp16ToFp32(kd[outTotal + pos + i])
        let s1 = fp16ToFp32(kSlab[pos + i])
        let s1V = fp16ToFp32(vSlab[pos + i])
        let s0 = fp16ToFp32(kSlab[posL0 + i])
        let s0V = fp16ToFp32(vSlab[posL0 + i])
        rt = max(rt, abs(a - s1))
        rt = max(rt, abs(aV - s1V))
        cross = max(cross, abs(a - s0))
        cross = max(cross, abs(aV - s0V))
  (rt, cross)

proc checkKvQuant(): bool =
  ## Two fixed (bits, compander) combos on layer 1 of a layer-major
  ## pool with a gapped block table: the dequant rows must reconstruct
  ## layer 1's slab within the lossy bound and must NOT match layer
  ## 0's content. Layer 1 holds 100× layer 0's values: a missing layer
  ## term leaves layer 1's rows unwritten (zeros), which the rt ≤ 64
  ## bound rejects (amplitude 100–200), and a wrong page stride reads
  ## layer 0's rows, which the cross > 64 bound rejects.
  randomize(0x5EED)
  let numPages = 4
  let numLayers = 2
  let layer = 1
  let numSeqs = 2
  let maxPages = 2
  let rows = 300
  let kvHeads = 2
  let D = 128
  let tokenDim = kvHeads * D
  let blockTable = [0'i32, 3, 1, 2].toSeq()
  let poolTokens = numPages * numLayers * 256
  var kf = newSeq[float32](poolTokens * tokenDim)
  var vf = newSeq[float32](poolTokens * tokenDim)
  for p in 0 ..< numPages:
    for l in 0 ..< numLayers:
      for t in 0 ..< 256:
        for d in 0 ..< tokenDim:
          let base = 1.0'f32 + rand(1.0'f32)
          let idx = (p * numLayers + l) * 256 * tokenDim + t * tokenDim + d
          kf[idx] = (if l == 1: 100.0'f32 * base else: base)
          vf[idx] = (if l == 1: 100.0'f32 * base else: base)
  for (bits, compander) in [(8, 0), (8, 1)]:
    let (rt, cross) = kvRoundTrip(kf, vf, blockTable, numPages, numLayers,
                                  layer, numSeqs, maxPages, rows, kvHeads,
                                  D, bits, compander)
    echo &"  bits={bits} compander={compander}: rt worst |Δ| = {rt} cross |Δ| = {cross}"
    doAssert rt <= 64.0'f32, "round-trip error exceeds the lossy bound"
    doAssert cross > 64.0'f32, "dequant output matches layer 0's content"
  result = true

when isMainModule:
  runCppTest("kvquant round trip on a layer-major pool", checkKvQuant)
