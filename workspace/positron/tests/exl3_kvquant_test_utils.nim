## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared paged-KV quant/dequant test support: the deterministic paged
## geometries, the host reference implementation of the
## quant_block_x4 / dequant_block_x4 arithmetic (q_cache_kernels.cuh),
## and the per-combo engine runs the exl3 kvquant tests share.

import std/[strformat, strutils, math, random, sequtils]
import workspace/crucible
import ../../ceramic/tests/tile_test_utils
import ../src/kernels/ceramic/exl3_kvquant

# ════════════════════════════════════════
#  The metal block. One launcher per kernel dispatches over runtime
#  bits/compander int32 args to the 6 static instantiations (nested
#  if/else: the crucible codegen drops elif branches). First param =
#  the output buffer: engine.run binds the separate outBuf argument to
#  it and the args tuple to the rest. pageSize = 256 and D = 128 baked
#  into every call.
# ════════════════════════════════════════

const kvquantMsl* = metal:
  proc kvQuantD128(
      outBuf: ptr UncheckedArray[uint32],
      kIn, vIn: ptr UncheckedArray[uint16],
      block_table, cache_seqlens: ptr UncheckedArray[int32],
      num_seqs, max_pages, token_dim, groups_per_token, q_words_total,
      num_layers, layer: int32,
      compand_a: float32, bits: int32, compander: int32) {.global.} =
    if bits == 2:
      if compander == 0:
        kv_quant_fwd(outBuf, kIn, vIn, block_table, cache_seqlens,
                     num_seqs, max_pages, token_dim, groups_per_token,
                     q_words_total, num_layers, layer, compand_a, 2, 0, 256, 128)
      else:
        kv_quant_fwd(outBuf, kIn, vIn, block_table, cache_seqlens,
                     num_seqs, max_pages, token_dim, groups_per_token,
                     q_words_total, num_layers, layer, compand_a, 2, 1, 256, 128)
    else:
      if bits == 4:
        if compander == 0:
          kv_quant_fwd(outBuf, kIn, vIn, block_table, cache_seqlens,
                       num_seqs, max_pages, token_dim, groups_per_token,
                       q_words_total, num_layers, layer, compand_a, 4, 0, 256, 128)
        else:
          kv_quant_fwd(outBuf, kIn, vIn, block_table, cache_seqlens,
                       num_seqs, max_pages, token_dim, groups_per_token,
                       q_words_total, num_layers, layer, compand_a, 4, 1, 256, 128)
      else:
        if bits == 8:
          if compander == 0:
            kv_quant_fwd(outBuf, kIn, vIn, block_table, cache_seqlens,
                         num_seqs, max_pages, token_dim, groups_per_token,
                         q_words_total, num_layers, layer, compand_a, 8, 0, 256, 128)
          else:
            kv_quant_fwd(outBuf, kIn, vIn, block_table, cache_seqlens,
                         num_seqs, max_pages, token_dim, groups_per_token,
                         q_words_total, num_layers, layer, compand_a, 8, 1, 256, 128)

  proc kvDequantD128(
      outBuf: ptr UncheckedArray[uint16],
      qK, qV: ptr UncheckedArray[uint32],
      sK, sV: ptr UncheckedArray[uint16],
      block_table, cache_seqlens: ptr UncheckedArray[int32],
      num_seqs, max_pages, token_dim, groups_per_token, out_total,
      num_layers, layer: int32,
      compand_a: float32, bits: int32, compander: int32) {.global.} =
    if bits == 2:
      if compander == 0:
        kv_dequant_fwd(outBuf, qK, qV, sK, sV, block_table, cache_seqlens,
                       num_seqs, max_pages, token_dim, groups_per_token,
                       out_total, num_layers, layer, compand_a, 2, 0, 256, 128)
      else:
        kv_dequant_fwd(outBuf, qK, qV, sK, sV, block_table, cache_seqlens,
                       num_seqs, max_pages, token_dim, groups_per_token,
                       out_total, num_layers, layer, compand_a, 2, 1, 256, 128)
    else:
      if bits == 4:
        if compander == 0:
          kv_dequant_fwd(outBuf, qK, qV, sK, sV, block_table, cache_seqlens,
                         num_seqs, max_pages, token_dim, groups_per_token,
                         out_total, num_layers, layer, compand_a, 4, 0, 256, 128)
        else:
          kv_dequant_fwd(outBuf, qK, qV, sK, sV, block_table, cache_seqlens,
                         num_seqs, max_pages, token_dim, groups_per_token,
                         out_total, num_layers, layer, compand_a, 4, 1, 256, 128)
      else:
        if bits == 8:
          if compander == 0:
            kv_dequant_fwd(outBuf, qK, qV, sK, sV, block_table, cache_seqlens,
                           num_seqs, max_pages, token_dim, groups_per_token,
                           out_total, num_layers, layer, compand_a, 8, 0, 256, 128)
          else:
            kv_dequant_fwd(outBuf, qK, qV, sK, sV, block_table, cache_seqlens,
                         num_seqs, max_pages, token_dim, groups_per_token,
                         out_total, num_layers, layer, compand_a, 8, 1, 256, 128)

# ════════════════════════════════════════
#  Case data (deterministic, hash-seeded: the shared ShapeCases
#  stream)
# ════════════════════════════════════════

type XorShift = object
  s: uint32

func seedRng*(shapes: var ShapeCases): XorShift =
  ## Seeds the value PRNG from the shape stream's seed bytes.
  let b = shapes.nextBytes()
  var s = uint32(b[0]) or (uint32(b[1]) shl 8) or
          (uint32(b[2]) shl 16) or (uint32(b[3]) shl 24)
  if s == 0: s = 0x9E3779B9'u32
  result = XorShift(s: s)

func nextU16(r: var XorShift): uint16 =
  ## xorshift32, the test's deterministic value source.
  r.s = r.s xor (r.s shl 13)
  r.s = r.s xor (r.s shr 17)
  r.s = r.s xor (r.s shl 5)
  result = uint16(r.s shr 16)

func f16GridVal(r: var XorShift): float32 =
  ## Deterministic f16-exact value in [-2, 2): a 32-point grid, one
  ## stream position per element, so the pattern never aliases the
  ## 32-value group width or the 128-value group-block span.
  float32(nextU16(r) mod 32) / 8.0'f32 - 2.0'f32

type KvCase* = object
  numSeqs*: int
  kvHeads*: int                 # kv_heads (token_dim = kv_heads·128)
  tokenDim*: int
  groupsPerToken*: int          # kv_heads·128 div 32
  rows*: int                    # the rectangular batch append / cached rows
  maxPages*, numPages*: int
  blockTable*: seq[int32]       # (num_seqs, max_pages) dense, -1 padding
  kSlab*, vSlab*: seq[uint16]   # (num_pages, 256, token_dim) fp16

proc buildKvCase*(shapes: var ShapeCases): KvCase =
  ## One paged geometry: 1..3 seqs with a rectangular append of
  ## `rows` tokens each (the exllamav3 batch-append semantics), the
  ## row count taken from a partial/full-page mix at the production
  ## PAGE_SIZE 256, gapped physical page ids, kv_heads 1..4 (→ 4..16
  ## groups per token). The slabs are filled with f16-exact grid
  ## values at the physical page rows.
  let b = shapes.nextBytes()
  result.numSeqs = caseInRange(b, 0, 1, 3)
  result.kvHeads = [1, 2, 4][caseInRange(b, 1, 0, 2)]
  result.tokenDim = result.kvHeads * 128
  result.groupsPerToken = result.tokenDim div 32
  let rowChoices = [100, 256, 300, 512, 600, 257]
  result.rows = rowChoices[caseInRange(b, 2, 0, rowChoices.len - 1)]
  let usedPages = (result.rows + 255) div 256
  var tables = newSeq[seq[int32]](result.numSeqs)
  var idCounter = 0
  for s in 0 ..< result.numSeqs:
    var row = newSeq[int32](usedPages)
    for j in 0 ..< usedPages:
      row[j] = int32(idCounter)
      let gap = (if j == usedPages - 1: 0 else: caseInRange(shapes.nextBytes(), 3 + s, 0, 1))
      idCounter += 1 + gap
    tables[s] = row
  result.numPages = idCounter
  result.maxPages = usedPages
  result.blockTable = newSeq[int32](result.numSeqs * result.maxPages)
  for i in 0 ..< result.blockTable.len:
    result.blockTable[i] = -1'i32
  for s in 0 ..< result.numSeqs:
    for j in 0 ..< usedPages:
      result.blockTable[s * result.maxPages + j] = tables[s][j]
  var r = seedRng(shapes)
  let slabLen = result.numPages * 256 * result.tokenDim
  result.kSlab = newSeq[uint16](slabLen)
  result.vSlab = newSeq[uint16](slabLen)
  for i in 0 ..< slabLen:
    result.kSlab[i] = fp32ToFp16(f16GridVal(r) * 2.0'f32)
    result.vSlab[i] = fp32ToFp16(f16GridVal(r) * 2.0'f32)

func tokenPos(d: KvCase; batch, tokenIdx: int): int =
  ## The physical position of the logical token: the block-table page
  ## mapping (the kernel's addressing, mirrored host-side).
  let pageIdx = tokenIdx shr 8
  let pageId = d.blockTable[batch * d.maxPages + pageIdx].int
  pageId * 256 + (tokenIdx and 255)

# ════════════════════════════════════════
#  The host reference implementation: the quant_block_x4 /
#  dequant_block_x4 arithmetic (q_cache_kernels.cuh), formula by formula
# ════════════════════════════════════════

# host libm mirrors (the device builtins have empty bodies on the host)
proc floorf(x: float32): float32 {.importc: "floorf", header: "<math.h>".}
proc exp2f(x: float32): float32 {.importc: "exp2f", header: "<math.h>".}
proc sqrtf(x: float32): float32 {.importc: "sqrtf", header: "<math.h>".}
proc fmaf(x, y, z: float32): float32 {.importc: "fmaf", header: "<math.h>".}

const R32 = 0.17677669529663688110'f32   # 1/sqrt(32), q_cache_kernels.cuh

func refCbrt(x: float32): float32 =
  ## The host mirror of the op's `cbrtDefined` (same fp32 sequence,
  ## importc floorf/exp2f): the defined cube root the LMCubic encode
  ## uses in place of the CUDA cbrtf (MSL has no cbrt).
  if x == 0.0'f32:
    return x
  let sgn = (if x < 0.0'f32: -1.0'f32 else: 1.0'f32)
  let ax = (if x < 0.0'f32: -x else: x)
  var e = 0'i32
  var m = ax
  if m >= 1.8446744073709552e19'f32:
    m = m * 5.421010862427522e-20'f32
    e += 64
  if m >= 4294967296.0'f32:
    m = m * 2.3283064365386963e-10'f32
    e += 32
  if m >= 65536.0'f32:
    m = m * 1.52587890625e-05'f32
    e += 16
  if m >= 256.0'f32:
    m = m * 0.00390625'f32
    e += 8
  if m >= 16.0'f32:
    m = m * 0.0625'f32
    e += 4
  if m >= 4.0'f32:
    m = m * 0.25'f32
    e += 2
  if m >= 2.0'f32:
    m = m * 0.5'f32
    e += 1
  while m < 1.0'f32:
    m = m * 2.0'f32
    e -= 1
  let q = int32(floorf(float32(e) * 0.3333333432674408'f32))
  let r = e - 3 * q
  var y: float32
  block seed:
    let u = 2.0'f32 * m - 3.0'f32
    var b1 = 0.0'f32
    var b2 = 0.0'f32
    var bb = 2.0'f32 * u * b1 - b2 - 7.892353387775054e-05'f32
    b2 = b1
    b1 = bb
    bb = 2.0'f32 * u * b1 - b2 + 0.000702055045675154'f32
    b2 = b1
    b1 = bb
    bb = 2.0'f32 * u * b1 - b2 - 0.0073754847563900945'f32
    b2 = b1
    b1 = bb
    bb = 2.0'f32 * u * b1 - b2 + 0.1292479927845696'f32
    b2 = b1
    b1 = bb
    y = u * b1 - b2 + 1.1374176813308194'f32
  if r == 1:
    y = y * 1.2599210498948732'f32
  else:
    if r == 2:
      y = y * 1.5874010519681994'f32
  y = y * exp2f(float32(q))
  y = (2.0'f32 * y + ax / (y * y)) * 0.3333333432674408'f32
  y = (2.0'f32 * y + ax / (y * y)) * 0.3333333432674408'f32
  sgn * y

func hadamard32(v: var array[32, float32]) =
  ## The H32 rotation, in place: the H4 butterfly per 4-block
  ## (had_4_inreg) followed by the H8 butterfly over the 8 blocks
  ## (had_8_subgroup, 3 stages pairing block k with k^i, the exact
  ## pairwise (a+b, a−b) with one rounding each, q_cache_kernels.cuh
  ## had_4_inreg / had_8_subgroup).
  for k in 0 ..< 8:
    let s0 = v[4 * k + 0] + v[4 * k + 1]
    let d0 = v[4 * k + 0] - v[4 * k + 1]
    let s1 = v[4 * k + 2] + v[4 * k + 3]
    let d1 = v[4 * k + 2] - v[4 * k + 3]
    v[4 * k + 0] = s0 + s1
    v[4 * k + 1] = d0 + d1
    v[4 * k + 2] = s0 - s1
    v[4 * k + 3] = d0 - d1
  for i in [1, 2, 4]:
    var nv = v
    for k in 0 ..< 8:
      let p = k xor i
      if (k and i) == 0:
        for rr in 0 ..< 4:
          nv[4 * k + rr] = v[4 * k + rr] + v[4 * p + rr]
          nv[4 * p + rr] = v[4 * k + rr] - v[4 * p + rr]
    v = nv

func lmCubicEncodeHost(x, compandA: float32; bits: int): uint32 =
  ## The LMCubic forward (lmq.cuh encode): Cardano's formula:
  ## q_half = x·inv_b·0.5, delta = fma(q_half, q_half, p3_cub),
  ## s = sqrt(delta), t = cbrt(q_half + s) + cbrt(q_half − s),
  ## idx = floor(fma(t, half_n, half_n)) clamped to [0, 2^bits − 1].
  ## The derived constants follow the CUDA constructor's fp32 order.
  let b = 1.0'f32 - compandA
  let invB = 1.0'f32 / b
  let p3 = compandA * invB * (1.0'f32 / 3.0'f32)
  let p3cub = p3 * p3 * p3
  let qHalf = x * invB * 0.5'f32
  let delta = fmaf(qHalf, qHalf, p3cub)
  let sq = sqrtf(delta)
  let t = refCbrt(qHalf + sq) + refCbrt(qHalf - sq)
  let idx = int(floorf(fmaf(t, float32(1 shl (bits - 1)), float32(1 shl (bits - 1)))))
  let qmax = (1 shl bits) - 1
  uint32(max(min(idx, qmax), 0))

func lmCubicDecodeHost(idx: uint32; compandA: float32; bits: int): float32 =
  ## The LMCubic inverse (lmq.cuh decode): t = fma(2·idx + 1, 1/N,
  ## −1), t2 = t·t, returns t·fma(t2, b, a).
  let b = 1.0'f32 - compandA
  let invN = 1.0'f32 / float32(1 shl bits)
  let t = fmaf(2.0'f32 * float32(idx) + 1.0'f32, invN, -1.0'f32)
  let t2 = t * t
  t * fmaf(t2, b, compandA)

func refQuantGroup(x: array[32, uint16]; bits, compander: int;
                      compandA: float32): tuple[words: array[8, uint32],
                                                 scale: uint16] =
  ## One group of 32 consecutive values quantized, the
  ## quant_block_x4 arithmetic (q_cache_kernels.cuh): load in the CUDA
  ## lane order (value j = 4·sl + r), H4+H8, the 1/sqrt(32) scale, the
  ## absmax + 1e-10 group scale (fp16 round-to-nearest-even (RNE)),
  ## the linear midpoint grid
  ## (`fmaf(v·inv_s, mf, mf)` floor-clamped to [0, 2^bits − 1]) or
  ## LMCubic encode, and the bit-plane pack (value j occupies bits
  ## [j·bits, (j+1)·bits) of the group's `bits` uint32 words, the
  ## single plane for bits {2,4,8}).
  var v: array[32, float32]
  for j in 0 ..< 32:
    v[j] = fp16ToFp32(x[j])
  hadamard32(v)
  for j in 0 ..< 32:
    v[j] = v[j] * R32
  var s = max(max(abs(v[0]), abs(v[1])), max(abs(v[2]), abs(v[3])))
  for j in 4 ..< 32:
    s = max(s, abs(v[j]))
  s = s + 1e-10'f32
  let invS = 1.0'f32 / s
  var q: array[32, uint32]
  if compander == 0:
    let mf = float32(1 shl (bits - 1))
    let qmax = (1 shl bits) - 1
    for j in 0 ..< 32:
      let t = fmaf(v[j] * invS, mf, mf)
      let qi = int(floorf(t))
      q[j] = uint32(max(min(qi, qmax), 0))
  else:
    for j in 0 ..< 32:
      q[j] = lmCubicEncodeHost(v[j] * invS, compandA, bits)
  let mask = (1'u32 shl bits) - 1
  for w in 0 ..< bits:
    var word = 0'u32
    for j in 0 ..< 32:
      if (j * bits) div 32 == w:
        word = word or ((q[j] and mask) shl ((j * bits) mod 32))
    result.words[w] = word
  result.scale = fp32ToFp16(s)

func refDequantGroup(words: array[8, uint32]; scale: uint16;
                        bits, compander: int; compandA: float32): array[32, uint16] =
  ## One group of 32 values dequantized, the dequant_block_x4
  ## arithmetic (q_cache_kernels.cuh): the bit-plane unpack, the fp16
  ## scale × 1/sqrt(32), the linear ((q − mh)·(s·inv_mf)) or LMCubic
  ## inverse, the H4+H8 rotate back (same order as the quant), and the
  ## fp16 RNE store.
  var q: array[32, uint32]
  let mask = (1'u32 shl bits) - 1
  for w in 0 ..< bits:
    let word = words[w]
    for j in 0 ..< 32:
      if (j * bits) div 32 == w:
        q[j] = (word shr ((j * bits) mod 32)) and mask
  let s = fp16ToFp32(scale) * R32
  var v: array[32, float32]
  if compander == 0:
    let mh = float32(1 shl (bits - 1)) - 0.5'f32
    let sm = s * (1.0'f32 / float32(1 shl (bits - 1)))
    for j in 0 ..< 32:
      v[j] = (float32(q[j]) - mh) * sm
  else:
    for j in 0 ..< 32:
      v[j] = lmCubicDecodeHost(q[j], compandA, bits) * s
  hadamard32(v)
  for j in 0 ..< 32:
    result[j] = fp32ToFp16(v[j])

# ════════════════════════════════════════
#  The per-combo check: quant bit-exact, dequant bit-exact, round trip
# ════════════════════════════════════════

func fnv1a(acc: var uint32; data: seq[uint32]) =
  ## FNV-1a over a uint32 stream (the quant output), byte-wise.
  for i in 0 ..< data.len:
    let b0 = uint32(data[i] and 0xFF'u32)
    let b1 = uint32((data[i] shr 8) and 0xFF'u32)
    let b2 = uint32((data[i] shr 16) and 0xFF'u32)
    let b3 = uint32((data[i] shr 24) and 0xFF'u32)
    acc = (acc xor b0) * 0x01000193'u32
    acc = (acc xor b1) * 0x01000193'u32
    acc = (acc xor b2) * 0x01000193'u32
    acc = (acc xor b3) * 0x01000193'u32

func fnv1a(acc: var uint32; data: seq[uint16]) =
  ## FNV-1a over a uint16 stream (the dequant output), byte-wise.
  for i in 0 ..< data.len:
    let b0 = uint32(data[i] and 0xFF'u16)
    let b1 = uint32(data[i] shr 8)
    acc = (acc xor b0) * 0x01000193'u32
    acc = (acc xor b1) * 0x01000193'u32

proc refQuant(d: KvCase; bits, compander: int; compandA: float32;
                 prefix: seq[int32]): seq[uint32] =
  ## The full reference quant over the case: for each seq and each
  ## appended row `y` (logical token y + prefix[s]), the K and V
  ## group quantizations written to the binding-0 layout
  ## [K words | V words | K scales | V scales]. Returns the expected
  ## quant buffer (the test sizes the regions by the PHYSICAL page
  ## pool: the kernel indexes by physical token position, and the
  ## gapped block tables leave unused holes that stay zero on both
  ## sides).
  let groups = d.groupsPerToken
  let poolTokens = d.numPages * 256
  let sTotal = poolTokens * groups
  let qWordsTotal = poolTokens * groups * bits
  let quantLen = 2 * qWordsTotal + 2 * sTotal
  result = newSeq[uint32](quantLen)
  for b in 0 ..< d.numSeqs:
    for y in 0 ..< d.rows:
      let tp = tokenPos(d, b, y + prefix[b])
      for g in 0 ..< groups:
        var xg: array[32, uint16]
        for j in 0 ..< 32:
          xg[j] = d.kSlab[tp * d.tokenDim + g * 32 + j]
        let r = refQuantGroup(xg, bits, compander, compandA)
        for w in 0 ..< bits:
          result[(tp * groups + g) * bits + w] = r.words[w]
        result[2 * qWordsTotal + tp * groups + g] = uint32(r.scale)
    for y in 0 ..< d.rows:
      let tp = tokenPos(d, b, y + prefix[b])
      for g in 0 ..< groups:
        var xg: array[32, uint16]
        for j in 0 ..< 32:
          xg[j] = d.vSlab[tp * d.tokenDim + g * 32 + j]
        let r = refQuantGroup(xg, bits, compander, compandA)
        for w in 0 ..< bits:
          result[qWordsTotal + (tp * groups + g) * bits + w] = r.words[w]
        result[2 * qWordsTotal + sTotal + tp * groups + g] = uint32(r.scale)

proc refDequant(d: KvCase; expQ: seq[uint32]; bits, compander: int;
                   compandA: float32; rows: int): seq[uint16] =
  ## The full reference dequant over the case's first `rows` logical
  ## tokens of each seq: K rows to [0, out_total), V rows to
  ## [out_total, 2·out_total), reading the reference quant's binding-0
  ## layout.
  let groups = d.groupsPerToken
  let poolTokens = d.numPages * 256
  let sTotal = poolTokens * groups
  let qWordsTotal = poolTokens * groups * bits
  let outTotal = poolTokens * d.tokenDim
  result = newSeq[uint16](2 * outTotal)
  for b in 0 ..< d.numSeqs:
    for y in 0 ..< rows:
      let tp = tokenPos(d, b, y)
      for g in 0 ..< groups:
        var words: array[8, uint32]
        for w in 0 ..< bits:
          words[w] = expQ[(tp * groups + g) * bits + w]
        let scale = uint16(expQ[2 * qWordsTotal + tp * groups + g] and 0xFFFF'u32)
        let row = refDequantGroup(words, scale, bits, compander, compandA)
        for j in 0 ..< 32:
          result[tp * d.tokenDim + g * 32 + j] = row[j]
    for y in 0 ..< rows:
      let tp = tokenPos(d, b, y)
      for g in 0 ..< groups:
        var words: array[8, uint32]
        for w in 0 ..< bits:
          words[w] = expQ[qWordsTotal + (tp * groups + g) * bits + w]
        let scale = uint16(expQ[2 * qWordsTotal + sTotal + tp * groups + g] and
                           0xFFFF'u32)
        let row = refDequantGroup(words, scale, bits, compander, compandA)
        for j in 0 ..< 32:
          result[outTotal + tp * d.tokenDim + g * 32 + j] = row[j]

type KvStats* = object
  nQuantWords*: int
  nQuantScales*: int
  nDequantRows*: int
  worstRoundTrip*: float32

proc runKvCombo*(engine: var auto; d: KvCase; bits, compander: int;
                 compandA: float32; tag: string;
                 st: var KvStats; hAll: var uint32): tuple[worstRt: float32, hash: uint32] =
  ## One (bits, compander) round-trip run over the case: the reference
  ## quant + the kernel quant (cache_seqlens = 0, binding-0 sliced:
  ## [K words | V words | K scales | V scales]), then the reference
  ## dequant of the reference quant + the kernel dequant of the kernel
  ## quant (cache_seqlens = rows, binding-0 [K rows | V rows]). Runs at
  ## layer 0 of a 1-layer pool (the plain pool). Checks: quant planes
  ## and scales and dequant
  ## rows 100% bit-exact vs the reference implementation (NaN =
  ## failure), the dequant |Δ| ≤ 1e-2 bound, and the round-trip worst
  ## |Δ| ≤ 64 bound (a backstop against NaN/inf/wild out-of-slab
  ## reads). Addressing/scale bugs are caught by the bit-exact
  ## compares. Returns the round-trip worst |Δ| (the KERNEL path) and
  ## the FNV hash of the combined quant + dequant outputs.
  let groups = d.groupsPerToken
  let poolTokens = d.numPages * 256
  let sTotal = poolTokens * groups
  let qWordsTotal = poolTokens * groups * bits
  let quantLen = 2 * qWordsTotal + 2 * sTotal
  let outTotal = poolTokens * d.tokenDim
  var prefix = newSeq[int32](d.numSeqs)

  # ── the reference quant ──
  let expQ = refQuant(d, bits, compander, compandA, prefix)

  # ── the kernel quant (cache_seqlens = 0: the appended rows) ──
  var kq = newSeq[uint32](quantLen)
  var rowsZero = newSeq[int32](d.numSeqs)
  engine.run << (grid: (groups div 4, d.rows, d.numSeqs), blk: (32, 1)) >> (
    "kvQuantD128", kq,
    (d.kSlab, d.vSlab, d.blockTable, rowsZero,
     int32(d.numSeqs), int32(d.maxPages), int32(d.tokenDim), int32(groups),
     int32(qWordsTotal), int32(1), int32(0),
     compandA, int32(bits), int32(compander)))

  # ── bit-exact planes + scales ──
  var nBitW = 0
  var nBitS = 0
  for i in 0 ..< 2 * qWordsTotal:
    if kq[i] == expQ[i]: inc nBitW
  for i in 0 ..< 2 * sTotal:
    if (kq[2 * qWordsTotal + i] and 0xFFFF'u32) ==
       (expQ[2 * qWordsTotal + i] and 0xFFFF'u32):
      inc nBitS

  # ── the reference dequant (of the reference quant) ──
  let expD = refDequant(d, expQ, bits, compander, compandA, d.rows)

  # ── the kernel dequant (of the kernel quant, the real path) ──
  var qK = newSeq[uint32](qWordsTotal)
  var qV = newSeq[uint32](qWordsTotal)
  for i in 0 ..< qWordsTotal:
    qK[i] = kq[i]
    qV[i] = kq[qWordsTotal + i]
  var sK = newSeq[uint16](sTotal)
  var sV = newSeq[uint16](sTotal)
  for i in 0 ..< sTotal:
    sK[i] = uint16(kq[2 * qWordsTotal + i] and 0xFFFF'u32)
    sV[i] = uint16(kq[2 * qWordsTotal + sTotal + i] and 0xFFFF'u32)
  var rowsCached = newSeq[int32](d.numSeqs)
  for i in 0 ..< d.numSeqs:
    rowsCached[i] = int32(d.rows)
  var kd = newSeq[uint16](2 * outTotal)
  engine.run << (grid: (groups div 4, d.rows, d.numSeqs), blk: (32, 1)) >> (
    "kvDequantD128", kd,
    (qK, qV, sK, sV, d.blockTable, rowsCached,
     int32(d.numSeqs), int32(d.maxPages), int32(d.tokenDim), int32(groups),
     int32(outTotal), int32(1), int32(0),
     compandA, int32(bits), int32(compander)))

  # ── the dequant bit-exact check + the round-trip |Δ| (kernel path) ──
  var nBitD = 0
  var worst = 0.0'f32
  var nBad = 0
  for i in 0 ..< 2 * outTotal:
    if kd[i] == expD[i]:
      inc nBitD
    else:
      let dv = fp16ToFp32(kd[i])
      let ov = fp16ToFp32(expD[i])
      let dabs = abs(dv - ov)
      if dabs != dabs or dabs > 1e-2'f32:      # NaN is a failure, not a pass
        inc nBad
      if dabs > worst: worst = dabs
  var worstRt = 0.0'f32
  var nGross = 0
  for b in 0 ..< d.numSeqs:
    for y in 0 ..< d.rows:
      let tp = tokenPos(d, b, y)
      for i in 0 ..< d.tokenDim:
        let got = fp16ToFp32(kd[tp * d.tokenDim + i])
        let orig = fp16ToFp32(d.kSlab[tp * d.tokenDim + i])
        let dabs = abs(got - orig)
        if dabs != dabs: inc nGross
        if dabs > worstRt: worstRt = dabs
        if dabs > 64.0'f32: inc nGross
      for i in 0 ..< d.tokenDim:
        let got = fp16ToFp32(kd[outTotal + tp * d.tokenDim + i])
        let orig = fp16ToFp32(d.vSlab[tp * d.tokenDim + i])
        let dabs = abs(got - orig)
        if dabs != dabs: inc nGross
        if dabs > worstRt: worstRt = dabs
        if dabs > 64.0'f32: inc nGross

  # ── the hash (quant + dequant outputs, FNV-1a) ──
  var h = 0x811C9DC5'u32
  fnv1a(h, kq)
  fnv1a(h, kd)

  st.nQuantWords += 2 * qWordsTotal
  st.nQuantScales += 2 * sTotal
  st.nDequantRows += 2 * outTotal
  if worstRt > st.worstRoundTrip: st.worstRoundTrip = worstRt
  fnv1a(hAll, kq)
  fnv1a(hAll, kd)

  let compName = (if compander == 0: "linear" else: "LMCubic")
  echo &"  [{tag}] bits={bits} {compName}: quant planes {nBitW}/{2*qWordsTotal} " &
       &"bit-exact, scales {nBitS}/{2*sTotal} bit-exact, dequant rows " &
       &"{nBitD}/{2*outTotal} bit-exact, round-trip worst |Δ| = {worstRt:.3e} " &
       &"(gross > 64: {nGross}), hash 0x{h:08X}"
  if nBitW != 2 * qWordsTotal or nBitS != 2 * sTotal or nBitD != 2 * outTotal:
    echo "  FAIL: quant planes/scales or dequant rows not bit-exact vs the " &
         "reference implementation"
    quit 1
  if nBad != 0:
    echo &"  FAIL: {nBad}/{2*outTotal} dequant elements beyond the 1e-2 bound " &
         "or NaN"
    quit 1
  if nGross != 0:
    echo "  FAIL: round-trip |Δ| > 64 (a backstop against NaN/inf/wild " &
         "out-of-slab reads; addressing/scale bugs are caught by the " &
         "bit-exact compares)"
    quit 1
  result = (worstRt, h)

# ════════════════════════════════════════
#  Fixed-value checks
# ════════════════════════════════════════

proc checkConvDrift*() =
  ## Checks the shared fp16 conversion helpers this test's three sides
  ## (buffer build, reference implementation, output readback) all
  ## consume: a silent rounding-mode drift must fail loudly here, not
  ## shift every case.
  doAssert fp32ToFp16(1.0005'f32) == 0x3C01'u16      # RNE: 1.0005 -> 1.0009765625
  doAssert fp32ToFp16(-1.0005'f32) == 0xBC01'u16
  doAssert fp32ToFp16(0.1'f32) == 0x2E66'u16        # -> 0.0999755859375
  doAssert fp32ToFp16(1.00048828125'f32) == 0x3C00'u16  # RNE tie -> even mantissa
  doAssert fp16ToFp32(0x3C01'u16) == 1.0009765625'f32
  doAssert fp16ToFp32(0x2E66'u16) == 0.0999755859375'f32

proc checkKvDrift*() =
  ## Checks the "exl3_kvquant" shape stream's first seed bytes and the
  ## value generator's first outputs: a silent drift must fail loudly
  ## here, not silently change every case's coverage.
  var shapes = initShapeCases("exl3_kvquant")
  let b0 = shapes.nextBytes()
  doAssert b0[0] == 0xF6'u8 and b0[1] == 0x01'u8 and b0[2] == 0x5E'u8,
    "exl3_kvquant:0 seed bytes changed"
  doAssert caseInRange(b0, 0, 1, 3) == 1, "exl3_kvquant:0 num_seqs changed"
  doAssert caseInRange(b0, 1, 0, 2) == 1, "exl3_kvquant:0 kv_heads changed"
  var r = seedRng(shapes)
  doAssert f16GridVal(r) == 1.625'f32, "kvquant value generator drifted"
  doAssert f16GridVal(r) == -0.375'f32, "kvquant value generator drifted"

# ════════════════════════════════════════
#  The guard cases
# ════════════════════════════════════════

proc runAppendCase(engine: var auto; shapes: var ShapeCases;
                   st: var KvStats; hAll: var uint32) =
  ## The append-semantics case: the quant kernel runs with a per-seq
  ## cached prefix (logical token y + cache_seqlens[s]), so the
  ## appended rows' regions must match the reference implementation
  ## and the prefix regions must stay untouched (marker 0xDEADBEEF).
  ## Checks the addressing the round-trip cases (all prefixes 0)
  ## cannot see.
  echo "\n── the append-semantics case: quant with per-seq cached prefixes ──"
  let b = shapes.nextBytes()
  let numSeqs = caseInRange(b, 0, 1, 3)
  let kvHeads = [1, 2, 4][caseInRange(b, 1, 0, 2)]
  let tokenDim = kvHeads * 128
  let groups = tokenDim div 32
  let rows = 64 + 32 * caseInRange(b, 2, 0, 3)   # 64..160 appended rows
  let prefix = 200 + caseInRange(b, 3, 0, 55)    # 200..255 cached rows
  # the per-seq prefixes grow by s, so the table must cover the largest
  # logical token: rows − 1 + prefix + num_seqs − 1. The prefix ≥ 200
  # with rows ≥ 64 puts every appended span across the 256-token page
  # boundary, so the composed y + cache_seqlens + page math is
  # exercised on both pages
  let totalRows = rows + prefix + numSeqs - 1
  let usedPages = (totalRows + 255) div 256
  var tables = newSeq[seq[int32]](numSeqs)
  var idCounter = 0
  for s in 0 ..< numSeqs:
    var row = newSeq[int32](usedPages)
    for j in 0 ..< usedPages:
      row[j] = int32(idCounter)
      let gap = (if j == usedPages - 1: 0 else: caseInRange(shapes.nextBytes(), 4 + s, 0, 1))
      idCounter += 1 + gap
    tables[s] = row
  let numPages = idCounter
  var blockTable = newSeq[int32](numSeqs * usedPages)
  for i in 0 ..< blockTable.len:
    blockTable[i] = -1'i32
  for s in 0 ..< numSeqs:
    for j in 0 ..< usedPages:
      blockTable[s * usedPages + j] = tables[s][j]
  var r = seedRng(shapes)
  let slabLen = numPages * 256 * tokenDim
  var kSlab = newSeq[uint16](slabLen)
  var vSlab = newSeq[uint16](slabLen)
  for i in 0 ..< slabLen:
    kSlab[i] = fp32ToFp16(f16GridVal(r) * 2.0'f32)
    vSlab[i] = fp32ToFp16(f16GridVal(r) * 2.0'f32)
  let d = KvCase(numSeqs: numSeqs, kvHeads: kvHeads, tokenDim: tokenDim,
                 groupsPerToken: groups, rows: rows, maxPages: usedPages,
                 numPages: numPages, blockTable: blockTable,
                 kSlab: kSlab, vSlab: vSlab)
  var prefixSeq = newSeq[int32](numSeqs)
  for s in 0 ..< numSeqs:
    prefixSeq[s] = int32(prefix + s)
  for bits in [2, 4, 8]:
    for compander in [0, 1]:
      let groupsL = d.groupsPerToken
      let poolTokens = d.numPages * 256
      let sTotal = poolTokens * groupsL
      let qWordsTotal = poolTokens * groupsL * bits
      let quantLen = 2 * qWordsTotal + 2 * sTotal
      let expQ = refQuant(d, bits, compander, 0.65'f32, prefixSeq)
      var kq = newSeq[uint32](quantLen)
      for i in 0 ..< quantLen:
        kq[i] = 0xDEADBEEF'u32
      engine.run << (grid: (groupsL div 4, d.rows, d.numSeqs), blk: (32, 1)) >> (
        "kvQuantD128", kq,
        (d.kSlab, d.vSlab, d.blockTable, prefixSeq,
         int32(d.numSeqs), int32(d.maxPages), int32(d.tokenDim), int32(groupsL),
         int32(qWordsTotal), int32(1), int32(0),
         0.65'f32, int32(bits), int32(compander)))
      var nBitW = 0
      var nBitS = 0
      var nMarkW = 0
      var nMarkS = 0
      for i in 0 ..< 2 * qWordsTotal:
        if kq[i] == expQ[i]:
          inc nBitW
        elif kq[i] == 0xDEADBEEF'u32:
          inc nMarkW
      for i in 0 ..< 2 * sTotal:
        let idx = 2 * qWordsTotal + i
        if (kq[idx] and 0xFFFF'u32) == (expQ[idx] and 0xFFFF'u32):
          inc nBitS
        elif kq[idx] == 0xDEADBEEF'u32:
          inc nMarkS
      let writtenW = 2 * qWordsTotal - nMarkW
      let writtenS = 2 * sTotal - nMarkS
      let compName = (if compander == 0: "linear" else: "LMCubic")
      echo &"  bits={bits} {compName}: appended-region words {nBitW}/{writtenW} " &
           &"and scales {nBitS}/{writtenS} bit-exact vs the reference " &
           &"implementation, {nMarkW + nMarkS} untouched prefix slots"
      if nBitW != writtenW or nBitS != writtenS:
        echo "  FAIL: the appended rows' quant regions must match the reference implementation"
        quit 1
      if nMarkW + nMarkS == 0:
        echo "  FAIL: the prefix regions must stay untouched (marker)"
        quit 1
      var h = 0x811C9DC5'u32
      fnv1a(h, kq)
      fnv1a(hAll, kq)
      st.nQuantWords += nBitW
      st.nQuantScales += nBitS
      echo &"    append hash 0x{h:08X}"

proc runBoundaryCase(engine: var auto; shapes: var ShapeCases;
                     st: var KvStats; hAll: var uint32) =
  ## The dequant boundary case: cache_seqlens < rows into a
  ## 0x7C00-prefilled uint16 buffer. Rows ≥ cache_seqlens must stay
  ## 0x7C00 and rows < cache_seqlens must match the reference dequant
  ## of the reference quant. Checks the dequant kernel's row guard.
  echo "\n── the dequant boundary case: cache_seqlens < rows, 0x7C00 prefill ──"
  let b = shapes.nextBytes()
  let numSeqs = caseInRange(b, 0, 1, 3)
  let kvHeads = [1, 2, 4][caseInRange(b, 1, 0, 2)]
  let tokenDim = kvHeads * 128
  let groups = tokenDim div 32
  let rows = 64 + 32 * caseInRange(b, 2, 0, 3)
  let usedPages = (rows + 255) div 256
  var tables = newSeq[seq[int32]](numSeqs)
  var idCounter = 0
  for s in 0 ..< numSeqs:
    var row = newSeq[int32](usedPages)
    for j in 0 ..< usedPages:
      row[j] = int32(idCounter)
      let gap = (if j == usedPages - 1: 0 else: caseInRange(shapes.nextBytes(), 3 + s, 0, 1))
      idCounter += 1 + gap
    tables[s] = row
  let numPages = idCounter
  var blockTable = newSeq[int32](numSeqs * usedPages)
  for i in 0 ..< blockTable.len:
    blockTable[i] = -1'i32
  for s in 0 ..< numSeqs:
    for j in 0 ..< usedPages:
      blockTable[s * usedPages + j] = tables[s][j]
  var r = seedRng(shapes)
  let slabLen = numPages * 256 * tokenDim
  var kSlab = newSeq[uint16](slabLen)
  var vSlab = newSeq[uint16](slabLen)
  for i in 0 ..< slabLen:
    kSlab[i] = fp32ToFp16(f16GridVal(r) * 2.0'f32)
    vSlab[i] = fp32ToFp16(f16GridVal(r) * 2.0'f32)
  let d = KvCase(numSeqs: numSeqs, kvHeads: kvHeads, tokenDim: tokenDim,
                 groupsPerToken: groups, rows: rows, maxPages: usedPages,
                 numPages: numPages, blockTable: blockTable,
                 kSlab: kSlab, vSlab: vSlab)
  # the covered prefix per seq: rows − (1 + s) keeps the guard live
  var cached = newSeq[int32](numSeqs)
  for s in 0 ..< numSeqs:
    cached[s] = int32(rows - 1 - s)
  for bits in [2, 4, 8]:
    for compander in [0, 1]:
      let groupsL = d.groupsPerToken
      let poolTokens = d.numPages * 256
      let sTotal = poolTokens * groupsL
      let qWordsTotal = poolTokens * groupsL * bits
      let outTotal = poolTokens * d.tokenDim
      let quantLen = 2 * qWordsTotal + 2 * sTotal
      var prefix = newSeq[int32](d.numSeqs)
      let expQ = refQuant(d, bits, compander, 0.65'f32, prefix)
      let expD = refDequant(d, expQ, bits, compander, 0.65'f32, rows)
      # the kernel quant covers all rows (cache_seqlens = 0)
      var kq = newSeq[uint32](quantLen)
      var rowsZero = newSeq[int32](d.numSeqs)
      engine.run << (grid: (groupsL div 4, d.rows, d.numSeqs), blk: (32, 1)) >> (
        "kvQuantD128", kq,
        (d.kSlab, d.vSlab, d.blockTable, rowsZero,
         int32(d.numSeqs), int32(d.maxPages), int32(d.tokenDim), int32(groupsL),
         int32(qWordsTotal), int32(1), int32(0),
         0.65'f32, int32(bits), int32(compander)))
      var qK = newSeq[uint32](qWordsTotal)
      var qV = newSeq[uint32](qWordsTotal)
      for i in 0 ..< qWordsTotal:
        qK[i] = kq[i]
        qV[i] = kq[qWordsTotal + i]
      var sK = newSeq[uint16](sTotal)
      var sV = newSeq[uint16](sTotal)
      for i in 0 ..< sTotal:
        sK[i] = uint16(kq[2 * qWordsTotal + i] and 0xFFFF'u32)
        sV[i] = uint16(kq[2 * qWordsTotal + sTotal + i] and 0xFFFF'u32)
      var kd = newSeq[uint16](2 * outTotal)
      for i in 0 ..< kd.len:
        kd[i] = 0x7C00'u16
      engine.run << (grid: (groupsL div 4, d.rows, d.numSeqs), blk: (32, 1)) >> (
        "kvDequantD128", kd,
        (qK, qV, sK, sV, d.blockTable, cached,
         int32(d.numSeqs), int32(d.maxPages), int32(d.tokenDim), int32(groupsL),
         int32(outTotal), int32(1), int32(0),
         0.65'f32, int32(bits), int32(compander)))
      # The covered rows must match the reference implementation. The guard rows must stay 0x7C00.
      var nBit = 0
      var nGuard = 0
      var nGuardBad = 0
      for s in 0 ..< d.numSeqs:
        for y in 0 ..< int(cached[s]):
          let tp = tokenPos(d, s, y)
          for i in 0 ..< d.tokenDim:
            if kd[tp * d.tokenDim + i] == expD[tp * d.tokenDim + i]:
              inc nBit
            else:
              echo "  FAIL: a covered K row differs from the reference implementation"
              quit 1
            if kd[outTotal + tp * d.tokenDim + i] ==
               expD[outTotal + tp * d.tokenDim + i]:
              inc nBit
            else:
              echo "  FAIL: a covered V row differs from the reference implementation"
              quit 1
        for y in int(cached[s]) ..< d.rows:
          let tp = tokenPos(d, s, y)
          for i in 0 ..< d.tokenDim:
            inc nGuard
            if kd[tp * d.tokenDim + i] != 0x7C00'u16:
              inc nGuardBad
            if kd[outTotal + tp * d.tokenDim + i] != 0x7C00'u16:
              inc nGuardBad
      let compName = (if compander == 0: "linear" else: "LMCubic")
      echo &"  bits={bits} {compName}: covered rows {nBit} bit-exact vs the " &
           &"reference implementation, {nGuard} guard slots untouched " &
           &"({nGuardBad} violations)"
      if nGuardBad != 0:
        echo "  FAIL: the dequant row guard wrote past cache_seqlens"
        quit 1
      st.nDequantRows += nBit
      var h = 0x811C9DC5'u32
      fnv1a(h, kd)
      fnv1a(hAll, kd)
      echo &"    boundary hash 0x{h:08X}"

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
  engine.run << (grid: (groups div 4, rows, numSeqs), blk: (32, 1)) >> (
    "kvQuantD128", outBuf,
    (kSlab, vSlab, blockTable, cacheZero,
     int32(numSeqs), int32(maxPages), int32(tokenDim),
     int32(groups), int32(qWordsTotal),
     int32(numLayers), int32(layer), compandA,
     int32(bits), int32(compander)))
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
  engine.run << (grid: (groups div 4, rows, numSeqs), blk: (32, 1)) >> (
    "kvDequantD128", kd,
    (qK, qV, sK, sV, blockTable, cacheRows,
     int32(numSeqs), int32(maxPages), int32(tokenDim),
     int32(groups), int32(outTotal),
     int32(numLayers), int32(layer), compandA,
     int32(bits), int32(compander)))
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

proc checkKvQuant() =
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

proc runCompanderSpot(engine: var auto) =
  ## The LMCubic compander's runtime `a` at a second operating point
  ## (a = 0.5, bits 2): the quant planes/scales + dequant rows must stay
  ## bit-exact vs the host reference implementation at a = 0.5. The
  ## fixed kv_heads=1 geometry (4 groups, grid.x = 1, 2 pages crossing
  ## 255/256, gapped page ids) also checks the group-count-1 launch
  ## shape end-to-end. Its slab comes from a fixed seed so the shape
  ## stream is untouched. Local accumulator, deliberately outside
  ## `hAll` and `st`, so this check adds no re-verification burden.
  echo "\n── the LMCubic compander at a = 0.5 (kv_heads 1, bits 2, LMCubic) ──"
  let maxPagesA05 = 2
  let numPagesA05 = 6
  let blockTableA05 = @[0'i32, 2'i32, 3'i32, 5'i32]  # seq0: pages 0,2 · seq1: pages 3,5
  var rA05 = XorShift(s: 0xA05A05'u32)
  let slabLenA05 = numPagesA05 * 256 * 128
  var kSlabA05 = newSeq[uint16](slabLenA05)
  var vSlabA05 = newSeq[uint16](slabLenA05)
  for i in 0 ..< slabLenA05:
    kSlabA05[i] = fp32ToFp16(f16GridVal(rA05) * 2.0'f32)
    vSlabA05[i] = fp32ToFp16(f16GridVal(rA05) * 2.0'f32)
  let dA05 = KvCase(numSeqs: 2, kvHeads: 1, tokenDim: 128,
                    groupsPerToken: 4, rows: 300, maxPages: maxPagesA05,
                    numPages: numPagesA05, blockTable: blockTableA05,
                    kSlab: kSlabA05, vSlab: vSlabA05)
  var stA05: KvStats
  var hA05 = 0x811C9DC5'u32
  discard runKvCombo(engine, dA05, 2, 1, 0.5'f32, tag = "a05", stA05, hA05)

# ════════════════════════════════════════
#  runKvQuantSuite
# ════════════════════════════════════════

proc runKvQuantSuite*(engine: var auto; shapes: var ShapeCases): uint32 =
  ## The full kvquant suite: the drift checks, the 3 round-trip paged
  ## geometries × bits {2,4,8} × {linear, LMCubic} (quant planes,
  ## scales and dequant rows bit-exact vs the host reference
  ## implementation, round-trip |Δ| bounded), the append-semantics
  ## and dequant-boundary guard cases, the layer-major separation
  ## spot, and the LMCubic a = 0.5 spot. Prints per-combo
  ## diagnostics, the round-trip error table and the aggregate.
  ## Returns the combined FNV hash of the round-trip, append and
  ## boundary outputs (the caller asserts it against the pinned
  ## value). Each combo gates its own checks with a hard `quit 1`,
  ## so the returned hash is only reached when every check passed.
  checkConvDrift()
  checkKvDrift()

  echo "\n── paged KV quant/dequant vs the q_cache_kernels.cuh reference implementation ──"
  var worstByBits: array[9, array[2, float32]]
  var st: KvStats
  var hAll = 0x811C9DC5'u32
  for dIdx in 0 ..< 3:
    var d = buildKvCase(shapes)
    echo &"  case {dIdx}: num_seqs={d.numSeqs} kv_heads={d.kvHeads} " &
         &"token_dim={d.tokenDim} groups_per_token={d.groupsPerToken} " &
         &"rows={d.rows} pages_per_seq={(d.rows + 255) div 256} " &
         &"num_pages={d.numPages} (gapped)"
    for bits in [2, 4, 8]:
      for compander in [0, 1]:
        let res = runKvCombo(engine, d, bits, compander, 0.65'f32,
                             tag = &"case{dIdx}", st, hAll)
        worstByBits[bits][compander] = max(worstByBits[bits][compander], res.worstRt)
        if dIdx == 0:
          # hash checks: the first case's combos must never drift
          let knownHashes: array[9, array[2, uint32]] = [
            [0x00000000'u32, 0x00000000'u32],   # bits 0 (unused)
            [0x00000000'u32, 0x00000000'u32],
            [0x8FF6034F'u32, 0x8D0CE614'u32],   # bits 2: linear, LMCubic
            [0x00000000'u32, 0x00000000'u32],
            [0x9CB3D317'u32, 0xD54573C3'u32],   # bits 4: linear, LMCubic
            [0x00000000'u32, 0x00000000'u32],
            [0x00000000'u32, 0x00000000'u32],
            [0x00000000'u32, 0x00000000'u32],
            [0x1138BC67'u32, 0x3C05071A'u32]]   # bits 8: linear, LMCubic
          doAssert res.hash == knownHashes[bits][compander],
            &"case 0 bits={bits} compander={compander} hash drifted"
  runAppendCase(engine, shapes, st, hAll)
  runBoundaryCase(engine, shapes, st, hAll)
  checkKvQuant()
  runCompanderSpot(engine)

  echo "\n── round-trip reconstruction error per bits (kernel path, worst |Δ| " &
       "over all cases) ──"
  echo "  bits |      linear |      LMCubic | LMCubic-vs-linear spread"
  for bits in [2, 4, 8]:
    let lin = worstByBits[bits][0]
    let cub = worstByBits[bits][1]
    echo &"  {bits:>4} | {lin:>10.3e} | {cub:>10.3e} | {cub - lin:+.3e}"
  echo "\n── aggregate ──"
  echo &"  quant words {st.nQuantWords} bit-exact, scales {st.nQuantScales} " &
       &"bit-exact, dequant rows {st.nDequantRows} bit-exact, " &
       &"round-trip worst |Δ| = " &
       &"{st.worstRoundTrip:.3e}, combined hash 0x{hAll:08X}"
  echo "\nkvquant check: quant planes + scales and dequant rows BIT-EXACT vs the " &
       "q_cache_kernels.cuh reference implementation across bits {2,4,8} × " &
       "{linear, LMCubic} at head_dim 128 on the 3 round-trip geometries " &
       "(the kernel dequant consumes the kernel quant — the real path); the " &
       "append-semantics case checks the y + cache_seqlens addressing with " &
       "marker-guarded prefix regions; the dequant boundary case checks the " &
       "row guard with a 0x7C00 prefill; the round-trip |Δ| table is the " &
       "lossy quantizer's documented reconstruction error"
  result = hAll
