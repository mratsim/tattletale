## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Paged KV-cache quant/dequant (exl3_kvquant): Tile API port
#
# ############################################################

## Paged KV-cache quant/dequant kernels on the ceramic Tile API,
## a port of the exllamav3 `cache/q_cache_kernels.cuh` mechanism:
## quant (fp16 K/V rows → packed planes + fp16 absmax scales)
## and dequant (planes → fp16 rows), bits {2, 4, 8} × {linear, LMCubic}.
## The kernels are opt-in (the production cache stays plain fp16).
##
## Quant chain (one group-block = 128 values, 4 groups of 32):
##
##   fp16 rows --> per-lane 4 values --> H4 in-reg --> H8 subgroup
##     --> × 1/sqrt(32) --> ÷ (absmax + 1e-10)  (fp32 subgroup max)
##     --> quantize: linear midpoint grid | LMCubic compander
##     --> bit-plane combine
##   out: packed uint32 words + fp16 group scales (RNE, lane-0)
##
## Dequant chain (the mirror):
##
##   packed words + fp16 scales --> unpack codes --> × 1/sqrt(32)
##     --> linear grid | LMCubic inverse --> H4 in-reg --> H8 subgroup
##     --> fp16 rows (RNE store)
##
## Frozen contract (from q_cache_kernels.cuh):
##   - a token row is quantized in groups of 32 consecutive values
##   - the H32 rotation is the H4 in-register over the lane's 4 values
##     plus the H8 over the 8-lane subgroup (shuffle stages at lane
##     xor 1/2/4, exact ±1 sign multiply)
##   - the group scale is absmax + 1e-10 over the 32 values, one fp16
##     per group (round-to-nearest-even), the fp32 reduce
##     order-independent, written by the subgroup's lane-0
##   - the quantizer is the linear midpoint grid or the LMCubic
##     compander, both clamped to [0, 2^bits − 1]
##   - the packed representation is the CUDA bit-plane layout: value j
##     of a group occupies bits [j·bits, (j+1)·bits) of the group's
##     uint32 words, one plane per bits {2, 4, 8}
##   - the quant is LOSSY by design: round-trip error is reported per
##     bits, never asserted ≤ 1 ulp
##
## Paged layout: the K/V slabs are the layer-major pool (num_pages,
## num_layers, PAGE_SIZE, token_dim) fp16 with row stride
## `token_dim = kv_heads·head_dim` (head-dim contiguous). The block table
## is the dense (num_seqs, max_pages) int32 table with -1 padding,
## and `cache_seqlens` (num_seqs) marks the cached prefix per seq.
## PAGE_SIZE is the static 256 (the exllamav3 CQ_PAGE_SIZE),
## so the page math is shift/mask (zero div/mod in the kernel body).
## A token's physical position in the layer-major pool:
## `pageId·(num_layers·PAGE_SIZE) + layer·PAGE_SIZE + (tokenIdx and 255)`.
## The packed planes/scales buffers follow the same layer-major
## addressing: `q_words_total = num_pages·num_layers·PAGE_SIZE·groups·bits`
## spans the whole pool. A token's quantized row:
## groups_per_token·bits uint32 words (groups of 32 consecutive
## values along the full token row) plus groups_per_token fp16 scales.
##
## Binding 0 (the engine reads back only binding 0). The quant kernel
## packs all four outputs into the uint32 buffer:
## [K words | V words | K scales | V scales], the scale slots carrying
## the f16 bit pattern in bits [0, 16). The K-word total
## `q_words_total` and the derived scale total
## `q_words_total shr log2(bits)` locate the regions. The dequant
## kernel packs the two fp16 row streams into the uint16 buffer:
## [K rows | V rows], the V half at `out_total` elements. The kernel
## signatures carry the buffer bindings in order: outBuf first,
## then the inputs, then the runtime scalars.
## `bits`/`compander` are STATIC (the per-bits/per-compander
## instantiations). `pageSize` = 256 and `D` = 128, both static.
##
## Tile-program note. The group-block is a per-lane 4-value chunk
## (lane l owns values 4·l..4·l+3 of the 128-span), not an mma
## fragment layout, so the loads/stores are per-lane global accesses.
## The ops own the lane machinery (H4/H8 shuffles, the subgroup scale
## reduce, the in-register plane combine), and the kernel body sees no
## lane bit, no shuffle, no packing word. There is no shared memory
## (the CUDA sh_pack staging has no `{.device.}` equivalent).
##
## Known production gaps (documented, not fixed):
## - token_dim: 128-multiple, groups_per_token: 4-multiple.
##   Grid.x covers whole group-blocks only. Partial spans fall
##   outside the contract.
## - Compander `a` is runtime (0.65 production). The derived LMCubic
##   constants are recomputed per block in fp32 (the CUDA constructor
##   order).

import workspace/crucible
import workspace/ceramic

# ════════════════════════════════════════
#  Device math builtins (module-local forwards to the MSL natives)
# ════════════════════════════════════════

proc fma(x, y, z: float32): float32 {.builtin.} = discard
  ## The IEEE fused multiply-add `x·y + z` with one rounding,
  ## forwarded to the MSL (Metal Shading Language) native `fma`
  ## (bit-exact vs host `fmaf` on the finite-normal domain).

proc floor(x: float32): float32 {.builtin.} = discard
  ## Device-side `floor`, forwarded to MSL's native `floor`
  ## (bit-identical to host `floorf`).

proc sqrt(x: float32): float32 {.builtin.} = discard
  ## Device-side `sqrt`, forwarded to MSL's native `sqrt`
  ## (correctly rounded, bit-identical to host `sqrtf` on the finite-normal domain).

proc fabs(x: float32): float32 {.builtin.} = discard
  ## Device-side `abs`, forwarded to MSL's native `fabs`. The
  ## distinct name avoids the stdlib float `abs` overload (whose
  ## generic body carries a `when nimvm` that crucible cannot lower).

proc fmax(a, b: float32): float32 {.builtin.} = discard
  ## Device-side `max`, forwarded to MSL's native `fmax`. Same
  ## rationale as `fabs`. The values here are never NaN, so the MSL
  ## fmax is bit-identical to the CUDA fmaxf.

# ════════════════════════════════════════
#  The H32 rotation: the had_4_inreg / had_8_subgroup pair (q_cache_kernels.cuh)
# ════════════════════════════════════════

proc had4InReg(v: var array[4, float32]) {.device.} =
  ## The H4 butterfly over the lane's 4 consecutive values, in place:
  ## the q_cache_kernels.cuh `had_4_inreg` arithmetic in fp32.
  let s0 = v[0] + v[1]
  let d0 = v[0] - v[1]
  let s1 = v[2] + v[3]
  let d1 = v[2] - v[3]
  v[0] = s0 + s1
  v[1] = d0 + d1
  v[2] = s0 - s1
  v[3] = d0 - d1

proc had8Subgroup(v: var array[4, float32]; lane: uint32) {.device.} =
  ## The H8 butterfly over the 8-lane subgroup, in place on the lane's
  ## 4 values: three shuffle stages at deltas 1/2/4 (lane xor delta,
  ## which stays inside the subgroup). The lane with the stage bit set
  ## negates its own value by an exact ±1 multiply (the CUDA sign-mask
  ## trick) before adding the partner's, the q_cache_kernels.cuh
  ## `had_8_subgroup` sequence. The shuffle reads the partner's
  ## PRE-stage value (SIMT lockstep: the shuffle instruction runs
  ## before any lane's write).
  let sgn1 = 1.0'f32 - 2.0'f32 * float32(lane and 1'u32)
  for s in 0'i32 ..< 4:
    v[s] = v[s] * sgn1 + simdShuffle(v[s], lane xor 1'u32)
  let sgn2 = 1.0'f32 - 2.0'f32 * float32((lane and 2'u32) shr 1)
  for s in 0'i32 ..< 4:
    v[s] = v[s] * sgn2 + simdShuffle(v[s], lane xor 2'u32)
  let sgn4 = 1.0'f32 - 2.0'f32 * float32((lane and 4'u32) shr 2)
  for s in 0'i32 ..< 4:
    v[s] = v[s] * sgn4 + simdShuffle(v[s], lane xor 4'u32)

# ════════════════════════════════════════
#  The defined cbrt: the LMCubic encode's cube root (MSL has no cbrt)
# ════════════════════════════════════════

const
  CbC0 = 1.1374176813308194'f32
  CbC1 = 0.1292479927845696'f32
  CbC2 = -0.0073754847563900945'f32
  CbC3 = 0.000702055045675154'f32
  CbC4 = -7.892353387775054e-05'f32
  CbCbrt2 = 1.2599210498948732'f32
  CbCbrt4 = 1.5874010519681994'f32
  CbThird = 0.3333333432674408'f32
  CbInv3 = 0.3333333432674408'f32

proc cbrtSeed(m: float32): float32 {.device.} =
  ## The degree-4 Chebyshev seed for cbrt(m), m in [1, 2): Clenshaw
  ## on u = 2m − 3, max relative error ~1.3e-5 (measured), which the
  ## two Newton iterations reduce to the final accuracy.
  let u = 2.0'f32 * m - 3.0'f32
  var b1 = 0.0'f32
  var b2 = 0.0'f32
  var b = 2.0'f32 * u * b1 - b2 + CbC4
  b2 = b1
  b1 = b
  b = 2.0'f32 * u * b1 - b2 + CbC3
  b2 = b1
  b1 = b
  b = 2.0'f32 * u * b1 - b2 + CbC2
  b2 = b1
  b1 = b
  b = 2.0'f32 * u * b1 - b2 + CbC1
  b2 = b1
  b1 = b
  u * b1 - b2 + CbC0

proc cbrtDefined(x: float32): float32 {.device.} =
  ## Defined fp32 cube root, the shared stand-in for the CUDA `cbrtf`
  ## (MSL has no cbrt): exponent split x = m·2^e with m in [1, 2) via an exact
  ## power-of-two ladder, the seed cbrt(m)·2^(r/3) with r = e mod 3 folded
  ## through the cbrt(2)/cbrt(4) constants, two Newton iterations
  ## `(2y + x/y²)/3` on the full value. ~2 fp32 ulp vs the correctly rounded
  ## cbrtf (measured).
  ## The ascending ladder is a while loop (≤ 4 iterations in the
  ## LMCubic domain, |input| ≥ ~0.08, ≤ 126 in general). Subnormal
  ## inputs never reach the ladder (the LMCubic inputs are
  ## qHalf ± sqrt(qHalf² + p3_cub), |input| ≥ ~0.08).
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
  let q = int32(floor(float32(e) * CbInv3))
  let r = e - 3 * q
  var y = cbrtSeed(m)
  if r == 1:
    y = y * CbCbrt2
  else:
    if r == 2:
      y = y * CbCbrt4
  y = y * exp2(float32(q))
  y = (2.0'f32 * y + ax / (y * y)) * CbThird
  y = (2.0'f32 * y + ax / (y * y)) * CbThird
  (if x == 0.0'f32: x else: sgn * y)

# ════════════════════════════════════════
#  The LMCubic compander: the lmq.cuh forward + inverse (a = 0.65)
# ════════════════════════════════════════

proc lmCubicEncode(x: float32; compandA: float32; bits: static int): uint32 {.device.} =
  ## The LMCubic forward (lmq.cuh `lm_cubic_encode`): Cardano's
  ## formula for the depressed cubic `b·t³ + a·t = x` (the cbrt pair
  ## on qHalf ± sqrt(qHalf² + p3_cub)), the index clamped to
  ## [0, 2^bits − 1]. `compandA` is the runtime `a` (0.65 production).
  ## The derived constants b = 1−a, inv_b, p3 = a·inv_b/3, p3_cub are
  ## computed in fp32 exactly as the CUDA constructor does.
  let b = 1.0'f32 - compandA
  let invB = 1.0'f32 / b
  let p3 = compandA * invB * (1.0'f32 / 3.0'f32)
  let p3cub = p3 * p3 * p3
  let qHalf = x * invB * 0.5'f32
  let delta = fma(qHalf, qHalf, p3cub)
  let sq = sqrt(delta)
  let t = cbrtDefined(qHalf + sq) + cbrtDefined(qHalf - sq)
  let idx = int32(floor(fma(t, float32(1 shl (bits - 1)), float32(1 shl (bits - 1)))))
  let qmax = (1 shl bits) - 1
  uint32(if idx < 0: 0 else: (if idx > qmax: qmax else: idx))

proc lmCubicDecode(idx: uint32; compandA: float32; bits: static int): float32 {.device.} =
  ## The LMCubic inverse (lmq.cuh `lm_cubic_decode`), the fp32
  ## sequence in the CUDA order.
  let b = 1.0'f32 - compandA
  let invN = 1.0'f32 / float32(1 shl bits)
  let t = fma(2.0'f32 * float32(idx) + 1.0'f32, invN, -1.0'f32)
  let t2 = t * t
  t * fma(t2, b, compandA)

# ════════════════════════════════════════
#  kvQuantBlock: one group-block (4 groups = 128 values) quantized
# ════════════════════════════════════════

proc kvQuantBlock(
    outBuf: ptr UncheckedArray[uint32],   # packed words + fp16 scale bits (low 16)
    inBuf: ptr UncheckedArray[uint16],    # the fp16 rows
    inBase, wordBase, scaleBase: int32,   # per-token bases
    bits: static int, compander: static int, compandA: float32) {.device.} =
  ## Quantizes the 128-value span at `inBase` (4 groups of 32
  ## consecutive values) into the group-block's packed words at
  ## `wordBase` (groups·bits uint32 per token) and the fp16 group
  ## scales at `scaleBase` (one u32 slot per group, the f16 bits in
  ## [0, 16), the binding-0 packing). Lane `l` owns the 4 consecutive
  ## values at 4·l. `bits` ∈ {2, 4, 8} (one plane each), `compander`
  ## 0 = linear / 1 = LMCubic.
  static: doAssert bits in {2, 4, 8}
  static: doAssert compander in {0, 1}
  let lane = uint32(thread_index_in_threadgroup)
  let l = int32(lane)
  let sg = l shr 3
  let sl = l and 7
  var v: array[4, float32]
  for r in 0'i32 ..< 4:
    v[r] = inBuf[inBase + 4 * l + r].asFp16().to(float32)
  had4InReg(v)
  had8Subgroup(v, lane)
  for r in 0'i32 ..< 4:
    v[r] = v[r] * 0.17677669529663688110'f32
  var s = fabs(v[0])
  for r in 1'i32 ..< 4:
    s = fmax(s, fabs(v[r]))
  s = s + 1e-10'f32
  s = fmax(s, simdShuffle(s, lane xor 1'u32))
  s = fmax(s, simdShuffle(s, lane xor 2'u32))
  s = fmax(s, simdShuffle(s, lane xor 4'u32))
  let invS = 1.0'f32 / s
  var q: array[4, uint32]
  when compander == 0:
    let mf = float32(1 shl (bits - 1))
    let qmax = (1 shl bits) - 1
    for r in 0'i32 ..< 4:
      let t = fma(v[r] * invS, mf, mf)
      let qi = int32(floor(t))
      q[r] = uint32(if qi < 0: 0 else: (if qi > qmax: qmax else: qi))
  else:
    for r in 0'i32 ..< 4:
      q[r] = lmCubicEncode(v[r] * invS, compandA, bits)
  when bits == 8:
    let field = q[0] or (q[1] shl 8) or (q[2] shl 16) or (q[3] shl 24)
    outBuf[wordBase + sg * 8 + sl] = field
  elif bits == 4:
    let field = q[0] or (q[1] shl 4) or (q[2] shl 8) or (q[3] shl 12)
    var f = field
    let pv = simdShuffle(f, lane xor 1'u32)
    if (sl and 1) == 0:
      f = f or (pv shl 16)
      outBuf[wordBase + sg * 4 + (sl shr 1)] = f
  else:
    let field = q[0] or (q[1] shl 2) or (q[2] shl 4) or (q[3] shl 6)
    var f = field
    let pv1 = simdShuffle(f, lane xor 1'u32)
    if (sl and 1) == 0:
      f = f or (pv1 shl 8)
    let pv2 = simdShuffle(f, lane xor 2'u32)
    if (sl and 2) == 0:
      f = f or (pv2 shl 16)
    if (sl and 3) == 0:
      outBuf[wordBase + sg * 2 + (sl shr 2)] = f
  if sl == 0:
    outBuf[scaleBase + sg] = uint32(s.to(float16).asU16())

# ════════════════════════════════════════
#  kvDequantBlock: one group-block (4 groups = 128 values) dequantized
# ════════════════════════════════════════

proc kvDequantBlock(
    outBuf: ptr UncheckedArray[uint16],   # the fp16 rows
    inWords: ptr UncheckedArray[uint32],  # the packed words
    inScales: ptr UncheckedArray[uint16], # the fp16 group scales
    wordBase, scaleBase, outBase: int32,
    bits: static int, compander: static int, compandA: float32) {.device.} =
  ## Dequantizes one group-block back to fp16 rows: the lane's 4 codes
  ## unpacked from the group's words (single global reads, the CUDA
  ## two-word funnel window, no cross-lane traffic for the payload),
  ## the fp16 scale × 1/sqrt(32), the linear / LMCubic inverse, the
  ## H4+H8 rotate back (same order as the quant), and the fp16 RNE
  ## store at `outBase` + 4·lane. `bits` ∈ {2, 4, 8}, `compander`
  ## 0 = linear / 1 = LMCubic.
  static: doAssert bits in {2, 4, 8}
  static: doAssert compander in {0, 1}
  let lane = uint32(thread_index_in_threadgroup)
  let l = int32(lane)
  let sg = l shr 3
  let sl = l and 7
  var q: array[4, uint32]
  when bits == 8:
    let w = inWords[wordBase + sg * 8 + sl]
    q[0] = w and 0xFF'u32
    q[1] = (w shr 8) and 0xFF'u32
    q[2] = (w shr 16) and 0xFF'u32
    q[3] = (w shr 24) and 0xFF'u32
  elif bits == 4:
    let w = inWords[wordBase + sg * 4 + (sl shr 1)] shr (16 * (sl and 1))
    q[0] = w and 0xF'u32
    q[1] = (w shr 4) and 0xF'u32
    q[2] = (w shr 8) and 0xF'u32
    q[3] = (w shr 12) and 0xF'u32
  else:
    let w = inWords[wordBase + sg * 2 + (sl shr 2)] shr (8 * (sl and 3))
    q[0] = w and 0x3'u32
    q[1] = (w shr 2) and 0x3'u32
    q[2] = (w shr 4) and 0x3'u32
    q[3] = (w shr 6) and 0x3'u32
  let s = inScales[scaleBase + sg].asFp16().to(float32) *
          0.17677669529663688110'f32
  var v: array[4, float32]
  when compander == 0:
    let mh = float32(1 shl (bits - 1)) - 0.5'f32
    let sm = s * (1.0'f32 / float32(1 shl (bits - 1)))
    for r in 0'i32 ..< 4:
      v[r] = (float32(q[r]) - mh) * sm
  else:
    for r in 0'i32 ..< 4:
      v[r] = lmCubicDecode(q[r], compandA, bits) * s
  had4InReg(v)
  had8Subgroup(v, lane)
  for r in 0'i32 ..< 4:
    outBuf[outBase + 4 * l + r] = v[r].to(float16).asU16()

# ════════════════════════════════════════
#  The kernels
# ════════════════════════════════════════

proc kv_quant_fwd*(
    outBuf: ptr UncheckedArray[uint32],   # binding 0: [K words | V words | K scales | V scales]
    kIn, vIn: ptr UncheckedArray[uint16], # (num_pages, num_layers, PAGE_SIZE, token_dim) fp16 rows
    block_table: ptr UncheckedArray[int32],   # (num_seqs, max_pages) dense, -1 padding
    cache_seqlens: ptr UncheckedArray[int32], # (num_seqs)
    num_seqs, max_pages, token_dim, groups_per_token, q_words_total,
    num_layers, layer: int32,
    compand_a: float32,
    bits: static int, compander: static int,
    pageSize: static int, D: static int) {.device.} =
  ## Quantizes the appended K/V rows (one group-block per threadgroup)
  ## into the packed planes + fp16 scales, the exllamav3 paged quant
  ## semantics on the layer-major slab layout. The logical token
  ## `y + cache_seqlens[z]` maps through the block table and the
  ## `layer` scalar to the physical layer-major position. The
  ## group-block's 128-value span is quantized by `kvQuantBlock` into
  ## the binding-0 regions (see the module doc).
  ## Grid: (groups_per_token div 4, tokens, num_seqs), 32 lanes.
  ## x = group-block, y = token, z = seq.
  static: doAssert bits in {2, 4, 8}
  static: doAssert compander in {0, 1}
  static: doAssert pageSize == 256
  static: doAssert D == 128
  let batch = int32(threadgroup_position_in_grid.z)
  let token = int32(threadgroup_position_in_grid.y)
  let gBlock = int32(threadgroup_position_in_grid.x)
  let tokenIdx = token + cache_seqlens[batch]
  let pageIdx = tokenIdx shr 8
  let pageId = block_table[batch * max_pages + pageIdx]
  let tokenPos = pageId * num_layers * 256 + layer * 256 + (tokenIdx and 255)
  let inBase = tokenPos * token_dim + gBlock * 128
  let wordBase = tokenPos * (groups_per_token * bits) + gBlock * (4 * bits)
  let scaleBase = tokenPos * groups_per_token + gBlock * 4
  when bits == 2:
    let sTotal = q_words_total shr 1
  elif bits == 4:
    let sTotal = q_words_total shr 2
  else:
    let sTotal = q_words_total shr 3
  kvQuantBlock(outBuf, kIn, inBase, wordBase, scaleBase + 2 * q_words_total,
               bits, compander, compand_a)
  kvQuantBlock(outBuf, vIn, inBase, wordBase + q_words_total,
               scaleBase + 2 * q_words_total + sTotal,
               bits, compander, compand_a)

proc kv_dequant_fwd*(
    outBuf: ptr UncheckedArray[uint16],   # binding 0: [K rows | V rows] fp16
    qK, qV: ptr UncheckedArray[uint32],   # the packed planes (num_pages, num_layers, PAGE_SIZE, groups·bits)
    sK, sV: ptr UncheckedArray[uint16],   # the fp16 scales (num_pages, num_layers, PAGE_SIZE, groups)
    block_table: ptr UncheckedArray[int32],   # (num_seqs, max_pages) dense, -1 padding
    cache_seqlens: ptr UncheckedArray[int32], # (num_seqs)
    num_seqs, max_pages, token_dim, groups_per_token, out_total,
    num_layers, layer: int32,
    compand_a: float32,
    bits: static int, compander: static int,
    pageSize: static int, D: static int) {.device.} =
  ## Dequantizes the cached K/V rows (one group-block per threadgroup)
  ## from the packed planes + fp16 scales back to fp16 rows, the
  ## round-trip mirror of `kv_quant_fwd`. The logical token `y` (the
  ## seq's cached rows [0, cache_seqlens[z])) maps through the block
  ## table and the `layer` scalar to the physical layer-major
  ## position. `kvDequantBlock` writes the K row stream to [0,
  ## out_total) and the V stream to [out_total, 2·out_total) of the
  ## binding-0 buffer.
  ## Grid: (groups_per_token div 4, tokens, num_seqs), 32 lanes.
  ## x = group-block, y = token, z = seq.
  static: doAssert bits in {2, 4, 8}
  static: doAssert compander in {0, 1}
  static: doAssert pageSize == 256
  static: doAssert D == 128
  let batch = int32(threadgroup_position_in_grid.z)
  let token = int32(threadgroup_position_in_grid.y)
  let gBlock = int32(threadgroup_position_in_grid.x)
  if token >= cache_seqlens[batch]:
    return
  let tokenIdx = token
  let pageIdx = tokenIdx shr 8
  let pageId = block_table[batch * max_pages + pageIdx]
  let tokenPos = pageId * num_layers * 256 + layer * 256 + (tokenIdx and 255)
  let outBase = tokenPos * token_dim + gBlock * 128
  let wordBase = tokenPos * (groups_per_token * bits) + gBlock * (4 * bits)
  let scaleBase = tokenPos * groups_per_token + gBlock * 4
  kvDequantBlock(outBuf, qK, sK, wordBase, scaleBase, outBase,
                 bits, compander, compand_a)
  kvDequantBlock(outBuf, qV, sV, wordBase, scaleBase, outBase + out_total,
                 bits, compander, compand_a)
