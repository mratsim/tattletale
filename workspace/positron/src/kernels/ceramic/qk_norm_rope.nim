## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#     Fused qk-norm + NEOX rope (qk_norm_rope_fwd): Tile API port
#
# ############################################################

## Fused qk-norm + NEOX rotary forward on the ceramic Tile API. Per
## 8-row tile (one 32-lane threadgroup), the two-rounding qk-norm
## (`RNE(x·rmf)` then `RNE(γ16·v)`, the `rms_norm` arithmetic from
## exllamav3's fused rope kernel) followed by the NEOX half-tile
## rotation. The body has zero branches and zero runtime div/mod (the
## `xColBase div 1024` lowers to a shift: constant power of two).
##
## Tile geometry: one 8×128 tile per threadgroup, grid (1, tokens,
## headBlocks). The X view's token stride is `xTokenStride` (the qkv
## row width for the layer's q/k views). Each tile's 8 rows are one
## token's 8-head block (`threadgroup_position_in_grid.z` is the block
## index) plus the head-column offset `xColBase` (0 for q, H·D for
## k). The k tile's slots past Nkv read the v region, which the caller
## discards. The cos/sin tables use the same (token, head-block)
## decomposition with `cosTokenStride` rows per token. The output tile
## index is `gidY·headBlocks + gidZ`, so the output is
## (tokens·headBlocks·8, 128) row-major. The flat `(Mp, 128)` input
## contract is the special case xTokenStride = 8·128, cosTokenStride =
## 8, headBlocks = 1, xColBase = 0, grid (1, Mp div 8, 1).
##
## The four views (gd + local_tile_dyn addressing, origin components
## in tile-extent units: element (r, c) of an R×C tile at origin
## (o0, o1, o2, o3) reads ptr[o0·s0 + o1·s1 + o2·s2·R + o3·s3·C +
## r·s2 + c·s3]):
## - glX = gd(X, stride (xTokenStride, 0, 128, 1)), origin
##   (gidY, 0, gidZ + xColBase div 1024, 0): the tile's 8 rows are the
##   token's head rows gidZ·8 + xColBase div 128 + r, cols the 128 dims
##   (xColBase is a 128-multiple, so xColBase div 1024 = H div 8 rows
##   of block offset)
## - glG = gd(G, stride (0, 0, 0, 1)), origin (0, 0, 0, 0): rows
##   broadcast, cols the 128 dims
## - glCos/glSin = gd(..., stride (cosTokenStride·64, 0, 64, 1)),
##   origin (gidY, 0, gidZ, 0): the fp32 tables, 8×64 tiles at the
##   token's head-block rows
## - glOut = gd(Out, stride (8·128, 0, 128, 1)), origins (tile, 0, 0, 0)
##   and (tile, 0, 0, 1): the two 8×64 stores at column origins 0 and 1
##   of the tile (64 columns per half, one RNE fp16 round at the store)
##
## Norm sequence (fp32 state): x² tile, `row_sum` (rv), ·1/128, +eps,
## rsqrt (the rms col-vec ops), `mul_row`. Then the fp16 two-rounding:
## `x16 = fp16(x·rmf)` (one RNE round), `x16 = x16 · γ16` (fp16×fp16
## single rounding, the per-element `mulF16`, γ loaded as an fp16
## broadcast tile). Rotation (fp32 state, fp16 output): the norm
## tile's column halves [0, 64) and [64, 128) split by a per-lane
## register copy (`splitHalfFloat32`, same-lane, no shuffle), the cos/sin fp32 loads,
## the sin products, and the two rotation adds as explicit IEEE fused multiply-adds
## (`fma`: the Metal compiler does not contract cross-statement arithmetic,
## so the explicit form keeps each rotation add a single IEEE rounding).
## The results quantize to fp16 at the store (one RNE round each).
## The cos/sin come from host-precomputed fp32 tables (no in-kernel sincos).
##
## Known production gaps (documented, not fixed):
## - D fixed at 128 (the EXL3 head dim, which the kernel hardcodes).
## - H should be an 8-multiple: grid.z covers whole 8-head blocks
##   (the composed q view requires H % 8 == 0).
## - Concrete fp16 in / fp16 out only: no fp32 path.

import workspace/crucible
import workspace/ceramic

# The kernel is non-generic (the contract has no static params), so its tile types
# cannot take the `rt_l`/`rv` default atoms: those defaults
# call `getTileConfig`, which asserts a `metal:`/`cuda:` block context,
# and a non-generic proc body is typechecked on the host import.
# The explicit universal-atom enum members below are exactly what the defaults
# resolve to on both the Metal and CUDA backends.

# ═════════════════════════════════════════════════════════════════════
#  Local device extensions: the per-element arithmetic the fusion needs
#  ═════════════════════════════════════════════════════════════════════

proc fma(x, y, z: float32): float32 {.builtin.} = discard
  ## The IEEE fused multiply-add `x·y + z` with one rounding: the `{.builtin.}` pragma
  ## forwards the plain name to the backend, and MSL has `fma` natively.
  ## The two rotation adds spell the fma explicitly because the Metal compiler
  ## does not contract cross-statement fp32 arithmetic.

proc mulF16[A: static MmaAtom](
    dst: var RtLeft[float16, 8, 128, A],
    a, b: RtLeft[float16, 8, 128, A]) {.device.} =
  ## Per-element fp16 multiply with one rounding (the `mul` op
  ## semantics): the fp32 product of the promoted operands, one RNE
  ## fp16 round. The frag walk covers exactly the elements the loads
  ## produced (same lane→element mapping as `loadTile`), so a, b and dst agree elementwise.
  ## Aliasing dst with a or b is safe: the element reads complete before the element write.
  const rowTiles = 8 div A.getM()
  const colTiles = 128 div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst.frags[n][m].frag[v] =
          (a.frags[n][m].frag[v].to(float32) *
           b.frags[n][m].frag[v].to(float32)).to(float16)

proc splitHalfFloat32[A16, AF32: static MmaAtom](
    x1, x2: var RtLeft[float32, 8, 64, AF32],
    x16: RtLeft[float16, 8, 128, A16]) {.device.} =
  ## Splits the fp16 norm tile's column halves into two fp32 8×64
  ## tiles by a per-lane register copy: each lane copies its own fragment slots,
  ## `frags[0][m].frag[v]` with m ∈ 0..<8 → x1, and m ∈ 8..<16 → x2
  ## (same-lane, no shuffle: both atoms are 8×8×8 universal atoms with the same lane→cell mapping,
  ## asserted below).
  static:
    doAssert A16.getM() == AF32.getM() and A16.getN() == AF32.getN() and
      A16.getVpt() == AF32.getVpt(),
      "splitHalfFloat32: both atoms must share the lane→fragment cell mapping"
  const rowTiles = 8 div A16.getM()
  const colTiles = 64 div A16.getN()
  const vpt = A16.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        x1.frags[n][m].frag[v] = x16.frags[n][m].frag[v].to(float32)
        x2.frags[n][m].frag[v] = x16.frags[n][m + colTiles].frag[v].to(float32)

proc fmaMapFloat32[A: static MmaAtom](
    dst: var RtLeft[float32, 8, 64, A],
    a, b, c: RtLeft[float32, 8, 64, A]) {.device.} =
  ## Per-element fused multiply-add over one fp32 register tile:
  ## `dst = fma(a, b, c)` = a·b + c with one rounding (the `fmaMulAdd` rotary-add form).
  ## The frag walk covers exactly the elements the loads produced.
  const rowTiles = 8 div A.getM()
  const colTiles = 64 div A.getN()
  const vpt = A.getVpt()
  for n in 0 ..< rowTiles:
    for m in 0 ..< colTiles:
      for v in 0 ..< vpt:
        dst.frags[n][m].frag[v] =
          fma(a.frags[n][m].frag[v], b.frags[n][m].frag[v],
              c.frags[n][m].frag[v])

# ═════════════════════════════════════════════════════════════════════
#  The kernel
#  ═════════════════════════════════════════════════════════════════════

proc qk_norm_rope_fwd*(
    Out: ptr UncheckedArray[float16],    # (tiles·8, 128) fp16: the rope'd norm output
    X: ptr UncheckedArray[float16],      # (tokens·xTokenStride) fp16: the qkv view
    G: ptr UncheckedArray[float16],      # (128,) fp16: the norm weight
    Cos: ptr UncheckedArray[float32],    # (tokens·cosTokenStride, 64) fp32: the cos table
    Sin: ptr UncheckedArray[float32],    # (tokens·cosTokenStride, 64) fp32: the sin table
    xTokenStride: int32,                 # the X row width per token (nQkv, or 8·128 flat)
    cosTokenStride: int32,               # the cos/sin rows per token (slotCount, or 8 flat)
    headBlocks: int32,                   # the 8-head blocks per token (grid.z extent)
    xColBase: int32,                     # the head-column offset (0 for q, H·D for k)
    eps: float32) {.device.} =
  ## Computes one 8×128 qk-norm+rope tile per threadgroup, the module doc's sequence:
  ## the `rms_norm` arithmetic (x² tile, `row_sum`, the rstd col-vec ops, `mul_row`),
  ## the fp16 two-rounding (fp32→fp16 convert, then the fp16×fp16 γ `mulF16`),
  ## the rotary half-tile sequence (the `splitHalfFloat32` register copy,
  ## the cos/sin loads, the sin products, the two fused multiply-adds),
  ## and the two fp16 8×64 stores at column origins 0 and 1 of the tile `gidY·headBlocks + gidZ`.
  ## All norm state is fp32. The rotation state is fp32 with the final fp16 rounding at the store.
  ## `xColBase` must be a 1024-multiple (0 or H·D with D = 128 and H an 8-multiple).
  ## The flat path passes 0.
  let gidY = int32(threadgroup_position_in_grid.y)
  let gidZ = int32(threadgroup_position_in_grid.z)
  let tile = gidY * headBlocks + gidZ
  let xRowBlock = gidZ + xColBase div 1024
  let glX = gd(X, shape = (-1, -1, -1, -1), stride = (xTokenStride, 0, 128, 1))
  let glG = gd(G, shape = (-1, -1, -1, -1), stride = (0, 0, 0, 1))
  let glCos = gd(Cos, shape = (-1, -1, -1, -1), stride = (cosTokenStride * 64, 0, 64, 1))
  let glSin = gd(Sin, shape = (-1, -1, -1, -1), stride = (cosTokenStride * 64, 0, 64, 1))
  let glOut = gd(Out, shape = (-1, -1, -1, -1), stride = (8 * 128, 0, 128, 1))
  var xReg: rt_l(float32, 8, 128, UNIVERSAL_8x8x8_F32F32F32F32)
  var sq: rt_l(float32, 8, 128, UNIVERSAL_8x8x8_F32F32F32F32)
  var ss: rv(float32, 8, 128, UNIVERSAL_8x8x8_F32F32F32F32)
  var x16: rt_l(float16, 8, 128, UNIVERSAL_8x8x8_F32F16F16F32)
  var gamma16: rt_l(float16, 8, 128, UNIVERSAL_8x8x8_F32F16F16F32)
  loadTile(xReg, glX, (gidY, 0, xRowBlock, 0))
  sq.mul(xReg, xReg)
  ss.row_sum(sq)
  let invC = 1.0'f32 / 128.0'f32
  ss.mul(ss, invC)
  ss.add(ss, eps)
  ss.rsqrt(ss)
  xReg.mul_row(xReg, ss)
  convert(x16, xReg)
  loadTile(gamma16, glG, (0, 0, 0, 0))
  mulF16(x16, x16, gamma16)
  var x1: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var x2: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var x1s: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var x2s: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var cosT: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var sinT: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var t1: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var t2: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  splitHalfFloat32(x1, x2, x16)
  loadTile(cosT, glCos, (gidY, 0, gidZ, 0))
  loadTile(sinT, glSin, (gidY, 0, gidZ, 0))
  x1s.mul(x1, sinT)
  x2s.mul(x2, sinT)
  x2s.mul(x2s, -1.0'f32)
  fmaMapFloat32(t1, x1, cosT, x2s)
  fmaMapFloat32(t2, x2, cosT, x1s)
  storeTile(glOut, t1, (tile, 0, 0, 0))
  storeTile(glOut, t2, (tile, 0, 0, 1))
