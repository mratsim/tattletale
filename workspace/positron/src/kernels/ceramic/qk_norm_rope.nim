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

## Fused qk-norm + NEOX rotary forward on the ceramic Tile API.
##
## Contract: one 8×128 tile computes the two-rounding qk-norm
## and the NEOX half-tile rotation. The qk-norm rounds twice:
## `RNE(x·rmf)` then the fp16×fp16 γ multiply.
##
## Dataflow:
##
##     x --> x² --> row_sum --> ·1/128 --> +ε --> rsqrt --> x·rmf
##     x·rmf --> fp16 round --> γ16 multiply --> x16
##     x16 --> halves (x1, x2)
##     x1 --> Out[0, 64)   = x1·cos - x2·sin
##     x2 --> Out[64, 128) = x2·cos + x1·sin
##
## All norm state is fp32. The rotation runs in fp32 with explicit
## fused multiply-adds. The results quantize to fp16 at the store.
## The cos/sin come from host-precomputed fp32 tables.
##
## Buffers:
##   - Out: (tokens·headBlocks·8, 128) fp16 row-major output
##   - X: (tokens·xTokenStride) fp16, the qkv view
##   - G: (128,) fp16 norm weight, rows broadcast
##   - Cos, Sin: (tokens·cosTokenStride, 64) fp32 rotary tables
##
## Tile facts:
##   - each tile's 8 rows are one token's 8-head block
##     at the head-column offset `xColBase` (0 for q, H·D for k)
##   - the k tile's slots past Nkv read the v region, which the caller
##     discards
##   - the flat (Mp, 128) input contract is the special case
##     `xTokenStride = 8·128, cosTokenStride = 8, headBlocks = 1, xColBase = 0`
##
## Known production gaps (documented, not fixed):
## - D fixed at 128 (the EXL3 head dim, which the kernel hardcodes).
## - H should be an 8-multiple. The composed q view requires H % 8 == 0.
## - Concrete fp16 in / fp16 out only: no fp32 path.

import workspace/crucible
import workspace/ceramic
import ./exl3_ops

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
  ## Computes the module doc's contract for one 8×128 tile:
  ##   - the `rms_norm` arithmetic (x², row_sum, rsqrt, mul_row)
  ##   - the fp16 two-rounding γ multiply
  ##   - the rotary half-tile rotation
  ## The two fp16 8×64 stores write to column origins 0 and 1.
  ## All norm state is fp32.
  ## The rotation state is fp32 with the final fp16 rounding at the store.
  ## `xColBase` must be a 1024-multiple (0 or H·D with D = 128 and H an 8-multiple).
  ## The flat path passes 0.
  let gidY = int32(threadgroup_position_in_grid.y)
  let gidZ = int32(threadgroup_position_in_grid.z)
  let tile = gidY * headBlocks + gidZ
  let xRowBlock = gidZ + xColBase div 1024
  let glX = X.gd(shape = (-1, -1, -1, -1), stride = (xTokenStride, 0, 128, 1))
  let glG = G.gd(shape = (-1, -1, -1, -1), stride = (0, 0, 0, 1))
  let glCos = Cos.gd(shape = (-1, -1, -1, -1), stride = (cosTokenStride * 64, 0, 64, 1))
  let glSin = Sin.gd(shape = (-1, -1, -1, -1), stride = (cosTokenStride * 64, 0, 64, 1))
  let glOut = Out.gd(shape = (-1, -1, -1, -1), stride = (8 * 128, 0, 128, 1))
  var xReg: rt_l(float32, 8, 128, UNIVERSAL_8x8x8_F32F32F32F32)
  var sq: rt_l(float32, 8, 128, UNIVERSAL_8x8x8_F32F32F32F32)
  var ss: rv(float32, 8, 128, UNIVERSAL_8x8x8_F32F32F32F32)
  var x16: rt_l(float16, 8, 128, UNIVERSAL_8x8x8_F32F16F16F32)
  var gamma16: rt_l(float16, 8, 128, UNIVERSAL_8x8x8_F32F16F16F32)
  xReg.loadTile(glX, (gidY, 0, xRowBlock, 0))
  sq.mul(xReg, xReg)
  ss.row_sum(sq)
  let invC = 1.0'f32 / 128.0'f32
  ss.mul(ss, invC)
  ss.add(ss, eps)
  ss.rsqrt(ss)
  xReg.mul_row(xReg, ss)
  x16.convert(xReg)
  gamma16.loadTile(glG, (0, 0, 0, 0))
  x16.mulF16(x16, gamma16)
  var x1: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var x2: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var x1s: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var x2s: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var cosT: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var sinT: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var t1: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  var t2: rt_l(float32, 8, 64, UNIVERSAL_8x8x8_F32F32F32F32)
  x1.splitHalfFloat32(x2, x16)
  cosT.loadTile(glCos, (gidY, 0, gidZ, 0))
  sinT.loadTile(glSin, (gidY, 0, gidZ, 0))
  x1s.mul(x1, sinT)
  x2s.mul(x2, sinT)
  x2s.mul(x2s, -1.0'f32)
  t1.fmaMapFloat32(x1, cosT, x2s)
  t2.fmaMapFloat32(x2, cosT, x1s)
  glOut.storeTile(t1, (tile, 0, 0, 0))
  glOut.storeTile(t2, (tile, 0, 0, 1))
