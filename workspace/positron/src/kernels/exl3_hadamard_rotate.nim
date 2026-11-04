# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## `hadamard_rotate_128_fwd` computes, per 128-block of the input axis,
## `dst = fp16round(FWHT(src * preScale) * norm) * postScale`, matching the exllamav3 op `ext.had_r_128`.
##
## - 4 fp16 values per lane, storage fp16 and compute fp32
## - one RNE round at each fp16 boundary: the pre-scale product, the norm product, the post-scale product
## - an fp32 butterfly kernel matches the f64 reference bit for bit only under narrow addend exponent spread, up to 13 binades
## - the domain keeps every partial sum inside the 24-bit fp32 significand, the validation inputs must respect it
##
## Butterfly stages, every output keeps its input position:
##
##   stages 1-2 (step 1, 2):     in-register over the lane's 4 slots
##   stages 3-7 (step 4 .. 64):  5 xor-shuffle rounds over the lanes,
##                               mask 2^r at round r
##
## Stage-3..7 exchange mechanics:
##
## - each lane xors the sign bit into its own contribution before the add,
##   an exact sign negation with no rounding
##
## fp32-input variants for the exllamav3 output-side sv pass
## (hadamard_inner.cuh `had_ff_r_128_inner` / `had_fh_r_128_inner`, post scale only):
##
## - `hadamard_rotate_128_fwd_ff`: fp32 in, fp32 out
##   `dst = (FWHT(src) * norm) * fp32(postScale)`, one fp32 round per product
## - `hadamard_rotate_128_fwd_fh`: fp32 in, fp16 out
##   `dst = fp16round(FWHT(src) * norm)`, then one fp16 postScale multiply
## - fp32 input keeps the GEMM accumulators' precision up to the sv store,
##   the sv consumer rotates the fp32 image `exl3_gemm` produces
##
## Launch shape:
##
## - grid `(M, C div 128)`, threadgroup `(32, 1)`, one simdgroup of 32 lanes per 128-block
## - preconditions: `M` is positive and `C` is a positive multiple of 128. The caller checks both before launching
## - a `C` below the contract drops the tail blocks to `C div 128`, a non-positive `M` leaves the grid empty
## - the int32 addressing caps `M * C` below 2^32, the bench sweep's 65536 x 16384 case reaches 2^30

import workspace/crucible
import workspace/ceramic/src/tile_algebra

const
  fwhtBlock = 128
    ## FWHT block width, the fused linear kernel's unexported `linear_exl3_core.fwhtBlock` is the same constant, importing it drags the GEMM

func scaleF16(a, b: float16): float16 {.inline.} =
  ## An fp32 product of two fp16 values is exact, one RNE round back to fp16 is the native fp16 multiply.
  to(to(a, float32) * to(b, float32), float16)

func butterflyXor32(h0, h1, h2, h3: var float32, lane: int32) {.inline.} =
  ## 5 xor-shuffle rounds over the 32 lanes, masks 1, 2, 4, 8, 16.
  ##
  ## A lane whose id sets the round mask xors the sign bit into its own value
  ## (an exact fp32 sign negation) and adds the partner lane's value.
  ## Every output keeps its input position.
  for r in 0'i32 ..< 5:
    let mask = 1'i32 shl r
    let
      p0 = cast[uint32](h0).simd_shuffle_xor(mask)
      p1 = cast[uint32](h1).simd_shuffle_xor(mask)
      p2 = cast[uint32](h2).simd_shuffle_xor(mask)
      p3 = cast[uint32](h3).simd_shuffle_xor(mask)
    let signBit = (0'u32 - uint32(lane and mask)) and 0x80000000'u32
    h0 = cast[float32](cast[uint32](h0) xor signBit) + cast[float32](p0)
    h1 = cast[float32](cast[uint32](h1) xor signBit) + cast[float32](p1)
    h2 = cast[float32](cast[uint32](h2) xor signBit) + cast[float32](p2)
    h3 = cast[float32](cast[uint32](h3) xor signBit) + cast[float32](p3)

proc hadamard_rotate_128_fwd*(
    dst, src, preScale, postScale: ptr UncheckedArray[float16],
    M, C: int32, norm: float32,
    withPreScale, withPostScale: static bool) {.device.} =
  ## One fused Hadamard rotate launch, the module doc's semantics and launch contract.
  ##
  ## Expected input:
  ##
  ## - dst, src: (M, C) fp16 tensors laid out contiguously, element (m, c) at offset m * C + c
  ## - preScale, postScale: C-element fp16 scale vectors, one value per axis position, shared by every tensor line
  ## - a scale slot whose static switch is false is never dereferenced, any valid pointer fills it
  ## - norm: the post-transform normalization factor, 1/sqrt(128) for EXL3
  ##
  ## Output:
  ##
  ## - dst, fully overwritten with the module doc's per-block transform
  let
    blocksPerRow = C div fwhtBlock
    work = M * blocksPerRow
    lane = int32(getLocalThreadID())
    first = getLocalTileID((M, blocksPerRow))
    ngrid = getGridSize()
  for w in countup(first, work - 1, ngrid):
    let
      row = w div blocksPerRow
      blk = w mod blocksPerRow
      lane = int32(getLocalThreadID())
    let
      srcU = cast[ptr UncheckedArray[uint32]](src)
      dstU = cast[ptr UncheckedArray[uint32]](dst)
      base = row * (C div 2) + blk * (fwhtBlock div 2) + 2 * lane
    let
      w0 = srcU[base]
      w1 = srcU[base + 1]
    let
      r0 = cast[float16](uint16(w0 and 0xFFFF'u32))
      r1 = cast[float16](uint16(w0 shr 16))
      r2 = cast[float16](uint16(w1 and 0xFFFF'u32))
      r3 = cast[float16](uint16(w1 shr 16))
    var
      h0 = r0.to(float32)
      h1 = r1.to(float32)
      h2 = r2.to(float32)
      h3 = r3.to(float32)
    when withPreScale:
      let scUPre = cast[ptr UncheckedArray[uint32]](preScale)
      let
        s0 = scUPre[blk * (fwhtBlock div 2) + 2 * lane]
        s1 = scUPre[blk * (fwhtBlock div 2) + 2 * lane + 1]
      h0 = scaleF16(r0, cast[float16](uint16(s0 and 0xFFFF'u32))).to(float32)
      h1 = scaleF16(r1, cast[float16](uint16(s0 shr 16))).to(float32)
      h2 = scaleF16(r2, cast[float16](uint16(s1 and 0xFFFF'u32))).to(float32)
      h3 = scaleF16(r3, cast[float16](uint16(s1 shr 16))).to(float32)
    let
      sum01 = h0 + h1
      diff01 = h0 - h1
      sum23 = h2 + h3
      diff23 = h2 - h3
    h0 = sum01 + sum23
    h1 = diff01 + diff23
    h2 = sum01 - sum23
    h3 = diff01 - diff23
    butterflyXor32(h0, h1, h2, h3, lane)
    var
      v0 = (h0 * norm).to(float16)
      v1 = (h1 * norm).to(float16)
      v2 = (h2 * norm).to(float16)
      v3 = (h3 * norm).to(float16)
    when withPostScale:
      let scUPost = cast[ptr UncheckedArray[uint32]](postScale)
      let
        t0 = scUPost[blk * (fwhtBlock div 2) + 2 * lane]
        t1 = scUPost[blk * (fwhtBlock div 2) + 2 * lane + 1]
      v0 = scaleF16(v0, cast[float16](uint16(t0 and 0xFFFF'u32)))
      v1 = scaleF16(v1, cast[float16](uint16(t0 shr 16)))
      v2 = scaleF16(v2, cast[float16](uint16(t1 and 0xFFFF'u32)))
      v3 = scaleF16(v3, cast[float16](uint16(t1 shr 16)))
    dstU[base] = uint32(cast[uint16](v0)) or (uint32(cast[uint16](v1)) shl 16)
    dstU[base + 1] = uint32(cast[uint16](v2)) or (uint32(cast[uint16](v3)) shl 16)

proc hadamard_rotate_128_fwd_ff*(
    dst, src: ptr UncheckedArray[float32],
    postScale: ptr UncheckedArray[float16],
    M, C: int32, norm: float32) {.device.} =
  ## fp32-output sv pass per the module doc's fp32-input variant contract.
  ##
  ## Expected input:
  ##
  ## - dst, src: (M, C) fp32 tensors, element (m, c) at offset m * C + c,
  ##   laid out contiguously, in place is legal, a lane reads its whole block
  ##   before it stores
  ## - postScale: C-element fp16 scale vector, one value per axis position,
  ##   shared by every tensor line
  ## - norm: the post-transform normalization factor, 1/sqrt(128) for EXL3
  ##
  ## Output:
  ##
  ## - dst, fully overwritten with `dst = (FWHT(src) * norm) * fp32(postScale)`
  ##   per 128-block, one fp32 round per product
  let
    blocksPerRow = C div fwhtBlock
    work = M * blocksPerRow
    first = getLocalTileID((M, blocksPerRow))
    ngrid = getGridSize()
  for w in countup(first, work - 1, ngrid):
    let
      row = w div blocksPerRow
      blk = w mod blocksPerRow
      lane = int32(getLocalThreadID())
      base = row * C + blk * fwhtBlock + 4 * lane
      scBase = blk * fwhtBlock + 4 * lane
    var
      h0 = src[base]
      h1 = src[base + 1]
      h2 = src[base + 2]
      h3 = src[base + 3]
    let
      sum01 = h0 + h1
      diff01 = h0 - h1
      sum23 = h2 + h3
      diff23 = h2 - h3
    h0 = sum01 + sum23
    h1 = diff01 + diff23
    h2 = sum01 - sum23
    h3 = diff01 - diff23
    butterflyXor32(h0, h1, h2, h3, lane)
    h0 = h0 * norm * postScale[scBase].to(float32)
    h1 = h1 * norm * postScale[scBase + 1].to(float32)
    h2 = h2 * norm * postScale[scBase + 2].to(float32)
    h3 = h3 * norm * postScale[scBase + 3].to(float32)
    dst[base] = h0
    dst[base + 1] = h1
    dst[base + 2] = h2
    dst[base + 3] = h3

proc hadamard_rotate_128_fwd_fh*(
    dst: ptr UncheckedArray[float16],
    src: ptr UncheckedArray[float32],
    postScale: ptr UncheckedArray[float16],
    M, C: int32, norm: float32) {.device.} =
  ## fp16-output sv pass per the module doc's fp32-input variant contract.
  ##
  ## Expected input:
  ##
  ## - src: (M, C) fp32 tensor, element (m, c) at offset m * C + c, laid out
  ##   contiguously
  ## - dst: (M, C) fp16 tensor, same shape, never aliased with src
  ## - postScale: C-element fp16 scale vector, one value per axis position,
  ##   shared by every tensor line
  ## - norm: the post-transform normalization factor, 1/sqrt(128) for EXL3
  ##
  ## Output:
  ##
  ## - dst, fully overwritten with `fp16round(FWHT(src) * norm)` then one fp16
  ##   postScale multiply per 128-block
  let
    blocksPerRow = C div fwhtBlock
    work = M * blocksPerRow
    first = getLocalTileID((M, blocksPerRow))
    ngrid = getGridSize()
  for w in countup(first, work - 1, ngrid):
    let
      row = w div blocksPerRow
      blk = w mod blocksPerRow
      lane = int32(getLocalThreadID())
      base = row * C + blk * fwhtBlock + 4 * lane
      scBase = blk * fwhtBlock + 4 * lane
    var
      h0 = src[base]
      h1 = src[base + 1]
      h2 = src[base + 2]
      h3 = src[base + 3]
    let
      sum01 = h0 + h1
      diff01 = h0 - h1
      sum23 = h2 + h3
      diff23 = h2 - h3
    h0 = sum01 + sum23
    h1 = diff01 + diff23
    h2 = sum01 - sum23
    h3 = diff01 - diff23
    butterflyXor32(h0, h1, h2, h3, lane)
    var
      v0 = (h0 * norm).to(float16)
      v1 = (h1 * norm).to(float16)
      v2 = (h2 * norm).to(float16)
      v3 = (h3 * norm).to(float16)
    v0 = scaleF16(v0, postScale[scBase])
    v1 = scaleF16(v1, postScale[scBase + 1])
    v2 = scaleF16(v2, postScale[scBase + 2])
    v3 = scaleF16(v3, postScale[scBase + 3])
    let
      dstU = cast[ptr UncheckedArray[uint32]](dst)
      base16 = row * (C div 2) + blk * (fwhtBlock div 2) + 2 * lane
    dstU[base16] = uint32(cast[uint16](v0)) or (uint32(cast[uint16](v1)) shl 16)
    dstU[base16 + 1] = uint32(cast[uint16](v2)) or (uint32(cast[uint16](v3)) shl 16)
