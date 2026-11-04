# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 trellis-dequant GEMM on the Metal surface, ported from the exllamav3
## kernel family: exl3_gemm_kernel.cuh, exl3_gemm_inner.cuh, exl3_dq.cuh, codebook.cuh.
##
## Computes `D = A @ dequant(bTrellis)` over an EXL3 trellis B of shape
## (k/16, n/16, 16*bits) uint16, read as little-endian uint32 words.
## A is (m, k) fp16, D is (m, n) fp16 or fp32,
## with k a multiple of 16 and n a multiple of 128.
##
## Metal adaptations, no cooperative launch and no grid.sync:
##
## - the m axis walks in-kernel, grid.y indexes the 16-position m chunks,
##   one launch replaces the CUDA chunk loop and its grid.sync()
## - per-chunk state (accumulator, register tiles) resets in-kernel per chunk
## - one simdgroup per output tile walks the whole k axis, fp32 accumulators
##   never round-trip through global memory, so the CUDA kernel's split-k
##   locks and fp16 intermediate sums have no counterpart here
##
## Trellis decode contract, per 16x16 tile of 256 values at `bits` bits.
##
## Stream layout (pack.cu):
##
## - the exllamav3 packer's output, 8*bits little-endian uint32 words per tile
## - each 16-weight span packs its codes MSB-first into `bits` uint16 words
## - the span's first code occupies the word's top bits
## - word t of the tile emits as the half-swapped pair of uint16 spans
##   2t and 2t+1
##
## Decode order:
##
## - value index i enumerates the tile in the packer's m16n8k16 B-fragment
##   order (two fragments along n per tile)
## - `i(k, n) = 32*(n mod 8) + 8*((k mod 8) div 2) + (k mod 2) + 2*(k div 8)
##   + 4*(n div 8)`, the v1 decode's tensor-core-shuffle word
##   (quant_exl3_ops.nim)
## - window: the 16-bit ring window at bit `b0 = (i+1)*bits - 16 + 256*bits`,
##   the exl3_dq.cuh funnel: words i0 = b0 div 32, i1 = (b0+15) div 32 wrap
##   circularly mod 8*bits, merging as (a shl 32) or b and shifting right
##   by s0 = (i1+1)*32 - (b0+16)
## - the window's low `bits` bits carry the value's code, the higher bits
##   the neighbouring codes (tail-biting ring)
##
## Decoding is exact integer arithmetic plus one round-to-nearest-even
## fp16 conversion, kernel and CPU reference agree bit for bit.
##
## su/sv scaling points:
##
##   A --[su: hadamard_rotate_128_fwd, preScale suh]--> A_had
##   A_had --[exl3_gemm]--> fp32 image
##   image --[sv: hadamard_rotate_128_fwd_ff | _fh, postScale svh]--> d
##
## - su: host-composed pre-pass into a scratch copy of the activations
## - sv: host-composed post-pass over the fp32 accumulator image: d itself
##   under fp32 out, an fp32 scratch before the fp16 store
## - sv stays two-launch: in-kernel fusion needs one 128-column mma tile
##   holding double the accumulator registers, and the exact fp32 image
##   round-trip preserves the accumulator numerics

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_layout_gemm

const
  exl3GemmMT* = 16
    ## m positions per chunk and per output tile, the CUDA kernels' TILESIZE_M
  exl3GemmNT* = 64
    ## Output columns per threadgroup tile
  exl3GemmKT* = 8
    ## k step per mma, one Apple 8x8x8 atom cell
  exl3Norm128* = 0.088388347648'f32
    ## 1/sqrt(128), the Hadamard rotate normalization of the su scaling pass

type
  Exl3Codebook* = enum
    ## EXL3's three codebooks, exllamav3 codebook.cuh, a runtime model property.
    cb3inst
      ## 3INST map: the window folds through a 32-bit linear map, the two
      ## 16-bit lanes reinterpret as fp16 values sharing the exponent field 0x3b6x
    cbMcg
      ## mcg: same linear fold as 3INST with its own multiplier
    cbMul1
      ## mul1: a byte-sum fold and one fp16 fma

const
  cb3instId = 0'u32
  cbMcgId = 1'u32
  cbMul1Id = 2'u32
    ## Codebook ordinals for the runtime dispatch: cirv2 rejects enum field
    ## symbols as device values, enum types lower to 32-bit storage

static: doAssert ord(cb3inst) == 0 and ord(cbMcg) == 1 and ord(cbMul1) == 2
## Ties the ordinal constants to the enum declaration

func trellisValueIndex*(kInTile, nInTile: int): int {.inline.} =
  ## Value index inside one 16x16 trellis tile for element (kInTile, nInTile),
  ## both coordinates in 0..15.
  ##
  ## Contract:
  ##
  ## - m16n8k16 fragment order, two fragments along n per tile:
  ##   lane = 4*(n mod 8) + ((k mod 8) div 2), slot = (k mod 2) + 2*(k div 8)
  ##   + 4*(n div 8), index = 8*lane + slot
  ## - identical word to the v1 production decode's tensor-core-shuffle formula
  ##   (quant_exl3_ops.nim) at v = 0 and to `exl3_encode.encodeValueIndex`
  ## - bijection over the 256 cells, `trellisValueCoords` inverts it exactly
  8*(4*(nInTile mod 8) + ((kInTile mod 8) div 2)) + (kInTile mod 2) + 2*(kInTile div 8) + 4*(nInTile div 8)

func trellisValueCoords*(valueIndex: int): (int, int) {.inline.} =
  ## Inverse of `trellisValueIndex`: (k, n) inside the 16x16 tile.
  let
    lane = valueIndex div 8
    slot = valueIndex mod 8
  ((slot mod 2) + 8*((slot div 2) mod 2) + 2*(lane mod 4),
   (lane div 4) + 8*(slot div 4))

func decode3inst*(w: uint32, cb: Exl3Codebook): float16 {.inline.} =
  ## One trellis value from its 16-bit window `w` through codebook `cb`.
  ##
  ## Runtime codebook dispatch, contract:
  ##
  ## - the branch is perfectly predictable and the mma dominates runtime,
  ##   monomorphization buys nothing for 3x codesize
  ## - cirv2 rejects case statements in device bodies: the dispatch lowers
  ##   as an if chain over the codebook ordinals
  if uint32(cb) == cbMul1Id:
    let x = w * 0x83DCD12D'u32
    let s = (x and 0xFF'u32) + ((x shr 8) and 0xFF'u32) +
            ((x shr 16) and 0xFF'u32) + ((x shr 24) and 0xFF'u32) + 0x6400'u32
    let h = cast[float16](uint16(s))
    to(to(h, float32) * to(cast[float16](0x1EEE'u16), float32) +
       to(cast[float16](0xC931'u16), float32), float16)
  else:
    let x = if uint32(cb) == cb3instId: w * 89226354'u32 + 64248484'u32
            else: w * 0xCBAC1FED'u32
    # Lane words bind to uint16 temporaries first: the Metal backend widens
    # bitwise results to int and as_type<half> rejects int operands
    let
      loBits: uint16 = (uint16(x and 0xFFFF'u32) and 0x8FFF'u16) xor 0x3B60'u16
      hiBits: uint16 = (uint16(x shr 16 and 0xFFFF'u32) and 0x8FFF'u16) xor 0x3B60'u16
      lo = cast[float16](loBits)
      hi = cast[float16](hiBits)
    to(lo.to(float32) + hi.to(float32), float16)

func trellisWindow*(words: ptr UncheckedArray[uint32], tileWordsBase, valueIndex, bits: int): uint32 {.inline.} =
  ## 16-bit decode window of `valueIndex` inside one trellis tile:
  ## exl3_dq.cuh's `dq` funnel over the packed words.
  ##
  ## Expected input:
  ##
  ## - words: the trellis bitstream as little-endian uint32 words
  ## - tileWordsBase: the tile's first uint32 word, a multiple of 8*bits
  ## - bits: the tile's bits per value, 1..8
  let
    b0 = valueIndex * bits + bits - 16 + 256 * bits
    b1 = b0 + 16
    i0 = b0 div 32
    i1 = (b1 - 1) div 32
    s0 = (i1 + 1) * 32 - b1
    nWords = 8 * bits
    a = words[tileWordsBase + i0 mod nWords]
    b = words[tileWordsBase + i1 mod nWords]
  uint32((uint64(a) shl 32 or uint64(b)) shr s0 and 0xFFFF'u64)

func dequantTrellisSlot*(words: ptr UncheckedArray[uint32], trellisTilesN, k, n, bits: int, cb: Exl3Codebook): float16 {.inline.} =
  ## Dequantized value of the trellis element at global coordinates (k, n).
  ##
  ## Runtime `bits` and `cb`: word offsets are ordinary runtime arithmetic,
  ## mod and div over the stream words, no type-level dependency on either.
  ##
  ## Expected input:
  ##
  ## - words: the trellis bitstream as little-endian uint32 words, tile
  ##   (k div 16, n div 16) at word offset (k div 16)*trellisTilesN*8*bits
  let
    tk = k div 16
    tn = n div 16
    w = trellisWindow(words, (tk * trellisTilesN + tn) * (8 * bits),
      trellisValueIndex(k mod 16, n mod 16), bits)
  decode3inst(w, cb)

proc exl3_dequant_tile*(
    dst, bTrellis: distinct TensorView, bits: int, cb: Exl3Codebook) =
  ## Dequantizes one 16x16 trellis tile per threadgroup into fp16.
  ##
  ## Expected input:
  ##
  ## - bTrellis: flat uint32 bitstream, tile t starts at word offset t*8*bits
  ##
  ## Output:
  ##
  ## - dst: flat fp16, element (k, n) of tile t at offset t*256 + k*16 + n
  ##
  ## Launch: grid (tile count), threadgroup (32, 1), lane l decodes values 8l..8l+7.
  let
    tileId = threadgroup_position_in_grid.x
    lane = getLocalThreadID()
    words = bTrellis.data
    tileWords = tileId * (8 * bits)
  for slot in 0 ..< 8:
    let
      valueIndex = lane * 8 + slot
      (k, n) = trellisValueCoords(valueIndex)
      w = trellisWindow(words, tileWords, valueIndex, bits)
    dst.data[tileId * 256 + k * 16 + n] = decode3inst(w, cb)

proc exl3_gemm*(d, a, bTrellis: distinct TensorView, bits: int, cb: Exl3Codebook) =
  ## `D = A @ dequant(B)` over the whole m axis, one launch.
  ##
  ## Expected input:
  ##
  ## - a: (sizeM, sizeK) fp16, contiguous
  ## - bTrellis: flat uint32 bitstream over sizeK div 16 by sizeN div 16 tiles,
  ##   little-endian words of the (k/16, n/16, 16*bits) uint16 stream
  ## - sizeK divisible by 16, sizeN divisible by exl3GemmNT, both derived from
  ##   the views and asserted on the host launcher
  ##
  ## Output:
  ##
  ## - d: (sizeM, sizeN), fp16 or fp32 per d's element type, every m position
  ##   overwritten with its product, m positions past sizeM never touched
  ##
  ## Launch contract:
  ##
  ## - grid (n tile count, m chunk count) with grid-stride walks on both axes,
  ##   a narrower grid still covers every output tile
  ## - threadgroup (32, 1), fp32 accumulation over the whole k axis
  ## - per-chunk state (accumulator, register tiles) resets per m chunk
  ##   in-kernel: the counterpart of the CUDA kernel's grid.sync() chunk loop
  const
    mma = tiledMma(float32, float16, float16, exl3GemmMT, exl3GemmNT, exl3GemmKT)
    aMap = mma.getThreadValueMap(opA)
    bMap = mma.getThreadValueMap(opB)
    dMap = mma.getThreadValueMap(opC)
    # Register tiles type with the raw mma layouts, the thread->value maps
    # drive the load/store and the per-slot tile coordinates
    aLayout = mma.getMmaLayout(opA)
    bLayout = mma.getMmaLayout(opB)
    dLayout = mma.getMmaLayout(opC)

  let
    nTile0 = threadgroup_position_in_grid.x
    mChunk0 = threadgroup_position_in_grid.y
    (nStride, mStride, _) = (threadgroups_per_grid.x, threadgroups_per_grid.y, threadgroups_per_grid.z)
    thrID = getLocalThreadID()
    sizeM = a.shape[0]
    sizeK = a.shape[1]
    trellisTilesK = sizeK div 16
    trellisTilesN = bTrellis.shape[0] div (8 * bits * trellisTilesK)
    sizeN = trellisTilesN * 16
    outTilesN = sizeN div exl3GemmNT
    mChunks = sizeM.ceil_div(exl3GemmMT)

  let
    words = bTrellis.data
    # Per-slot tile coordinates of this lane's B-fragment registers
    bThrCrd = idx2crd(bMap.getThreadPartitioning(), thrID)
    bRegCrd = bMap.layout.slice(((bThrCrd, _), _))
    bCrdBase = bMap.layout.crd2idx(((bThrCrd, _), _))

  # Each threadgroup starts at its own m chunk and steps by the grid height,
  # so a grid shorter than the chunk count still covers every chunk
  for mChunk in countup(mChunk0, mChunks - 1, mStride):
    let
      m0 = mChunk * exl3GemmMT
      chunkM = min(exl3GemmMT, sizeM - m0)
      aChunk = a(m0 ..< m0 + chunkM, _)
      dChunk = d(m0 ..< m0 + chunkM, _)

    # Each threadgroup starts at its own output tile and steps by the grid width,
    # so a grid smaller than the tile count still covers every tile
    for nTile in countup(nTile0, outTilesN - 1, nStride):
      # Per-tile state: the accumulator and the dequantized B registers are
      # rebuilt for every grid-stride iteration, a stale accumulator would
      # double-count tiles 1 ..< outTilesN
      var acc = zeros(tile(float32, dLayout))
      let n0 = nTile * exl3GemmNT

      var k0 = 0
      while k0 < sizeK:
        var aReg: tile(float16, aLayout)
        aReg.load(aChunk, aMap, thrID, (0, k0 div exl3GemmKT))

        var bReg: tile(float16, bLayout)
        forEach idx, var e in bReg:
          let (nCol, kRow) = bCrdBase + bRegCrd.crd2idx(idx)
          e = dequantTrellisSlot(words, trellisTilesN, k0 + kRow, n0 + nCol, bits, cb)

        mma.atom.gemm_tile(acc, aReg, bReg, acc)
        k0 += exl3GemmKT

      when d.T is float32:
        dChunk.store(acc, dMap, thrID, (0, nTile))
      else:
        dChunk.store(acc.downcast(float16), dMap, thrID, (0, nTile))
