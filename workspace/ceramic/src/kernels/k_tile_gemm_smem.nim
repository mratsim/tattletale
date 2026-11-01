# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared-memory staging for the block-tiled matmul.
##
## - A and B stage per block in shared memory, double-buffered over
##   the reduction: async copy chunks fill one stage while the warps
##   consume the other, with `commit_group` and `wait_group`
##   pipelining the transfers
## - stage rows keep k-contiguous elements padded by one 16-byte unit.
##   Every staged chunk stays 16-byte aligned, and the padded row
##   stride maps each warp's fragment load to its own bank group.
## - the mma fragments load from the stage through the atom
##   thread->value map. The register tiles share that map.
## - ragged M, N, K go through coordinate predication with zero-fill
##   for out-of-bounds chunks and an elementwise in-bounds prefix
##   copy for the k-tail chunk, so no edge tile needs
##   a scalar pass

import workspace/math
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/ceramic/src/kernels/k_layout_copy_gpu
import workspace/ceramic/src/hardware/h_copy_dispatch
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu

template stepTiledMma*[TAcc: typedesc, TA: typedesc, TB: typedesc, MT: static int, NT: static int, KT: static int](
    mma: static TiledMma[TAcc, TA, TB, MT, NT, KT]): untyped =
  ## Warp tile schedule narrowed to one atom cell along k.
  TiledMma[TAcc, TA, TB, MT, NT, 16](atom: mma.atom)

# ═══════════════════════════════════════════════════════════════════════
#  Stage geometry
# ═══════════════════════════════════════════════════════════════════════

const stagePadElements* = 8
  ## Stage rows pad by one 16-byte unit, 8 bf16/f16 elements.
  ##
  ## - every staged chunk stays 16-byte aligned under the padded stride
  ## - the padded row stride maps each warp's fragment load to banks
  ##   of its own group per row
  ##
  ## The cp.async store side is a DMA write whose banking is not
  ## instruction-serialized, the fragment loads are.

func stageRowStride*(stageK: static int): static int {.inline.} =
  ## Elements between two stage rows, `stageK` padded by one 16-byte unit
  stageK + stagePadElements

func stageChunksPerRow*(stageK: static int): static int {.inline.} =
  ## 16-byte chunks along one stage row, `stageK` div 8
  stageK div 8

func stageBankWord*(row, k: int, stageK: static int): int {.inline.} =
  ## 4-byte word index of stage element (row, k). Bank analysis uses
  ## it directly: bank = word mod 32
  (row * stageRowStride(stageK) + k) div 2

# ═══════════════════════════════════════════════════════════════════════
#  Fragment loads
# ═══════════════════════════════════════════════════════════════════════

template loadFragTile*(dst: untyped, src: TensorView, tvm: static ThreadValueMap, threadIdx: int, tileID: (int, int)) =
  ## Loads one thread->value tile of `src` into the register tile
  ## `dst`, the tile.load fast path over a shared-memory stage.
  ##
  ## Contract:
  ## - the read is always in bounds. The staging covers every slice
  ##   coordinate with real data or with the zero the mma consumes.
  ##   The guarded fallback of tile.load has no work here
  ## - the source view points into threadgroup storage, so the loader
  ##   stays a template: a harvested proc fixes its parameter space
  ##   (caDevice) and rejects the smem-space argument
  ##
  ## Pure layout math and field reads only, every step inlines.
  let lay = src.getLayout()
  let tiled = lay.logical_divide(tvm.layout)
  let thrCrd = idx2crd(tvm.getThreadPartitioning(), threadIdx)
  let thrSel = (((thrCrd, _), _), tileID)
  let srcReg = tiled.slice(thrSel)
  for i in 0 ..< srcReg.size():
    dst[i] = src.data[crd2idx(srcReg, i)]

# ═══════════════════════════════════════════════════════════════════════
#  Staging
# ═══════════════════════════════════════════════════════════════════════

template stageSmemTile*(stageWords: untyped, src: TensorView, tid: int, stageRows: static int, stageK: static int, wordBase, rowOrigin, kOrigin, rowBound, kBound: int) =
  ## Stages one (stageRows, stageK) tile of `src` into the word-backed
  ## shared-memory `stageWords` array at word offset `wordBase`.
  ##
  ## Args:
  ## - stageRows, stageK: the tile shape, 64 staging threads copy it,
  ##   8 chunks of 8 k-contiguous elements per staging thread
  ## - rowOrigin, kOrigin: the global coordinates of the tile origin
  ## - rowBound, kBound: the tensor bounds.
  ##   Out-of-bounds chunks zero-fill through the src-size operand
  ##   that the async copy applies
  ##
  ## `src` is k-major over the tile: tile element (k, row) reads
  ## `src` at row `rowOrigin + row`, column `kOrigin + k`.
  const
    chunkElems = 8
    chunkWords = 4
    chunksPerRow = stageChunksPerRow(stageK)
    stageThreads = 64
    chunksPerThread = (stageRows * chunksPerRow) div stageThreads
  let
    elems = cast[ptr UncheckedArray[typeof(src.data[0])]](stageWords)
    zeroVal = zero(typeof(src.data[0]))
  const atom = getCopyAsyncAtom(uint32)
  for i in 0 ..< chunksPerThread:
    let c = tid + i * stageThreads
    let kc = c mod chunksPerRow
    let rc = c div chunksPerRow
    let elemOff = rc * stageRowStride(stageK) + kc * chunkElems
    let kIn = max(min(kBound - (kOrigin + kc * chunkElems), chunkElems), 0)
    let rowIn = rowOrigin + rc < rowBound
    let dstChunk = make_view(addr stageWords[wordBase + elemOff div 2],
                             make_layout((chunkWords,)))
    if not rowIn:
      # outside the row bounds: zero-fill, no global read
      copyIf(atom, dstChunk, dstChunk, false, chunkWords)
    elif kIn == chunkElems:
      # fully in bounds: one async chunk, the gmem side re-viewed as words
      let srcFlat = src.getLayout().crd2idx(
        (rowOrigin + rc, kOrigin + kc * chunkElems))
      let srcChunk = make_view(
        cast[ptr UncheckedArray[uint32]](addr src.data[srcFlat]),
        make_layout((chunkWords,)))
      copyIf(atom, dstChunk, srcChunk, true, chunkWords)
    elif kIn > 0:
      # K-tail chunk: the in-bounds prefix copies, the rest zeroes
      let srcFlat = src.getLayout().crd2idx(
        (rowOrigin + rc, kOrigin + kc * chunkElems))
      for j in 0 ..< kIn:
        elems[wordBase * 2 + elemOff + j] = src.data[srcFlat + j]
      for j in kIn ..< chunkElems:
        elems[wordBase * 2 + elemOff + j] = zeroVal
    else:
      copyIf(atom, dstChunk, dstChunk, false, chunkWords)
