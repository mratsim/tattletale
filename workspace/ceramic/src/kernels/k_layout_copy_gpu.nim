## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU-suitable copy kernels: divmod-based flat-index iteration.
##
## These use `dst(i) = src(i)` which calls `crd2idx` per element.
## `crd2idx` is unfortunately implemented in terms of slow div+mod,
## however on GPU there is branch-free alternative.
## Any branch would potentially lead to warp divergence per dimension of the tensors involved.
##
## `copyFrom` copies the span decomposition of the coalesced common layout
## of src and dst:
##   - each span as 16/8/4/2-element flat chunks capped at 128 bits
##   - a scalar per-span remainder
## Layouts the decomposition does not fully cover copy elementwise.
##
## On CPU, use `k_layout_copy_cpu` (`copySameShape_cpu`/`copyPermuted_cpu`)
## which avoids divmod entirely via if/else branching and can fuse contiguous accesses.

import std/[macros, math, typetraits]

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_unsanctioned_helpers
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_copy_registry
import workspace/ceramic/src/hardware/h_copy_properties
import workspace/ceramic/src/hardware/h_copy_dispatch
import workspace/crucible

{.experimental: "callOperator".}

macro guardWriteDisjointness(shD, stD, shS, stS: typed) =
  ## Static error on an ambiguous multi-write:
  ##   - a dst leaf with static stride 0 and static shape > 1
  ##   - paired with a non-zero static src leaf stride
  ## i.e. a write on a broadcasted tensor
  let
    shDv = toSeqStaticInts(shD.getTypeInst())
    stDv = toSeqStaticInts(stD.getTypeInst())
    shSv = toSeqStaticInts(shS.getTypeInst())
    stSv = toSeqStaticInts(stS.getTypeInst())
  let R = min(min(min(shDv.len, stDv.len), shSv.len), stSv.len)
  result = newStmtList()
  for d in 0 ..< R:
    if stDv[d] == 0 and shDv[d] > 1 and
        stSv[d] != DynamicSentinel and stSv[d] != 0:
      result.add newTree(nnkPragma,
        newTree(nnkExprColonExpr, ident"error",
          newLit("copyFrom: ambiguous multi-write, dst leaf " & $d &
                 " has stride 0 with shape > 1 and src stride " & $stSv[d] &
                 " is non-zero")))

func copyChunks[T, ShC, StC](C: Layout[ShC, StC]; d: static int;
                             dstP, srcP: ptr UncheckedArray[T];
                             dstOff, srcOff, srcStride: int;
                             W, vecCap: int; elemBits: static int) {.inline.} =
  ## Copies dst <- src by chunks
  when d == rank(C):
    let
      aDst = if dstOff == 0: vecCap else: dstOff and -dstOff
      aSrc = if srcOff == 0: vecCap else: srcOff and -srcOff
      w = min(min(vecCap, aDst), aSrc)
    template tier(width: static int) =
      when 128 div elemBits >= width:
        if vecCap >= width and w >= width and w < width * 2:
          type Chunk = array[width, T]
          let
            dstChunks = cast[ptr UncheckedArray[Chunk]](addr dstP[dstOff])
            srcChunks = cast[ptr UncheckedArray[Chunk]](addr srcP[srcOff])
            chunks = W div width
          for c in 0 ..< chunks:
            dstChunks[c] = srcChunks[c]
          for i in chunks * width ..< W:
            dstP[dstOff + i] = srcP[srcOff + i]
    tier(16)
    tier(8)
    tier(4)
    tier(2)
    if w < 2:
      for i in 0 ..< W:
        dstP[dstOff + i] = srcP[srcOff + i]
  else:
    let
      cdI = toInt(C.shape[d])
      sdI = toInt(C.stride[d])
    for j in 0 ..< cdI:
      copyChunks(C, d + 1, dstP, srcP,
                 dstOff + sdI * j, srcOff + srcStride * j, srcStride * cdI,
                 W, vecCap, elemBits)


func copyElementwise[T, ShD, StD, ShS, StS](
    dst: var TensorView[T, ShD, StD];
    src: TensorView[T, ShS, StS]) {.inline.} =
  for i in 0 ..< size(src):
    dst(i) = src(i)

func copyFrom*[T, ShD, StD, ShS, StS](
    dst: var (TensorView[T, ShD, StD] or TensorOwned[T, ShD, StD]);
    src: (TensorView[T, ShS, StS] or TensorOwned[T, ShS, StS])) {.inline.} =
  ## Copies every element of src to dst, dst(flat k) = src(flat k).
  ##
  ## For runtime strides, assumes 128B alignment for vectorized copies
  guardWriteDisjointness(dst.shape, dst.stride,
                         src.shape, src.stride)
  var dstV = when typeof(dst) is TensorOwned: view(dst) else: dst
  let srcV = when typeof(src) is TensorOwned: view(src) else: src
  let R = right_inverse(srcV.getLayout())
  let C = coalesce(compose(dstV.getLayout(), R))
  const elemBits = sizeof(T) * 8
  let
    s0 = when typeof(C.stride) is tuple: C.stride[0] else: C.stride
    wV = when typeof(C.shape) is tuple: toInt(C.shape[0]) else: toInt(C.shape)
    vecCap = min(wV and -wV, 128 div elemBits)
  let spansCover = size(R) === size(srcV)
  if spansCover and s0 === 1:
    copyChunks(C, 1, dstV.data, srcV.data, 0, 0, wV, wV, vecCap, elemBits)
  else:
    copyElementwise(dstV, srcV)


func copyFromIfAsync*[T, Sh, StA, StB, StP](
    dst: var TensorView[T, Sh, StB];
    src: TensorView[T, Sh, StA];
    predicate: (TensorView[bool, Sh, StP] or TensorOwned[bool, Sh, StP])) {.inline.} =
  ## Predicated **async** copy
  ##
  ## This requires commit_group to actually enqueue the copy
  ## and wait_group to wait for its completion

  const atom = getCopyAsyncAtom(T)
  const chunkElems = atom.getVecBytes() div sizeof(T)
  when Sh.rank == 1:
    copyIf(atom, dst, src, predicate.data[0], chunkElems)
  else:
    for i in 0 ..< size(predicate):
      copyIf(atom, dst(_, i), src(_, i), predicate(_, i).data[0], chunkElems)

# ═════════════════════════════════════════════════════════════════════════
#   Partitioned copies
# ═════════════════════════════════════════════════════════════════════════
#
#  Partitioned copy set up data for Tensor-Cores (MMA):
#
#    gmem --------> smem --------> registers
#    partition_S    partition_D    partition_A/B/C
#
#  The hardware shapes each leg:
#  - cp.async copies exactly 16 bytes per instruction
#  - the tile splits into 16-byte aligned chunks, shared between the threads
#  - the MMA reads per-thread register fragments with fixed shapes and register order per operand
#
#  partition_A/B/C (tensors/tensors_mma_partitioning) slice the smem tile into per-thread register fragments for the MMA atom.
#  partition_S / partition_D slice the gmem source and the smem destination into per-thread 16-byte chunks.
#  The two sides are separate because the partition derives from each tensor's own strides.
#  The padded gmem source and the compact smem destination produce different offsets despite the same shape structure.
#  A/B/C name the MMA operands, S/D the copy's Source and Destination.

func thrfrg_copy*[Sh, St, Atom](L: Layout[Sh, St];
                          atom: typedesc[Atom];
                          blockSize: static int): auto {.inline.} =
  ## Returns the copy partition: the tile split between the threads into 16-byte chunks.
  ## Index it with (thread id, chunk index) to get the chunk's offset in the tile.
  ##
  ## The tile is a grid of 16-byte chunks, chunkCols columns and tileK rows.
  ## Thread (tc, tr) takes the chunks at column tc, rows tr, tr + kRows, tr + 2·kRows, and so on.
  ##
  ## The layout shape ((chunkCols, kRows), 1, tileK div kRows):
  ## - (chunkCols, kRows), the thread grid, indexed by the flat thread id
  ## - 1, the single chunk per thread position
  ## - tileK div kRows, the thread's chunks along k
  ##
  ## The flat thread id decomposes as (tc, tr) against the grid.
  ## Thread (tc, tr) owns the chunks at column tc and k-rows tr + i·kRows,
  ## for i in 0 ..< tileK div kRows, flat chunk position c = tid + i·blockSize.
  ##
  ## Numbers:
  ## - chunkWidth = numPacked(atom), 16 div sizeof(T) elements:
  ##   4 for int32, 16 for int8
  ## - chunkCols = tileM div chunkWidth, the tile's chunk-columns
  ##   (tileM = the first dimension, M for A, N for B)
  ## - kRows = blockSize div chunkCols, the grid's k-rows
  ##
  ## Example: a (16, 8) int32 tile with 8 threads has chunkWidth 4,
  ## chunkCols 4, kRows 2, layout ((4, 2), 1, 4). The chunk grid
  ## (4 chunk-columns × 8 k-rows) with the owner thread per chunk:
  ##
  ##        k →  0   1   2   3   4   5   6   7
  ##   m 0-3    T0  T4  T0  T4  T0  T4  T0  T4
  ##   ↓ 4-7    T1  T5  T1  T5  T1  T5  T1  T5
  ##     8-11   T2  T6  T2  T6  T2  T6  T2  T6
  ##     12-15   T3  T7  T3  T7  T3  T7  T3  T7
  ##
  ## Thread 4 (column 0, k-rows 1, 3, 5, 7) owns the chunks at
  ## element offsets m + 16·k = 16, 48, 80, 112.
  const
    chunkWidth = numPacked(atom)   # the elements packed in the 16-byte chunk
    tileM = Sh.default[0]
    tileK = Sh.default[1]
    chunkCols = tileM div chunkWidth   # the chunk columns of the tile
    kRows = blockSize div chunkCols    # the thread rows along k
  static:
    doAssert tileM mod chunkWidth === 0,
      "thrfrg_copy: the tile row dim must be a multiple of the chunk width"
    doAssert blockSize mod chunkCols === 0,
      "thrfrg_copy: the thread grid must tile the chunk grid evenly"
    doAssert tileK mod kRows === 0,
      "thrfrg_copy: the tile K dim must tile the thread-grid rows evenly"
  # CuTe tiles with a static Tiler_MN{} (copy_atom.hpp tile2thrfrg, partitioner.hpp thrfrg), the chunk tiler is a hardware fact.
  # makeIntTuple promotes the compile-time-known leaves so the Int[N] markers propagate statically, runtime leaves stay runtime.
  # The (chunkCols, kRows) thread grid derives from the tile shape and the thread count, the same static fact.
  let ur = zipped_divide(L, makeIntTuple(tilerMN(atom)))
  tiled_divide(dimension(ur, 1), makeIntTuple((chunkCols, kRows)))

func partition_S*[T, ShA, StA, Atom](src: TensorView[T, ShA, StA];
                             atom: typedesc[Atom];
                             blockSize: static int;
                             thrIdx: int): auto =
  ## Slice the copy partition (thrfrg_copy) of the source tile at the flat thread id:
  ## the thread's chunks to copy.
  ## S = Source, the gmem side of the copy.
  ##
  ## In use, each thread slices its source and destination chunks,
  ## then copyFromIfAsync issues one atom chunk per chunk position:
  ##
  ##   let srcChunks = partition_S(tileA, atom, blockSize, threadIdx)
  ##   var dstChunks = partition_D(stageA, atom, blockSize, threadIdx)
  ##   copyFromIfAsync(dstChunks, srcChunks, predChunks)
  ##
  ## Example: a (16, 8) int8 tile with 4 threads, thread 2 gets the chunks at flat positions c = 2, 6, element offsets 32, 96.
  let thrTensor = make_view(src.data, thrfrg_copy(src.getLayout(), atom, blockSize))
  thrTensor(thrIdx, _, _)

func partition_D*[T, ShB, StB, Atom](dst: TensorView[T, ShB, StB];
                             atom: typedesc[Atom];
                             blockSize: static int;
                             thrIdx: int): auto =
  ## Slice the copy partition (thrfrg_copy) of the destination tile at the flat thread id:
  ## the thread's chunks to receive the copy.
  ## D = Destination, the smem side of the copy.
  ##
  ## In use, each thread slices its source and destination chunks,
  ## then copyFromIfAsync issues one atom chunk per chunk position:
  ##
  ##   let srcChunks = partition_S(tileA, atom, blockSize, threadIdx)
  ##   var dstChunks = partition_D(stageA, atom, blockSize, threadIdx)
  ##   copyFromIfAsync(dstChunks, srcChunks, predChunks)
  ##
  ## Example: a (16, 8) int8 tile with 4 threads, thread 2 gets the chunks at flat positions c = 2, 6, element offsets 32, 96.
  let thrTensor = make_view(dst.data, thrfrg_copy(dst.getLayout(), atom, blockSize))
  thrTensor(thrIdx, _, _)
