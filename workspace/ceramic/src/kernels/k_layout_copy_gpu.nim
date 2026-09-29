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
## `copyFrom` auto-vectorizes on fully static layouts.
## It copies the common contiguous run of src and dst in multi-element chunks.
##
## On CPU, use `k_layout_copy_cpu` (`copySameShape_cpu`/`copyPermuted_cpu`)
## which avoids divmod entirely via if/else branching and can fuse contiguous accesses.

import std/[math, typetraits]

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_copy_registry
import workspace/ceramic/src/hardware/h_copy_properties
import workspace/ceramic/src/hardware/h_copy_dispatch
import workspace/crucible

{.experimental: "callOperator".}

func basePtr[T, Sh, St](t: (TensorView[T, Sh, St] or TensorOwned[T, Sh, St])): ptr T {.inline.} =
  when typeof(t) is TensorOwned:
    addr t.data[0]
  else:
    cast[ptr T](t.data)

func getContiguity[T, ShD, StD, ShS, StS](
    dst: (TensorView[T, ShD, StD] or TensorOwned[T, ShD, StD]);
    src: (TensorView[T, ShS, StS] or TensorOwned[T, ShS, StS])): int {.inline.} =
  ## Largest N with dst(flat k) == k == src(flat k) for all 0 <= k < N,
  ## flat k in the element operator's column-major order, dimension 0
  ## walks fastest, unequal shapes keep the min of the two sizes.
  const R = min(tupleLen(ShD), tupleLen(ShS))
  var
    span = 1
    stop = false
  staticFor d, 0, R:
    if not stop:
      let shD = dst.layout.shape[d]
      let stD = dst.layout.stride[d]
      let shS = src.layout.shape[d]
      let stS = src.layout.stride[d]
      if stD === span and stS === span:
        span = min(span * shD, span * shS)
      else:
        stop = true
  span

func copyElementwise[T, ShD, StD, ShS, StS](
    dst: var (TensorView[T, ShD, StD] or TensorOwned[T, ShD, StD]);
    src: (TensorView[T, ShS, StS] or TensorOwned[T, ShS, StS])) {.inline.} =
  ## Element loop, one scalar load and store per logical element.
  for i in 0 ..< size(src):
    dst(i) = src(i)

func copyFlatChunks[T, ShD, StD, ShS, StS](
    dst: var (TensorView[T, ShD, StD] or TensorOwned[T, ShD, StD]);
    src: (TensorView[T, ShS, StS] or TensorOwned[T, ShS, StS]);
    W: static int;
    n: int) {.inline.} =
  let
    dstBase = basePtr(dst)
    srcBase = basePtr(src)
  type Chunk = array[W, T]
  let
    dstChunks = cast[ptr UncheckedArray[Chunk]](dstBase)
    srcChunks = cast[ptr UncheckedArray[Chunk]](srcBase)
    dstElems = cast[ptr UncheckedArray[T]](dstBase)
    srcElems = cast[ptr UncheckedArray[T]](srcBase)
    chunks = n div W
  for c in 0 ..< chunks:
    dstChunks[c] = srcChunks[c]
  for i in chunks * W ..< n:
    dstElems[i] = srcElems[i]

func copyFrom*[T, ShD, StD, ShS, StS](
    dst: var (TensorView[T, ShD, StD] or TensorOwned[T, ShD, StD]);
    src: (TensorView[T, ShS, StS] or TensorOwned[T, ShS, StS])) {.inline.} =
  ## Copies every element of src to dst, dst(flat k) = src(flat k).
  ##
  ## The first commonSpan elements run as flat vector chunks, the rest
  ## loops elementwise, vector granularity up to 128 bits, base pointers
  ## assumed 16-byte aligned.
  const elemBits = sizeof(T) * 8
  let commonSpan = getContiguity(dst, src)
  let vecElems = min(commonSpan and -commonSpan, 128 div elemBits)
  when 128 div elemBits >= 16:
    if vecElems >= 16:
      copyFlatChunks(dst, src, 16, commonSpan)
  when 128 div elemBits >= 8:
    if vecElems < 16 and vecElems >= 8:
      copyFlatChunks(dst, src, 8, commonSpan)
  when 128 div elemBits >= 4:
    if vecElems < 8 and vecElems >= 4:
      copyFlatChunks(dst, src, 4, commonSpan)
  when 128 div elemBits >= 2:
    if vecElems < 4 and vecElems >= 2:
      copyFlatChunks(dst, src, 2, commonSpan)
  if vecElems < 2:
    copyFlatChunks(dst, src, 1, commonSpan)
  for i in commonSpan ..< size(src):
    dst(i) = src(i)

func copyFromIfAsync*[T, Sh, StA, StB, StP](
    dst: var TensorView[T, Sh, StB];
    src: TensorView[T, Sh, StA];
    predicate: AnyTensor[bool, Sh, StP]) {.inline.} =
  ## Predicated **async** copy
  ##
  ## This requires commit_group to actually enqueue the copy
  ## and wait_group to wait for its completion

  const atom = getCopyAsyncAtom(T)
  when Sh.rank == 1:
    copyIf(atom, dst, src, predicate.data[0])
  else:
    for i in 0 ..< size(predicate):
      copyIf(atom, dst(_, i), src(_, i), predicate(_, i).data[0])

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
  let thrTensor = make_view(src.data, thrfrg_copy(src.layout, atom, blockSize))
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
  let thrTensor = make_view(dst.data, thrfrg_copy(dst.layout, atom, blockSize))
  thrTensor(thrIdx, _, _)
