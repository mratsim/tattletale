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
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_copy_registry
import workspace/ceramic/src/hardware/h_copy_properties
import workspace/ceramic/src/hardware/h_copy_dispatch
import workspace/crucible

{.experimental: "callOperator".}

func basePtr[T, Sh, St](t: (TensorView[T, Sh, St] or TensorOwned[T, Sh, St])): ptr UncheckedArray[T] {.inline.} =
  when typeof(t) is TensorOwned:
    cast[ptr UncheckedArray[T]](addr t.data[0])
  else:
    cast[ptr UncheckedArray[T]](t.data)

macro guardWriteDisjointness(shD, stD, shS, stS: typed) =
  ## Static error on an ambiguous multi-write:
  ##   - a dst leaf with static stride 0 and static shape > 1
  ##   - paired with a non-zero static src leaf stride
  ## Several flats then write one dst element with different values.
  ## Order-dependent, rejected before any copy code is emitted.
  ## A src stride-0 leaf is a broadcast read and stays legal.
  ## Dynamic leaves cannot prove ambiguity and stay legal.
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

func copyFlatChunks[T](dstP, srcP: ptr UncheckedArray[T]; dstOff, srcOff: int;
                       W: static int; n: int) {.inline.} =
  ## Copies n flat elements at (dstOff, srcOff) as W-element chunks plus a scalar tail.
  let chunks = n div W
  type Chunk = array[W, T]
  let
    dstChunks = cast[ptr UncheckedArray[Chunk]](addr dstP[dstOff])
    srcChunks = cast[ptr UncheckedArray[Chunk]](addr srcP[srcOff])
    dstElems = cast[ptr UncheckedArray[T]](addr dstP[dstOff])
    srcElems = cast[ptr UncheckedArray[T]](addr srcP[srcOff])
  for c in 0 ..< chunks:
    dstChunks[c] = srcChunks[c]
  for i in chunks * W ..< n:
    dstElems[i] = srcElems[i]

template spanCopyAt(dstP, srcP: untyped; dstOff, srcOff: int;
                    W, vecCap: int; elemBits: static int) =
  ## Copies one span of W elements at (dstOff, srcOff) element offsets:
  ## 16/8/4/2-element flat chunks capped at 128 bits, scalar remainder.
  ## Chunk width never exceeds the 2-adic valuation of a span base offset.
  let
    aDst = if dstOff == 0: vecCap else: dstOff and -dstOff
    aSrc = if srcOff == 0: vecCap else: srcOff and -srcOff
    w = min(min(vecCap, aDst), aSrc)
  when 128 div elemBits >= 16:
    if vecCap >= 16:
      if w >= 16:
        copyFlatChunks(dstP, srcP, dstOff, srcOff, 16, W)
  when 128 div elemBits >= 8:
    if vecCap >= 8:
      if w < 16 and w >= 8:
        copyFlatChunks(dstP, srcP, dstOff, srcOff, 8, W)
  when 128 div elemBits >= 4:
    if vecCap >= 4:
      if w < 8 and w >= 4:
        copyFlatChunks(dstP, srcP, dstOff, srcOff, 4, W)
  when 128 div elemBits >= 2:
    if vecCap >= 2:
      if w < 4 and w >= 2:
        copyFlatChunks(dstP, srcP, dstOff, srcOff, 2, W)
  if w < 2:
    copyFlatChunks(dstP, srcP, dstOff, srcOff, 1, W)

template copySpanRec(C: Layout; d: static int; dstP, srcP: untyped;
                     dstOff, srcOff, srcStride: int;
                     W, vecCap: int; elemBits: static int) =
  ## Visits C's span index space, dimensions 1 ..< rank(C), nested loops.
  ##   - dst span base = the C.stride entries dotted with the span index
  ##   - src span base = W times the column-major flat span index
  when d == rank(C):
    spanCopyAt(dstP, srcP, dstOff, srcOff, W, vecCap, elemBits)
  else:
    let
      cd = C.shape[d]
      sd = C.stride[d]
    for j in 0 ..< cd:
      copySpanRec(C, d + 1, dstP, srcP,
                  dstOff + sd * j, srcOff + srcStride * j, srcStride * cd,
                  W, vecCap, elemBits)

template copyCommonSpanBody(C: Layout; dst, src: untyped; elemBits: static int) =
  ## W-ladder over C's span decomposition, dimension 0 = the span at stride 1.
  block:
    when typeof(C.stride) is tuple:
      when typeof(C.shape[0]) is Int:
        const
          wV = typeof(C.shape[0]).V
          vecCap = min(wV and -wV, 128 div elemBits)
        copySpanRec(C, 1, basePtr(dst), basePtr(src), 0, 0, wV, wV, vecCap, elemBits)
      else:
        let
          wV = C.shape[0]
          vecCap = min(wV and -wV, 128 div elemBits)
        copySpanRec(C, 1, basePtr(dst), basePtr(src), 0, 0, wV, wV, vecCap, elemBits)
    else:
      when typeof(C.stride) is Int:
        const
          wV = typeof(C.shape).V
          vecCap = min(wV and -wV, 128 div elemBits)
        copySpanRec(C, 1, basePtr(dst), basePtr(src), 0, 0, wV, wV, vecCap, elemBits)
      else:
        let
          wV = C.shape
          vecCap = min(wV and -wV, 128 div elemBits)
        copySpanRec(C, 1, basePtr(dst), basePtr(src), 0, 0, wV, wV, vecCap, elemBits)

func copyCommonSpans[T, ShD, StD, ShS, StS, ShR, StR](
    dst: var (TensorView[T, ShD, StD] or TensorOwned[T, ShD, StD]);
    src: (TensorView[T, ShS, StS] or TensorOwned[T, ShS, StS]);
    R: Layout[ShR, StR]) {.inline.} =
  ## Copies the span decomposition of the coalesced common layout
  ## C = coalesce(compose(dst.layout, R)), R the quasi-inverse of src.layout.
  ## Requires size(R) == size(src), checked by the caller.
  ##   - a dst-consecutive dimension 0 with static stride 1 is the span
  ##   - anything else falls back to the element loop
  const elemBits = sizeof(T) * 8
  let C = coalesce(compose(dst.layout, R))
  when typeof(C.stride) is tuple:
    when typeof(C.stride[0]) is Int:
      when typeof(C.stride[0]).V == 1:
        copyCommonSpanBody(C, dst, src, elemBits)
      else:
        copyElementwise(dst, src)
    else:
      if C.stride[0] === 1:
        copyCommonSpanBody(C, dst, src, elemBits)
      else:
        copyElementwise(dst, src)
  else:
    when typeof(C.stride) is Int:
      when typeof(C.stride).V == 1:
        copyCommonSpanBody(C, dst, src, elemBits)
      else:
        copyElementwise(dst, src)
    else:
      if C.stride === 1:
        copyCommonSpanBody(C, dst, src, elemBits)
      else:
        copyElementwise(dst, src)

func copyElementwise[T, ShD, StD, ShS, StS](
    dst: var (TensorView[T, ShD, StD] or TensorOwned[T, ShD, StD]);
    src: (TensorView[T, ShS, StS] or TensorOwned[T, ShS, StS])) {.inline.} =
  ## Element loop, one scalar load and store per logical element.
  for i in 0 ..< size(src):
    dst(i) = src(i)


func copyFrom*[T, ShD, StD, ShS, StS](
    dst: var (TensorView[T, ShD, StD] or TensorOwned[T, ShD, StD]);
    src: (TensorView[T, ShS, StS] or TensorOwned[T, ShS, StS])) {.inline.} =
  ## Copies every element of src to dst, dst(flat k) = src(flat k).
  ##
  ## For runtime strides, assumes 128B alignment for vectorized copies
  guardWriteDisjointness(dst.layout.shape, dst.layout.stride,
                         src.layout.shape, src.layout.stride)
  block:
    let R = right_inverse(src.layout)
    type
      SizeR = typeof(size(R))
      SizeS = typeof(size(src.layout))
    when SizeR is Int and SizeS is Int:
      when SizeR.V == SizeS.V:
        copyCommonSpans(dst, src, R)
      else:
        copyElementwise(dst, src)
    else:
      if size(R) === size(src.layout):
        copyCommonSpans(dst, src, R)
      else:
        copyElementwise(dst, src)

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
