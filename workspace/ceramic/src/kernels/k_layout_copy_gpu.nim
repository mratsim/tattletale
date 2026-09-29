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

import std/macros
import std/math

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_copy_registry
import workspace/ceramic/src/hardware/h_copy_properties
import workspace/ceramic/src/hardware/h_copy_dispatch
import workspace/crucible

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════
#  Layout staticness and alignment facts (copy-kernel side)
# ═══════════════════════════════════════════════════════════════
#
#  Facts consumed by copyFrom to size the copy chunks,
#  per CuTe copy.hpp AutoVectorizingCopyWithAssumedAlignment:
#  - is_static<Layout> → `isStaticLayout` from the layout algebra
#  - max_alignment(Layout) → `max_alignment` below
#  - max_common_vector(a, b) comes from the layout algebra.

template max_alignment*(L: Layout): int =
  ## Maximum alignment of a layout, in elements:
  ## - the largest N for which `upcast<N>(L)` is valid
  ## - i.e. the largest chunk granularity that respects every static
  ##   shape and stride of L
  ##
  ## Compile-time. Requires a fully static layout (checked by callers).
  ##
  ## Examples:
  ##   max_alignment(make_layout((32, 16), (1, 32)))  # → 512, coalesces to (512):(1)
  ##   max_alignment(make_layout((32, 16), (32, 1)))  # → 16, columns are 32 elements apart
  ##
  ## Contract (cute layout.hpp max_alignment):
  ## - dynamic leaves are masked out, shape → 1 and stride → 0
  ## - only the static component of the layout is trusted
  ## - dynamic strides are assumed to be large multiples of the result
  block:
    let flat = coalesce(L)
    let filterL = mapLeavesWith(flat):
      when it_sh is Int:
        (it_sh, when it_st is Int: it_st else: Int[0]())
      else:
        (Int[1](), when it_st is Int: it_st else: Int[0]())
    let permuted = logical_divide(filterL, right_inverse(filterL))
    let leadingSize = size(make_layout(permuted.shape[0], permuted.stride[0]))
    let trailingStride = permuted.stride[1]
    when typeof(trailingStride) is tuple:
      gcd(toIntVal(leadingSize), toIntVal(flatten(trailingStride)[0]))
    else:
      gcd(toIntVal(leadingSize), toIntVal(trailingStride))

template copyFrom*[T, ShD, StD, ShS, StS](
    dst: var (TensorView[T, ShD, StD] or TensorOwned[T, ShD, StD]);
    src: AnyTensor[T, ShS, StS]) =
  ## Copy every logical element from src to dst.
  ## Unpredicated whole-tensor copy with no predicate:
  ## `dst(flat k) = src(flat k)` for all k.
  ##
  ## Flat-index iteration (`dst(i) = src(i)`) is divmod-based, slow but
  ## unavoidable on GPU as if/else-based indexing would trigger warp
  ## divergence per dimension.
  ##
  ## For fully static layouts (all shape and stride leaves compile-time)
  ## the copy auto-vectorizes per CuTe's
  ## `AutoVectorizingCopyWithAssumedAlignment<128>` (copy.hpp):
  ##
  ##   vec_bits = gcd(max_common_vector(dst, src)·elem_bits,
  ##                  max_alignment(dst), max_alignment(src), 128)
  ##
  ## - `max_common_vector(dst, src)` is the longest contiguous run present
  ##   in both element orders
  ## - the 128-bit term is an assumption on the data pointers, not a check
  ## - when the chunk is larger than one element, both tensors are recast
  ##   to `array[vecElems, T]` chunks via `upcast` and copied with one
  ##   multi-byte copy per chunk
  ##
  ## With any dynamic leaf (CuTe's 8-bit alignment tier), the assumed
  ## alignment is 8 bits, which never exceeds an element, so the copy
  ## degrades to the plain element loop below.
  when isStaticLayout(dst.layout) and isStaticLayout(src.layout):
    # The layout facts live in the types (`Int[N]` leaves), so rebuild
    # compile-time layout values from the type parameters.
    const
      elemBits = sizeof(T) * 8
      dstL = default(Layout[ShD, StD])
      srcL = default(Layout[ShS, StS])
      commonElems = max_common_vector(dstL, srcL)
      alignBits = gcd(gcd(max_alignment(dstL), max_alignment(srcL)), 128)
      vecBits = gcd(commonElems * elemBits, alignBits)
    when vecBits mod 8 == 0 and vecBits mod elemBits == 0 and vecBits > elemBits:
      const vecElems = vecBits div elemBits
      when toIntVal(size(dst)) mod vecElems == 0:
        type Chunk = array[vecElems, T]
        let dstChunks = make_view(
          when dst is TensorOwned:
            cast[ptr UncheckedArray[Chunk]](addr dst.data[0])
          else:
            cast[ptr UncheckedArray[Chunk]](dst.data),
          upcast(dst.layout, vecElems))
        let srcChunks = make_view(
          when src is TensorOwned:
            cast[ptr UncheckedArray[Chunk]](addr src.data[0])
          else:
            cast[ptr UncheckedArray[Chunk]](src.data),
          upcast(src.layout, vecElems))
        static:
          doAssert toIntVal(size(srcChunks)) == toIntVal(size(dstChunks)),
            "copyFrom: recast chunk counts of src and dst disagree; " &
            "the copy is not chunkable, use the element loop"
        for i in 0 ..< size(dstChunks):
          dstChunks(i) = srcChunks(i)
      else:
        for i in 0 ..< size(dst):
          dst(i) = src(i)
    else:
      for i in 0 ..< size(dst):
        dst(i) = src(i)
  else:
    for i in 0 ..< size(dst):
      dst(i) = src(i)

template copyFromIfAsync*[T, Sh, StA, StB, StP](
    dst: var TensorView[T, Sh, StB];
    src: TensorView[T, Sh, StA];
    predicate: AnyTensor[bool, Sh, StP]) =
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
#  The copy partition
# ═════════════════════════════════════════════════════════════════════════
#
#  The copy partition is the gmem → smem leg of the GEMM pipeline, the
#  counterpart of the MMA partition on the smem → register leg:
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
  ## Numbers:
  ## - chunkWidth = numPacked(atom), 16 div sizeof(T) elements:
  ##   4 for int32, 16 for int8
  ## - chunkCols = tileM div chunkWidth, the tile's chunk-columns
  ##   (tileM = the first dimension, M for A, N for B)
  ## - kRows = blockSize div chunkCols, the grid's k-rows
  ##
  ## The flat thread id decomposes as (tc, tr) against the grid.
  ## Thread (tc, tr) owns the chunks at column tc and k-rows tr + i·kRows,
  ## for i in 0 ..< tileK div kRows, flat chunk position c = tid + i·blockSize.
  ##
  ## Example: a (16, 8) int32 tile with 8 threads has chunkWidth 4,
  ## chunkCols 4, kRows 2, layout ((4, 2), 1, 4). The chunk grid
  ## (4 chunk-columns × 8 k-rows) with the owner thread per chunk:
  ##
  ##        k →  0   1   2   3   4   5   6   7
  ##   m 0-3    T0  T4  T0  T4  T0  T4  T0  T4
  ##   ↓ 4-7    T1  T5  T1  T5  T1  T5  T1  T5
  ##     8-11   T2  T6  T2  T6  T2  T6  T2  T6
  ##     12-15  T3  T7  T3  T7  T3  T7  T3  T7
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
