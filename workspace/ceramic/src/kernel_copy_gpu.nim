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
## On CPU, use `kernel_copy_cpu` (`copySameShape_cpu`/`copyPermuted_cpu`)
## which avoids divmod entirely via if/else branching and can fuse contiguous accesses.

import std/macros

import ./int_tuples
import ./layouts
import ./layout_constructors
import ./layout_algebra
import ./tensors
import ./atoms_copy
import workspace/crucible

{.experimental: "callOperator".}

template copyFrom*[T, ShD, StD, ShS, StS](
    dst: var (TensorView[T, ShD, StD] or Tensor[T, ShD, StD]);
    src: AnyTensor[T, ShS, StS]) =
  ## Copy every logical element from src to dst.
  ## Uses flat-index iteration (`dst(i) = src(i)`)
  ## which is divmod-based.
  ##
  ## This is slow but unavoidable on GPU as if/else-based indexing
  ## would trigger warp-divergence.
  for i in 0 ..< size(dst):
    dst(i) = src(i)

# ── Simdgroup fragments (Apple GPU, Metal): hardware gather/scatter ────────
# simdgroup_load/store is a tile-level all-32-lane gather: the view must be
# the TILE view (base pointer, row-length stride), never the partition view
# (per-thread offset), col-major, with MSL transpose arg = not isLayoutLeft.

template copyFrom*[T, Sh, St; isLayoutLeft: static bool](
    dst: var SimdgroupFragment[T, isLayoutLeft];
    src: AnyTensor[T, Sh, St]) =
  ## Metal simdgroup gather: `simdgroup_load(dst, src.data, stride, 0, not isLayoutLeft)`
  ## with stride the source tile's row length (`St.default[1]` of the
  ## col-major view). `src` is the tile view, not the partition view.
  simdgroupLoad(dst, src.data, uint32(toIntVal(St.default[1])), 0'u32, not isLayoutLeft)

template copyFrom*[T; isLayoutLeftD: static bool; isLayoutLeftS: static bool](
    dst: var SimdgroupFragment[T, isLayoutLeftD];
    src: SimdgroupFragment[T, isLayoutLeftS]) =
  ## Fragment-to-fragment copy: a plain simdgroup matrix assignment (the
  ## explicit-destination form seeds the accumulator with a copy of the
  ## input fragment).
  dst = src

template copyFrom*[T, Sh, St; isLayoutLeft: static bool](
    dst: AnyTensor[T, Sh, St];
    frag: SimdgroupFragment[T, isLayoutLeft]) =
  ## Metal simdgroup scatter, destination first:
  ## `simdgroup_store(frag, dst.data, stride, 0, not isLayoutLeft)` with stride
  ## the destination tile's row length. `dst` is the tile view, not the
  ## partition view.
  simdgroupStore(frag, dst.data, uint32(toIntVal(St.default[1])), 0'u32, not isLayoutLeft)

template copyFromIfAsync*[T, Sh, StA, StB, StP](
    dst: var TensorView[T, Sh, StB];
    src: TensorView[T, Sh, StA];
    predicate: AnyTensor[bool, Sh, StP]) =
  ## Predicated **async** copy
  ##
  ## This requires cp.async.commit_group to actually enqueue the copy
  ## and cp.async.wait_group to wait for its completion

  when Sh.rank == 1:
    cp.async.cg_shared_global_16B(dst, src, if predicate.data[0]: 16 else: 0)
  else:
    for i in 0 ..< size(predicate):
      cp.async.cg_shared_global_16B(dst(_, i), src(_, i), if predicate(_, i).data[0]: 16 else: 0)

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
#  partition_A/B/C (atoms_mma_partitioning) slice the smem tile into per-thread register fragments for the MMA atom.
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
  let ur = zipped_divide(L, tilerMN(atom))
  tiled_divide(mode(ur, 1), (chunkCols, kRows))

func partition_S*[T, ShA, StA, Atom](src: TensorView[T, ShA, StA];
                             atom: typedesc[Atom];
                             blockSize: static int;
                             thrIdx: int): auto =
  ## Slice the copy partition (thrfrg_copy) of the source tile at the flat thread id:
  ## the thread's chunks to copy.
  ## S = Source, the gmem side of the copy.
  ##
  ## In use, each thread slices its source and destination chunks,
  ## then copyFromIfAsync issues one 16-byte cp.async per chunk:
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
  ## then copyFromIfAsync issues one 16-byte cp.async per chunk:
  ##
  ##   let srcChunks = partition_S(tileA, atom, blockSize, threadIdx)
  ##   var dstChunks = partition_D(stageA, atom, blockSize, threadIdx)
  ##   copyFromIfAsync(dstChunks, srcChunks, predChunks)
  ##
  ## Example: a (16, 8) int8 tile with 4 threads, thread 2 gets the chunks at flat positions c = 2, 6, element offsets 32, 96.
  let thrTensor = make_view(dst.data, thrfrg_copy(dst.layout, atom, blockSize))
  thrTensor(thrIdx, _, _)
