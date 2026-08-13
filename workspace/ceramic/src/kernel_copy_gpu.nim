## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU-suitable copy kernels: divmod-based flat-index iteration.
##
## These use `dst(i) = src(i)` which calls `crd2idx` per element
## (divmod for flat→coord decomposition). Acceptable on GPU where
## divmod is relatively cheap and warp divergence from wheel-winding
## would be catastrophic.
##
## On CPU, use `kernel_copy_cpu` (`copySameShape_cpu`/`copyPermuted_cpu`)
## which avoids divmod entirely via contiguity-fused copyMem.

import std/macros

import ./int_tuples
import ./layouts
import ./layout_constructors
import ./layout_algebra
import ./tensors
import ./atoms_copy
import workspace/crucible

{.experimental: "callOperator".}

template copyFrom*[T, ShA, StA, ShB, StB](
    dst: var TensorView[T, ShB, StB];
    src: TensorView[T, ShA, StA]) =
  ## Copy every logical element from src to dst.
  ## Uses flat-index iteration (`dst(i) = src(i)`) which calls crd2idx
  ## per element, acceptable on GPU, slow on CPU.
  for i in 0 ..< size(dst):
    dst(i) = src(i)

template copyFrom*[T, ShA, StA, ShB, StB](
    dst: var Tensor[T, ShB, StB];
    src: AnyTensor[T, ShA, StA]) =
  ## Owning-tensor dst form: the fragment tensors
  ## (make_fragment_A/B, make_tensor/make_tensor_like).
  ## The flat-index `dst(i) = src(i)` is coordinate semantics:
  ## crd2idx
  ## decodes `i` through each tensor's own shape and maps through
  ## its own strides. The fragment (V = atom register order,
  ## stride-1) receives the element at the same logical coordinate
  ## as src, whatever src's layout, row-major included.
  ## The fragment's physical order follows the fragment's layout:
  ## V fastest, matching gemm_atom's data[k·VA+i] read.
  for i in 0 ..< size(dst):
    dst(i) = src(i)

# ═════════════════════════════════════════════════════════════════════════
#  The copy partition and the predicated tiled copy
# ═════════════════════════════════════════════════════════════════════════
#
#  A tiled copy pairs the 16-byte cp.async atom with a thread layout.
#  The thread layout assigns each thread its slice. partition_S
#  cuts the source tensor, partition_D the destination.
#  The identity partition gives each copy unit its tile coordinate.
#  copy_if iterates the predicate tensor. The copy atom reads
#  the unit addresses from the tensors, issuing one 16-byte cp.async
#  per unit, the predicate as the copy size. A false predicate
#  zeroes the copy size, the ZFILL zero-fill.
#
#  The chunk is 4 row-consecutive elements at one k, matching
#  the atom's 16-byte span. The flat (tileM, tileK) order makes
#  the chunk a contiguous 16 bytes in both the gmem k-tile
#  and the compact smem stage, so no swizzle is needed for
#  the 16-byte alignment. The padded-allocation contract
#  covers the ragged gmem reads. The thread layout is
#  the strided chunk
#  sequence c = threadIdx + i·blockSize over the flat chunk grid.
#  The partition layout zips the tile by the chunk unit
#  and the thread unit, its first mode the flat thread-index
#  space. The partition slice (partition_S / partition_D) cuts
#  it at the flat thread index with the underscore.
#  crd2idx decomposes the index against the thread-grid mode,
#  the chunk column and the first k of the thread units. Each
#  unit view carries the addresses, the copy atom's
#  &dst(0) / &src(0).

func thrfrg_copy*[Sh, St, Atom](L: Layout[Sh, St];
                          atom: typedesc[Atom];
                          blockSize: static int): auto {.inline.} =
  ## The copy-partition layout of the (M, K) k-tile: the tile
  ## zipped by the atom's chunk tiler, then the rest tiled
  ## by the thread grid. The first mode is the thread-grid
  ## space, its shape chunkCols·kRows == blockSize: the
  ## partition slice
  ## (partition_S / partition_D) decomposes the flat thread index
  ## against it inside crd2idx, the chunk column and the first k
  ## of the thread's units. The value mode is shape 1, the copy
  ## atom's single chunk per predicate element, the rest mode
  ## the units, the k coordinate advancing by tileK div kRows
  ## per unit.
  ## The chunk width is the atom's NumPacked (16 div sizeof(T):
  ## int32 gives 4, int8 gives 16), the chunk tiler the atom's
  ## Tiler_MN, the thread grid (chunkCols, kRows) the chunk columns
  ## and the thread rows along k. The tile dims come from
  ## the layout's own static shape.
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
  ## The thread's copy units of the gmem k-tile: the partition
  ## layout cut at the flat thread index, the units kept.
  ## The one-liner is the operator() with the mixed
  ## scalar/underscore coordinate: the flat index goes into
  ## the underscore slice, and crd2idx decomposes it against
  ## the thread-grid mode, the chunk column and the first k
  ## of the thread's units, never hand-rolled. The slice moves
  ## the data pointer, so the unit view carries
  ## its own addresses, the copy atom's &src(0).
  let thrTensor = make_view(src.data, thrfrg_copy(src.layout, atom, blockSize))
  thrTensor(thrIdx, _, _)

func partition_D*[T, ShB, StB, Atom](dst: TensorView[T, ShB, StB];
                             atom: typedesc[Atom];
                             blockSize: static int;
                             thrIdx: int): auto =
  ## The thread's copy units of the smem stage. See partition_S.
  let thrTensor = make_view(dst.data, thrfrg_copy(dst.layout, atom, blockSize))
  thrTensor(thrIdx, _, _)

template copyFromIf*[T, Sh, StA, StB, StP](
    dst: var TensorView[T, Sh, StB];
    src: TensorView[T, Sh, StA];
    predicate: AnyTensor[bool, Sh, StP]) =
  ## Predicated tiled copy: one 16-byte cp.async per predicate
  ## element, from the src view to the dst view.

  when Sh.rank == 1:
    cp.async.cg_shared_global_16B(dst, src, predicate)
  else:
    for i in 0 ..< size(predicate):
      cp.async.cg_shared_global_16B(dst(_, i), src(_, i), predicate(_, i))
