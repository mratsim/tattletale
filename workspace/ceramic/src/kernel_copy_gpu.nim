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
import workspace/crucible

{.experimental: "callOperator".}

template copyFrom*[T, ShA, StA, ShB, StB](
    dst: var TensorView[T, ShB, StB];
    src: TensorView[T, ShA, StA]) =
  ## Copy every logical element from src to dst.
  ## Uses flat-index iteration (`dst(i) = src(i)`) which calls crd2idx
  ## per element — acceptable on GPU, slow on CPU.
  for i in 0 ..< size(dst):
    dst(i) = src(i)

template copyFrom*[T, ShA, StA, ShB, StB](
    dst: var Tensor[T, ShB, StB];
    src: AnyTensor[T, ShA, StA]) =
  ## Owning-tensor dst form — the fragment tensors (make_fragment_A/B,
  ## make_tensor/make_tensor_like). The flat-index `dst(i) = src(i)` is
  ## coordinate semantics: crd2idx decodes `i` through each tensor's own
  ## shape (mode order) then maps through its own strides, so the
  ## fragment (V = atom register order, stride-1) receives the element at
  ## the same logical coordinate as src, whatever src's layout (row-major
  ## included). The fragment's physical order follows the fragment's
  ## layout: V fastest, matching gemm_atom's data[k·VA+i] read.
  for i in 0 ..< size(dst):
    dst(i) = src(i)

template copyFromIf*[T, ShA, StA, ShB, StB](
    dst: var TensorView[T, ShB, StB];
    src: TensorView[T, ShA, StA];
    predicate: typed;
    defaultVal: T) =
  ## Copy elements where predicate(i) is true, fill rest with defaultVal.
  for i in 0 ..< size(dst):
    if predicate(i):
      dst(i) = src(i)
    else:
      dst(i) = defaultVal

# ═════════════════════════════════════════════════════════════════════════
#  cp.async — the NVIDIA async-copy builtins (SM80+)
# ═════════════════════════════════════════════════════════════════════════
#
#  The builtins follow the CUDA instruction names through a namespace
#  type-chain, so the calls read `cp.async.cg_shared_global(...)`,
#  `cp.async.commit_group()` and `cp.async.wait_group(N)`, the PTX
#  names, the asm texts from copy_sm80.hpp:57-70, 100-120 and 167-191.
#  The type-mapping func `async` turns the type `cp` into `CpAsync`,
#  and the copy, commit and wait resolve on that type. The func form
#  of the type-mapping returns void on the pinned Nim (2.2.10), so
#  `async` is a template with the same call syntax (DEV-002).
#
#  Each builtin emits one GCC extended-asm statement through the same
#  path as gemm_mma (kernel_gemm/nvidia_tensor_cores.nim): the asm text
#  is a single literal string, the operand identifiers are Nim locals
#  the asm statement resolves, and crucible emits the C asm statement.
#  A Nim asm statement with no output operands is implicitly volatile in
#  GCC (kept, never deleted), matching CUTLASS's `asm volatile`. The
#  memory clobber keeps the compiler from reordering the asm across
#  smem/gmem accesses.
#
#  The smem destination address is a uint32 shared address. The CUDA
#  intrinsic __cvta_generic_to_shared provides it. The gmem source is a
#  64-bit pointer. The copy size is the immediate 16, and the predicate
#  the runtime size 16 or 0.
#  Both addresses must be 16-byte aligned for the 128-bit copy, the smem element offset and
#  the gmem leading stride multiples of 4 elements.

type
  cp* = object
  CpAsync* = object

template async*(_: type cp): untyped = CpAsync

func cg_shared_global*[T, ShA, StA, ShB, StB](
    _: type CpAsync;
    dstSmem: TensorView[T, ShA, StA];
    srcGmem: TensorView[T, ShB, StB];
    predicate: bool) {.inline.} =
  ## One 16-byte predicated cp.async.cg copy, gmem → smem. The
  ## predicate is the SIZE operand, CuTe's SM80_CP_ASYNC_CACHEGLOBAL_ZFILL
  ## copy. A false predicate makes src_size 0, so the smem destination
  ## is zero-filled (ZFILL), no separate clear needed.
  ##
  ## Args:
  ##   dstSmem: the smem destination slice, the unit view at one chunk
  ##   srcGmem: the gmem source slice, the unit view at one chunk
  ##   predicate: the copy predicate, false → smem zero-fill (no gmem read)
  ##
  ## The addresses are the sliced views' data pointers, the copy atom's
  ## &dst(0) / &src(0) (copy_sm80.hpp:100-120). The {.inline.} func's
  ## locals bind the asm backtick names directly, with no template
  ## hygiene to dodge. Each call site inlines to its own scope, so the
  ## chunk loop's expansions do not collide.
  let smemInt = cvtaGenericToShared(dstSmem.data)
  let gmemPtr = srcGmem.data
  let size = if predicate: 16 else: 0
  asm "\"cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\" :: \"r\"(`smemInt`), \"l\"(`gmemPtr`), \"n\"(16), \"r\"(`size`) : \"memory\""

func commit_group*(_: type CpAsync) {.inline.} =
  ## One cp.async.commit_group: commit the group of cp.async copies
  ## issued since the previous commit. CuTe calls it cp_async_fence, the
  ## asm from copy_sm80.hpp:167-191. The empty output and input lists
  ## keep the statement implicitly volatile, the memory clobber orders
  ## it against smem/gmem accesses. DEV-001: the literal `: "memory"`
  ## form puts "memory" in the output position, which GCC and NVRTC
  ## reject, so the landed form has empty outputs and inputs.
  asm "\"cp.async.commit_group;\" :: : \"memory\""

func wait_group*(_: type CpAsync; N: static int) {.inline.} =
  ## One cp.async.wait_group N: block until all but N of the most
  ## recent commit groups have landed in smem. CuTe calls it
  ## cp_async_wait, the asm from copy_sm80.hpp:167-191. N = 0 waits for
  ## every committed group, the single-stage pipeline. The asm
  ## statement takes only a string literal, so the stage waits are
  ## baked as literal branches.
  when N == 0:
    asm "\"cp.async.wait_group %0;\" :: \"n\"(0) : \"memory\""
  elif N == 1:
    asm "\"cp.async.wait_group %0;\" :: \"n\"(1) : \"memory\""
  else:
    {.error: "wait_group: only the pipeline waits 0 and 1 have baked asm literals".}

# ═════════════════════════════════════════════════════════════════════════
#  The copy partition and the predicated tiled copy
#  (CuTe: TV_Tiler, partition_S / partition_D, copy_if)
# ═════════════════════════════════════════════════════════════════════════
#
#  CuTe partitions a copy across the threads with a TiledCopy.
#  The TiledCopy is a 16-byte cp.async.cg atom plus a thread layout.
#  The layout assigns each thread its slice via get_slice(thread_idx).
#  partition_S cuts the source tensor, partition_D the destination,
#  and the identity partition gives each copy unit its tile coordinate.
#  This is the tAcA of sm80_mma_multistage.hpp:506-533. copy_if,
#  copy.hpp:75-105, iterates the predicate tensor.
#  The copy atom reads the unit addresses from the tensors, and issues
#  one 16-byte cp.async per unit, the predicate as the copy size.
#  This is the ZFILL zero-fill.
#
#  The ceramic chunk is 4 row-consecutive elements at one k, the value
#  layout of the copy atom, instead of CUTLASS's 4 k-consecutive
#  elements. The flat (tileM, tileK) order makes the chunk a contiguous
#  16 bytes in both the gmem k-tile and the compact smem stage, so no
#  swizzle is needed for the 16-byte alignment. The padded-allocation
#  contract covers the ragged gmem reads. The thread layout is the
#  strided chunk sequence c = threadIdx + i·blockSize over the flat
#  chunk grid: the chunk column is fixed per thread, the k coordinate
#  advances by tileK div units per unit. The partition layout is the
#  zipped_divide of the tile by the chunk unit and by the thread unit
#  (the thrfrg_A construction), sliced at the thread's chunk position,
#  so the unit views carry their own addresses, the copy atom's
#  &dst(0) / &src(0).

func thrfrg_copy*[Sh, St](L: Layout[Sh, St];
                          tileM, tileK, blockSize: static int): auto {.inline.} =
  ## The copy-partition layout of the (M, K) k-tile, CuTe's
  ## TV_Tiler::apply. The mode pair (C, tileK div units) by (1, units)
  ## maps to the tile offset. C = tileM div 4 chunk columns, units =
  ## the 16-byte chunks per thread. The first mode is the thread's
  ## chunk position, the chunk column and the first k coordinate of
  ## its units. The second mode is the unit within the thread's share,
  ## the k coordinate advancing by tileK div units.
  const
    C = tileM div 4
    units = (tileM * tileK) div (4 * blockSize)
  static:
    doAssert tileM mod 4 == 0,
      "thrfrg_copy: the tile row dim must be a multiple of 4 elements"
    doAssert tileM * tileK mod (4 * blockSize) == 0,
      "thrfrg_copy: the k-tile must partition evenly into 16-byte chunks per thread"
    doAssert blockSize mod C == 0 and tileK mod units == 0,
      "thrfrg_copy: the thread layout must tile the chunk grid evenly"
  let ur = zipped_divide(L, (4, 1))
  let tp = zipped_divide(mode(ur, 1), (C, tileK div units))
  make_layout((mode(tp, 0).shape, mode(tp, 1).shape),
              (mode(tp, 0).stride, mode(tp, 1).stride))

func partition_S*[T, ShA, StA](src: TensorView[T, ShA, StA];
                             tileM, tileK, blockSize: static int;
                             thrIdx: int): auto =
  ## The thread's copy units of the gmem k-tile. CuTe's partition_S,
  ## sm80_mma_multistage.hpp:506-510, cuts the thrfrg_copy layout at
  ## the thread's chunk position. The slice moves the data pointer, so
  ## the unit view carries its own addresses, the copy atom's &src(0).
  const
    C = tileM div 4
    units = (tileM * tileK) div (4 * blockSize)
  let thrTensor = make_view(src.data, thrfrg_copy(src.layout, tileM, tileK, blockSize))
  let cm = thrIdx mod C
  let k0base = thrIdx div C
  let rsel = mapLeavesWith((1, units)): X()
  thrTensor(((cm, k0base), rsel))

func partition_D*[T, ShB, StB](dst: TensorView[T, ShB, StB];
                             tileM, tileK, blockSize: static int;
                             thrIdx: int): auto =
  ## The thread's copy units of the smem stage. CuTe's partition_D,
  ## sm80_mma_multistage.hpp:506-510. See partition_S.
  const
    C = tileM div 4
    units = (tileM * tileK) div (4 * blockSize)
  let thrTensor = make_view(dst.data, thrfrg_copy(dst.layout, tileM, tileK, blockSize))
  let cm = thrIdx mod C
  let k0base = thrIdx div C
  let rsel = mapLeavesWith((1, units)): X()
  thrTensor(((cm, k0base), rsel))

template copyFromIf*[T, ShA, StA, ShB, StB, ShP, StP](
    dst: var TensorView[T, ShB, StB];
    src: TensorView[T, ShA, StA];
    predicate: AnyTensor[bool, ShP, StP]) =
  ## The predicated tiled copy (CuTe: copy_if, copy.hpp:75-105): one
  ## 16-byte cp.async per predicate element, from the src view to the
  ## dst view. The predicate is the copy size (ZFILL): a false element
  ## makes the copy size 0, so the smem destination is zero-filled, no
  ## branch in the loop. Each unit's addresses are the sliced views'
  ## data pointers, the copy atom's &dst(0) / &src(0).
  for i in 0 ..< size(predicate):
    let subDst = dst(_, i)
    let subSrc = src(_, i)
    let p = predicate[i]
    cp.async.cg_shared_global(subDst, subSrc, p)
