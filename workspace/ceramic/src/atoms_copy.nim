## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Copy atoms: the cp.async chunk type and the builtins that issue
## the instruction.
##
## The cp.async instruction copies one 16-byte chunk per issue.
## The L2::128B mnemonic fixes the chunk at 128 bits.
## The element count follows from the element type, 16 div
## sizeof(T): int32 and float32 give 4, int8 gives 16, int64
## and float64 give 2.
##
## The atom carries these static members:
##   NumPacked: the elements packed in the 16-byte chunk,
##   AtomNumThr: the threads per atom issue, 1
##   AtomNumVal: the values per atom issue, 1
##   Tiler_MN: the chunk tiler over the tile, (NumPacked, 1),
##     a chunk of NumPacked consecutive elements in the first
##     tile mode
##
## The copy partition reads these members, never hand-rolled
## arithmetic on the tile shape.
##
## The builtins are the asm statements behind the atom: the copy,
## the commit and the wait, resolved through the `cp.async`
## namespace type-chain so the calls read the PTX names.

import ./int_tuples
import ./layouts
import ./layout_constructors
import ./tensors
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  The cp.async atom
# ═════════════════════════════════════════════════════════════════════════

type CpAsyncAtomImpl*[T; NumPacked: static int] = object
  ## One 16-byte predicated cp.async.cg chunk. T is the element type
  ## and NumPacked the elements packed in the 16-byte chunk.
  ## The type parameters are the members the copy partition reads.

type CpAsyncAtom*[T] = CpAsyncAtomImpl[T, 16 div sizeof(T)]
  ## The 16-byte cp.async chunk atom for the element type T, with
  ## the NumPacked count following from sizeof(T): int32 gives 4,
  ## gives 16, int64 gives 2.

template numPacked*[T; NumPacked: static int](_: typedesc[CpAsyncAtomImpl[T, NumPacked]]): int =
  static:
    doAssert NumPacked * sizeof(T) === 16,
      "CpAsyncAtom: the chunk must be 16 bytes (the cp.async L2::128B)"
  NumPacked

template atomNumThr*(_: typedesc[CpAsyncAtomImpl]): static int =
  ## The threads per atom issue: 1.
  1

template atomNumVal*(_: typedesc[CpAsyncAtomImpl]): static int =
  ## The values per atom issue: 1.
  1

template tilerMN*[T; NumPacked: static int](_: typedesc[CpAsyncAtomImpl[T, NumPacked]]): auto =
  ## The chunk tiler over the tile,
  ## the (NumPacked, 1) chunk: NumPacked elements along the first
  ## mode.
  (NumPacked, 1)

# ═════════════════════════════════════════════════════════════════════════
#  cp.async: the NVIDIA async-copy builtins (SM80+)
# ═════════════════════════════════════════════════════════════════════════
#
#  The builtins follow the CUDA instruction names through a namespace
#  type-chain, so the calls read `cp.async.cg_shared_global_16B(...)`,
#  `cp.async.commit_group()` and `cp.async.wait_group(N)`, the PTX
#  names, the asm texts of the PTX instructions. The type-mapping `async`
#  turns the type `cp` into `CpAsync`, and the copy, commit and wait
#  resolve on that type. The func form of the type-mapping returns
#  void on the pinned Nim (2.2.10), so `async` is a template
#  with the same call syntax.
#
#  Each builtin emits one GCC extended-asm statement through the same
#  path as gemm_mma (kernel_gemm/nvidia_tensor_cores.nim): the asm
#  text is a single literal string, the operand identifiers are Nim
#  locals the asm statement resolves, and crucible emits the C asm
#  statement. A Nim asm statement with no output operands is
#  implicitly volatile in GCC (kept, never deleted), matching
#  the `asm volatile` semantics. The memory clobber keeps
#  the compiler from reordering the asm across smem/gmem
#  accesses.
#
#  The smem destination address is a uint32 shared address. The CUDA
#  intrinsic __cvta_generic_to_shared provides it. The gmem source
#  is a 64-bit pointer. The copy size is the immediate 16,
#  the predicate the runtime size 16 or 0.
#  Both addresses must be 16-byte aligned for the 128-bit
#  copy, the smem element offset and the gmem leading stride
#  multiples of 4 elements.

type
  cp* = object
  CpAsync* = object

template async*(_: type cp): untyped = CpAsync

func cg_shared_global_16B*[T, ShA, StA, ShB, StB, ShP, StP](
    _: type CpAsync;
    dstSmem: TensorView[T, ShA, StA];
    srcGmem: TensorView[T, ShB, StB];
    predicate: TensorView[bool, ShP, StP]) {.inline.} =
  ## One 16-byte predicated cp.async.cg copy, gmem → smem.
  ## The instruction's L2::128B fixes the chunk at 128 bits.
  ## The size operand is always 16 whatever the element type,
  ## and the element count per chunk follows from sizeof(T):
  ## int32 and float32 give 4, int8 gives 16, int64 and float64
  ## give 2. The predicate is the size operand, and a false
  ## predicate makes src_size 0, so the smem destination
  ## is zero-filled (ZFILL), no separate clear needed.
  ##
  ## Args:
  ##   dstSmem: the smem destination slice, the unit view at one chunk
  ##   srcGmem: the gmem source slice, the unit view at one chunk
  ##   predicate: the copy predicate, the (1,)-shaped view at the chunk,
  ##     its (0) element the bool (false → smem zero-fill, no gmem read)
  ##
  ## The addresses are the sliced views' data pointers, the copy
  ## atom's &dst(0) / &src(0). The {.inline.} func's locals
  ## bind the asm backtick names directly, with no template
  ## hygiene to dodge. Each call site inlines to its own scope,
  ## so the chunk loop's expansions do not collide.
  let smemInt = cvtaGenericToShared(dstSmem.data)
  let gmemPtr = srcGmem.data
  let size = if predicate.data[0]: 16 else: 0
  asm "\"cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\" :: \"r\"(`smemInt`), \"l\"(`gmemPtr`), \"n\"(16), \"r\"(`size`) : \"memory\""

func commit_group*(_: type CpAsync) {.inline.} =
  ## One cp.async.commit_group: commit the group of cp.async copies
  ## issued since the previous commit. The statement has no output
  ## or input operands, which makes it implicitly volatile.
  ## The memory clobber orders it against smem/gmem accesses.
  ## The literal `: "memory"` form puts "memory" in the output
  ## position, which GCC and NVRTC reject, so the landed form
  ## has empty outputs and inputs.
  asm "\"cp.async.commit_group;\" :: : \"memory\""

func wait_group*(_: type CpAsync; N: static int) {.inline.} =
  ## One cp.async.wait_group N: block until all but N of the most
  ## recent commit groups have landed in smem. N = 0 waits for
  ## every committed group, the single-stage pipeline. The asm
  ## statement takes only a string literal, so the stage waits
  ## are baked as literal branches.
  when N == 0:
    asm "\"cp.async.wait_group %0;\" :: \"n\"(0) : \"memory\""
  elif N == 1:
    asm "\"cp.async.wait_group %0;\" :: \"n\"(1) : \"memory\""
  else:
    {.error: "wait_group: only the pipeline waits 0 and 1 have baked asm literals".}
