## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Copy atom: Low-level memory copy primitive.

import ./int_tuples
import ./layouts
import ./layout_constructors
import ./tensors
import workspace/crucible

# ═════════════════════════════════════════════════════════════════════════
#  The cp.async atom
# ═════════════════════════════════════════════════════════════════════════

type CpAsyncAtomImpl[T; NumPacked: static int] = object
  ## One 16-byte cp.async.cg chunk: NumPacked elements of type T.

type CpAsyncAtom*[T] = CpAsyncAtomImpl[T, 16 div sizeof(T)]
  ## 16-byte cp.async chunk atom for element type T, NumPacked = 16 div sizeof(T).

template numPacked*[T; NumPacked: static int](_: typedesc[CpAsyncAtomImpl[T, NumPacked]]): int =
  static:
    doAssert NumPacked * sizeof(T) === 16,
      "CpAsyncAtom: the chunk must be 16 bytes (the cp.async L2::128B)"
  NumPacked

template tilerMN*[T; NumPacked: static int](_: typedesc[CpAsyncAtomImpl[T, NumPacked]]): auto =
  ## The chunk tiler: (NumPacked, 1), NumPacked consecutive elements.
  (NumPacked, 1)

# ═════════════════════════════════════════════════════════════════════════
#  cp.async: the NVIDIA async-copy builtins (SM80+)
# ═════════════════════════════════════════════════════════════════════════
#
#  Expose Cuda intrinsics:
#  - cp.async.cg.shared.global.L2::128B (Prepare up to 16B / 128b async copy, with zero masking)
#  - cp.async.commit_group (Launch the non-blocking copies)
#  - cp.async.wait_group (Block until copies are finished)

type
  cp* = object
  CpAsync* = object

template async*(_: type cp): untyped = CpAsync

func cg_shared_global_16B*[T, ShA, StA, ShB, StB](
    _: type CpAsync;
    dstSmem: TensorView[T, ShA, StA];
    srcGmem: TensorView[T, ShB, StB];
    srcSize: uint32) {.inline.} =
  ## Prepare a 16-byte cp.async.cg copy
  ##   from global memory (gmem)
  ##   to the per-warp shared memory (smem),
  ## issued asynchronously with other prepared copies in the same commit_group.
  ## The copies from the same commit_group are waited for with wait_group.
  ## A srcSize of 0 fills the chunk with zeros instead of gmem data.
  ##
  ## This uses hardware resources separate from Cuda cores
  ## and allows latency hiding (pipelining)
  ## by overlapping data movement and compute.
  ##
  ## Args:
  ##   dstSmem: the smem destination slice, the unit view at one chunk
  ##   srcGmem: the gmem source slice, the unit view at one chunk
  ##   srcSize: the instruction's src-size operand, 16 or 0
  ##
  ## Both addresses must be 16-byte aligned for the 128-bit copy.
  let smemInt = cvtaGenericToShared(dstSmem.data)
  let gmemPtr = srcGmem.data
  asm "\"cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\" :: \"r\"(`smemInt`), \"l\"(`gmemPtr`), \"n\"(16), \"r\"(`srcSize`) : \"memory\""

func commit_group*(_: type CpAsync) {.inline.} =
  ## Commit the cp.async copies prepared since the previous commit
  asm "\"cp.async.commit_group;\" :: : \"memory\""

func wait_group*(_: type CpAsync; N: static int) {.inline.} =
  ## Block until all but N of the recent copy groups are fully copied to shared memory.
  ## The buffering depth is N + 1 stages:
  ##   N = 0: single-buffered
  ##   N = 1: double-buffered
  ##   N = 2: triple-buffered
  when N == 0:
    asm "\"cp.async.wait_group %0;\" :: \"n\"(0) : \"memory\""
  elif N == 1:
    asm "\"cp.async.wait_group %0;\" :: \"n\"(1) : \"memory\""
  elif N == 2:
    asm "\"cp.async.wait_group %0;\" :: \"n\"(2) : \"memory\""
  else:
    {.error: "wait_group: only N = 0, 1 and 2 have baked asm literals".}
