## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## MMA fragment partitioning: partition_A / partition_B / partition_C.
##
## A TiledMma (an MMA atom plus a thread layout) computes one tile of the output:
##   - A tile (tileM, tileK)
##   - B tile (tileN, tileK)
##   - C tile (tileM, tileN)
## Each thread holds a fragment: the tile elements it reads into registers (A, B) or accumulates (C)
## in the order the hardware instruction expects.
##
## Example (m16n8k8 tf32 atom, SM80_16x8x8_F32TF32TF32F32_TN):
##   - one atom: A tile (16, 8), thread 0 holds (0,0) (8,0) (0,4) (8,4), its 4 values in register order
##   - 2×2 tiled: A tile (32, 8), thread 32 (atom (1, 0)) holds (16,0) (24,0) (16,4) (24,4), the same pattern shifted by 16
##
## The fragment layout maps (thread, value, remainder) to the element's col-major offset in the tile
## (the address the kernel indexes shared / global memory with).
## The remainder counts how many times the thread layout repeats to fill the tile:
##   the (2, 2, 1) thread layout covers (32, 8) of the A tile, so a (64, 16) tile has remainder (2, 2).
##   Each repetition shifts a thread's fragment by (32, 8).
## It is not the tiles of the kernel's K-loop, and it is not a partial tile at the problem edge (those are bounds-predicated).
## It is built by thrfrg_A/B/C and cut per thread by get_slice + partition_A/B/C.
## make_fragment_A/B/C allocate the register buffers, and cStoreMask predicates the C store.

import ./int_tuples
import ./layouts
import ./layout_constructors
import ./layout_indexing
import ./layout_algebra
import ./tensors
import ./atoms

# ═════════════════════════════════════════════════════════════════════════
#  thrfrg_A/B/C: the fragment layout of an operand tile
# ═════════════════════════════════════════════════════════════════════════
#
#  For every (thread, value, tile position) the layout gives the offset in the tile, in the operand's own strides.
#  It nests three blocks:
#    (T, V)         the atom's layout: its T threads × V values mapped onto the atom block (atomM × atomK)
#    (ThrM, ThrK)   the atom block of the thread layout this thread belongs to
#                   (each block shifts the atom pattern by (tm·atomM, tk·atomK))
#                   [(ThrM, ThrN) for C]
#    (RepeatM, RepeatK) how many times the thread layout repeats to fill a tile larger than the thread layout's coverage:
#                   the (2, 2, 1) thread layout covers (32, 8) of the A tile, so a (64, 16) tile has Rest (2, 2)
#                   (each repetition shifts a thread's fragment by (32, 8))
#
#  Build:
#    - split the tile into (unit, remainder) by one thread's coverage (thread layout × atom)
#    - split the unit into (atom block, atom positions)
#    - compose the atom's (T, V) layout into the atom block

func thrfrg_A*[Sh, St](tma: static TiledMma; L: Layout[Sh, St]): auto {.inline.} =
  ## The fragment layout of the (M, K) A tensor:
  ## ((T, V), (ThrM, ThrK), (RepeatM, RepeatK)) → offset in the tile.
  ## T: threads per atom. V: registers per thread. Thr*: the thread layout.
  ## Repeat*: how many times the thread layout repeats across the tile
  ## (the (2, 2, 1) thread layout covers (32, 8), so a (64, 16) tile has Rest (2, 2)).
  const
    aLayout = tma.atom.aLayout
    atomM = tma.atom.mnk.m
    atomK = tma.atom.mnk.k
    thrM  = tma.threadLayout.shape[0]
    thrK  = tma.threadLayout.shape[2]
  static:
    doAssert cosize(aLayout) === atomM * atomK,
      "thrfrg_A: A fragment layout cosize (" & $cosize(aLayout) &
      ") != atom M·K (" & $atomM & "·" & $atomK & ") — the (T, V) layout must tile the operand"
    doAssert St.default[0] === 1,
      "thrfrg_A: operand must be col-major (stride (1, k-stride)), row-major thread" &
      " offsets in the partition are not layout-correct yet (fragment construction is," &
      " but the partition's T-mode strides are not)"
  const unitM = thrM * atomM
  const unitK = thrK * atomK
  let ur = zipped_divide(L, (unitM, unitK))
  let ap = zipped_divide(ur.mode(0), (atomM, atomK))
  let fragPart = compose(ap.mode(0), aLayout)
  make_layout((fragPart.shape, ap.mode(1).shape, ur.mode(1).shape),
              (fragPart.stride, ap.mode(1).stride, ur.mode(1).stride))

func thrfrg_B*[Sh, St](tma: static TiledMma; L: Layout[Sh, St]): auto {.inline.} =
  ## The fragment layout of the (N, K) B tensor:
  ## ((T, V), (ThrN, ThrK), (RepeatN, RepeatK)) → offset in the tile. See thrfrg_A.
  const
    bLayout = tma.atom.bLayout
    atomN = tma.atom.mnk.n
    atomK = tma.atom.mnk.k
    thrN  = tma.threadLayout.shape[1]
    thrK  = tma.threadLayout.shape[2]
  static:
    doAssert cosize(bLayout) === atomN * atomK,
      "thrfrg_B: B fragment layout cosize (" & $cosize(bLayout) &
      ") != atom N·K (" & $atomN & "·" & $atomK & ") — the (T, V) layout must tile the operand"
    doAssert St.default[0] === 1,
      "thrfrg_B: operand must be col-major (stride (1, k-stride)), row-major thread" &
      " offsets in the partition are not layout-correct yet (fragment construction is," &
      " but the partition's T-mode strides are not)"
  const unitN = thrN * atomN
  const unitK = thrK * atomK
  let ur = zipped_divide(L, (unitN, unitK))
  let ap = zipped_divide(ur.mode(0), (atomN, atomK))
  let fragPart = compose(ap.mode(0), bLayout)
  make_layout((fragPart.shape, ap.mode(1).shape, ur.mode(1).shape),
              (fragPart.stride, ap.mode(1).stride, ur.mode(1).stride))

func thrfrg_C*[Sh, St](tma: static TiledMma; L: Layout[Sh, St]): auto {.inline.} =
  ## The fragment layout of the (M, N) C tensor:
  ## ((T, V), (ThrM, ThrN), (RepeatM, RepeatN)) → offset in the tile. See thrfrg_A.
  ## Unlike A and B, C has no col-major requirement: a stride-0 rows view
  ## (the epilogue's broadcast bias) partitions to the same per-column offsets.
  const
    cLayout = tma.atom.cLayout
    atomM = tma.atom.mnk.m
    atomN = tma.atom.mnk.n
    thrM  = tma.threadLayout.shape[0]
    thrN  = tma.threadLayout.shape[1]
  static:
    doAssert cosize(cLayout) === atomM * atomN,
      "thrfrg_C: C fragment layout cosize (" & $cosize(cLayout) &
      ") != atom M·N (" & $atomM & "·" & $atomN & ") — the (T, V) layout must tile the operand"
  const unitM = thrM * atomM
  const unitN = thrN * atomN
  let ur = zipped_divide(L, (unitM, unitN))
  let ap = zipped_divide(ur.mode(0), (atomM, atomN))
  let fragPart = compose(ap.mode(0), cLayout)
  make_layout((fragPart.shape, ap.mode(1).shape, ur.mode(1).shape),
              (fragPart.stride, ap.mode(1).stride, ur.mode(1).stride))

# ═════════════════════════════════════════════════════════════════════════
#  get_slice: one thread's coordinates in the fragment layout
# ═════════════════════════════════════════════════════════════════════════
#
#  The thrfrg layouts cover all threads. A kernel thread needs its own slice:
#  get_slice splits the flat thread index once, then partition_A/B/C cut the thrfrg layout at the result.
#  Example (m16n8k8 atom, (2, 2, 1) thread layout, A tile (32, 8)):
#    get_slice(32) → tv 0, tm 1, tn 0, tk 0
#    partition_A cuts the ((32, 4), (2, 1), (1, 1)) layout at (0, 1, 0):
#      - T dropped, V kept
#      - (ThrM, ThrK) cut at (1, 0)
#      - remainder kept whole
#    → the thread's (4, 1, 1) fragment view, its 4 values in register order

type ThrSlice* = object
  ## One thread's position in the TiledMma: tv is the thread's position within one atom (0 ..< T).
  ## tm/tn/tk are which atom block of the thread layout the thread belongs to.
  tv*, tm*, tn*, tk*: int

func get_slice*(tma: static TiledMma; threadIdx: int): ThrSlice {.inline.} =
  ## Split the flat thread index into the coordinates above, once per thread.
  ## The thread-within-atom index varies fastest: threadIdx = tv + T·(tm + ThrM·(tn + ThrN·tk))
  const T = tma.atom.threadCount(opA)
  let coords = idx2crd(flatten(tma.threadLayout.shape), threadIdx div T)
  ThrSlice(
    tv: threadIdx mod T,
    tm: coords[0],
    tn: coords[1],
    tk: coords[2])

# ═════════════════════════════════════════════════════════════════════════
#  partition_A / partition_B / partition_C: the thread's fragment of a tensor
# ═════════════════════════════════════════════════════════════════════════
#
#  Cut a thrfrg layout at one thread's coordinates:
#    - keep all V values
#    - drop the thread's T coordinate
#    - cut the thread block at (tm, tk), or (tm, tn) for C
#    - keep the remainder whole
#  The result is the thread's fragment view, in register order.

func partition_A*[T, Sh, St](
    tma: static TiledMma; thr: ThrSlice;
    A: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto {.inline.} =
  ## The thread's A-fragment view of a tensor, method-call syntax: `tma.partition_A(thr, A)`.
  ## It cuts the thrfrg_A layout at the thread's coordinates, keeping every V value.
  ## The flat order is the register order the gather copies into the mma fragment.
  const
    atomM = tma.atom.mnk.m
    atomK = tma.atom.mnk.k
    thrM  = tma.threadLayout.shape[0]
    thrK  = tma.threadLayout.shape[2]
    rshape = (Sh.default[0] div (thrM * atomM), Sh.default[1] div (thrK * atomK))
  let thrTensor = make_view(A.data, thrfrg_A(tma, A.layout))
  let tsel = idx2crd(tma.atom.aLayout.shape[0], thr.tv)
  let vsel = mapLeavesWith(tma.atom.aLayout.shape[1]): X()
  let rsel = mapLeavesWith(rshape): X()
  thrTensor(((tsel, vsel), (thr.tm, thr.tk), rsel))

func partition_B*[T, Sh, St](
    tma: static TiledMma; thr: ThrSlice;
    B: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto {.inline.} =
  ## The thread's B-fragment view of a tensor, method-call syntax: `tma.partition_B(thr, B)`.
  ## See partition_A. The (ThrN, ThrK) block is cut at (tn, tk).
  const
    atomN = tma.atom.mnk.n
    atomK = tma.atom.mnk.k
    thrN  = tma.threadLayout.shape[1]
    thrK  = tma.threadLayout.shape[2]
    rshape = (Sh.default[0] div (thrN * atomN), Sh.default[1] div (thrK * atomK))
  let thrTensor = make_view(B.data, thrfrg_B(tma, B.layout))
  let tsel = idx2crd(tma.atom.bLayout.shape[0], thr.tv)
  let vsel = mapLeavesWith(tma.atom.bLayout.shape[1]): X()
  let rsel = mapLeavesWith(rshape): X()
  thrTensor(((tsel, vsel), (thr.tn, thr.tk), rsel))

func partition_C*[T, Sh, St](
    tma: static TiledMma; thr: ThrSlice;
    C: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto {.inline.} =
  ## The thread's C view of a tensor, method-call syntax: `tma.partition_C(thr, C)`. See partition_A.
  ## The (ThrM, ThrN) block is cut at (tm, tn). C has no col-major requirement (thrfrg_C), so a stride-0 rows view
  ## (the epilogue's broadcast bias) partitions to the same per-column offsets.
  const
    atomM = tma.atom.mnk.m
    atomN = tma.atom.mnk.n
    thrM  = tma.threadLayout.shape[0]
    thrN  = tma.threadLayout.shape[1]
    rshape = (Sh.default[0] div (thrM * atomM), Sh.default[1] div (thrN * atomN))
  let thrTensor = make_view(C.data, thrfrg_C(tma, C.layout))
  let tsel = idx2crd(tma.atom.cLayout.shape[0], thr.tv)
  let vsel = mapLeavesWith(tma.atom.cLayout.shape[1]): X()
  let rsel = mapLeavesWith(rshape): X()
  thrTensor(((tsel, vsel), (thr.tm, thr.tn), rsel))


# ═════════════════════════════════════════════════════════════════════════
#  cStoreMask: final data movement + predication
# ═════════════════════════════════════════════════════════════════════════
#
#  cStoreMask predicates the store on each C-fragment element's
#  coordinate in the tile.

func cStoreMask*(tma: static TiledMma; threadIdx: int;
                 tileM, tileN: static int; validM, validN: int): int =
  ## Return a predication mask for selective copy
  const cLayout = tma.atom.cLayout
  const atomM = tma.atom.mnk.m
  const atomN = tma.atom.mnk.n
  const atomL = make_layout((atomM, atomN), (1, atomM))
  const fragSize = toIntVal(product(cLayout.shape[1]))
  const blockSize = tma.threadCount()
  static:
    doAssert fragSize <= 63,
      "cStoreMask: the C fragment (" & $fragSize &
      " elements per thread) exceeds the 63-bit store mask"
    doAssert tileM === tma.thrM * atomM and tileN === tma.thrN * atomN,
      "cStoreMask: the tile dims must be the thread layout's exact coverage" &
      " (the partition contract gives the fragment no rest modes)"
  if validM <= 0 or validN <= 0:
    return 0

  #  Two compile-time tables map fragment elements to tile coordinates (div/mod is too slow at runtime):
  #  - coordMap: value index → (row, col) within the atom block, from the atom's C layout
  #  - origin: thread → the (row, col) of its fragment's top-left element in the tile
  const coordMap = block:
    var a: array[fragSize, (int, int)]
    for v in 0 ..< fragSize:
      a[v] = idx2crd(atomL, toIntVal(crd2idx(cLayout, (0, v))))
    a
  const origin = block:
    var a: array[blockSize, (int, int)]
    for tid in 0 ..< blockSize:
      let s = tma.get_slice(tid)
      let tsel = idx2crd(cLayout.shape[0], s.tv)
      let f0 = idx2crd(atomL, toIntVal(crd2idx(cLayout, (tsel, 0))))
      a[tid] = (s.tm * atomM + f0[0], s.tn * atomN + f0[1])
    a

  let o = origin[threadIdx]
  let resM = validM - o[0]
  let resN = validN - o[1]
  result = 0
  for i in 0 ..< fragSize:
    if coordMap[i][0] < resM and coordMap[i][1] < resN:
      result = result or (1 shl i)

# ═════════════════════════════════════════════════════════════════════════
#  make_fragment_A/B/C: register buffers in hardware fragment order
# ═════════════════════════════════════════════════════════════════════════
#
#  A partition view is a window into a tensor with the tensor's strides. A fragment is a register buffer:
#  its own layout, in the order the hardware instruction reads and writes registers.
#  make_fragment builds that buffer from a partition view:
#    - the V values become stride-1, the hardware register order, regardless of the operand's strides (row-major included)
#    - the remaining positions keep the view's order, packed after the V registers
#  The result has the same shape as the view, so copying between the two moves the thread's elements in flat register order.

template make_fragment_A*[T, Sh, St](
    mma: static MmaAtom; t: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto =
  ## The thread's A fragment: a register buffer with the V values in hardware order (stride-1).
  ## The remaining positions keep the view's order. V is the atom's register count (aLayout.shape[1] values per thread).
  make_tensor(T, make_fragment_like(t.layout, mma.aLayout.shape[1]))

template make_fragment_B*[T, Sh, St](
    mma: static MmaAtom; t: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto =
  ## The thread's B fragment (see make_fragment_A).
  make_tensor(T, make_fragment_like(t.layout, mma.bLayout.shape[1]))

template make_fragment_C*[T, Sh, St](
    mma: static MmaAtom; t: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto =
  ## The thread's C fragment: a register buffer with the V values in hardware order (stride-1).
  ## The remaining positions keep the view's order. V is the atom's register count (cLayout.shape[1] values per thread).
  make_tensor(T, make_fragment_like(t.layout, mma.cLayout.shape[1]))
