## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## MMA fragment partitioning — partition_A / partition_B / partition_C.
##
## A TiledMma (atom + thread layout) tiles an operand tile of shape
## (tileM, tileK) for A, (tileN, tileK) for B, (tileM, tileN) for C. Each
## thread holds a FRAGMENT: the elements of the tile it reads into
## registers / accumulates, in the register order the hardware expects.
##
## A fragment ELEMENT is one such tile element: its (row, col) coordinate
## in the operand tile. Examples (m16n8k8 tf32 atom, SM80_16x8x8_...):
##   - single atom (1×1 tiled), A tile (16, 8), thread 0 holds
##       (0,0) (8,0) (0,4) (8,4)   — v = 0..3, register order
##   - 2×2 tiled, A tile (32, 8), thread 32 (atom position (1,0)) holds
##       (16,0) (24,0) (16,4) (24,4) — the same v pattern shifted by 16
## The partition layout maps the fragment coordinates to the element's
## col-major OFFSET in the tile (row + col·tileRows), which is what the
## kernel indexes shared/gmem with.
##
## Construction — CuTe thrfrg chain (mma_atom.hpp), with ceramic
## layout algebra, no loops, no seq, no manual div/mod:
##   zipped_divide(tileLayout, (unitM, unitK))        # unit | rest
##   zipped_divide(unit, (atomM, atomK))              # atom | positions
##   fragment (T, V) part: compose(mode(ap, 0), atom.aLayout) — CuTe's
##   `a_tensor.compose(AtomLayoutA_TV{}, _)`: the atom mode (AtomM, AtomK)
##   at tile strides composed with the atom's (T, V) layout. compose's
##   unwrap (CuTe `composition_impl`'s `unwrap`) merges the composed
##   leaves flat, so the fragment comes out in the tile's col-major
##   (k-stride atomM → tileM) with the same nesting CuTe produces.
##
## Result: a rank-3 layout ((T, V), (ThrM, ThrK), (RestM, RestK)) → tile
## offset. The per-thread fragment (CuTe partition_A on a thread index) is
## two indexings: the thread's base = layout(threadCoords, zeros) and the
## per-v offsets = layout(threadCoords, (v, rest)) — no div/mod anywhere,
## the thread coordinate decomposition (tv, tm, tn, tk) is the caller's
## (kernel threadIdx / idx2crd of the thread layout).
##
## Strides: the partition shape (which thread owns which coordinate) is
## determined by the atom and the tile dims and is always static. The
## stride values may be static Int[N] (offsets baked at compile time) or
## runtime int. A runtime leading stride from the launcher's problem shape
## keeps the offset arithmetic runtime — the algebra is uniform, there is
## no static/runtime branch (NVRTC folds the static leaves).
##
## Reference semantics: tensor-layouts `tile_mma_grid` (layout_utils.py)
## enumerates the same (thread, v, rest) → offset grid; the values here
## are cross-checked against CuTe's own partition output (host-extracted).

import ./int_tuples
import ./layouts
import ./layout_constructors
import ./layout_indexing
import ./layout_algebra
import ./tensors
import ./atoms

# ═════════════════════════════════════════════════════════════════════════
#  thrfrg_A/B/C — the thread-fragment layouts (CuTe: ThrMMA::thrfrg_*)
# ═════════════════════════════════════════════════════════════════════════
#
#  CuTe: thrfrg_A (mma_atom.hpp) then slice at the thread:
#    logical_divide(tile, tiled-unit) → zipped_divide(unit, atom) →
#    compose(atom (T,V) layout, _) → zipped_divide(_, (ThrM, ThrK))
#  Ceramic (tuple tilers, quotient-first zipped_divide):
#    zipped_divide(tile, unit) → mode 0 = unit at tile strides,
#                                mode 1 = rest at unit strides
#    zipped_divide(unit, atom) → mode 0 = atom at tile strides,
#                                mode 1 = atom positions at atom strides
#  The (T, V) fragment layout is the atom's (T, V) layout re-strided into
#  the tile's col-major (see module doc).
#
#  The thrfrg_* functions build the partition layout from the operand's
#  OWN layout, so its strides are the tile's real strides (static Int[N]
#  leaves fold at compile time, a runtime leading stride keeps the offset
#  leaves runtime — the algebra is uniform, no branch). The partition_*
#  functions cut it at the thread's coordinates (CuTe: thr_tensor(
#  thr_vmk, _) = slice_and_offset): the (T,V) mode is cut at tv
#  (T-leaves dropped, V kept), the (Thr) mode at (tm/tn, tk), the Rest
#  kept whole. The result is the thread's fragment view (V·, RestM,
#  RestK) — offset inside the view, no linearization.

func thrfrg_A*[Sh, St](tma: static TiledMma; L: Layout[Sh, St]): auto {.inline.} =
  ## The thread-fragment layout of the (M, K) A tensor (CuTe:
  ## ThrMMA::thrfrg_A): ((T, V), (ThrM, ThrK), (RestM, RestK)) → tile
  ## offset. T = threads per atom, V = registers per thread, Thr* = the
  ## thread tiling, Rest* = the tile positions beyond one thread's unit.
  ## The strides come from A's own layout, see the section doc.
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
  let ap = zipped_divide(mode(ur, 0), (atomM, atomK))
  # CuTe: a_tensor.compose(AtomLayoutA_TV, _) — the (T, V) layout
  # composed into the atom mode at tile strides.
  let fragPart = compose(mode(ap, 0), aLayout)
  make_layout((fragPart.shape, mode(ap, 1).shape, mode(ur, 1).shape),
              (fragPart.stride, mode(ap, 1).stride, mode(ur, 1).stride))

func thrfrg_B*[Sh, St](tma: static TiledMma; L: Layout[Sh, St]): auto {.inline.} =
  ## The thread-fragment layout of the (N, K) B tensor (CuTe:
  ## ThrMMA::thrfrg_B): ((T, V), (ThrN, ThrK), (RestN, RestK)) → tile
  ## offset. See thrfrg_A.
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
  let ap = zipped_divide(mode(ur, 0), (atomN, atomK))
  # CuTe: b_tensor.compose(AtomLayoutB_TV, _)
  let fragPart = compose(mode(ap, 0), bLayout)
  make_layout((fragPart.shape, mode(ap, 1).shape, mode(ur, 1).shape),
              (fragPart.stride, mode(ap, 1).stride, mode(ur, 1).stride))

func thrfrg_C*[Sh, St](tma: static TiledMma; L: Layout[Sh, St]): auto {.inline.} =
  ## The thread-fragment layout of the (M, N) C tensor (CuTe:
  ## ThrMMA::thrfrg_C): ((T, V), (ThrM, ThrN), (RestM, RestN)) → tile
  ## offset. See thrfrg_A. Unlike A and B, C has no col-major
  ## requirement: a stride-0 rows view (the epilogue's broadcast bias)
  ## partitions to the same per-column offsets.
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
  let ap = zipped_divide(mode(ur, 0), (atomM, atomN))
  # CuTe: c_tensor.compose(AtomLayoutC_TV, _)
  let fragPart = compose(mode(ap, 0), cLayout)
  make_layout((fragPart.shape, mode(ap, 1).shape, mode(ur, 1).shape),
              (fragPart.stride, mode(ap, 1).stride, mode(ur, 1).stride))

# ═════════════════════════════════════════════════════════════════════════
#  get_slice — the per-thread decomposition (CuTe: ThrMMA)
# ═════════════════════════════════════════════════════════════════════════
#
#  CuTe: thr_mma = mma.get_slice(threadIdx.x);  tCsA = thr_mma.partition_A(sA)
#  The partition layouts above are the (T,V),(Thr,Rest) grid; the kernel
#  needs the THREAD'S slice of that grid. get_slice decomposes the flat
#  thread index ONCE (CuTe: thr_layout_vmnk.get_flat_coord), the
#  partition_* overloads below cut the partition at the decomposed
#  coordinates (CuTe: thr_tensor(thr_vmk, _) = slice_and_offset).

type ThrSlice* = object
  ## The thread's view of the TiledMma (CuTe: ThrMMA) — the decomposed
  ## thread coordinates.
  ##   tv*: the thread within the atom (the T-mode of the fragment
  ##        layouts — CuTe: ThrV)
  ##   tm*, tn*, tk*: the atom coordinates over the threadLayout
  ##        (CuTe: ThrM, ThrN, ThrK)
  tv*, tm*, tn*, tk*: int

func get_slice*(tma: static TiledMma; threadIdx: int): ThrSlice {.inline.} =
  ## Decompose the flat thread index once per thread (CuTe: mma.get_slice(
  ## threadIdx.x)). threadIdx in 0 ..< T·ThrM·ThrN·ThrK; the decomposition
  ## order matches CuTe's (ThrV fastest):
  ##   threadIdx = tv + T·(tm + ThrM·(tn + ThrN·tk))
  const T = tma.atom.threadCount(opA)
  let coords = idx2crd(flatten(tma.threadLayout.shape), threadIdx div T)
  ThrSlice(
    tv: threadIdx mod T,
    tm: coords[0],
    tn: coords[1],
    tk: coords[2])

# ═════════════════════════════════════════════════════════════════════════
#  partition_A / partition_B / partition_C — the thread's operand views
# ═════════════════════════════════════════════════════════════════════════
#
#  CuTe: thr_mma.partition_A(atensor) = make_tensor(data, thrfrg_A(layout))
#  then thr_tensor(thr_vmk, _) — the partition layout cut at the thread's
#  coordinates with the rest left full. The result is the thread's
#  fragment view (V·, RestM, RestK) — offset inside the view, no
#  linearization.

func partition_A*[T, Sh, St](
    tma: static TiledMma; thr: ThrSlice;
    A: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto {.inline.} =
  ## The thread's A-fragment view (CuTe: thr_mma.partition_A(atensor)) —
  ## method-call syntax: `tma.partition_A(thr, A)`.
  ## The partition layout is thrfrg_A; this slices it at the thread's
  ## (T, (ThrM, ThrK)) coordinate, keeping every value mode. The result's
  ## flat order is the register-fragment order the gather copies straight
  ## into the mma fragment.
  const
    atomM = tma.atom.mnk.m
    atomK = tma.atom.mnk.k
    thrM  = tma.threadLayout.shape[0]
    thrK  = tma.threadLayout.shape[2]
    rshape = (Sh.default[0] div (thrM * atomM), Sh.default[1] div (thrK * atomK))
  let thrTensor = make_view(A.data, thrfrg_A(tma, A.layout))   # CuTe: make_tensor(data, thrfrg_A(layout))
  let tsel = idx2crd(tma.atom.aLayout.shape[0], thr.tv) # CuTe: get<0>(thr_vmnk_)
  let vsel = mapLeavesWith(tma.atom.aLayout.shape[1]): X()
  let rsel = mapLeavesWith(rshape): X()
  thrTensor(((tsel, vsel), (thr.tm, thr.tk), rsel))     # CuTe: thr_tensor(thr_vmk, _)

func partition_B*[T, Sh, St](
    tma: static TiledMma; thr: ThrSlice;
    B: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto {.inline.} =
  ## The thread's B-fragment view (CuTe: thr_mma.partition_B(btensor)) —
  ## method-call syntax: `tma.partition_B(thr, B)`.
  ## See partition_A. The (ThrN, ThrK) group is cut at (tn, tk).
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
  ## The thread's C view (CuTe: thr_mma.partition_C(ctensor)) — method-call
  ## syntax: `tma.partition_C(thr, C)`.
  ## See partition_A. The (ThrM, ThrN) group is cut at (tm, tn). C has no
  ## col-major requirement (thrfrg_C), so a stride-0 rows view (the
  ## epilogue's broadcast bias) partitions to the same per-column offsets.
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


# ═══════════════════════════════════════════════════════════════
#  make_fragment_A/B/C — fragment tensors from partition views
# ═══════════════════════════════════════════════════════════════
#  The fragment is a tensor shaped like the partition view, with the V
#  mode in the atom's register order (stride-1, the hardware enumeration
#  of the atom's registers) and the rest modes kept in the view's order.
#  The partition view's mode-0 is the atom's (T, V) layout composed in
#  partition_A/B/C above, so make_fragment_like(view.layout, V shape)
#  pins V to stride-1 and keeps the rest modes' order, decoupling the
#  fragment from the operand's strides (row-major included).
#
#  API: make_fragment_A(mma, partitionView). The V boundary comes from
#  the atom's aLayout (mma.aLayout.shape[1] is passed to
#  make_fragment_like internally).

template make_fragment_A*[T, Sh, St](
    mma: static MmaAtom; t: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto =
  ## The thread's A fragment: V flattened to atom register order
  ## (stride-1), rest modes compact by the view's order. The V boundary
  ## comes from the atom's aLayout (V shape value).
  make_tensor(T, make_fragment_like(t.layout, mma.aLayout.shape[1]))

template make_fragment_B*[T, Sh, St](
    mma: static MmaAtom; t: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto =
  ## The thread's B fragment (see make_fragment_A).
  make_tensor(T, make_fragment_like(t.layout, mma.bLayout.shape[1]))

template make_fragment_C*[T, Sh, St](
    mma: static MmaAtom; t: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto =
  ## The thread's C fragment: the compact (V, RestM, RestN) fragment
  ## derived from the partition view with V = atom cLayout order.
  make_tensor(T, make_fragment_like(t.layout, mma.cLayout.shape[1]))

# ═════════════════════════════════════════════════════════════════════════
#  Fragment tile coordinates — the ragged-tile predication
# ═════════════════════════════════════════════════════════════════════════
#
#  A boundary tile (problem M or N not a multiple of the tile dims)
#  computes the full static tile with zeros outside the problem, and only
#  the elements inside the problem are stored. The gather and the store
#  predicate each fragment element on its coordinate in the tile. The
#  coordinate of the fragment element at flat index i comes from the
#  layouts, never from hand-rolled register-pattern math:
#
#    v = i mod V                        the register position (the V-mode
#                                       is the innermost fragment mode)
#    off = crd2idx(p, coords)           the tile offset from the partition
#                                       layout at the thread's coordinates
#    (row, col) = idx2crd(tileL, off)   the tile coordinate from the tile
#                                       layout
#
#  The partition layout is built with compact strides: the tile
#  coordinates are stride-independent. The exact-coverage contract
#  (tileM === thrM·atomM, tileN === thrN·atomN, asserted below and in
#  gemm_tiled) makes the fragment rest modes empty, so the rest
#  coordinates are always (0, 0).

func fragmentTileCoord*[A, P, TL](atomLayout: A; tsel: auto;
                        thrCoords: (int, int); p: P; tileL: TL;
                        i: int): (int, int) =
  ## The tile (row, col) of fragment element i, decoded through the
  ## layouts (see the section doc). `atomLayout` is the atom's (T, V)
  ## fragment layout, `p` the partition layout, `tileL` the tile layout.
  let V = product(atomLayout.shape[1])
  let vcoords = idx2crd(atomLayout.shape[1], i mod V)
  let off = crd2idx(p, ((tsel, vcoords), thrCoords, (0, 0)))
  idx2crd(tileL, off)

func aFragmentTileRow*(tma: static TiledMma; thr: ThrSlice;
                       tileM: static int; i: int): int =
  ## The tile row of the A-fragment element at flat index i. This is the
  ## ragged-M gather predicate: the element reads gmem only when its row
  ## is inside the problem (row < validM), otherwise it gathers 0.
  ## See fragmentTileCoord for the decode.
  const
    aLayout = tma.atom.aLayout
    atomM = tma.atom.mnk.m
    atomK = tma.atom.mnk.k
    thrM = tma.threadLayout.shape[0]
    thrK = tma.threadLayout.shape[2]
    unitK = thrK * atomK
  static:
    doAssert tileM === thrM * atomM,
      "aFragmentTileRow: tileM (" & $tileM & ") != thrM·atomM (" & $thrM & "·" & $atomM &
        "). The partition contract gives the fragment no rest modes"
  const tileL = make_layout((tileM, unitK), (1, tileM))
  const pA = thrfrg_A(tma, tileL)
  let tsel = idx2crd(aLayout.shape[0], thr.tv)
  fragmentTileCoord(aLayout, tsel, (thr.tm, thr.tk), pA, tileL, i)[0]

func bFragmentTileCol*(tma: static TiledMma; thr: ThrSlice;
                       tileN: static int; i: int): int =
  ## The tile column of the B-fragment element at flat index i. This is
  ## the ragged-N gather predicate: the element reads gmem only when its
  ## column is inside the problem (col < validN), otherwise it gathers 0.
  ## See fragmentTileCoord for the decode.
  const
    bLayout = tma.atom.bLayout
    atomN = tma.atom.mnk.n
    atomK = tma.atom.mnk.k
    thrN = tma.threadLayout.shape[1]
    thrK = tma.threadLayout.shape[2]
    unitK = thrK * atomK
  static:
    doAssert tileN === thrN * atomN,
      "bFragmentTileCol: tileN (" & $tileN & ") != thrN·atomN (" & $thrN & "·" & $atomN &
        "). The partition contract gives the fragment no rest modes"
  const tileL = make_layout((tileN, unitK), (1, tileN))
  const pB = thrfrg_B(tma, tileL)
  let tsel = idx2crd(bLayout.shape[0], thr.tv)
  fragmentTileCoord(bLayout, tsel, (thr.tn, thr.tk), pB, tileL, i)[0]

func cStoreMask*(tma: static TiledMma; thr: ThrSlice;
                 tileM, tileN: static int; validM, validN: int): int =
  ## The store-predication bitmask over the C fragment: bit i set = the
  ## element's tile coordinate is inside the valid (M, N) range of the
  ## tile and may be stored. gemm_cta computes it from the tile's valid
  ## extents (the min of the tile dim and the remaining problem extent)
  ## and stores it on the op, whose finalStore skips the masked-off
  ## stores (see kernel_gemm_epilogues). All bits set (no
  ## predication) when the tile is fully inside the problem. See
  ## fragmentTileCoord for the coordinate decode.
  const
    cLayout = tma.atom.cLayout
    atomM = tma.atom.mnk.m
    atomN = tma.atom.mnk.n
    thrM = tma.threadLayout.shape[0]
    thrN = tma.threadLayout.shape[1]
    thrK = tma.threadLayout.shape[2]
    blockSize = tma.atom.threadCount(opA) * thrM * thrN * thrK
    fragSize = tileM * tileN div blockSize
  static:
    doAssert fragSize <= 63,
      "cStoreMask: the C fragment (" & $fragSize &
      " elements per thread) exceeds the 63-bit store mask"
    doAssert tileM === thrM * atomM and tileN === thrN * atomN,
      "cStoreMask: the tile dims must be the thread layout's exact coverage" &
      " (the partition contract gives the fragment no rest modes)"
  const tileL = make_layout((tileM, tileN), (1, tileM))
  const pC = thrfrg_C(tma, tileL)
  let tsel = idx2crd(cLayout.shape[0], thr.tv)
  result = 0
  for i in 0 ..< fragSize:
    let (m, n) = fragmentTileCoord(cLayout, tsel, (thr.tm, thr.tn),
                                   pC, tileL, i)
    if m <= validM - 1 and n <= validN - 1:
      result = result or (1 shl i)

# ═════════════════════════════════════════════════════════════════════════
#  make_fragment_A/B/C — fragment tensors from partition views
# ═════════════════════════════════════════════════════════════════════════
#  The fragment is a tensor shaped like the partition view, with the V
#  mode in the atom's register order (stride-1, the hardware enumeration
#  of the atom's registers) and the rest modes kept in the view's order.
#  The partition view's mode-0 is the atom's (T, V) layout composed in
#  partition_A/B/C above, so make_fragment_like(view.layout, V shape)
#  pins V to stride-1 and keeps the rest modes' order, decoupling the
#  fragment from the operand's strides (row-major included).
#
#  API: make_fragment_A(mma, partitionView). The V boundary comes from
#  the atom's aLayout (mma.aLayout.shape[1] is passed to
#  make_fragment_like internally).

template make_fragment_A*[T, Sh, St](
    mma: static MmaAtom; t: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto =
  ## The thread's A fragment: V flattened to atom register order
  ## (stride-1), rest modes compact by the view's order. The V boundary
  ## comes from the atom's aLayout (V shape value).
  make_tensor(T, make_fragment_like(t.layout, mma.aLayout.shape[1]))

template make_fragment_B*[T, Sh, St](
    mma: static MmaAtom; t: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto =
  ## The thread's B fragment (see make_fragment_A).
  make_tensor(T, make_fragment_like(t.layout, mma.bLayout.shape[1]))

template make_fragment_C*[T, Sh, St](
    mma: static MmaAtom; t: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto =
  ## The thread's C fragment: the compact (V, RestM, RestN) fragment
  ## derived from the partition view with V = atom cLayout order.
  make_tensor(T, make_fragment_like(t.layout, mma.cLayout.shape[1]))
