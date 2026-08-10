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
## Construction — CuTe thrfrg chain (mma_atom.hpp:288-314), with ceramic
## layout algebra, no loops, no seq, no runtime div/mod, all Int[N]:
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
#  Partition — full layout per operand
# ═════════════════════════════════════════════════════════════════════════
#
#  CuTe: thrfrg_A (mma_atom.hpp:288-314) then slice at the thread:
#    logical_divide(tile, tiled-unit) → zipped_divide(unit, atom) →
#    compose(atom (T,V) layout, _) → zipped_divide(_, (ThrM, ThrK))
#  Ceramic (tuple tilers, quotient-first zipped_divide):
#    zipped_divide(tile, unit) → mode 0 = unit at tile strides,
#                                mode 1 = rest at unit strides
#    zipped_divide(unit, atom) → mode 0 = atom at tile strides,
#                                mode 1 = atom positions at atom strides
#  The (T, V) fragment layout is the atom's (T, V) layout re-strided into
#  the tile's col-major (see module doc).

func partition_A*(mma: static TiledMma; tileM, tileK: static int): auto {.inline.} =
  ## A partition layout: ((T, V), (ThrM, ThrK), (RestM, RestK)) → tile
  ## offset (m + k·tileM). T = threads per atom, V = registers per thread.
  ## Thread coordinate for the (ThrM, ThrK) group: (tm, tk).
  const atomM = mma.atom.mnk.m
  const atomK = mma.atom.mnk.k
  const thrM  = mma.threadLayout.shape[0]
  const thrK  = mma.threadLayout.shape[2]
  static:
    doAssert toIntVal(cosize(mma.atom.aLayout)) == atomM * atomK,
      "partition_A: A fragment layout cosize (" & $toIntVal(cosize(mma.atom.aLayout)) &
      ") != atom M·K (" & $atomM & "·" & $atomK & ") — the (T, V) layout must tile the operand"
  const unitM = thrM * atomM
  const unitK = thrK * atomK
  let tileL = make_layout((tileM, tileK))
  let ur = zipped_divide(tileL, (unitM, unitK))
  let ap = zipped_divide(mode(ur, 0), (atomM, atomK))
  # CuTe: a_tensor.compose(AtomLayoutA_TV, _) — the (T, V) layout
  # composed into the atom mode at tile strides.
  let fragPart = compose(mode(ap, 0), mma.atom.aLayout)
  make_layout((fragPart.shape, mode(ap, 1).shape, mode(ur, 1).shape),
              (fragPart.stride, mode(ap, 1).stride, mode(ur, 1).stride))

func partition_B*(mma: static TiledMma; tileN, tileK: static int): auto {.inline.} =
  ## B partition layout: ((T, V), (ThrN, ThrK), (RestN, RestK)) → tile
  ## offset (n + k·tileN). Thread coordinate for the (ThrN, ThrK) group:
  ## (tn, tk).
  const atomN = mma.atom.mnk.n
  const atomK = mma.atom.mnk.k
  const thrN  = mma.threadLayout.shape[1]
  const thrK  = mma.threadLayout.shape[2]
  static:
    doAssert toIntVal(cosize(mma.atom.bLayout)) == atomN * atomK,
      "partition_B: B fragment layout cosize (" & $toIntVal(cosize(mma.atom.bLayout)) &
      ") != atom N·K (" & $atomN & "·" & $atomK & ") — the (T, V) layout must tile the operand"
  const unitN = thrN * atomN
  const unitK = thrK * atomK
  let tileL = make_layout((tileN, tileK))
  let ur = zipped_divide(tileL, (unitN, unitK))
  let ap = zipped_divide(mode(ur, 0), (atomN, atomK))
  # CuTe: b_tensor.compose(AtomLayoutB_TV, _)
  let fragPart = compose(mode(ap, 0), mma.atom.bLayout)
  make_layout((fragPart.shape, mode(ap, 1).shape, mode(ur, 1).shape),
              (fragPart.stride, mode(ap, 1).stride, mode(ur, 1).stride))

func partition_C*(mma: static TiledMma; tileM, tileN: static int): auto {.inline.} =
  ## C partition layout: ((T, V), (ThrM, ThrN), (RestM, RestN)) → tile
  ## offset (m + n·tileM). Thread coordinate for the (ThrM, ThrN) group:
  ## (tm, tn).
  const atomM = mma.atom.mnk.m
  const atomN = mma.atom.mnk.n
  const thrM  = mma.threadLayout.shape[0]
  const thrN  = mma.threadLayout.shape[1]
  static:
    doAssert toIntVal(cosize(mma.atom.cLayout)) == atomM * atomN,
      "partition_C: C fragment layout cosize (" & $toIntVal(cosize(mma.atom.cLayout)) &
      ") != atom M·N (" & $atomM & "·" & $atomN & ") — the (T, V) layout must tile the operand"
  const unitM = thrM * atomM
  const unitN = thrN * atomN
  let tileL = make_layout((tileM, tileN))
  let ur = zipped_divide(tileL, (unitM, unitN))
  let ap = zipped_divide(mode(ur, 0), (atomM, atomN))
  # CuTe: c_tensor.compose(AtomLayoutC_TV, _)
  let fragPart = compose(mode(ap, 0), mma.atom.cLayout)
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
  const T = toIntVal(tma.atom.threadCount(opA))
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
#  coordinates with the rest left full. Here: the (T,V) mode is cut at tv
#  (T-leaves dropped, V kept), the (Thr) mode at (tm/tn, tk), the Rest
#  kept whole. The result is the thread's fragment view
#  (V·, RestM, RestK) — offset inside the view, no linearization.

func partition_A*[T, Sh, St](
    tma: static TiledMma; thr: ThrSlice;
    A: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto {.inline.} =
  ## The thread's A-fragment view (CuTe: thr_mma.partition_A(atensor)) —
  ## method-call syntax: `tma.partition_A(thr, A)`.
  const
    pA = partition_A(tma, toIntVal(Sh.default[0]), toIntVal(Sh.default[1]))
    tshape = pA.shape[0][0]      # the atom's T-mode
    vshape = pA.shape[0][1]      # the atom's V-mode
    rshape = pA.shape[2]         # the (RestM, RestK) mode
  let tsel = idx2crd(tshape, thr.tv)
  let vsel = mapLeavesWith(vshape): X()
  let rsel = mapLeavesWith(rshape): X()
  let sel  = ((tsel, vsel), (thr.tm, thr.tk), rsel)
  make_view(A.data +% toIntVal(crd2idx(pA, sel)), slice(pA, sel))

func partition_B*[T, Sh, St](
    tma: static TiledMma; thr: ThrSlice;
    B: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto {.inline.} =
  ## The thread's B-fragment view (CuTe: thr_mma.partition_B(btensor)) —
  ## method-call syntax: `tma.partition_B(thr, B)`.
  const
    pB = partition_B(tma, toIntVal(Sh.default[0]), toIntVal(Sh.default[1]))
    tshape = pB.shape[0][0]      # the atom's T-mode
    vshape = pB.shape[0][1]      # the atom's V-mode
    rshape = pB.shape[2]         # the (RestN, RestK) mode
  let tsel = idx2crd(tshape, thr.tv)
  let vsel = mapLeavesWith(vshape): X()
  let rsel = mapLeavesWith(rshape): X()
  let sel  = ((tsel, vsel), (thr.tn, thr.tk), rsel)
  make_view(B.data +% toIntVal(crd2idx(pB, sel)), slice(pB, sel))

func partition_C*[T, Sh, St](
    tma: static TiledMma; thr: ThrSlice;
    C: TensorView[T, Sh, St] or Tensor[T, Sh, St]): auto {.inline.} =
  ## The thread's C view (CuTe: thr_mma.partition_C(ctensor)) — method-call
  ## syntax: `tma.partition_C(thr, C)`.
  const
    pC = partition_C(tma, toIntVal(Sh.default[0]), toIntVal(Sh.default[1]))
    tshape = pC.shape[0][0]      # the atom's T-mode
    vshape = pC.shape[0][1]      # the atom's V-mode
    rshape = pC.shape[2]         # the (RestM, RestN) mode
  let tsel = idx2crd(tshape, thr.tv)
  let vsel = mapLeavesWith(vshape): X()
  let rsel = mapLeavesWith(rshape): X()
  let sel  = ((tsel, vsel), (thr.tm, thr.tn), rsel)
  make_view(C.data +% toIntVal(crd2idx(pC, sel)), slice(pC, sel))
