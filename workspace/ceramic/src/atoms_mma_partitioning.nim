## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## MMA fragment partitioning — partition_A/B/C and fragment derivation.
##
## Given a TiledMma (atom + thread layout) and a tile shape, compute the
## per-thread fragment: which elements of A/B/C each thread holds in
## registers, and in what order (the V order IS the register order the
## hardware expects).
##
## Model: tensor-layouts `tile_mma_grid` (layout_utils.py:193) — enumerate
## atom positions × (T, V) → offset, via the atom's own fragment layouts.
## The CuTe thrfrg chain (logical_divide → zipped_divide → compose →
## zipped_divide) computes the same mapping in layout-composition form;
## the direct enumeration is used here because it is the form emission
## consumes (per-thread register addresses). Coverage/disjointness is a
## test-only concern and lives in the tests (verifyFragments).
##
## Offsets are col-major in the operand tile: A (m + k·M), B (n + k·N),
## C (m + n·M). Flat (T,V) index for crd2idx: t + T·v (mode 0 fastest).

import ./int_tuples
import ./layouts
import ./layout_constructors
import ./layout_indexing
import ./atoms

# ═════════════════════════════════════════════════════════════════════════
#  FragmentElement
# ═════════════════════════════════════════════════════════════════════════

type
  FragmentElement* = object
    ## One register-held element of a fragment.
    atomIdx*: array[3, int]   ## (am, an, ak) atom position in the thread tiling
    row*, col*: int           ## global (m, k)/(n, k)/(m, n) coord in the tile
    valIdx*: int              ## v index — the register order within the atom
    offset*: int              ## layout offset within the atom tile

  Fragment* = seq[FragmentElement]
    ## A thread's fragment: ordered by atom position, then by v.

# ═════════════════════════════════════════════════════════════════════════
#  Partition — get_slice mapping
# ═════════════════════════════════════════════════════════════════════════
#
#  The global thread id maps to ONE atom position (CuTe get_slice):
#    flat = tv + T·(tm + ThrM·(tn + ThrN·tk))   — tiled_product(ThrID, AtomLayout)
#  so atom position = thread div T, decomposed col-major over the operand's
#  atom tiling (A: (ThrM, ThrK), B: (ThrN, ThrK), C: (ThrM, ThrN)),
#  and the local thread = thread mod T.

func partitionA*[A, TL](mma: TiledMma[A, TL];
                        tileShape: (int, int);  # (M, K) of the tile
                        thread: int): Fragment =
  ## Per-thread A fragment: (m, k) coords, register order.
  let
    mAtoms = mma.threadLayout.shape[0].toIntVal()  # ThrM
    kAtoms = mma.threadLayout.shape[2].toIntVal()  # ThrK
    mAtom  = mma.atom.mnk.m
    kAtom  = mma.atom.mnk.k
    tCount = mma.atom.threadCount(opA).toIntVal()
    vCount = mma.atom.fragmentValsPerThread(opA).toIntVal()
  let
    tileM = mAtoms * mAtom
    tileK = kAtoms * kAtom
  doAssert tileShape[0] mod tileM == 0 and tileShape[1] mod tileK == 0,
    "tile shape must be a multiple of the tiled-mma unit"
  let
    restM = tileShape[0] div tileM
    restK = tileShape[1] div tileK
    atomIdx = thread div tCount
    tm = atomIdx mod mAtoms
    tn = (atomIdx div mAtoms) mod mma.threadLayout.shape[1].toIntVal()
    tk = atomIdx div (mAtoms * mma.threadLayout.shape[1].toIntVal())
    am = tm
    ak = tk
    localT = thread mod tCount
  ## Rest positions (value tiling beyond the tiled-mma unit) are ordered
  ## mma-position-outer, col-major over (RestM, RestK) — the CuTe fragment
  ## shape (MMA, M, K) with MMA the slowest mode; v (register order) inner.
  for rk in 0 ..< restK:
    for rm in 0 ..< restM:
      for v in 0 ..< vCount:
        let off = crd2idx(mma.atom.aLayout, localT + tCount * v).toIntVal()
        let lm = off mod mAtom
        let lk = off div mAtom
        result.add FragmentElement(
          atomIdx: [am, 0, ak],
          row: rm * tileM + am * mAtom + lm,
          col: rk * tileK + ak * kAtom + lk,
          valIdx: v,
          offset: off)

func partitionB*[A, TL](mma: TiledMma[A, TL];
                        tileShape: (int, int);  # (N, K) of the tile
                        thread: int): Fragment =
  ## Per-thread B fragment: (n, k) coords, register order. See partitionA.
  let
    nAtoms = mma.threadLayout.shape[1].toIntVal()  # ThrN
    kAtoms = mma.threadLayout.shape[2].toIntVal()  # ThrK
    nAtom  = mma.atom.mnk.n
    kAtom  = mma.atom.mnk.k
    tCount = mma.atom.threadCount(opB).toIntVal()
    vCount = mma.atom.fragmentValsPerThread(opB).toIntVal()
  let
    tileN = nAtoms * nAtom
    tileK = kAtoms * kAtom
  doAssert tileShape[0] mod tileN == 0 and tileShape[1] mod tileK == 0,
    "tile shape must be a multiple of the tiled-mma unit"
  let
    restN = tileShape[0] div tileN
    restK = tileShape[1] div tileK
    atomIdx = thread div tCount
    tm = atomIdx mod mma.threadLayout.shape[0].toIntVal()
    tn = (atomIdx div mma.threadLayout.shape[0].toIntVal()) mod nAtoms
    tk = atomIdx div (mma.threadLayout.shape[0].toIntVal() * nAtoms)
    an = tn
    ak = tk
    localT = thread mod tCount
  for rk in 0 ..< restK:
    for rn in 0 ..< restN:
      for v in 0 ..< vCount:
        let off = crd2idx(mma.atom.bLayout, localT + tCount * v).toIntVal()
        let ln = off mod nAtom
        let lk = off div nAtom
        result.add FragmentElement(
          atomIdx: [0, an, ak],
          row: rn * tileN + an * nAtom + ln,
          col: rk * tileK + ak * kAtom + lk,
          valIdx: v,
          offset: off)

func partitionC*[A, TL](mma: TiledMma[A, TL];
                        tileShape: (int, int);  # (M, N) of the tile
                        thread: int): Fragment =
  ## Per-thread C fragment: (m, n) coords, register order. See partitionA.
  let
    mAtoms = mma.threadLayout.shape[0].toIntVal()  # ThrM
    nAtoms = mma.threadLayout.shape[1].toIntVal()  # ThrN
    mAtom  = mma.atom.mnk.m
    nAtom  = mma.atom.mnk.n
    tCount = mma.atom.threadCount(opC).toIntVal()
    vCount = mma.atom.fragmentValsPerThread(opC).toIntVal()
  let
    tileM = mAtoms * mAtom
    tileN = nAtoms * nAtom
  doAssert tileShape[0] mod tileM == 0 and tileShape[1] mod tileN == 0,
    "tile shape must be a multiple of the tiled-mma unit"
  let
    restM = tileShape[0] div tileM
    restN = tileShape[1] div tileN
    atomIdx = thread div tCount
    am = atomIdx mod mAtoms
    an = (atomIdx div mAtoms) mod nAtoms
    localT = thread mod tCount
  for rn in 0 ..< restN:
    for rm in 0 ..< restM:
      for v in 0 ..< vCount:
        let off = crd2idx(mma.atom.cLayout, localT + tCount * v).toIntVal()
        let lm = off mod mAtom
        let ln = off div mAtom
        result.add FragmentElement(
          atomIdx: [am, an, 0],
          row: rm * tileM + am * mAtom + lm,
          col: rn * tileN + an * nAtom + ln,
          valIdx: v,
          offset: off)
