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
## consumes (per-thread register addresses) and the form the verification
## layer checks (disjointness, coverage).
##
## Offsets are col-major in the operand tile: A (m + k·M), B (n + k·N),
## C (m + n·M). Flat (T,V) index for crd2idx: t + T·v (mode 0 fastest).

import std/strformat
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
    tCount = mma.atom.threadCount(opA)
    vCount = mma.atom.fragmentValsPerThread(opA)
  doAssert tileShape == (mAtoms * mAtom, kAtoms * kAtom), "tile shape vs tiling"
  let
    atomIdx = thread div tCount
    tm = atomIdx mod mAtoms
    tn = (atomIdx div mAtoms) mod mma.threadLayout.shape[1].toIntVal()
    tk = atomIdx div (mAtoms * mma.threadLayout.shape[1].toIntVal())
    am = tm
    ak = tk
    localT = thread mod tCount
  for v in 0 ..< vCount:
    let off = crd2idx(mma.atom.aLayout, localT + tCount * v).toIntVal()
    let lm = off mod mAtom
    let lk = off div mAtom
    result.add FragmentElement(
      atomIdx: [am, 0, ak],
      row: am * mAtom + lm,
      col: ak * kAtom + lk,
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
    tCount = mma.atom.threadCount(opB)
    vCount = mma.atom.fragmentValsPerThread(opB)
  doAssert tileShape == (nAtoms * nAtom, kAtoms * kAtom), "tile shape vs tiling"
  let
    atomIdx = thread div tCount
    tm = atomIdx mod mma.threadLayout.shape[0].toIntVal()
    tn = (atomIdx div mma.threadLayout.shape[0].toIntVal()) mod nAtoms
    tk = atomIdx div (mma.threadLayout.shape[0].toIntVal() * nAtoms)
    an = tn
    ak = tk
    localT = thread mod tCount
  for v in 0 ..< vCount:
    let off = crd2idx(mma.atom.bLayout, localT + tCount * v).toIntVal()
    let ln = off mod nAtom
    let lk = off div nAtom
    result.add FragmentElement(
      atomIdx: [0, an, ak],
      row: an * nAtom + ln,
      col: ak * kAtom + lk,
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
    tCount = mma.atom.threadCount(opC)
    vCount = mma.atom.fragmentValsPerThread(opC)
  doAssert tileShape == (mAtoms * mAtom, nAtoms * nAtom), "tile shape vs tiling"
  let
    atomIdx = thread div tCount
    am = atomIdx mod mAtoms
    an = (atomIdx div mAtoms) mod nAtoms
    localT = thread mod tCount
  for v in 0 ..< vCount:
    let off = crd2idx(mma.atom.cLayout, localT + tCount * v).toIntVal()
    let lm = off mod mAtom
    let ln = off div mAtom
    result.add FragmentElement(
      atomIdx: [am, an, 0],
      row: am * mAtom + lm,
      col: an * nAtom + ln,
      valIdx: v,
      offset: off)

# ═════════════════════════════════════════════════════════════════════════
#  Verification — disjointness + coverage
# ═════════════════════════════════════════════════════════════════════════

proc verifyFragments*[A, TL](mma: TiledMma[A, TL];
                             tileShape: (int, int);
                             operand: MmaOperand) =
  ## Every tile element appears in EXACTLY the expected number of threads'
  ## fragments, with no duplicates within one thread group:
  ##   A: ThrN copies (A is shared across N-atoms)
  ##   B: ThrM copies (B is shared across M-atoms)
  ##   C: 1 copy (C atoms tile (ThrM, ThrN) — all distinct)
  ## Checks: per-thread count == vCount; per-element multiplicity == expected.
  let
    tCount = mma.atom.threadCount(operand)
    vCount = mma.atom.fragmentValsPerThread(operand)
    tileSize = tileShape[0] * tileShape[1]
    thrM = mma.threadLayout.shape[0].toIntVal()
    thrN = mma.threadLayout.shape[1].toIntVal()
    expected = case operand
               of opA: thrN
               of opB: thrM
               of opC: 1
  var counts = newSeq[int](tileSize)
  for thread in 0 ..< tCount * thrM * thrN:
    let frag = case operand
               of opA: mma.partitionA(tileShape, thread)
               of opB: mma.partitionB(tileShape, thread)
               of opC: mma.partitionC(tileShape, thread)
    doAssert frag.len == vCount, &"per-thread fragment count (thread {thread})"
    for el in frag:
      doAssert el.row >= 0 and el.row < tileShape[0]
      doAssert el.col >= 0 and el.col < tileShape[1]
      counts[el.row + el.col * tileShape[0]].inc
  for i, c in counts:
    doAssert c == expected, &"element {i}: multiplicity {c}, expected {expected}"
