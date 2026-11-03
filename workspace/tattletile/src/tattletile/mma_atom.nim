# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Mma atom layer: hardware mma instruction descriptors, fragment layouts, and the per-thread fragment program.
##
## An atom is one hardware instruction: an MNK cell computed per invocation, the operand datatypes, the fragment maps.
##
## Value-level and backend-agnostic:
## - fragment layouts are tattletile algebra Layout values, ported from ceramic's macro-time h_mma_registry
## - atomFor either selects an atom, deriving the operand thread-value layouts, or reports no selection
## - no selection is a fallback signal: the abstract per-thread FMA lowering stays in charge
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/b8 --outdir:build/b8
##     workspace/tattletile/tests/test_mma_atom.nim

import ./layout_algebra
import ./tile_ir

type
  MmaAtomKind* = enum
    ## Hardware atoms tattletile can select: one Metal atom to start, the one the hardware POC proved end to end.
    akApple8x8x8F32

  MmaAtom* = object
    ## One hardware mma instruction descriptor.
    ##
    ## Cell layouts map (thread, value-slot) pairs to flat offsets inside the atom 8x8 operand cells, col-major:
    ## low 3 offset bits index the first cell coordinate, high 3 the second.
    ## - A cell is (M, K): first coordinate m, second k
    ## - B cell is (N, K): first coordinate n, second k
    ## - C/D cell is (M, N): first coordinate m, second n
    kind*: MmaAtomKind
    m*, n*, k*: int
    threadCount*: int
    valuesPerThread*: int
    aCell*, bCell*, cCell*: Layout

  BitLinear* = object
    ## Cell offset of one fragment value over the thread bits:
    ## o(t, v) = base + sum_i bit_i(t) * s[i] + v * sV, bit_i(t) = (t / 2^i) mod 2, base = offset at (0, 0).
    s*: array[5, int]
    sV*: int

  RepAxis* = enum
    ## Repetition axes of an mmaf tile
    raM
    raN
    raK

  FragmentMap* = object
    ## Cell offset to view coordinates for one operand:
    ##
    ## - rowIsLow: the first cell coordinate is the low offset bits (o mod 8),
    ##   else the high bits (o div 8)
    ## - repRowAxis / repColAxis: repetition axis scaling the first resp.
    ##   second cell coordinate
    ## - rowViewDim / colViewDim: view dimension indexed by the first resp.
    ##   second cell coordinate
    rowIsLow*: bool
    repRowAxis*: RepAxis
    repColAxis*: RepAxis
    rowViewDim*: int
    colViewDim*: int

  MmaAtomSel* = object
    ## One atom selection for one abstract mmaf op, or no selection.
    ##
    ## Tv layouts are the full thread-value layouts of the three operands over the mmaf tile:
    ## ((T, V), reps) -> tile flat coordinate, computed through the layout algebra.
    ## - the tv layouts are the algebra-level record of the partition
    ## - the bit-linear forms and fragment maps are the emission view of the same partition
    selected*: bool
    atom*: MmaAtom
    repsM*, repsN*, repsK*: int
    aTv*, bTv*, cTv*: Layout
    aBits*, bBits*, cBits*: BitLinear
    aMap*, bMap*, cMap*: FragmentMap

# ═══════════════════════════════════════════════════════════════════════
#  The atom registry
# ═══════════════════════════════════════════════════════════════════════

proc apple8x8x8F32*(): MmaAtom =
  ## Apple simdgroup f32 atom: 8x8x8 cell, one simdgroup of 32 lanes, 2 values per thread per fragment, f32 accumulate.
  ##
  ## Fragment layout numbers ported from ceramic h_mma_registry (Apple8x8_AC_Layout / Apple8x8_B_Layout),
  ## verified on hardware by the lane-map POC:
  ## - A/C: (T32, V2) -> col-major cell offset (first coordinate + 8 * second), V steps along the column axis (K for A, N for C/D)
  ## - B: (T32, V2) -> col-major cell offset n + 8*k, V steps along N
  const acShape = it(it(2, 2, 2, 2, 2), itv(2))
  const acStride = it(it(16, 1, 2, 32, 4), itv(8))
  const bShape = it(it(2, 2, 2, 2, 2), itv(2))
  const bStride = it(it(2, 8, 16, 4, 32), itv(1))
  MmaAtom(
    kind: akApple8x8x8F32,
    m: 8, n: 8, k: 8,
    threadCount: 32,
    valuesPerThread: 2,
    aCell: makeLayout(acShape, acStride),
    bCell: makeLayout(bShape, bStride),
    cCell: makeLayout(acShape, acStride))

# ═══════════════════════════════════════════════════════════════════════
#  Fragment program math
# ═══════════════════════════════════════════════════════════════════════

proc bitLinearForm*(l: Layout, what: string): BitLinear =
  ## Bit-linear form of a fragment layout, extracted by evaluating the layout
  ## through the algebra and verified to reproduce all 32 x V layout values.
  ##
  ## Raises:
  ## - LayoutError on a fragment layout not bit-linear in the thread id, a fragment the per-thread emission cannot address
  let base = l.callInt(it(0, 0))
  result.sV = l.callInt(it(0, 1)) - base
  for v in 0 ..< 2:
    for i in 0 ..< 5:
      result.s[i] = l.callInt(it(1 shl i, v)) - l.callInt(it(0, v))
    for t in 0 ..< 32:
      var o = base + v * result.sV
      for i in 0 ..< 5:
        o += ((t div (1 shl i)) mod 2) * result.s[i]
      if o != l.callInt(it(t, v)):
        raise layoutError(what & ": fragment layout is not bit-linear in the " &
          "thread id at thread " & $t & " value " & $v)

proc operandTvLayout*(cell: Layout, cellOuter, cellLast, tileOuter, repOuter, repLast: int): Layout =
  ## Full thread-value layout of one mma operand over its tile, the algebra composition ceramic getMatmulThreadmap performs:
  ## - the atom cell layout composed with `(cellOuter, cellLast):(1, tileOuter)` flattens one atom cell into tile flats
  ## - the repetition modes attach with one stride per cell size
  ##
  ## Result shape ((T, V), (repOuter, repLast)), call form `l(it(it(t, v), it(r0, r1)))` -> tile flat coordinate.
  ## Tile flats are column-major: the first cell coordinate steps by 1, the second by tileOuter.
  ##
  ## Example, the C operand of a 16x16 tile over the Apple 8x8x8 atom: cellOuter = 8 (M), cellLast = 8 (N),
  ## tileOuter = 16, reps (2, 2), thread 0 owns tile flats {0, 8, 16, 24, 128, 136, 144, 152}, the (m, n) pairs
  ## {(0,0), (8,0), (0,1), (8,1), (0,8), (8,8), (0,9), (8,9)}.
  let atomFlat = compose(makeLayout(it(cellOuter, cellLast), it(1, tileOuter)),
                         cell)
  makeLayout(it(atomFlat.shape, it(repOuter, repLast)),
             it(atomFlat.stride, it(cellOuter, tileOuter * cellLast)))

proc fragmentMap*(rowIsLow: bool, repRow, repCol: RepAxis, rowViewDim, colViewDim: int): FragmentMap =
  FragmentMap(rowIsLow: rowIsLow, repRowAxis: repRow, repColAxis: repCol,
              rowViewDim: rowViewDim, colViewDim: colViewDim)

# ═══════════════════════════════════════════════════════════════════════
#  Selection
# ═══════════════════════════════════════════════════════════════════════

proc atomFor*(elem: ScalarElem, shapeM, shapeN, shapeK, threads: int): MmaAtomSel =
  ## Atom selection for one abstract mmaf with accumulator tile (M, N),
  ## A operand (M, K) and B operand (K, N), per-thread value element type
  ## `elem`, `threads` threads per threadgroup.
  ##
  ## No selection (selected = false) is the fallback signal, not an error:
  ## the abstract per-thread FMA lowering stays in charge. Atom applies if all hold:
  ##
  ## - the element type is f32
  ## - the thread count equals the atom thread group (32, one simdgroup)
  ## - M, N, K are positive multiples of the atom cell dims, so every fragment
  ##   predicate is statically true
  ##
  ## Operand thread-value layouts and fragment maps:
  ##
  ## - A: cell (M, K), reps (M/8, K/8), the m coordinate indexes view dim 0,
  ##   the k coordinate view dim 1
  ## - B: cell (N, K), reps (N/8, K/8), the n coordinate indexes view dim 1,
  ##   the k coordinate view dim 0
  ## - C/D: cell (M, N), reps (M/8, N/8), the m coordinate indexes view dim 0,
  ##   the n coordinate view dim 1
  if elem != eF32:
    return
  if threads != 32:
    return
  const atom = apple8x8x8F32()
  if shapeM <= 0 or shapeN <= 0 or shapeK <= 0:
    return
  if shapeM mod atom.m != 0 or shapeN mod atom.n != 0 or shapeK mod atom.k != 0:
    return
  result.selected = true
  result.atom = atom
  result.repsM = shapeM div atom.m
  result.repsN = shapeN div atom.n
  result.repsK = shapeK div atom.k
  result.aTv = operandTvLayout(atom.aCell, atom.m, atom.k, shapeM,
                               result.repsM, result.repsK)
  result.bTv = operandTvLayout(atom.bCell, atom.n, atom.k, shapeN,
                               result.repsN, result.repsK)
  result.cTv = operandTvLayout(atom.cCell, atom.m, atom.n, shapeM,
                               result.repsM, result.repsN)
  result.aBits = bitLinearForm(atom.aCell, "A fragment")
  result.bBits = bitLinearForm(atom.bCell, "B fragment")
  result.cBits = bitLinearForm(atom.cCell, "C fragment")
  result.aMap = fragmentMap(true, raM, raK, 0, 1)
  result.bMap = fragmentMap(false, raK, raN, 0, 1)
  result.cMap = fragmentMap(true, raM, raN, 0, 1)
