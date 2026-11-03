# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Mma atom layer: selection, fragment layouts, thread-value partitions
##
## Checks on the gemm fixture abstract mmaf:
## - the Apple 8x8x8 f32 simdgroup atom selects at 32 threads only
## - operand thread-value layouts cover their tiles exactly once with the hardware ownership tables
## - bit-linear fragment forms match the ceramic registry numbers
## - unselected shapes fall back to the abstract mma without refusals
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/b8
##     --outdir:build/b8 workspace/tattletile/tests/test_mma_atom.nim

import std/os
import std/algorithm
import std/tables

import workspace/tattletile/tattletile

const FixturesDir = currentSourcePath().parentDir().parentDir() / "fixtures"

proc gemmEntry(): Entry =
  parseTileIr(readFile(FixturesDir / "gemm.tileir")).getEntry("gemm")

proc checkSelection() =
  ## Gemm mmaf (16x16x16) selects the Apple atom at 32 threads only
  let entry = gemmEntry()
  let views = entry.resolveViews()
  let layouts = entry.inferThreadValueLayouts(threads = 32)
  let sels = entry.selectMmaAtoms(layouts, threads = 32)
  doAssert "acc" in sels, "the gemm mmaf has an atom selection"
  let sel = sels["acc"]
  doAssert sel.selected, "selection is active"
  doAssert sel.atom.kind == akApple8x8x8F32, "the Apple atom"
  doAssert (sel.atom.m, sel.atom.n, sel.atom.k) == (8, 8, 8), "atom cell"
  doAssert sel.atom.threadCount == 32, "one simdgroup per atom"
  doAssert (sel.repsM, sel.repsN, sel.repsK) == (2, 2, 2), "2x2x2 repetitions"

  # no selection is the fallback signal, never a refusal
  let sels4 = entry.selectMmaAtoms(layouts, threads = 4)
  doAssert sels4.len == 0, "4 threads select no atom, the FMA fallback stays"
  doAssert atomFor(eI32, 16, 16, 16, 32).selected == false,
    "i32 values select no atom"
  doAssert atomFor(eF32, 12, 12, 12, 32).selected == false,
    "tile sizes off the atom cell select no atom"
  doAssert atomFor(eF32, 8, 8, 8, 32).selected == true,
    "a single atom cell selects"
  let one = atomFor(eF32, 8, 8, 8, 32)
  doAssert (one.repsM, one.repsN, one.repsK) == (1, 1, 1), "single-cell reps"

proc checkPipelineWiring() =
  ## runTilePipeline attaches the selection to the program
  doAssert gemmEntry().runTilePipeline(threads = 32).atomMmas.hasKey("acc"),
    "the program carries the atom selection"
  doAssert gemmEntry().runTilePipeline(threads = 4).atomMmas.len == 0,
    "the 4-thread program falls back to the abstract mma"

proc allTileCoords(n: int): seq[int] =
  ## 0 ..< n, the exact-once coverage target
  for i in 0 ..< n:
    result.add i

proc checkTvPartition() =
  ## Each operand tv layout covers its 16x16 tile exactly once, thread 0 owning the hardware table the lane-map POC verified
  let entry = gemmEntry()
  let layouts = entry.inferThreadValueLayouts(threads = 32)
  let sel = entry.selectMmaAtoms(layouts, threads = 32)["acc"]
  # column-major tile flats (see mma_atom.operandTvLayout):
  # - A (flat = m + 16*k) and C (flat = m + 16*n): thread 0 owns
  #   m in {0, 8} x (k or n) in {0, 1, 8, 9}, one 8x8 cell per repetition
  # - B (flat = n + 16*k): thread 0 owns n in {0, 1, 8, 9} x k in {0, 8}
  const Thread0Ac = @[0, 8, 16, 24, 128, 136, 144, 152]
  const Thread0B = @[0, 1, 8, 9, 128, 129, 136, 137]
  for i, tv in [sel.aTv, sel.bTv, sel.cTv]:
    let Thread0Owned = if i == 1: Thread0B else: Thread0Ac
    var covered: seq[int] = @[]
    for t in 0 ..< 32:
      for v in 0 ..< 2:
        for r0 in 0 ..< 2:
          for r1 in 0 ..< 2:
            covered.add tv.callInt(it(it(t, v), it(r0, r1)))
    covered.sort()
    doAssert covered == allTileCoords(256),
      "the tv partition must cover the tile exactly once"
    var row0: seq[int] = @[]
    for v in 0 ..< 2:
      for r0 in 0 ..< 2:
        for r1 in 0 ..< 2:
          row0.add tv.callInt(it(it(0, v), it(r0, r1)))
    row0.sort()
    doAssert row0 == Thread0Owned,
      "thread 0 owns one 8x8 cell per repetition, got " & $row0

proc checkBitLinearForms() =
  ## Per-thread fragment address forms match the ceramic registry fragment layout numbers the lane-map POC verified on hardware
  let entry = gemmEntry()
  let layouts = entry.inferThreadValueLayouts(threads = 32)
  let sel = entry.selectMmaAtoms(layouts, threads = 32)["acc"]
  doAssert sel.aBits.s == [16, 1, 2, 32, 4] and sel.aBits.sV == 8,
    "A fragment: v steps along K, thread bits spread m and k"
  doAssert sel.bBits.s == [2, 8, 16, 4, 32] and sel.bBits.sV == 1,
    "B fragment: v steps along N"
  doAssert sel.cBits.s == [16, 1, 2, 32, 4] and sel.cBits.sV == 8,
    "C fragment: v steps along N"
  # fragment maps: A and C cell first coordinates are the low offset bits and index view dim 0,
  # B cell first coordinates are the high bits (n) indexing view dim 1
  doAssert sel.aMap.rowIsLow and sel.aMap.rowViewDim == 0 and
    sel.aMap.colViewDim == 1, "A map: (m, k) -> view dims (0, 1)"
  doAssert not sel.bMap.rowIsLow and sel.bMap.rowViewDim == 0 and
    sel.bMap.colViewDim == 1, "B map: (k, n) -> view dims (0, 1)"
  doAssert sel.cMap.rowIsLow and sel.cMap.rowViewDim == 0 and
    sel.cMap.colViewDim == 1, "C map: (m, n) -> view dims (0, 1)"
  # semantic coordinates of one fragment, thread 0 value 0 of the A operand:
  # cell offset 0 -> (m, k) = (0, 0) of the repetition (0, 0)
  doAssert sel.aTv.callInt(it(it(0, 0), it(0, 0))) == 0, "A fragment origin"

when isMainModule:
  checkSelection()
  checkPipelineWiring()
  checkTvPartition()
  checkBitLinearForms()
  echo "[OK] mma atom layer: selection, tv partitions, fragment forms"
