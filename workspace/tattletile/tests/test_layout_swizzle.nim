# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Swizzle concretization: F2 arithmetic, the Swizzle functor, swizzled layouts
##
## Test that the F2 arithmetic, the Swizzle functor and the swizzled layouts match
## the vendored pycute reference values, the functor form an involution and every
## aligned 32-element segment of the canonical 128B swizzled tile hitting all 32
## banks exactly once.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/b8
##     --outdir:build/b8 workspace/tattletile/tests/test_layout_swizzle.nim

import std/tables

import workspace/tattletile/tattletile

proc checkF2Arithmetic() =
  ## F2 examples from the pycute swizzle.py F2 docstring
  doAssert f2Add(0b1010, 0b1100) == 0b0110, "F2 addition is XOR"
  doAssert f2Mul(3, 0b1010) == 0b11110, "an integer operand acts through its bits"
  doAssert f2Mul(0b11, 0b11) == 0b101, "carry-less, where 3 * 3 == 9 in Z"
  doAssert f2Add(0b1, 0) == 0b1, "0 is still the identity"
  doAssert f2Mul(0b1010, 0b1100) == 0b1111000, "carry-less 10 * 12"
  doAssert f2Mul(9, 3) == 0b11011, "carry-less 9 * 3"

proc checkSwizzleFunctor() =
  ## Swizzle examples from the pycute swizzle.py Swizzle docstring
  let sw = initSwizzle(1, 0, 1)
  doAssert sw.apply(0b00) == 0b00, "identity on the zero offset"
  doAssert sw.apply(0b10) == 0b11, "the base bit XORs the shifted bit"
  doAssert sw.apply(sw.apply(0b10)) == 0b10, "an involution on non-overlapping fields"
  doAssert $initSwizzle(3, 3, 3) == "SW3_3_3", "print form"
  doAssert sw.apply(0b11) == 0b10, "the XOR runs both ways"
  try:
    discard initSwizzle(-1, 0, 1)
    doAssert false, "negative bits must refuse"
  except LayoutError:
    discard
  try:
    discard initSwizzle(2, 0, -1)
    doAssert false, "a negative shift with abs(shift) < bits must refuse"
  except LayoutError:
    discard
  try:
    discard initSwizzle(1, 0, 0)
    doAssert false, "a zero shift with abs(shift) < bits must refuse"
  except LayoutError:
    discard
  try:
    discard initSwizzle(2, 0, 1)
    doAssert false, "overlapping fields must refuse"
  except LayoutError:
    discard

proc checkSwizzleReferenceTables() =
  ## Swizzle offset tables lifted from the pycute test suite, exact values
  let sw202 = initSwizzle(2, 0, 2)
  var got202: seq[int] = @[]
  for i in 0 ..< 16:
    got202.add sw202.apply(i)
  doAssert got202 == @[0, 1, 2, 3, 5, 4, 7, 6, 10, 11, 8, 9, 15, 14, 13, 12],
    "SW2_0_2 offsets " & $got202
  let sw101 = initSwizzle(1, 0, 1)
  var got101: seq[int] = @[]
  for i in 0 ..< 8:
    got101.add sw101.apply(i)
  doAssert got101 == @[0, 1, 3, 2, 4, 5, 7, 6], "SW1_0_1 offsets " & $got101
  # SW2_3_3 XORs bit fields above bit 6 into bits 4..3, offsets below 64 keep the YYY mask zero
  let sw233 = initSwizzle(2, 3, 3)
  for i in 0 ..< 64:
    doAssert sw233.apply(i) == i, "SW2_3_3 passes " & $i & " through"

proc checkSwizzledAxisCodomain() =
  ## A swizzle applies to integer offsets only: a layout whose codomain runs over
  ## coordinate axes refuses the call
  let axes = makeLayout(it(4, 8), itNode(@[itLeaf(eUnit(@[0])), itLeaf(eUnit(@[1]))]))
  let swizzled = swizzleLayout(axes, initSwizzle(2, 0, 2))
  try:
    discard swizzled.call(it(1, 1))
    doAssert false, "a swizzle over a coordinate-axis codomain must refuse"
  except LayoutError:
    discard

proc checkF2Coshape() =
  ## F2 codomain bounds, the pycute test_layout.py coshape table
  doAssert f2SpanBits(@[8, 8], @[1, 9]) == 64, "(8, 8):(F2(1), F2(9))"
  doAssert f2SpanBits(@[8, 8], @[9, 1]) == 64, "(8, 8):(F2(9), F2(1))"
  doAssert f2SpanBits(@[4, 8], @[1, 4]) == 32, "(4, 8):(F2(1), F2(4))"
  doAssert f2SpanBits(@[4, 4], @[1, 1]) == 4, "(4, 4):(F2(1), F2(1)), non-injective"
  doAssert f2SpanBits(@[8], @[0]) == 1, "broadcast"
  doAssert f2SpanBits(@[4], @[3]) == 8, "the image reaches 6, the bound spans it"
  # defining property: the bound covers every offset the layout produces
  for caseIdx in 0 ..< 3:
    let (lengths, strides) = case caseIdx
      of 0: (@[8, 8], @[1, 9])
      of 1: (@[4, 8], @[1, 4])
      else: (@[4], @[3])
    let bound = f2SpanBits(lengths, strides)
    var maxOff = 0
    if lengths.len == 2:
      for a in 0 ..< lengths[0]:
        for b in 0 ..< lengths[1]:
          maxOff = max(maxOff, f2Add(f2Mul(strides[0], a), f2Mul(strides[1], b)))
    else:
      for a in 0 ..< lengths[0]:
        maxOff = max(maxOff, f2Mul(strides[0], a))
    doAssert bound > maxOff, "the F2 bound must exceed every offset"

proc f2LayoutCall(lengths, strides: seq[int], a, b: int): int =
  ## One evaluation of an F2-strided 2-d layout: XOR of the carry-less products
  f2Add(f2Mul(strides[0], a), f2Mul(strides[1], b))

proc checkFunctorF2Equivalence() =
  ## Functor form and F2-strided form agree elementwise on the canonical 8x8 smem swizzle,
  ## two representations of one permutation
  let swizzled = swizzleLayout(makeLayout(it(8, 8), it(1, 8)), initSwizzle(3, 0, 3))
  for m in 0 ..< 8:
    for k in 0 ..< 8:
      doAssert swizzled.call(it(m, k)) == f2LayoutCall(@[8, 8], @[1, 9], m, k),
        "SW3_0_3 over the col-major tile equals the (F2(1), F2(9)) layout " &
        "at (" & $m & ", " & $k & ")"
  # readme_figures pair: F2(9)/F2(1) strides XOR the m/k offset, the transposed functor form matches the transposed F2 layout
  let swizzledT = swizzleLayout(makeLayout(it(8, 8), it(8, 1)), initSwizzle(3, 0, 3))
  for m in 0 ..< 8:
    for k in 0 ..< 8:
      doAssert swizzledT.call(it(m, k)) == f2LayoutCall(@[8, 8], @[9, 1], m, k),
        "SW3_0_3 over the row-major tile equals the (F2(9), F2(1)) layout"

proc checkBankConflictFree() =
  ## Canonical 128B swizzled smem tile: 8 x 64 f32 elements, SW3_3_3 XORs the first-dimension
  ## bits into the second-dimension bits 3..5. Property: every aligned 32-element segment hits
  ## all 32 banks exactly once, the conflict-free warp access pattern.
  let tile = swizzleLayout(makeLayout(it(8, 64), it(64, 1)), initSwizzle(3, 3, 3))
  doAssert tile.size == 8 * 64, "the tile size is preserved"
  for row in 0 ..< 8:
    for seg in 0 ..< 2:
      var banks = initCountTable[int]()
      for col in seg * 32 ..< (seg + 1) * 32:
        banks.inc tile.call(it(row, col)) mod 32
      for bank in 0 ..< 32:
        doAssert banks[bank] == 1,
          "row " & $row & " segment " & $seg & " bank " & $bank & " hit " &
          $banks[bank] & " times, conflict-free means exactly once"
  # permutation check: a bijection over the tile offsets the layout reaches
  var seen = initCountTable[int]()
  for row in 0 ..< 8:
    for col in 0 ..< 64:
      seen.inc tile.call(it(row, col))
  for row in 0 ..< 8:
    for col in 0 ..< 64:
      doAssert seen[tile.call(it(row, col))] == 1, "no two coordinates collide"

proc checkSwizzledAlgebraLayouts() =
  ## Any layout the algebra builds swizzles compositionally: a zipped-divided staging tile
  ## swizzles to a bijective offset permutation, size preserved
  let staging = zippedDivide(makeLayout(it(64, 64), it(64, 1)), it(8, 8))
  let swizzled = swizzleLayout(staging, initSwizzle(3, 3, 3))
  doAssert swizzled.size == staging.size, "the swizzle keeps the domain size"
  doAssert $swizzled == "SW3_3_3 o " & $staging, "print form shows the composition"
  var seen = initCountTable[int]()
  for i in 0 ..< staging.size:
    let off = swizzled.callInt(i)
    doAssert off >= 0, "the swizzled offset stays non-negative"
    seen.inc off
  for i in 0 ..< staging.size:
    doAssert seen[swizzled.callInt(i)] == 1, "the permutation stays injective"

when isMainModule:
  checkF2Arithmetic()
  checkSwizzleFunctor()
  checkSwizzleReferenceTables()
  checkSwizzledAxisCodomain()
  checkF2Coshape()
  checkFunctorF2Equivalence()
  checkBankConflictFree()
  checkSwizzledAlgebraLayouts()
  echo "[OK] swizzle concretization: F2, functor, coshape, equivalence, banks"
