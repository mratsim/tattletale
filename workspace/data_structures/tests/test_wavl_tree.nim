# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be published, modified, or distributed except according to those terms.

import std/random
import ../data_structures

proc wavlAssertValid(links: openArray[WavlLink]; root: int32; ctx: string) =
  ## Assert WAVL invariants
  wavlVerifyInvariants(links, root, ctx)

# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------

proc testBasic() =
  var data = @[30, 10, 50, 20, 40]
  var links = newSeq[WavlLink](5)
  var root: int32 = WavlNil

  let cmp = proc(a, b: int32): int =
    if data[a] < data[b]: -1 elif data[a] > data[b]: 1 else: 0

  # Insert
  for i in 0..4:
    wavlInsert(links, root, int32(i), cmp)
  wavlAssertValid(links, root, "after insert")

  # Find
  let idx20 = wavlFind(links, root, proc(idx: int32): int =
    if 20 < data[idx]: -1 elif 20 > data[idx]: 1 else: 0)
  doAssert idx20 >= 0
  doAssert data[idx20] == 20

  doAssert wavlFind(links, root, proc(idx: int32): int =
    if 99 < data[idx]: -1 elif 99 > data[idx]: 1 else: 0) == WavlNil

  # Min/Max
  doAssert data[wavlMin(links, root)] == 10
  doAssert data[wavlMax(links, root)] == 50

  # In-order traversal
  var ordered: seq[int]
  var curr = wavlMin(links, root)
  while curr >= 0:
    ordered.add data[curr]
    curr = wavlNext(links, curr)
  doAssert ordered == @[10, 20, 30, 40, 50], $ordered

  # Delete leaf (10 — index 1)
  wavlDelete(links, root, 1)
  wavlAssertValid(links, root, "after delete 10")

  # Delete middle (30 — index 0)
  wavlDelete(links, root, 0)
  wavlAssertValid(links, root, "after delete 30")

  # Delete max (50 — index 2)
  wavlDelete(links, root, 2)
  wavlAssertValid(links, root, "after delete 50")

  ordered.setLen(0)
  curr = wavlMin(links, root)
  while curr >= 0:
    ordered.add data[curr]
    curr = wavlNext(links, curr)
  doAssert ordered == @[20, 40], $ordered

  echo "  [PASS] testBasic"

# ---------------------------------------------------------------------------
# Stress test: monotonic keys
# ---------------------------------------------------------------------------

proc testMonotonic(N: int) =
  var data = newSeq[int](N)
  var links = newSeq[WavlLink](N)
  var root: int32 = WavlNil

  for i in 0..<N:
    data[i] = i

  let cmp = proc(a, b: int32): int =
    if data[a] < data[b]: -1 elif data[a] > data[b]: 1 else: 0

  for i in 0..<N:
    wavlInsert(links, root, int32(i), cmp)
  wavlAssertValid(links, root, "after monotonic insert")

  # Check in-order
  var ordered: seq[int]
  var curr = wavlMin(links, root)
  while curr >= 0:
    ordered.add data[curr]
    curr = wavlNext(links, curr)
  doAssert ordered.len == N
  for i in 0..<N:
    doAssert ordered[i] == i

  # Delete every other node (odd values)
  for i in countup(1, N-1, 2):
    let idx = wavlFind(links, root, proc(k: int32): int =
      if i < data[k]: -1 elif i > data[k]: 1 else: 0)
    doAssert idx >= 0, "should find " & $i
    wavlDelete(links, root, idx)
  wavlAssertValid(links, root, "after delete half")

  let remaining = N div 2
  ordered.setLen(0)
  curr = wavlMin(links, root)
  while curr >= 0:
    ordered.add data[curr]
    curr = wavlNext(links, curr)
  doAssert ordered.len == remaining
  for i in 0..<remaining:
    doAssert ordered[i] == i * 2

  echo "  [PASS] testMonotonic (", N, " nodes)"

# ---------------------------------------------------------------------------
# Stress test: random keys
# ---------------------------------------------------------------------------

proc testRandom(N: int) =
  var data = newSeq[int](N)
  var links = newSeq[WavlLink](N)
  var root: int32 = WavlNil

  for i in 0..<N:
    data[i] = rand(1_000_000)

  let cmp = proc(a, b: int32): int =
    if data[a] < data[b]: -1 elif data[a] > data[b]: 1 else: 0

  for i in 0..<N:
    wavlInsert(links, root, int32(i), cmp)
  wavlAssertValid(links, root, "after random insert")

  var ordered: seq[int]
  var curr = wavlMin(links, root)
  while curr >= 0:
    ordered.add data[curr]
    curr = wavlNext(links, curr)
  doAssert ordered.len == N
  for i in 1..<ordered.len:
    doAssert ordered[i-1] <= ordered[i]

  echo "  [PASS] testRandom (", N, " nodes)"

# ---------------------------------------------------------------------------
# removeChild dance test
# ---------------------------------------------------------------------------

proc testRemoveChild() =
  var data = @[50, 20, 80, 10, 30, 70, 90, 5, 25, 35]
  var links = newSeq[WavlLink](data.len)
  var root: int32 = WavlNil
  let cmp = proc(a, b: int32): int =
    if data[a] < data[b]: -1 elif data[a] > data[b]: 1 else: 0

  for i in 0 ..< data.len:
    wavlInsert(links, root, int32(i), cmp)
  wavlAssertValid(links, root, "after insert")

  # Remove child at index 2 (value 80) via the dance
  let delIdx: int32 = 2
  let lastIdx = int32(data.len - 1)

  wavlDelete(links, root, delIdx)
  data.del(delIdx)
  links.del(delIdx)
  fixLinksAfterDataDeletion(links, root, lastIdx, delIdx)
  wavlAssertValid(links, root, "after removeChild dance")

  doAssert data.len == 9

  # Verify remaining values
  var ordered: seq[int]
  var curr = wavlMin(links, root)
  while curr >= 0:
    ordered.add data[curr]
    curr = wavlNext(links, curr)
  doAssert ordered == @[5, 10, 20, 25, 30, 35, 50, 70, 90], $ordered

  echo "  [PASS] testRemoveChild"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

proc runTests*() =
  echo "WAVL Tree Tests"
  testBasic()
  testMonotonic(100)
  testMonotonic(1000)
  testRandom(100)
  testRandom(1000)
  testRemoveChild()
  echo "All tests passed."

when isMainModule:
  runTests()
