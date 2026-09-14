# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
##
## denseIdsOfSorted result-neutrality. The index-permutation sort
## (no key-copying tuple sequence) must produce exactly the dense ids
## the tuple-sort semantics produced, byte for byte, including duplicate ranks.
##
## - reference recomputed independently in this suite by the old
##   tuple-sort construction, materializing then rank-ascending sorting
##   the (key, rank) tuple sequence with the byte-lexicographic tiebreak,
##   then assigning each byte-sorted input position the first
##   dense slot of its rank plus the occurrence counter of that rank.

import std/[monotimes, times]
import std/[algorithm, tables]

import workspace/toktoktok/src/merge

var checkFailures = 0

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    checkFailures += 1

proc lcg(state: var uint64): uint64 =
  state = state * 6364136223846793005'u64 + 1442695040888963407'u64
  state

proc lcgHigh(state: var uint64): uint64 =
  ## High 32 bits of the LCG step, the low bits have short periods,
  ## a read pattern whose call count matches the period would emit
  ## the same value forever (reproducer-verified trap).
  lcg(state) shr 32

proc cmpBytesRef(a, b: openArray[byte]): int =
  ## Byte-wise lexicographic compare (unsigned), then by length.
  let n = min(a.len, b.len)
  for i in 0 ..< n:
    if a[i] != b[i]:
      return int(a[i]) - int(b[i])
  a.len - b.len

proc referenceDenseIds(keys: openArray[seq[byte]],
    keyRanks: openArray[int]): seq[int32] =
  ## Old tuple-sort semantics, verbatim:
  ## - the dense order is
  ##   the materialized (key, rank) sequence sorted rank-ascending
  ##   with the byte-lexicographic tiebreak, each byte-sorted input
  ##   position gets its rank's first dense slot plus the occurrence
  ##   counter of that rank so far.
  let n = keys.len
  var ranked = newSeq[(seq[byte], int)](n)
  for i in 0 ..< n:
    ranked[i] = (keys[i], keyRanks[i])
  ranked.sort(proc(a, b: (seq[byte], int)): int =
    if a[1] != b[1]:
      return cmp(a[1], b[1])
    cmpBytesRef(a[0], b[0]))
  result = newSeq[int32](n)
  var firstDense = initTable[int, int32](n)
  for d in 0 ..< n:
    if ranked[d][1] notin firstDense:
      firstDense[ranked[d][1]] = int32(d)
  var seen = initTable[int, int32](n)
  for i in 0 ..< n:
    let r = keyRanks[i]
    let j = seen.getOrDefault(r, int32(0))
    seen[r] = j + 1
    result[i] = firstDense[r] + j

proc runTests*() =
  ## Entry point, runs this file's checks in order.
  #
  # duplicate-rank tables:
  #   heavy cross-content rank ties, where
  #   a wrong occurrence counting or tie order shifts dense ids
  #
  block:
    const Rounds = 2000
    var state: uint64 = 0xBB3E68C1EEF02A11'u64
    var fails = 0
    for round in 0 ..< Rounds:
      var keys: seq[seq[byte]] = @[]
      var keyRanks: seq[int] = @[]
      # byte-sorted distinct keys over a 3-letter alphabet with ranks
      # drawn from a tiny range (heavy duplicate ranks), sorted by bytes
      var genRounds = 0
      while keys.len < 40:
        inc genRounds
        doAssert genRounds <= 10000, "key generator overran its bound"
        let n = 1 + int(lcgHigh(state) mod 5)
        var key: seq[byte] = @[]
        for i in 0 ..< n:
          key.add byte(97 + int(lcgHigh(state) mod 3))
        # keys must stay distinct, ranks may collide freely
        var dup = false
        for existing in keys.items:
          if existing == key:
            dup = true
            break
        if not dup:
          keys.add key
          keyRanks.add int(state mod 8)
      # byte-sort the keys (the caller contract is byte-sorted input)
      var order = newSeq[int](keys.len)
      for i in 0 ..< order.len:
        order[i] = i
      order.sort(proc(a, b: int): int = cmpBytesRef(keys[a], keys[b]))
      var sortedKeys: seq[seq[byte]] = @[]
      var sortedRanks: seq[int] = @[]
      for i in order.items:
        sortedKeys.add keys[i]
        sortedRanks.add keyRanks[i]
      let want = referenceDenseIds(sortedKeys, sortedRanks)
      echo "round ", round, " reference done"; flushFile(stdout)
      let got = denseIdsOfSorted(sortedKeys, sortedRanks)
      echo "round ", round, " candidate done"; flushFile(stdout)
      if got != want:
        inc fails
        if fails <= 3:
          echo "NEUTRALITY FAIL round ", round, ": want ", want, " got ", got
    check "result-neutral vs tuple-sort semantics (2000 duplicate-rank tables)",
      fails == 0, $fails & " fails"

  #
  # structured rows:
  #   every dense id is a permutation of 0..<n, rank
  # blocks are contiguous, and inside one rank the byte order holds
  #
  block:
    var structFails = 0
    for trial in 0 ..< 200:
      var state = uint64(0x51633E2D9F5C7F21'u64 + uint64(trial))
      var keys: seq[seq[byte]] = @[]
      var keyRanks: seq[int] = @[]
      var seen: seq[seq[byte]] = @[]
      while keys.len < 30:
        let n = 1 + int(lcgHigh(state) mod 4)
        var key: seq[byte] = @[]
        for i in 0 ..< n:
          key.add byte(97 + int(lcgHigh(state) mod 4))
        if key notin seen:
          seen.add key
          keys.add key
          keyRanks.add int(state mod 4)
      var order = newSeq[int](keys.len)
      for i in 0 ..< order.len:
        order[i] = i
      order.sort(proc(a, b: int): int = cmpBytesRef(keys[a], keys[b]))
      var sortedKeys: seq[seq[byte]] = @[]
      var sortedRanks: seq[int] = @[]
      for i in order.items:
        sortedKeys.add keys[i]
        sortedRanks.add keyRanks[i]
      let dense = denseIdsOfSorted(sortedKeys, sortedRanks)
      var perm: seq[bool]
      perm.setLen(sortedKeys.len)
      var okPerm = true
      for d in dense.items:
        if d < 0 or d >= sortedKeys.len or perm[d]:
          okPerm = false
        else:
          perm[d] = true
      if not okPerm:
        inc structFails
        continue
      # dense order == (rank, bytes) order over the sorted input
      for i in 0 ..< sortedKeys.len:
        for j in 0 ..< sortedKeys.len:
          let iFirst = dense[i] < dense[j]
          if sortedRanks[i] != sortedRanks[j]:
            if iFirst != (sortedRanks[i] < sortedRanks[j]):
              inc structFails
          else:
            if iFirst != (cmpBytesRef(sortedKeys[i], sortedKeys[j]) < 0):
              inc structFails
    check "dense order == (rank, byte-lex) order, permutation of 0..<n",
      structFails == 0, $structFails & " fails"

  #
  # determinism:
  #   identical inputs give identical outputs
  #
  block:
    var keys: seq[seq[byte]] = @[]
    for i in 0 ..< 64:
      var key: seq[byte] = @[]
      let n = 1 + i mod 4
      for j in 0 ..< n:
        key.add byte(97 + (i * 7 + j * 3) mod 5)
      var dup = false
      for existing in keys.items:
        if existing == key:
          dup = true
          break
      if not dup:
        keys.add key
    var order = newSeq[int](keys.len)
    for i in 0 ..< order.len:
      order[i] = i
    order.sort(proc(a, b: int): int = cmpBytesRef(keys[a], keys[b]))
    var sortedKeys: seq[seq[byte]] = @[]
    var sortedRanks: seq[int] = @[]
    for i in order.items:
      sortedKeys.add keys[i]
      sortedRanks.add int((i * 13) mod 6)   # duplicate ranks on purpose
    check "determinism: double call identical",
      denseIdsOfSorted(sortedKeys, sortedRanks) ==
        denseIdsOfSorted(sortedKeys, sortedRanks)

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall dense-ids neutrality checks passed"
when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
