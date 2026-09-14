# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
## Double-array vocab trie suite:
##
## equivalence vs the rank Table
## on every staged tiktoken table (r50k, p50k, cl100k, o200k, kimik2.5 with 163584 keys), fuzz keys and corpus-slice keys, byte-wise
## double-build determinism, lazy-build receipt + memory receipt.

import std/[os, strutils, algorithm, monotimes, times, tables, sequtils]

import workspace/zstd/zstd_highlevel
import workspace/toktoktok/src/deserializers
import workspace/toktoktok/src/scan
import workspace/toktoktok/src/merge

const
  TestsDir = currentSourcePath().parentDir().parentDir()
    ## tests/ root (the suite lives one level down)
  TokenizersDir = TestsDir / "tokenizers"
  CorpusDir = TestsDir / "corpus"

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

type
  SortedEntry = tuple[key: seq[byte], rank: int]

proc cmpEntry(a, b: SortedEntry): int =
  let n = min(a.key.len, b.key.len)
  for i in 0 ..< n:
    if a.key[i] != b.key[i]:
      return int(a.key[i]) - int(b.key[i])
  a.key.len - b.key.len

proc sortedEntries(encoder: Table[seq[byte], int]): seq[SortedEntry] =
  for k, v in encoder.pairs:
    result.add((k, v))
  result.sort(cmpEntry)

proc readCorpusPrefix(path: string, maxBytes: int): string =
  var text = readFile(path).zstdDecompress(string)
  if text.len > maxBytes:
    text = text[0 ..< maxBytes]
  text

proc main() =
  if not fileExists(TokenizersDir / "gpt2-tokenizer.json"):
    echo "MISSING tokenizer data: ", TokenizersDir
    echo "stage it with: python3 ",
      TestsDir /
      "fetch_test_tokenizers.py"
    quit(1)

  var families = [
    ("r50k", "r50k_base.tiktoken"),
    ("p50k", "p50k_base.tiktoken"),
    ("cl100k", "cl100k_base.tiktoken"),
    ("o200k", "o200k_base.tiktoken"),
    ("kimik2.5", "kimik2.5.tiktoken"),
  ]

  var heaviestMs = 0.0
  var heaviestLabel = ""
  var fails = 0

  for (label, file) in families.items:
    let codec = loadTiktokenCodec(TokenizersDir / file)
    let entries = sortedEntries(codec.ranks)
    doAssert entries.len == codec.ranks.len

    # determinism:
    #   two builds must produce identical arrays
    let memBefore = getTotalMem()
    let t0 = getMonoTime()
    let t1 = buildTrie(entries.mapIt(it.key), entries.mapIt(it.rank))
    let buildMs = (getMonoTime() - t0).inMicroseconds.float64 / 1000.0
    let tclock = getMonoTime()
    let trie2 = buildTrie(entries.mapIt(it.key), entries.mapIt(it.rank))
    let buildMs2 = (getMonoTime() - tclock).inMicroseconds.float64 / 1000.0
    let memAfter = getTotalMem()
    if t1.base != trie2.base or t1.chk != trie2.chk or t1.vals != trie2.vals:
      inc fails
      echo "DETERMINISM FAIL [", label, "]"
    check "double-build determinism [" & label & "]",
      t1.base == trie2.base and t1.chk == trie2.chk and t1.vals == trie2.vals

    # full-vocab equivalence:
    #   every key found with its exact rank,
    # prefixes of every key found only when themselves keys
    var eqFails = 0
    for (k, v) in entries.items:
      let got = t1.lookup(k)
      if got != v:
        inc eqFails
        if eqFails <= 3:
          echo "LOOKUP MISMATCH [", label, "] key bytes ", k, " want ", v,
            " got ", got
      # strict prefixes are keys only when they terminate
      if k.len > 1:
        var pfx: seq[byte] = k[0 ..< k.len - 1]
        let want = if pfx in codec.ranks: codec.ranks[pfx] else: RankNotFound
        if t1.lookup(pfx) != want:
          inc eqFails
    check "full-vocab equivalence vs Table [" & label & "]",
      eqFails == 0, $eqFails & " fails over " & $entries.len & " keys"

    # fuzz keys:
    #   absent random byte strings must miss, corpus slices
    # must agree with the Table both when present and absent
    var state: uint64 = 0x9E3779B97F4A7C15'u64
    var fuzzFails = 0
    for round in 0 ..< 2000:
      var k: seq[byte] = @[]
      let n = 1 + int(lcg(state) mod 12)
      for i in 0 ..< n:
        k.add byte(lcg(state) mod 256)
      let want = if k in codec.ranks: codec.ranks[k] else: RankNotFound
      if t1.lookup(k) != want:
        inc fuzzFails
    let sanguozhi = readCorpusPrefix(CorpusDir /
      "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt.zst", 40000)
    var corpusFails = 0
    for round in 0 ..< 2000:
      let pos = int(lcg(state) mod uint64(sanguozhi.len - 16))
      let n = 1 + int(lcg(state) mod 14)
      var k = cast[seq[byte]](sanguozhi[pos ..< pos + n])
      let want = if k in codec.ranks: codec.ranks[k] else: RankNotFound
      if t1.lookup(k) != want:
        inc corpusFails
    check "fuzz + corpus-slice lookup agreement [" & label & "]",
      fuzzFails == 0 and corpusFails == 0,
      $fuzzFails & " fuzz / " & $corpusFails & " corpus fails"

    # single-byte coverage:
    #   byte ids must equal the 1-byte ranks
    var byteFails = 0
    for b in 0 .. 255:
      let k = @[byte(b)]
      let want = if k in codec.ranks: codec.ranks[k] else: RankNotFound
      if t1.lookup(k) != want:
        inc byteFails
    check "single-byte rank coverage [" & label & "]", byteFails == 0

    if buildMs > heaviestMs:
      heaviestMs = buildMs
      heaviestLabel = label
    echo "receipt [", label, "]: ", entries.len, " keys, ",
      t1.nodeCount, " nodes, ", t1.base.len, " slots, ",
      t1.arrayBytes() div 1024, " KB arrays, build ",
      buildMs, " ms (rerun ", buildMs2, " ms), getTotalMem delta ",
      memAfter - memBefore, " bytes"

  check "trie builds green", fails == 0
  echo "heaviest trie build: ", heaviestLabel, " at ", heaviestMs, " ms"

  # empty-key discipline:
  #   an empty lookup always misses
  let demo = buildTrie([@[byte('a')]], @[7])
  check "empty key misses", demo.lookup("") == RankNotFound and
    demo.lookup(newSeq[byte]()) == RankNotFound

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall vocab trie checks passed"
when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  main()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
