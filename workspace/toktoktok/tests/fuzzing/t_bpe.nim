# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
##
## Named engine-contract rows for the served lazy-DP backtracking encoder
## (src/merge.nim encodeSegment) and its bt validity predicate:
##
## | class                | description                                                         |
## | -------------------- | ------------------------------------------------------------------- |
## | direct-merge invalid | both bytes of any 2-byte token merge, the pair must be rejected     |
## | no-merge valid       | two single-byte tokens whose byte concat is no token stay valid     |
## | frozen-invalid       | a valid split whose recorded split differs rejects as frozen        |
## | zero-alloc receipt   | a warm bt over a CJK-heavy input moves no memory across 2000 passes |
##
## The id values of the served encoders stay locked to ground truth through
## the recorded-fixture suites, and the per-piece semantics carry named
## rows in tests/unit/t_unit_bpe.nim.

{.experimental: "views".}
import std/[os, monotimes, times]

import workspace/toktoktok/src/deserializers
import workspace/toktoktok/src/merge
import workspace/toktoktok/src/scan

const
  TestsDir = currentSourcePath().parentDir().parentDir()
    ## tests/ root (the suite lives one level down)
  TokenizersDir = TestsDir / "tokenizers"

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    raise newException(AssertionDefect,
      "FAIL " & name & (if detail.len > 0: " " & detail else: ""))

proc main() =
  if not fileExists(TokenizersDir / "kimik2.5.tiktoken"):
    echo "MISSING tokenizer data: ", TokenizersDir
    echo "stage it with: python3 ",
      TestsDir /
      "fetch_test_tokenizers.py"
    quit(1)

  block:
    # named validity-predicate rows over kimik2.5, direct-merge
    # invalid rows, no-merge valid, frozen-invalid (valid split, unchosen)
    let tok = loadTiktokenCodec(TokenizersDir / "kimik2.5.tiktoken")
    let engine = BpeEngine.init(tok.ranks)
    engine.ensureBuilt()
    # direct-merge, any 2-byte token's two bytes merge, predicate false
    var directChecked = false
    for d in 0 ..< engine.mt.starts.len:
      if directChecked or int(engine.mt.lens[d]) != 2:
        continue
      let st = int(engine.mt.starts[d])
      let u = uint32(engine.trie.lookup(engine.mt.arena, st, st + 1))
      let v = uint32(engine.trie.lookup(engine.mt.arena, st + 1, st + 2))
      if u < 256 and v < 256:
        check "isValidTokenPair direct-merge bytes invalid",
          not isValidTokenPair(engine, u, v)
        directChecked = true
    # no-merge, two single-byte tokens whose byte concat is no token,
    # predicate true (dense ids 0..255 are the single-byte tokens, the concat key is read off the arena, not spelled from the ids)
    var noMergeChecked = false
    for u in 0 ..< 256:
      if noMergeChecked:
        break
      let bu = engine.mt.arena[int(engine.mt.starts[u])]
      for v in 0 ..< 256:
        let bv = engine.mt.arena[int(engine.mt.starts[v])]
        var key: array[2, byte] = [bu, bv]
        if engine.trie.lookup(key) < 0:
          check "isValidTokenPair no-merge byte pair valid",
            isValidTokenPair(engine, uint32(u), uint32(v)),
            "bytes (" & $bu & ", " & $bv & ") concat lookup " &
              $engine.trie.lookup(key) & " pairIndex " &
              $engine.mt.pairIndex.lookup(uint32(u), uint32(v)) &
              " split " & $engine.mt.splitLeft[u] & "," &
              $engine.mt.splitRight[u] & " / " & $engine.mt.splitLeft[v] &
              "," & $engine.mt.splitRight[v]
          noMergeChecked = true
          break
    # frozen-invalid:
    #   a valid split (u, v) of token d whose recorded
    # split differs -> the merge process would merge (u, v) -> false
    var frozenFound = false
    for d in 0 ..< engine.mt.starts.len:
      if frozenFound or int(engine.mt.lens[d]) < 2:
        continue
      if engine.mt.splitLeft[d] == int32(d):
        continue # self-split (unreachable-shape receipt case)
      let st = int(engine.mt.starts[d])
      let en = st + int(engine.mt.lens[d])
      for s in 1 ..< int(engine.mt.lens[d]):
        let a = engine.trie.lookup(engine.mt.arena, st, st + s)
        let b = engine.trie.lookup(engine.mt.arena, st + s, en)
        if a < 0 or b < 0:
          continue
        if a != int(engine.mt.splitLeft[d]) or b != int(engine.mt.splitRight[d]):
          check "isValidTokenPair frozen-invalid (valid unchosen split)",
            not isValidTokenPair(engine, uint32(a), uint32(b)),
            "token " & $d & " split (" & $a & ", " & $b & ") vs chosen (" &
              $engine.mt.splitLeft[d] & ", " & $engine.mt.splitRight[d] & ")"
          frozenFound = true
          break
    check "frozen-invalid pair exists in kimik2.5", frozenFound

  block:
    # zero-alloc receipt:
    #   warm bt steady state over a CJK-heavy input
    const CjkText = "中文字横書きカタカナ语料 1234567890 mixed 漢字 runs."
    let tok = loadTiktokenCodec(TokenizersDir / "cl100k_base.tiktoken")
    let engine = BpeEngine.init(tok.ranks)
    engine.ensureBuilt()
    var bt = BacktrackBuf.init()
    var cache = PreTokRegexCache()
    var r = initPreTokenizer(cache, famCl100k, "")
    var warm = 0
    block warmup:
      r.reset(cache, famCl100k, CjkText)
      var scratch: seq[int] = @[]
      let data = CjkText.toOpenArrayByte(0, CjkText.len - 1)
      for piece in r.items:
        scratch.setLen(0)
        encodeSegment(engine, scratch, bt, data, piece[0], piece[1])
        warm += scratch.len
    check "warmup drain emitted ids", warm > 0
    let before = getTotalMem()
    let data = CjkText.toOpenArrayByte(0, CjkText.len - 1)
    var scratch: seq[int] = @[]
    for i in 0 ..< 2000:
      r.reset(cache, famCl100k, CjkText)
      for piece in r.items:
        scratch.setLen(0)
        encodeSegment(engine, scratch, bt, data, piece[0], piece[1])
    let after = getTotalMem()
    check "zero-alloc receipt (2000 reset + drain passes, bt core)",
      after == before, "delta " & $(after - before)

  echo "\nall bt engine-contract rows passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  main()
  echo "\nwall ",
    (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
