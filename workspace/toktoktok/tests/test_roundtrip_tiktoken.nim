# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
##
import std/unittest
import std/os
import std/tables

import workspace/toktoktok/src/deserializers
import workspace/toktoktok/src/pipeline
import pytoktoktok

let
  SourcePathAbs = absolutePath(currentSourcePath())
  TokenizersDir = SourcePathAbs.parentDir() /
    "tokenizers"

const PullCap = 4096

var loadedCache: Table[string, PipelineRef]

proc loadOnce(name, filename, pattern: string): PipelineRef =
  ## One pipeline per checkpoint for the whole suite, the checkpoint
  ## json parse and the vocab trie build dominate the suite wall,
  ## the pipeline is reused across rows.
  if name notin loadedCache:
    loadedCache[name] = load_tokenizer_tiktoken(TokenizersDir / filename, pattern)
  loadedCache[name]

proc pullAll(pipeline: TokPipeline, text: string): seq[int] =
  ## Whole-input encode as a bounded-consumption loop, the one-shot
  ## convenience lives in the tests, never in the binding, so every row
  ## exercises bounded machine consumption and the drain discipline.
  pipeline.resetText(text)
  var taken = 0
  while true:
    var batch = 0
    for id in pipeline.items:
      result.add id
      inc taken
      inc batch
      if batch == PullCap:
        break
    if batch < PullCap:
      break

proc runTiktokenizerTests() =
  suite "Tiktokenizer Tests":

    test "load tiktokenizer file not found":
      expect TokenizerLoadError:
        discard loadTiktokenCodec("nonexistent.tiktoken")

    const TiktokenPairs = [
      ("r50k_base", "r50k_base.tiktoken", "r50k"),
      ("p50k_base", "p50k_base.tiktoken", "p50k"),
      ("cl100k_base", "cl100k_base.tiktoken", "cl100k"),
      ("o200k_base", "o200k_base.tiktoken", "o200k"),
      ("kimik2.5", "kimik2.5.tiktoken", "kimik2.5"),
    ]

    for tokenizerPair in TiktokenPairs:
      let (name, filename, pattern) = tokenizerPair
      let path = TokenizersDir / filename

      test "load and decode (" & name & ")":
        doAssert fileExists(path), name & " tiktokenizer not found: " & path
        let loaded = loadOnce(name, filename, pattern)
        let encoded = pullAll(loaded.pipe, "Hello, world!")
        check encoded.len > 0

        let decoded = decodeToString(loaded.codec, encoded)
        check decoded.len >= 5 and decoded[0 .. 4] == "Hello"

      test "byte encoding roundtrip (" & name & ")":
        let loaded = loadOnce(name, filename, pattern)
        let text = "Hello, world!"
        let encoded = pullAll(loaded.pipe, text)
        let decodedStr = decodeToString(loaded.codec, encoded)
        check decodedStr == text

      test "CJK roundtrip - Chinese (" & name & ")":
        let loaded = loadOnce(name, filename, pattern)
        let original = "你好世界"
        let encoded = pullAll(loaded.pipe, original)
        let decoded = decodeToString(loaded.codec, encoded)
        check decoded == original

      test "CJK roundtrip - Japanese (" & name & ")":
        let loaded = loadOnce(name, filename, pattern)
        let original = "こんにちは"
        let encoded = pullAll(loaded.pipe, original)
        let decoded = decodeToString(loaded.codec, encoded)
        check decoded == original

      test "CJK roundtrip - Korean (" & name & ")":
        let loaded = loadOnce(name, filename, pattern)
        let original = "안녕하세요 세계"
        let encoded = pullAll(loaded.pipe, original)
        let decoded = decodeToString(loaded.codec, encoded)
        check decoded == original

      test "Russian roundtrip (" & name & ")":
        let loaded = loadOnce(name, filename, pattern)
        let original = "Привет мир"
        let encoded = pullAll(loaded.pipe, original)
        let decoded = decodeToString(loaded.codec, encoded)
        check decoded == original

      test "Hebrew roundtrip (" & name & ")":
        let loaded = loadOnce(name, filename, pattern)
        let original = "שלום עולם"
        let encoded = pullAll(loaded.pipe, original)
        let decoded = decodeToString(loaded.codec, encoded)
        check decoded == original

      test "Khmer roundtrip (" & name & ")":
        let loaded = loadOnce(name, filename, pattern)
        let original = "សួស្តីពិភពលោក"
        let encoded = pullAll(loaded.pipe, original)
        let decoded = decodeToString(loaded.codec, encoded)
        check decoded == original

      test "Emoji roundtrip (" & name & ")":
        let loaded = loadOnce(name, filename, pattern)
        let original = "Hello 🌍 World! 🎉"
        let encoded = pullAll(loaded.pipe, original)
        let decoded = decodeToString(loaded.codec, encoded)
        check decoded == original

      test "Mixed CJK and English roundtrip (" & name & ")":
        let loaded = loadOnce(name, filename, pattern)
        let original = "Hello 世界 こんにちは 안녕"
        let encoded = pullAll(loaded.pipe, original)
        let decoded = decodeToString(loaded.codec, encoded)
        check decoded == original

      test "Chinese historical paragraph issue merging '。\\n' (" & name & ")":
        let loaded = loadOnce(name, filename, pattern)
        let original = "紅。白\n髮漁樵江渚上，慣看秋月春風。一壺濁酒喜相逢：古今多少事，都付笑談中。\n\n　　話說天下大勢，分久必合，合久必分：周末七國分爭，并" &
          "入於秦。及秦滅之後，楚\n、漢分爭，又并入於漢。漢朝自高祖斬白蛇而起義，一統天下。後來光武中興，傳至獻\n帝，遂分為三國。推其致亂之由，殆始" &
          "於桓、靈二帝。桓帝禁錮善類，崇信宦官。及桓\n帝崩，靈帝即位，大將軍竇武、太傅陳蕃，共相輔佐。時"
        let encoded = pullAll(loaded.pipe, original)
        let decoded = decodeToString(loaded.codec, encoded)
        check decoded == original

when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runTiktokenizerTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
