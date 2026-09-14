# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
## The GPT-2 byte-to-unicode remap (deserializers.nim procs/tables).
## 1. remap table == an independently computed GPT-2 bytes_to_unicode table,
##    and every mapped codepoint round-trips through `unmap`,
##    cross-checked against the deserializer's GPT-2 byte decoder,
## 2. corpus streams:
##   `bytesToUnicode` output unmaps back to the corpus
##    text byte-identically (the remap is a bijection over bytes),
## 3. prefix-space position:
##   add_prefix_space=true prepends one space
##    unless the input already opens with one,
## 4. the remap string is the full mapped alphabet:
##   nonempty for any
##    nonempty input and it unmaps back byte-identically.

import std/[monotimes, times]
import std/[os, strutils, tables]

import workspace/zstd/zstd_highlevel
import workspace/toktoktok/src/deserializers


const
  TestsDir = currentSourcePath().parentDir().parentDir()
    ## tests/ root (the suite lives one level down)
  CorpusDir = TestsDir / "corpus"

var checkFailures = 0

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    checkFailures += 1

proc readCorpusPrefix(path: string, maxBytes: int): string =
  var text = readFile(path).zstdDecompress(string)
  if text.len > maxBytes:
    text = text[0 ..< maxBytes]
  text

proc independentGpt2Table(): array[256, uint32] =
  ## Independent recomputation of the GPT-2 bytes_to_unicode table
  ## (list(range(ord("!"), ord("~")+1)) + list(range(0xA1, 0xAC+1)) + list(range(0xAE, 0xFF+1))),
  ##
  ## every other byte maps to 256 + its rank among the bytes outside
  ## the identity set, ascending.
  var n = 0'u32
  for b in 0 ..< 256:
    let inIdentity = (b >= 0x21 and b <= 0x7E) or (b >= 0xA1 and b <= 0xAC) or
      (b >= 0xAE and b <= 0xFF)
    if inIdentity:
      result[b] = uint32(b)
    else:
      result[b] = 256'u32 + n
      inc n

proc runTests*() =
  ## Entry point, runs this file's checks in order.
  #
  # 1. table identity + per-byte round-trip + decoder cross-check
  #
  block:
    let want = independentGpt2Table()
    check "remap table == independent GPT-2 bytes_to_unicode table",
      ByteLevelRemap == want
    let decoder = gpt2ByteDecoder()
    var tableFails = 0
    for b in 0 ..< 256:
      let cp = ByteLevelRemap[b]
      # the GPT-2 byte decoder maps the remapped codepoint back
      # to the input byte (its table is built from the same GPT-2 shape)
      if decoder.getOrDefault(cp, -1) != b:
        inc tableFails
      let mappedChars = bytesToUnicode("" & char(b))
      if unmap(mappedChars).len != 1 or
          unmap(mappedChars)[0] != char(b):
        inc tableFails
    check "per-byte round-trip + GPT-2 byte-decoder cross-check (256 bytes)",
      tableFails == 0, $tableFails & " fails"

  #
  # 2. corpus remap strings unmap back to the corpus text
  #
  let corpora = [
    ("sanguozhi", CorpusDir /
      "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt.zst", 20000),
    ("verne", CorpusDir /
      "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt.zst", 20000),
    ("shakespeare", CorpusDir / "pg100-shakespeare.txt.zst", 30000),
    ("sqlite", CorpusDir / "sqlite3.c.zst", 50000),
  ]
  for (cname, path, maxBytes) in corpora.items:
    let text = readCorpusPrefix(path, maxBytes)
    let mapped = bytesToUnicode(text)
    check "remap string unmaps to the corpus text [" & cname & "]",
      unmap(mapped) == text, $text.len & " bytes -> " & $mapped.len &
      " mapped bytes"

  #
  # 3. prefix-space position
  #
  block:
    let spaceCp = ByteLevelRemap[uint8(' ')]
    let spaceMapped: string = block:
      var s: string
      if spaceCp < 0x80'u32:
        s.add char(spaceCp)
      else:
        s.add char(0xC0'u8 or uint8(spaceCp shr 6))
        s.add char(0x80'u8 or uint8(spaceCp and 0x3F'u32))
      s
    check "prefix space prepended when input opens non-space",
      bytesToUnicode("abc", true) == spaceMapped & bytesToUnicode("abc", false)
    check "no prefix space when input opens with a space",
      bytesToUnicode(" x", true) == spaceMapped & bytesToUnicode("x", false)
    let withSpace = bytesToUnicode("abc", true)
    check "no prefix space when disabled",
      withSpace.len == spaceMapped.len + 3 and
      bytesToUnicode("abc", false) == withSpace[spaceMapped.len .. ^1]

  #
  # 4. the remap string is the full mapped alphabet
  #
  block:
    let mapped = bytesToUnicode("abc")
    check "remap string is the remapped alphabet (nonempty, 3 bytes)",
      mapped.len == 3 and unmap(mapped) == "abc"

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall bytelevel remap checks passed"
when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
