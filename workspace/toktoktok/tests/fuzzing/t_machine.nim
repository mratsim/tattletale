# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
## Machine-protocol identity machine over corpus prefixes:
## 1. identity machine stream == input bytes,
## 2. resume discipline, a stream broken mid-iteration resumes
##    from the first element not yet received (position state advances before each yield).

import std/[monotimes, times]
import std/[os, strutils]

import workspace/zstd/zstd_highlevel
import workspace/toktoktok/src/machine

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

proc collectIdentity(text: string): string =
  var m = identityStream(text)
  for c in m.items:
    result.add c

proc collectIdentityChunked(text: string, step: int): string =
  ## Breaks the stream every `step` bytes and re-iterates, exercising
  ## the resume contract (every received byte consumed exactly once).
  var m = identityStream(text)
  var taken = 0
  while taken < text.len:
    var batch = 0
    for c in m.items:
      result.add c
      inc taken
      inc batch
      if batch == step:
        break
  doAssert result.len == taken

proc runTests*() =
  ## Entry point, runs this file's checks in order.
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
    check "identity stream == input bytes [" & cname & "]",
      collectIdentity(text) == text, $text.len & " bytes"
    check "identity resume discipline [" & cname & "]",
      collectIdentityChunked(text, 997) == text,
      $text.len & " bytes, step 997"
  block:
    # empty input drains immediately
    check "identity over empty input yields nothing",
      collectIdentity("") == ""

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall stage-machine checks passed"
when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
