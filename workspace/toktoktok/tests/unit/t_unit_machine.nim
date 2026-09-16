# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
##
## Explicit machine-protocol unit rows:
##   named input, named expectation.
## 1. identity stream == input bytes over named inputs,
## 2. resume discipline:
##   a stream broken after a named byte count
##    resumes at the first byte not yet received,
## 3. empty-input rows.

import std/[monotimes, times]

import workspace/toktoktok/src/machine

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    raise newException(AssertionDefect,
      "FAIL " & name & (if detail.len > 0: " " & detail else: ""))

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
  block:
    check "identity stream 'tok' == 'tok'", collectIdentity("tok") == "tok"
    check "identity stream 'Hello, 你好!' == input",
      collectIdentity("Hello, 你好!") == "Hello, 你好!"

    # resume discipline:
    #   break after a named byte count, re-iterate
    var m = identityStream("toktok")
    var first = ""
    for c in m.items:
      first.add c
      if first.len == 2:
        break
    var rest = ""
    for c in m.items:
      rest.add c
    check "identity resume: 2 received bytes, rest is 'ktok'",
      first == "to" and rest == "ktok"
    check "identity resume: broken stream reassembles 'toktok'",
      first & rest == "toktok"

  block:
    check "identity chunked resume, step 3 over 'toktoktok'",
      collectIdentityChunked("toktoktok", 3) == "toktoktok"
    check "identity chunked resume, step 1 over 'ab'",
      collectIdentityChunked("ab", 1) == "ab"

  block:
    var m = identityStream("")
    var n = 0
    for c in m.items:
      inc n
    check "identity over empty input yields nothing", n == 0

  echo "\nall machine unit rows passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
