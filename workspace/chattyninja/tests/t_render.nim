# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Byte-exact rendering against the recorded corpus ground truth.
##
## Every ok row of `corpus/deepseekv2lite` is compared byte for byte with the `rendered` field
## recorded from HF `render_jinja_template`. Every `err_*` row must raise the recorded
## exception class and message.
##
## An engine that renders past an expected error fails in a way a happy-path test cannot see.
##
## The comparison is deliberately not `doAssert a == b`. `sameBytes` reports the first differing
## byte and both surroundings, and the suite checks that `sameBytes` rejects a mutated expectation,
## so a green run cannot come from an always-passing check.
##
## Run:
##   $ ./workspace/chattyninja/run_tests.sh t_render

import std/[os, strutils]
import cjn_types, cjn_values, cjn_parse, chattyninja
import rows

type RenderMismatch = ref object of CatchableError

proc fail(msg: string): void {.noreturn.} =
  var e = RenderMismatch()
  e.msg = msg
  raise e

func firstDiff(got, want: string): int =
  ## Returns the index of the first differing byte, or the shared length when one is a prefix.
  let n = min(got.len, want.len)
  while result < n and got[result] == want[result]:
    inc result

func surround(s: string, at: int): string =
  ## Returns a repr of the 24 bytes around `at`, for a failure message.
  s[max(0, at - 24) ..< min(s.len, at + 24)].replace("\n", "\\n").replace("\t", "\\t")

func sameBytes*(got, want: string): bool =
  ## Reports whether two renderings are byte-identical.
  got.len == want.len and firstDiff(got, want) == got.len

func report(got, want: string): string =
  if got.len != want.len and firstDiff(got, want) == min(got.len, want.len):
    "length " & $got.len & " vs " & $want.len & ", common prefix; got tail " &
        surround(got, got.len) & " | want tail " & surround(want, want.len)
  else:
    let k = firstDiff(got, want)
    "first difference at byte " & $k & " (got " & $got.len & " B, want " & $want.len & " B)\n" &
        "   got  " & surround(got, k) & "\n   want " & surround(want, k)

# The equality must reject a one-byte change, since a comparison that cannot fail makes
# the suite vacuous.
block equalityRejectsChange:
  doAssert sameBytes("abc", "abc")
  doAssert not sameBytes("abc", "abd"), "sameBytes accepted a changed byte"
  doAssert not sameBytes("abc", "abcd"), "sameBytes accepted a length change"
  doAssert not sameBytes("", "a"), "sameBytes accepted empty against non-empty"

proc renderRow(m: Machine, t: Tables, r: Row): string =
  ## Renders one row through the pull interface, whole.
  var d = newDriver(r.context, r.clock)
  pullAll(m, t, d)

# deepseekv2lite: byte-exact against the recorded bytes
# ---------------------------------------------------------------------------

block deepseekv2liteByteExact:
  let src = templateSource("deepseekv2lite")
  let (nodes, tables) = parseTemplate(src)
  let m = Machine(jinja: src, nodes: nodes)
  var checked = 0
  for r in rows("deepseekv2lite"):
    doAssert not r.expectError, "deepseekv2lite row " & r.row & " carries an expected error"
    let got = renderRow(m, tables, r)
    if not sameBytes(got, r.rendered):
      fail("deepseekv2lite/" & r.row & ": " & report(got, r.rendered))
    inc checked
  # 4 recorded ok rows exist. A suite that silently checked none would prove nothing.
  doAssert checked == 4, "expected 4 deepseekv2lite rows, checked " & $checked

# Every err row: the recorded error, not a wrong success
# ---------------------------------------------------------------------------

const errSuites = ["gemma3", "gptoss20b", "mistral7bv01", "qwen38flashnext"]

block errRowsRaiseWhatWasRecorded:
  var checked = 0
  for suite in errSuites:
    let src = templateSource(suite)
    let (nodes, tables) = parseTemplate(src)
    let m = Machine(jinja: src, nodes: nodes)
    for r in rows(suite):
      if not r.expectError:
        continue
      var raisedName = ""
      var raisedMsg = ""
      var wrongSuccess = ""
      try:
        wrongSuccess = renderRow(m, tables, r)
      except CatchableError as e:
        raisedName = $e.name
        raisedMsg = e.msg
      # Nim renders a CatchableError's class as `Name:ObjectType`. The corpus records the bare name.
      let cls =
        if ':' in raisedName: raisedName[0 ..< raisedName.find(':')]
        else: raisedName
      if wrongSuccess.len > 0:
        fail(suite & "/" & r.row & ": expected " & r.errorClass & " `" & r.errorMessage &
            "` but the render produced " & $wrongSuccess.len & " bytes")
      if cls != r.errorClass:
        fail(suite & "/" & r.row & ": expected exception " & r.errorClass & ", got " & cls &
            " (" & raisedMsg & ")")
      if raisedMsg != r.errorMessage:
        fail(suite & "/" & r.row & ": message mismatch\n   got  " & raisedMsg & "\n   want " &
            r.errorMessage)
      inc checked
  # PROVENANCE records 16 err rows across exactly these four suites.
  doAssert checked == 16, "expected 16 err rows, checked " & $checked

echo "t_render: deepseekv2lite 4 rows byte-exact, 16 err rows raise the recorded error"
