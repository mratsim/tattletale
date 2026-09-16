# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
##
## Hand-computed segmentation rows for the pattern-split scan
## (scan.nim pattern-split section), named input, named expectation,
## every row labeled with its hazard class:
##
## | class | description                                                                                              |
## | ----- | -------------------------------------------------------------------------------------------------------- |
## | H2    | trailing whitespace run segments as ONE piece, via the pat1-`$` path, never as the pat2+pat3 pair        |
## | H3    | the pat2 drop drops the last space of an interior run, and the scan resumes at it                        |
## | join  | single-space join rows (word, digit run, Han run)                                                        |
## | ws    | CRLF and multi-byte whitespace rows, per family \s variant (U+180E is PCRE2 \s but not Rust White_Space) |
##
## The machine driver over the same patterns is row-checked
## in tests/unit/t_pretok_chain.nim (parity canary rows).

import std/[monotimes, times]

import workspace/toktoktok/src/scan

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    raise newException(AssertionDefect,
      "FAIL " & name & (if detail.len > 0: " " & detail else: ""))

proc spans32ToStrings(text: string,
    spans: seq[tuple[lo, hi: int32]]): seq[string] =
  for s in spans.items:
    result.add text[int(s.lo) ..< int(s.hi)]

proc runTests*() =
  ## Entry point, runs this file's checks in order.
  block:
    let named = [
      (label: "H2 trailing run one piece (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a   ", want: @["a", "   "]),
      (label: "H3 drop-last resume (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a   b", want: @["a", "  ", " b"]),
      (label: "single space joins word (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a b", want: @["a", " b"]),
      (label: "single space before digit (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a 1", want: @["a", " ", "1"]),
      (label: "H2 trailing run one piece (r50k)", pat: R50kPat,
        rustWs: false, text: "a   ", want: @["a", "   "]),
      (label: "H3 drop-last resume (r50k)", pat: R50kPat,
        rustWs: false, text: "a   b", want: @["a", "  ", " b"]),
      (label: "single space joins digit run (r50k)", pat: R50kPat,
        rustWs: false, text: "a 1", want: @["a", " 1"]),
      (label: "tab run interior (r50k)", pat: R50kPat,
        rustWs: false, text: "a\t\t\tb", want: @["a", "\t\t", "\t", "b"]),
      (label: "CRLF mix (r50k)", pat: R50kPat,
        rustWs: false, text: "ab \n cd", want: @["ab", " \n", " cd"]),
      (label: "H3 drop-last resume (exaone)", pat: ExaoneStepPat,
        rustWs: true, text: "a   b", want: @["a", "  ", " b"]),
      (label: "H2 trailing run one piece (exaone)", pat: ExaoneStepPat,
        rustWs: true, text: "a   ", want: @["a", "   "]),
      (label: "space joins Han run (o200k)", pat: O200kPat,
        rustWs: false, text: "a 三國 b", want: @["a", " 三國", " b"]),
      (label: "CRLF run to window end (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "ab \n", want: @["ab", " \n"]),
      (label: "multi-byte ws run interior (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "  \u{00A0} x", want: @["  \u{00A0}", " x"]),
      (label: "U+180E is ws, PCRE2 variant (kimi)", pat: KimiK25Pat,
        rustWs: false, text: " \u{180E} x", want: @[" \u{180E}", " x"]),
      (label: "U+180E not ws, Rust variant (exaone)", pat: ExaoneStepPat,
        rustWs: true, text: " \u{180E} x", want: @[" \u{180E}", " x"]),
      (label: "tab after LF stays its own piece (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a \n\t b", want: @["a", " \n", "\t", " b"]),
      (label: "LF run reaches window end (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a\n\n\nb", want: @["a", "\n\n\n", "b"]),
    ]
    var namedFails = 0
    for row in named.items:
      let sp = splitLookaheadPattern(row.pat, row.rustWs, "named")
      var spans: seq[tuple[lo, hi: int32]]
      sp.scanSplit(spans, row.text, 0, row.text.len)
      let got = spans32ToStrings(row.text, spans)
      if got != row.want:
        inc namedFails
        echo "NAMED ROW MISMATCH [", row.label, "]: split ", got,
          " want ", row.want
    check "hand-computed segmentation rows (split == recorded pieces)",
      namedFails == 0, $namedFails & " fails / " & $named.len & " rows"

  echo "\nall hand-computed split rows passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 /
    1000.0, " s"
