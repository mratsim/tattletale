# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Regex pattern engine, the pattern parser, Thompson NFA compiler,
## leftmost-first matcher, the frontier-DFA deterministic compile,
## plus the Unicode property tables.
runnableExamples:
  import ./src/regex_core

  let p = compilePattern(r"\p{L}+|\p{N}+")
  var m = initRegexScanner(p, "ab12 c")
  var spans: seq[tuple[start, stop: int]]
  for s in m.items:
    spans.add s
  doAssert spans == @[(0, 2), (2, 4), (5, 6)]

import ./src/regex_core
import ./src/regex_unicode_tables

export regex_core, regex_unicode_tables
