# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Whitespace rules, comment and raw parsing of chattyninja, asserted at the rendered bytes,
## plus the located raises of the parse error contract.
##
## Byte-exact template output stays locked in `t_corpus` through the corpus ledger, one
## arena check remains here, the `nkIf` body-termination walk.
##
## Run:
##   $ nim test_chattyninja

import std/[os, strutils]
import cnj_types, jinja_data_model, cnj_parse, cnj_engine

const root = currentSourcePath().parentDir
let src = readFile(root / "corpus" / "deepseekv2lite" / "deepseekv2lite.jinja")
let (tmpl, symbols) = parseTemplate(src)
let nodes = tmpl.nodes

func ctx(pairs: varargs[(string, JinjaVal)]): JinjaVal =
  ## Builds one context value from `key`, value pairs, the shape the fixtures read.
  var d = DictVal()
  for (k, v) in pairs:
    dictSet(d, k, v)
  dictVal(d)

# `nkIf` is single-entry and single-activation. Walking a branch body forward by `succ` must reach
# the `if` node's own successor, which parse time backpatched past the whole chain, and must never
# land back on the `if` node itself.
for i, n in nodes:
  if n.kind != nkIf:
    continue
  var cur = n.child
  var steps = 0
  while cur != n.succ:
    doAssert cur != NoLink and cur >= 0'i32 and cur < nodes.len.int32,
        "nkIf " & $i & " body does not terminate past the chain"
    doAssert cur != int32 i, "nkIf " & $i & " is re-entered by its own body path"
    doAssert steps <= nodes.len, "nkIf " & $i & " body walks a cycle"
    inc steps
    cur = nodes[cur].succ
  doAssert n.succ != int32 i, "nkIf " & $i & " terminates on itself"

# Whitespace control and comments resolve at parse time, the fixtures assert the rendered
# bytes they produce.

const wsIfSrc = "A   {%- if x -%}\n   hello   \n{%- endif -%}   B\n"
doAssert renderToString(wsIfSrc, ctx(("x", boolVal(true)))) == "AhelloB",
    "dash markers strip the runs around an if body: got " &
    renderToString(wsIfSrc, ctx(("x", boolVal(true)))).escape
doAssert renderToString("A\n   {# plain #}\nB{#- gone -#}  C",
    JinjaVal(kind: vkUndefined)) == "A\nBC",
    "a comment carries lstrip_blocks, trim_blocks and the dash markers"
doAssert renderToString("a{# dropped #}b", JinjaVal(kind: vkUndefined)) == "ab",
    "a comment is erased, not preserved"
doAssert renderToString("keep  me  ", JinjaVal(kind: vkUndefined)) == "keep  me  ",
    "spacing survives byte for byte, only a trailing newline is dropped"

# A `{% raw %}` body renders verbatim, the closing tag's bytes never joining the body,
# a bare `endraw %}` inside it unable to close the block. Each row carries an independently
# written expected output string.
const rawShapes = [
  ("{% raw %}X{% endraw %}", "X"),
  ("{% raw %}B{%  endraw %}C", "BC"),
  ("{% raw %}{% endraw %}", ""),
  ("pre{% raw %}X{% endraw %}post", "preXpost"),
  ("{% raw %}A endraw %} B{% endraw %}", "A endraw %} B"),
  ("{% raw %}X{% endraw -%}  Y", "XY"),
  ("{% raw %}X{%- endraw -%}Y", "XY"),
  ("{% raw %}  X  {%- endraw %}", "  X"),
  ("{% raw %}don't{% endraw %}", "don't"),
  ("A{% raw %}X{% endraw %}B{% raw %}C{% endraw %}D", "AXBCD"),
  # `-%}` on the open tag strips the body's leading whitespace run, `{% endraw -%}`
  # the trailing one, the glued `endraw-%}` spelling the closer upstream accepts
  ("A{% raw -%}  X  {% endraw %}B", "AX  B"),
  ("X{% raw -%}\n  Y  {% endraw %}Z", "XY  Z"),
  ("A{% raw %} X {% endraw-%}B", "A X B"),
  ("A{% raw %}X{% endraw-%}\n  Y", "AXY"),
  # trim_blocks drops the one newline a plain `{% raw %}` opening carries into
  # the body, the dashed opening stripping the whole leading run already
  ("{% raw %}\nX{% endraw %}", "X"),
  ("{% raw %}\n\nX{% endraw %}", "\nX"),
  ("{% raw %}\n  X{% endraw %}", "  X"),
  ("A{% raw %}\nX{% endraw %}B", "AXB"),
]
for (rSrc, want) in rawShapes:
  let got = renderToString(rSrc, JinjaVal(kind: vkUndefined))
  doAssert got == want, "raw body: " & rSrc.escape & " rendered " & got.escape &
      ", want " & want.escape

# A comment body is a verbatim byte run, its close scan quote-blind, an odd quote count
# in the body neither raising nor deferring the scan into the template text after it.
doAssert renderToString("{# don't #}KEEPME{# it's fine #}",
    JinjaVal(kind: vkUndefined)) == "KEEPME", "the comment close is quote-blind"

# A for-iterable ending in `if` (motif, serif) is one word, the filter-clause word scan
# matching only at a word boundary, both shapes visible in the rendered bytes.
doAssert renderToString("{% for m in motif %}{{ m }}{% endfor %}",
    ctx(("motif", seqVal(@[strVal("aa"), strVal("bb")])))) == "aabb",
    "an `if`-suffixed iterable is not truncated at the word boundary"
doAssert renderToString("{% for x in xs if x %}{{ x }}{% endfor %}",
    ctx(("xs", seqVal(@[strVal("a"), intVal(0)])))) == "a",
    "the filter clause keeps only the truthy items and still splits from the iterable"

# A break outside every macro and every for is malformed and raises located. A macro body
# defers the check to its call site and parses.
try:
  discard parseTemplate("{% break %}")
  doAssert false, "a top-level break parsed instead of raising"
except JinjaError as e:
  doAssert "`{% break %}` is outside any `{% for %}`" in e.what, e.what
  doAssert e.offset == 3 and e.span == 5, "the raise locates the keyword: " &
      $e.offset & "+" & $e.span
# Construct nesting is capped at ParseNestingCap, a template nested past it raising located
# at the offending tag, never exhausting the dispatch stack. Nesting at the cap parses.
block nestingValve:
  block:
    discard parseTemplate(repeat("{% if x %}", ParseNestingCap) & "y" &
        repeat("{% endif %}", ParseNestingCap))
  var reported = ""
  var at = -1
  try:
    discard parseTemplate(repeat("{% if x %}", ParseNestingCap + 1) & "y" &
        repeat("{% endif %}", ParseNestingCap + 1))
    doAssert false, "nesting past the cap parsed instead of raising"
  except JinjaError as e:
    reported = e.what
    at = e.offset
  doAssert "ParseNestingCap" in reported, reported
  let tagStart = "{% if x %}".len * ParseNestingCap
  doAssert at >= tagStart and at < tagStart + "{% if x %}".len,
      "the cap raise located inside the offending tag: offset " & $at

# An `{% elif %}` chain recurses `parseIf` outside any body walk, so the chain depth counts
# toward ParseNestingCap too. A modest chain parses, a chain past the cap raising located.
block elifChainNesting:
  discard parseTemplate("{% if z %}a" & repeat("{% elif z %}a", 10) & "{% endif %}")
  var reported = ""
  try:
    discard parseTemplate("{% if z %}a" & repeat("{% elif z %}a", 20000) & "{% endif %}")
    doAssert false, "an elif chain past the cap parsed instead of raising"
  except JinjaError as e:
    reported = e.what
  doAssert "ParseNestingCap" in reported, reported

# Truncation is a located raise, never a spin. Every body walk consumes at least one tag,
# an unterminated construct reaching the end sentinel, its close check reporting it.
block truncationRaisesLocated:
  # Each test checks the raise offset inside the offending tag's span, so a raise
  # at an unrelated byte fails the test.
  for truncation in [("{% set x %}body", 0, 11), ("{% generation %}body", 0, 16),
      ("{% for x in xs %}{{ x }}", 0, 17), ("{% if x %}body", 0, 10),
      ("{{ x", 0, 4), ("{% x", 0, 4), ("{# c", 0, 4), ("{% raw %}body", 0, 9),
      ("{% macro f(a, b = 1 %}body{% endmacro %}", 0, 22),
      ("{% macro f(, %}body{% endmacro %}", 0, 15),
      ("{% macro f %}body{% endmacro %}", 0, 13), ("{% endfor %}", 0, 12),
      ("{% elif x %}", 0, 12)]:
    let (t, tagLo, tagHi) = truncation
    var reported = ""
    var at = -1
    try:
      discard parseTemplate(t)
      doAssert false, "a truncated or degenerate template parsed: " & t
    except JinjaError as e:
      reported = e.what
      at = e.offset
    doAssert reported.len > 0, "truncation raised nothing: " & t
    doAssert at >= tagLo and at < tagHi,
        "the truncation raise located inside the offending tag: " & t & " at " & $at

# Declared-gap terminators (`endmacro` without a matching macro) raise the gap message,
# which carries no location by the gap raise's contract.
try:
  discard parseTemplate("{% endmacro %}")
  doAssert false, "a lone endmacro parsed"
except JinjaError as e:
  doAssert e.cause == ceUnimplemented and "endmacro" in e.what, e.what

# A `set` tag with no target name raises at the tag, matching upstream's
# missing-name raise:
#   - a bare `{% set %}`
#   - `{% set = 3 %}` with no name at all
# A name ending exactly at the tag end, `{% set x%}`, is a block assignment whose
# body follows, so the raise there names the missing `{% endset %}`, never the target.
for setNoName in ["{% set %}A", "{% set = 3 %}"]:
  try:
    discard parseTemplate(setNoName)
    doAssert false, "a nameless set parsed: " & setNoName
  except JinjaError as e:
    doAssert "needs a target" in e.what, e.what
try:
  discard parseTemplate("{% set x%}3{{ x }}")
  doAssert false, "a set with no endset parsed"
except JinjaError as e:
  doAssert "endset" in e.what, e.what

# Whitespace-rule sweep at the rendered bytes, one row per rule and branch
# (`{% %}` block tag, `{{ }}` variable tag, comment, raw body, text run).
#
# | Rule            | Effect                                           |
# | --------------- | ------------------------------------------------ |
# | `trim_blocks`   | one newline after a block or comment tag's close |
# | `lstrip_blocks` | the blanks of a block or comment tag's line      |
# | `-` markers     | every whitespace run before or after their tag   |
# | plain raw open  | one body newline consumed                        |
#
# Byte-locked shapes, a consolidation of the rule sites moving one byte failing here,
# before the corpus ledger has to report it.
const wsShapes = [
  ("{% set y = 1 %}\nB", "B", "trim after a block tag's close"),
  ("A\n{% set y = 1 %}\nB", "A\nB", "trim drops the newline after the close"),
  ("A\n   {% set y = 1 %}\nB", "A\nB", "lstrip the blanks before a block tag"),
  ("A\n   {{ 1 }}\nB", "A\n   1\nB", "lstrip leaves a variable tag's line alone"),
  ("{{ 1 }}\nB", "1\nB", "trim leaves a variable tag's newline"),
  ("A   {{- 1 -}}   B", "A1B", "dashes strip the runs around a variable tag"),
  ("A   {%- set y = 1 -%}   B", "AB", "dashes strip the runs around a block tag"),
  ("A\n   {# c #}\nB", "A\nB", "a plain comment carries lstrip and trim"),
  ("A  {#- c -#}  B", "AB", "a dashed comment strips the runs"),
  ("{#- c -#}\nB", "B", "a dashed comment at the start"),
  ("A{% raw %}\nX{% endraw %}B", "AXB", "a plain raw opening consumes one newline"),
  ("{% raw %}\nX{% endraw %}", "X", "the raw newline consume with no run before"),
  ("A\n   {% raw %}\nX{% endraw %}\nB", "A\nXB", "raw under lstrip and trim"),
  ("A{% raw -%}\n  X  {% endraw %}B", "AX  B", "a dashed raw opening strips the body's leading run"),
  ("A{% raw %}\nX{% endraw -%}\nB", "AXB", "a dashed raw close strips the run after it"),
  ("A\n", "A", "the final newline dropped"),
  ("A\nB\n", "A\nB", "only the final newline dropped"),
  ("A {#-#}\n\nB", "AB", "a whole-dash comment interior loses both markers at once"),
  ("{#-#}   \n\nB", "B", "a whole-dash comment strips nothing extra after the tag"),
]
for (wSrc, want, label) in wsShapes:
  let got = renderToString(wSrc, JinjaVal(kind: vkUndefined))
  doAssert got == want, label & ": " & wSrc.escape & " rendered " & got.escape & ", want " & want.escape

echo "t_parse: ", nodes.len, " nodes, ", symbols.names.len, " interned names, verbatim spans final"
