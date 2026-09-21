# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Expression tier of the chattyninja engine, the fused walker over a `lo..hi` template-text span.
##
## Every case renders through `{{ }}`, so stringification and the pending-piece
## path sit on the observed path. Expected values come from the expression text.
##
## Branches the engine must skip carry a `raise_exception` call, so a wrongly entered
## branch raises the recorded error at that call.
##
## Run:
##   $ nim test_chattyninja

import std/strutils
import cnj_types, jinja_data_model, cnj_engine

proc render(expr: string, ctx = JinjaVal(kind: vkUndefined)): string =
  ## Renders one expression through `renderToString`.
  renderToString("{{ " & expr & " }}", ctx)

proc renderStmt(stmt: string, ctx = JinjaVal(kind: vkUndefined), clock = 0.0): string =
  ## Renders one statement through `renderToString`, the clock the driver receives.
  renderToString(stmt, ctx, clock)

func ctx(pairs: varargs[(string, JinjaVal)]): JinjaVal =
  var d = DictVal()
  for (k, v) in pairs:
    dictSet(d, k, v)
  dictVal(d)

let people = ctx(
    ("name", strVal("ada")),
    ("age", intVal(36)),
    ("tags", seqVal(@[strVal("x"), strVal("y")])),
    ("active", boolVal(true)))
let withPeople = ctx(("people", people))

# Literals, name lookup, attribute and subscript
# ---------------------------------------------------------------------------

doAssert render("42") == "42"
doAssert render("'text'") == "text"
doAssert render("'a\\nb'") == "a\nb", "escape decoding inside a literal"
doAssert render("true") == "True"
doAssert render("none") == "None"
doAssert render("True") == "True", "capitalized literal spelling"
doAssert render("False") == "False", "capitalized literal spelling"
doAssert render("None") == "None", "capitalized literal spelling"
doAssert render("1.5") == "1.5"
doAssert render("people.name", withPeople) == "ada"
doAssert render("people['name']", withPeople) == "ada"
doAssert render("people.tags[1]", withPeople) == "y"
doAssert render("people.tags[-1]", withPeople) == "y"
doAssert render("people.missing", withPeople) == "", "absence renders empty, it is not an error"
doAssert render("people.tags[1:]", withPeople) == "['y']"
doAssert render("'h\u00e9llo'[-1]") == "o", "a negative string subscript counts from the end"
doAssert render("'h\u00e9llo'[1]") == "\u00e9", "a multibyte string subscript returns its codepoint"

# Arithmetic and concatenation
# ---------------------------------------------------------------------------

doAssert render("1 + 2") == "3", "`+` on integers"
doAssert render("'a' + 'b'") == "ab", "`+` on strings"
doAssert render("1 + 2 == 3") == "True", "`+` binds tighter than `==`"
doAssert render("5 - 2 - 1") == "2", "`-` is left-associative"
doAssert render("-3 + 10") == "7"
doAssert render("'a' ~ 1 ~ none") == "a1None", "`~` stringifies both sides"
doAssert render("people.age + 1", withPeople) == "37"

# Comparison, membership
# ---------------------------------------------------------------------------

doAssert render("1 == 1") == "True"
doAssert render("'a' == 'a'") == "True"
doAssert render("1 == 1.0") == "True", "numbers compare across tiers"
doAssert render("'a' != 'b'") == "True"
doAssert render("1 < 2") == "True"
doAssert render("2 > 1") == "True"
doAssert render("'a' in ['a', 'b']") == "True"
doAssert render("'z' in ['a', 'b']") == "False"
doAssert render("'a' not in ['a', 'b']") == "False", "`not in` is `in` negated"
doAssert render("'z' not in ['a', 'b']") == "True"
doAssert render("'name' in people", withPeople) == "True", "`in` over a mapping tests keys"
doAssert render("'sub' in 'substring'") == "True", "`in` over a string is a substring test"
doAssert render("'ss' in 'substring'") == "False",
    "a substring test whose first byte matches keeps scanning, absent stays absent"
doAssert render("'ngs' in 'substring'") == "False", "an absent multi-byte-position needle stays absent"

# `~` binds tighter than any comparison, so `1 == 1 ~ 'x'` compares against the unrendered
# concatenation and raises. Parentheses hand the comparison result to `~`, and a precedence
# swap toward `==` would render the second form's bytes for the first expression.
doAssert render("(1 == 1) ~ 'x'") == "Truex", "parentheses give the comparison to `~`"
try:
  discard render("1 == 1 ~ 'x'")
  doAssert false, "a comparison over an unrendered concat did not raise"
except JinjaError as e:
  doAssert "emit position" in e.what, e.what

# A concat reads plain values in one place only, the argument list: `raise_exception`
# names its message from the materialized text. Every other non-emit position raises,
# and a set-bound concat streams when a later emit reaches it.
block concatConsumption:
  try:
    discard render("{'k': 'a' ~ 'b'}")
    doAssert false, "a concat in a dict literal did not raise"
  except JinjaError as e:
    doAssert "emit position" in e.what, e.what
  try:
    discard renderStmt("{% if 'a' ~ 'b' %}x{% endif %}")
    doAssert false, "a concat in a condition did not raise"
  except JinjaError as e:
    doAssert "emit position" in e.what, e.what
  try:
    discard render("('a' ~ 'b') | tojson")
    doAssert false, "a concat under a filter did not raise"
  except JinjaError as e:
    doAssert "emit position" in e.what, e.what
  doAssert renderStmt("{% set q = 'a' ~ 'b' %}{{ q }}") == "ab",
      "a set-bound concat did not stream on its later emit"
  try:
    discard renderStmt("{% set q = 'a' ~ 'b' %}{% if q %}x{% endif %}")
    doAssert false, "a truthiness test over a set-bound concat did not raise"
  except JinjaError as e:
    doAssert "emit position" in e.what, e.what
  try:
    discard renderStmt("{{ raise_exception('boom ' ~ 'bang') }}")
    doAssert false, "raise_exception did not raise"
  except JinjaError as e:
    doAssert e.what == "boom bang", e.what
  doAssert render("(1 ~ 2) ~ 3") == "123", "a grouped concat flattens into its parent"

# A lone `=` spells no infix in Jinja, so the expression must end before it and the unclosed
# tail is reported instead of comparing. Keyword arguments are matched in argument lists, not here.
block loneEqualsIsAnError:
  var reported = ""
  try:
    discard render("1 = 2")
  except JinjaError as e:
    reported = e.what
  doAssert "trailing text" in reported, reported

# `and` / `or` skip the operand they do not evaluate
# ---------------------------------------------------------------------------

doAssert render("false and raise_exception('left was skipped')") == "False",
    "`and` with a false left operand must not evaluate the right"
doAssert render("true or raise_exception('left was truthy')") == "True",
    "`or` with a truthy left operand must not evaluate the right"
doAssert render("true and 1 == 1") == "True"
doAssert render("false or 2 > 1") == "True"

# Ternary: one condition evaluation, exactly one branch
# ---------------------------------------------------------------------------

doAssert render("'yes' if true else raise_exception('else branch ran')") == "yes",
    "a ternary must not run the branch it did not take"
doAssert render("raise_exception('then branch ran') if false else 'no'") == "no",
    "a ternary must not run the branch it did not take"
doAssert render("'on' if people.active else 'off'", withPeople) == "on"
doAssert render("'on' if people.missing else 'off'", withPeople) == "off"
doAssert render("1 if 1 > 2 else 2 if 2 > 1 else 3") == "2", "a chained ternary takes one arm"
doAssert render("range(1 if true else 2) | length") == "1",
    "a ternary inside a call argument list takes one arm"
doAssert render("{'k': 't' if true else 'f'}['k']") == "t",
    "a ternary inside a dict literal value takes one arm"
doAssert render("('t' if false else 'f')") == "f",
    "a ternary inside a parenthesised group takes one arm"

# `is defined` set-guard
# ---------------------------------------------------------------------------

# The arena's entry must be the outermost construct, not a node inside a branch
# body. Entering the arena at index 0 has to reach the `if`, so the false branch
# is what comes out.
doAssert renderStmt("{% if false %}X{% else %}O{% endif %}") == "O",
    "an `if` whose condition is false must take the else body"
doAssert renderStmt("{% if 1 == 2 %}X{% else %}O{% endif %}") == "O"
doAssert renderStmt("{% if true %}X{% else %}O{% endif %}") == "X"
doAssert renderStmt("{% if false %}X{% endif %}after") == "after"
doAssert renderStmt("{% for x in [1, 2] %}{{ x }}{% endfor %}") == "12",
    "a `for` must be entered from the arena entry, not from its body"
doAssert renderStmt("{% if missing is defined %}X{% else %}O{% endif %}") == "O"
doAssert renderStmt("{% if people is defined %}X{% endif %}", ctx(("people", intVal(1)))) == "X"
doAssert renderStmt("{% if not people is defined %}X{% else %}O{% endif %}",
    ctx(("people", intVal(1)))) == "O"
doAssert render("people.missing is defined", withPeople) == "False"
doAssert render("people.name is defined", withPeople) == "True"

# A macro call in a boolean condition renders to its output text, whose bytes then decide
# the branch, matching upstream Jinja. A concat in a condition keeps raising (asserted above).
doAssert renderStmt("{% macro m() %}yes{% endmacro %}{% if m() %}A{% else %}B{% endif %}") == "A",
    "a truthy macro call takes the if body"
doAssert renderStmt("{% macro m() %}yes{% endmacro %}{% if 0 %}X{% elif m() %}E{% else %}O{% endif %}") == "E",
    "a truthy macro call takes the elif body"
doAssert renderStmt("{% macro m() %}{% endmacro %}{% if m() %}A{% else %}B{% endif %}") == "B",
    "an empty macro body renders falsy"
doAssert renderStmt("{% macro m() %}yes{% endmacro %}{% if not m() %}A{% else %}B{% endif %}") == "B"
doAssert renderStmt("{% macro m() %}yes{% endmacro %}{% if m() and 1 %}A{% else %}B{% endif %}") == "A"
doAssert renderStmt("{% macro m() %}yes{% endmacro %}{% if 0 or m() %}A{% else %}B{% endif %}") == "A"
doAssert renderStmt("{% macro m() %}yes{% endmacro %}{% if 1 or m() %}A{% else %}B{% endif %}") == "A",
    "a short-circuit or skips the second operand's forcing"
doAssert renderStmt("{% macro m() %}yes{% endmacro %}{% if 0 and m() %}A{% else %}B{% endif %}") == "B"
doAssert renderStmt("{% macro m() %}yes{% endmacro %}{{ 'a' if m() else 'b' }}") == "a",
    "a ternary condition renders the macro call"
doAssert renderStmt("{% macro m() %}yes{% endmacro %}{% for x in [1, 2] if m() %}{{ x }}{% endfor %}") == "12",
    "a for-filter condition renders the macro call"
# An integer constant after a dot subscripts, upstream's `m.content.0` spelling of `m.content[0]`.
doAssert render("['a', 'b'].0") == "a"
doAssert render("['a', 'b'].1") == "b"
doAssert render("'ab'.0") == "a", "a string's integer-constant subscript yields one codepoint"
doAssert render("{'0': 'z'}.0") == "z", "a mapping's integer-constant subscript reads the key"
doAssert render("people.tags.0", withPeople) == "x"
try:
  discard render("[1, 2].5")
  doAssert false, "an out-of-range integer-constant subscript did not raise"
except JinjaError as e:
  doAssert "out of range" in e.what, e.what

# A break unwinds to the nearest for-row and continues at its successor, the loop's after-node.
# A continue shares the node kind and re-enters the loop instead, one item skipped.
doAssert renderStmt("{% for x in [1, 2, 3] %}{{ x }}{% break %}{% endfor %}tail") == "1tail",
    "a break ends the loop after the body ran, control resuming at the for's successor"
doAssert renderStmt("{% for x in [1, 2, 3] %}{% if x == 2 %}{% break %}{% endif %}{{ x }}{% endfor %}") == "1",
    "a break inside an if body unwinds the if, which carries no row, and ends the loop"
doAssert renderStmt("{% for x in [1, 2, 3] %}{% if x == 2 %}{% continue %}{% endif %}{{ x }}{% endfor %}") == "13",
    "a continue leaves the for-row in place and re-enters it, skipping one item"
doAssert renderStmt(
    "{% macro m() %}a{% break %}b{% endmacro %}" &
    "{% for x in [1, 2] %}{{ m() }}{% endfor %}") == "aa",
    "a break stops at the macro-call boundary, the body's output so far draining on each call"
doAssert renderStmt(
    "{% macro m() %}{% for x in [1, 2] %}{% break %}{% endfor %}{% endmacro %}" &
    "{% if m() %}A{% else %}B{% endif %}") == "B",
    "a break inside a for inside a forced macro body unwinds that for"

# Registries: filters and tests dispatch by name
# ---------------------------------------------------------------------------

doAssert render("[1, 2] | length") == "2"
doAssert render("'  pad  ' | trim | length") == "3"
doAssert render("{'a': 1, 'b': 2} | tojson") == "{\"a\": 1, \"b\": 2}"
doAssert render("[1, 2] | tojson") == "[1, 2]"
doAssert render("'x' | tojson") == "\"x\""
doAssert render("'<' | tojson") == "\"\\u003c\"", "tojson applies the Jinja HTML escape"
# Plain `| tojson` renders non-ASCII as raw UTF-8, Jinja's default.
doAssert render("'東京' | tojson") == "\"東京\"", "tojson keeps non-ASCII verbatim by default"
doAssert renderStmt("{#- hello -#}\nA{{ name }}",
    ctx(("name", strVal("B")))) == "AB", "comment `-` markers strip the surrounding whitespace runs"
doAssert render("people.tags | join('-')", withPeople) == "x-y"
doAssert render("[1, 2] | join(x == '-')",
    ctx(("x", seqVal(@[strVal("a"), strVal("b")])))) == "1False2",
    "`==` inside a call argument list compares, it is not read as a keyword `=`"
block loneEqualsInCallArgsIsAnError:
  var reported = ""
  try:
    discard render("[1, 2] | join(1 = '-')")
  except JinjaError as e:
    reported = e.what
  doAssert reported.len > 0, "a lone `=` inside a call argument list must raise, not render"
doAssert render("'ab' | upper") == "AB"
doAssert render("'AB' | lower") == "ab"
doAssert render("'abc' | capitalize") == "Abc"
doAssert render("missing | default('d')") == "d"
doAssert render("people is mapping", withPeople) == "True"
doAssert render("people.tags is sequence", withPeople) == "True"
doAssert render("people is sequence", withPeople) == "False"
doAssert render("'x' is not string") == "False"

# Calls: methods and globals
# ---------------------------------------------------------------------------

doAssert render("people.keys()", withPeople) == "['name', 'age', 'tags', 'active']"
doAssert render("people.items()", withPeople)[0 ..< 15] == "[['name', 'ada'"
doAssert render("people.values()", withPeople)[0 ..< 10] == "['ada', 36"
doAssert render("people.get('name')", withPeople) == "ada"
doAssert render("people.get('nope', 'fb')", withPeople) == "fb"
doAssert render("'a b'.split(' ')") == "['a', 'b']"
doAssert render("'abc'.startswith('ab')") == "True"
doAssert render("'abc'.endswith('bc')") == "True"
doAssert render("'  x '.strip()") == "x"
doAssert render("'  x'.lstrip()") == "x"
doAssert render("'x  '.rstrip()") == "x"
doAssert render("[3, 1, 2] | list") == "[3, 1, 2]"
# `replace` is a declared filter name with no corpus site, so reaching it must report the gap
# rather than answer wrongly or silently.
block gapIsLoud:
  var reported = ""
  try:
    discard render("[2, 1] | map('int')")
  except JinjaError as e:
    reported = e.what
  doAssert "not implemented" in reported and "map" in reported, reported
doAssert render("range(3)") == "[0, 1, 2]"

# Slices: bounds, and the step of `messages[::-1]`
# ---------------------------------------------------------------------------

# Expected values are Python's own `list.__getitem__`/`str.__getitem__` results for the literal
# expression on the left, not values the engine computed.
# Bounds without a step keep the forward walk, including an empty range and out-of-range bounds.
doAssert render("[1, 2, 3][-2:]") == "[2, 3]", "a negative start counts from the end"
doAssert render("[1, 2, 3][:-1]") == "[1, 2]", "a negative stop counts from the end"
doAssert render("'abcde'[-3:]") == "cde"
doAssert render("'abcde'[:-1]") == "abcd"
doAssert render("[1, 2, 3][-99:]") == "[1, 2, 3]", "a start past the head clamps to the head"
doAssert render("[1, 2, 3][99:]") == "[]", "a start past the tail is empty"
doAssert render("[1, 2, 3][2:1]") == "[]", "a stop before the start is empty, not reversed"
doAssert render("[1, 2, 3][1:1]") == "[]"
doAssert render("[1, 2, 3][::-1]") == "[3, 2, 1]", "negative step reverses"
doAssert render("'abc'[::-1]") == "cba", "negative step reverses a string by codepoint"
doAssert render("[1, 2, 3, 4][::2]") == "[1, 3]", "step over the whole range"
doAssert render("[1, 2, 3, 4][1:4:2]") == "[2, 4]", "bounds and step together"
doAssert render("[1, 2, 3, 4][::-1]") == "[4, 3, 2, 1]"
doAssert render("[1, 2, 3][::-2]") == "[3, 1]", "negative step visits every second item"
doAssert render("[1, 2, 3][2:0:-1]") == "[3, 2]", "explicit bounds win over the reversal defaults"
doAssert render("[][::-1]") == "[]", "a reversed empty sequence is empty"
doAssert render("'abc'[2:1]") == "", "a string stop before the start is empty, not reversed"
doAssert render("'h\u00e9llo'[99:]") == "", "a string start past the tail is empty"
doAssert render("'abcdef'[1:5:2]") == "bd", "string slicing counts codepoints"
block zeroStepIsAnError:
  var reported = ""
  try:
    discard render("[1, 2, 3][::0]")
  except JinjaError as e:
    reported = e.what
  doAssert reported.len > 0, "a zero slice step must raise, not loop forever"

# Dict literals inside an expression
# ---------------------------------------------------------------------------

doAssert render("{'role': 'user'}['role']") == "user"
doAssert render("{'role': 'user'}") == "{'role': 'user'}"
doAssert render("[{'role': 'user'}] | length") == "1"

# Multibyte string edge cases beyond the recorded template forms
# ---------------------------------------------------------------------------

doAssert render("'h\u00e9llo'[1:3]") == "\u00e9l", "a step-1 string slice copies its byte window"
doAssert render("'h\u00e9llo'[::-1]") == "oll\u00e9h", "a reversed string slice walks codepoints"
doAssert render("'h\u00e9llo'[-2]") == "l", "a negative string subscript steps back over the multibyte rune"
doAssert render("'h\u00e9llo'[::-2]") == "olh", "a negative string step visits every second codepoint"
doAssert render("'a--b'.split('--')") == "['a', 'b']", "a multi-byte separator splits between occurrences"
doAssert render("'a-b'.split('--')") == "['a-b']", "a first-byte match that is no separator keeps the part whole"
doAssert render("'ab'.split('')") == "['a', 'b']", "an empty separator splits per codepoint"
doAssert render("'a b c'.split()") == "['a', 'b', 'c']", "the default separator is one space"
doAssert render("'\u00c9\u00c0'|lower") == "\u00c9\u00c0", "a non-ascii letter passes through lower unchanged"
doAssert render("'\u00c9\u00c0'|upper") == "\u00c9\u00c0", "a non-ascii letter passes through upper unchanged"
doAssert render("007") == "7", "a leading-zero literal parses as an integer"
doAssert render("1.5e3") == "1500.0", "an exponent literal parses as a float"
doAssert render("{'k': strVal}", ctx(("strVal", strVal("a'b\nc")))) == "{'k': 'a\\'b\\nc'}",
    "a container repr escapes quotes and control characters"
doAssert renderStmt("{{ strftime_now('%Y-%m-%d %H:%M:%S %j') }}") == "1970-01-01 00:00:00 1",
    "the epoch renders through the hand-rolled civil conversion"
doAssert renderStmt("{{ strftime_now('%Y-%m-%d %H:%M:%S %j') }}", undefinedVal(), 951782400.0) ==
    "2000-02-29 00:00:00 60", "a leap-day epoch renders the day of year"
doAssert renderStmt("{{ strftime_now('%Y-%m-%d %H:%M:%S %j') }}", undefinedVal(), 4107542400.0) ==
    "2100-03-01 00:00:00 60", "a non-leap century renders the day of year"

echo "t_expr: expression tier ok"
