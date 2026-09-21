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
import cnj_types, jinja_data_model, cnj_parse, cnj_engine

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

# Integer arithmetic checks its result, a value past int64 raising a located `JinjaError`,
# never an uncatchable `OverflowDefect`.
# Python answers with unbounded integers, a value kind without a slot in this tier.
# The raise is the contract here.
doAssert render("9223372036854775807") == "9223372036854775807", "int64 high renders"
doAssert render("0 - 9223372036854775807") == "-9223372036854775807",
    "int64 low renders through a subtraction"
block integerOverflowRaisesLocated:
  var reported = ""
  try:
    discard render("9223372036854775807 + 1")
  except JinjaError as e:
    reported = e.what
  doAssert "integer overflow" in reported, reported
  try:
    discard render("99999999999999999999")
  except JinjaError as e:
    reported = e.what
  doAssert "outside the int64 range" in reported, reported
  try:
    discard render("-x", ctx(("x", intVal(int64.low))))
  except JinjaError as e:
    reported = e.what
  doAssert "integer overflow in unary" in reported, reported

# A digit spelling of exactly 2^63 renders as int64.low under an immediate unary `-`,
# Python's rendering of `-9223372036854775808`. Every other past-int64 magnitude raises,
# the positive spelling and a re-negated one, a `--` shape overflowing unary.
doAssert render("-9223372036854775808") == "-9223372036854775808", "int64 low renders as its literal spelling"
doAssert renderStmt("{% set q = -9223372036854775808 %}{{ q }}") == "-9223372036854775808",
    "int64 low stays an int through a set"
doAssert render("-9223372036854775808 == x", ctx(("x", intVal(int64.low)))) == "True",
    "int64 low compares as the data's int64.low"
try:
  discard render("9223372036854775808")
  doAssert false, "the positive 2^63 spelling did not raise"
except JinjaError as e:
  doAssert "outside the int64 range" in e.what, e.what
try:
  discard render("--9223372036854775808")
  doAssert false, "the re-negated 2^63 spelling did not raise"
except JinjaError as e:
  doAssert "integer overflow in unary" in e.what, e.what

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

# A concat reads plain values in two places, the argument list and a boolean position.
#   `raise_exception` names its message from the materialized text, a condition reading
#   the rendered bytes. Every other non-emit position raises, and a set-bound concat
#   streams when a later emit reaches it.
block concatConsumption:
  try:
    discard render("{'k': 'a' ~ 'b'}")
    doAssert false, "a concat in a dict literal did not raise"
  except JinjaError as e:
    doAssert "emit position" in e.what, e.what
  doAssert renderStmt("{% if 'a' ~ 'b' %}x{% endif %}") == "x",
      "a concat in a condition reads its rendered bytes"
  try:
    discard render("('a' ~ 'b') | tojson")
    doAssert false, "a concat under a filter did not raise"
  except JinjaError as e:
    doAssert "emit position" in e.what, e.what
  doAssert renderStmt("{% set q = 'a' ~ 'b' %}{{ q }}") == "ab",
      "a set-bound concat did not stream on its later emit"
  doAssert renderStmt("{% set q = 'a' ~ 'b' %}{% if q %}x{% endif %}") == "x",
      "a set-bound concat reads its rendered bytes in a condition"
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

# `not` binds looser than the comparisons and membership, tighter than `and`/`or`,
# so `not a == 5` tests `a == 5`. Unary `-` and `+` keep their tighter binding.
doAssert renderStmt("{% set a = 0 %}{% if not a == 5 %}Y{% else %}N{% endif %}") == "Y",
    "`not` spans the comparison it negates"
doAssert renderStmt("{% set xs = [1, 2] %}{% if not 3 in xs %}Y{% else %}N{% endif %}") == "Y",
    "`not` spans the membership test it negates"
doAssert renderStmt("{% set a = 0 %}{% if not a == 5 and true %}Y{% else %}N{% endif %}") == "Y",
    "`and` still binds looser than `not`"
doAssert renderStmt("{% set a = 5 %}{% if -a == -5 %}Y{% else %}N{% endif %}") == "Y",
    "unary `-` keeps its tighter binding under a comparison"

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
# the branch, matching upstream Jinja.
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

# A macro call evaluates once at its call site, in `{% set %}` and the for-iterable
# position like in a condition, matching upstream Jinja.
doAssert renderStmt("{% set ns = namespace(c = 0) %}{% macro m() %}{% set ns.c = ns.c + 1 %}{% endmacro %}" &
    "{% set x = m() %}{{ ns.c }}") == "1",
    "a macro call bound by `set` ran exactly once at the set"
doAssert renderStmt("{% set ns = namespace(n = 0) %}{% macro m() %}{% set ns.n = ns.n + 1 %}X{% endmacro %}" &
    "{% set ns.c = m() %}{{ ns.n }}{{ ns.c }}") == "1X",
    "a namespace-attribute `set` of a macro call ran the body at the set"
doAssert renderStmt("{% macro m() %}ab{% endmacro %}{% for c in m() %}[{{ c }}]{% endfor %}") == "[a][b]",
    "a macro call as the for iterable iterates its rendered bytes"

# An empty macro body emits nothing. Its streamed call closes at once, the whole render
# tail delivering, and its forced capture is the empty string.
doAssert renderStmt("{% macro m() %}{% endmacro %}A{{ m() }}B") == "AB",
    "an empty macro body's streamed call delivers the render tail"
doAssert renderStmt("{% macro m() %}{% endmacro %}A{% if m() %}X{% else %}B{% endif %}C") == "ABC",
    "an empty macro body's forced capture is the empty string"

# Macro-argument binding raises where upstream raises:
#   a positional past the parameter list, a positional after a keyword,
#   a keyword naming no parameter, a keyword repeating a bound one.
# Well-formed positional, keyword and default binding keep rendering.
block macroArgBinding:
  proc reportedOf(src: string): string =
    try:
      discard renderStmt(src)
      doAssert false, "a misplaced macro argument did not raise: " & src
    except JinjaError as e:
      result = e.what
  doAssert "positional argument follows a keyword argument" in
      reportedOf("{% macro m(x) %}<{{ x }}>{% endmacro %}{{ m(x=1, 2) }}"),
      "positional after keyword"
  doAssert "takes no keyword argument" in
      reportedOf("{% macro m(x) %}<{{ x }}>{% endmacro %}{{ m(1, y=2) }}"),
      "unknown keyword"
  doAssert "got multiple values for argument" in
      reportedOf("{% macro m(x) %}<{{ x }}>{% endmacro %}{{ m(1, x=2) }}"),
      "keyword repeating a positionally bound parameter"
  doAssert "takes at most 1 positional argument" in
      reportedOf("{% macro m(x) %}<{{ x }}>{% endmacro %}{{ m(1, 2) }}"),
      "excess positional"
  doAssert renderStmt("{% macro m(x, y = 9) %}<{{ x }}{{ y }}>{% endmacro %}{{ m(1, y=2) }}") == "<12>",
      "positional, keyword and default binding still render"

# Filter, method and test calls share the macro carrier's argument order, so a positional
# after a keyword binds under no parameter and drops, the callee raising located.
# `getArg` binds the pos-th positional by its own count, and the raise keeps a carrier
# mis-ordered this way from ever reaching one.
block calleeArgOrder:
  proc reportedOf(src: string): string =
    try:
      discard renderStmt(src)
      doAssert false, "a misplaced callee argument did not raise: " & src
    except JinjaError as e:
      result = e.what
  doAssert "positional argument follows a keyword argument" in
      reportedOf("{{ [1, 2, 3] | join(sep = ',', '-') }}"), "positional after keyword in a filter call"
  doAssert "positional argument follows a keyword argument" in
      reportedOf("{{ 'a,b,c'.split(sep = ',', 1) | join('|') }}"), "positional after keyword in a method call"
  doAssert renderStmt("{{ [1, 2, 3] | join(',') }}") == "1,2,3",
      "well-ordered filter arguments keep binding"
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

# A continue or break crossing a `{% generation %}` row must not disturb the scope stack:
# the generation row records the enclosing scope's mark without pushing one, so the unwind
# pops nothing for it and every binding below survives.
block controlThroughGeneration:
  doAssert renderStmt(
      "{% set x = 'keep' %}{% for x in [1, 2] %}{% continue %}{% endfor %}{{ x }}") == "keep",
      "control case, a plain continue keeps the outer binding"
  doAssert renderStmt(
      "{% set x = 'keep' %}{% for x in [1, 2] %}{% generation %}{% continue %}{% endgeneration %}" &
      "{% endfor %}{{ x }}") == "keep",
      "a continue crossing a generation row pops no scope, the outer binding intact"

# A continue has no meaning at a macro-call boundary. A break stops there, a continue has
# no loop to re-enter through the boundary, so it raises located like a no-for break.
block continueAtMacroBoundary:
  var reported = ""
  var at = -1
  try:
    discard renderStmt("{% macro m() %}{% continue %}{% endmacro %}{{ m() }}")
    doAssert false, "a continue inside a macro body closed the body instead of raising"
  except JinjaError as e:
    reported = e.what
    at = e.offset
  doAssert "continue" in reported and "outside every" in reported, reported
  doAssert at >= 15 and at < 26, "the boundary raise located inside the continue tag: " & $at
  try:
    discard renderStmt(
        "{% macro m() %}{% continue %}{% endmacro %}" &
        "{% for x in [1, 2] %}{{ m() }}{% endfor %}")
    doAssert false, "a continue inside a macro body crossed the boundary into the caller's loop"
  except JinjaError as e:
    doAssert "outside every" in e.what, e.what

# A `{% generation %}` block renders its body byte-for-byte unchanged and records the span
# of the output it produced, byte coordinates into the render, read after the drain.
proc renderWithSpans(src: string): tuple[text: string, spans: seq[tuple[start, stop: int]]] =
  ## Renders whole and returns the bytes plus the driver's recorded generation spans.
  var (tmpl, sym) = parseTemplate(src)
  var d = startRender(tmpl, sym, JinjaVal(kind: vkUndefined))
  result.text = pullAll(d)
  result.spans = d.generationSpans()

doAssert renderWithSpans("{% generation %}A{% endgeneration %}").text == "A",
    "the body renders unchanged"
doAssert renderWithSpans("{% generation %}A{% endgeneration %}").spans == @[(start: 0, stop: 1)],
    "one block, the span over its bytes"
doAssert renderWithSpans("pre{% generation %}A{% endgeneration %}post").spans == @[(start: 3, stop: 4)],
    "the span sits at the body's output position"
block conditionalEntry:
  doAssert renderWithSpans("{% if false %}{% generation %}A{% endgeneration %}{% endif %}").spans.len == 0,
      "an untaken branch records no span"
  doAssert renderWithSpans("{% if true %}{% generation %}A{% endgeneration %}{% endif %}").spans ==
      @[(start: 0, stop: 1)], "a taken branch records one"
  doAssert renderWithSpans("{% if false %}{% generation %}A{% endgeneration %}{% endif %}x").text == "x",
      "the untaken branch emits nothing either"
doAssert renderWithSpans(
    "{% for x in [1, 2] %}{% generation %}{{ x }}{% endgeneration %}{% endfor %}").spans ==
    @[(start: 0, stop: 1), (start: 1, stop: 2)], "one span per block, source order"
doAssert renderWithSpans("{% generation %}{{ 'abc' }}{% endgeneration %}").spans ==
    @[(start: 0, stop: 3)], "the closing position counts the body's bytes"
doAssert renderWithSpans("{% generation %}{% endgeneration %}").spans == @[(start: 0, stop: 0)],
    "an empty body records an empty span"

# A break or continue abandoning a generation body still ran its bytes, the walk closing
# the body's span at the position reached, the span never silently dropped.
doAssert renderWithSpans(
    "{% for x in [1, 2] %}{% generation %}{{ x }}{% continue %}{% endgeneration %}{% endfor %}"
    ).text == "12", "a continue after the body's emit leaves the bytes on"
doAssert renderWithSpans(
    "{% for x in [1, 2] %}{% generation %}{{ x }}{% continue %}{% endgeneration %}{% endfor %}"
    ).spans == @[(start: 0, stop: 1), (start: 1, stop: 2)],
    "a continue abandoning a generation body closes its span at the position reached"
doAssert renderWithSpans(
    "{% for x in [1, 2] %}{% generation %}A{% if x == 2 %}{% break %}{% endif %}{% endgeneration %}" &
    "{% endfor %}").text == "AA"
doAssert renderWithSpans(
    "{% for x in [1, 2] %}{% generation %}A{% if x == 2 %}{% break %}{% endif %}{% endgeneration %}" &
    "{% endfor %}").spans == @[(start: 0, stop: 1), (start: 1, stop: 2)],
    "a break abandoning a generation body closes its span at the position reached"

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

# A step magnitude past the walk span visits the start element only, and the walk's
# index arithmetic stays inside int64 however extreme the step, Python's own answer.
doAssert render("'abcdef'[1::9223372036854775807]") == "b",
    "a forward step past the walk span visits the start only"
doAssert render("'abcdef'[:0:-9223372036854775807]") == "f",
    "a backward step past the walk span visits the start only"

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

# Zero-node templates and empty for bodies

doAssert renderStmt("") == "", "a template compiling to zero nodes renders empty"
doAssert renderStmt("{# hi #}") == "", "a comment-only template renders empty"

# An empty body completes a for at set-up, its row and scope closing exactly
# as an exhausted loop closes them:
#   the target and `loop` unbind with the scope, no row stays open for a break
#   to unwind into, and a filter clause still runs per remaining item.
doAssert renderStmt("A{% for x in [1] %}{% endfor %}B") == "AB",
    "an empty for at the top level advances past its successor"
doAssert renderStmt("{% for x in [1] %}{% endfor %}{{ x }}") == "",
    "an empty for unbinds its target with its scope, as an exhausted loop does"
doAssert renderStmt(
    "{% for x in [1, 2] %}{% for y in [1] %}{% endfor %}{% endfor %}Z") == "Z",
    "an empty for nested in a for completes at set-up, the enclosing loop advancing"
doAssert renderStmt(
    "{% macro m() %}{% for y in [1] %}{% endfor %}{% endmacro %}{{ m() }}A") == "A",
    "an empty for inside a macro body closes back on the definition node"
doAssert renderStmt("{% for x in [1, 2, 3] if x > 1 %}{% endfor %}ok") == "ok",
    "an empty body still runs its filter clause over every remaining item"
try:
  discard renderStmt("{% for x in [1] %}{% endfor %}{% break %}")
  doAssert false, "a break after an empty for found a leaked for-row"
except JinjaError as e:
  doAssert "`{% break %}` is outside any `{% for %}`" in e.what, e.what
# A filter clause runs on the loop's advance path, item 0 bound at set-up and kept
# unconditionally in either body shape, so a raising clause raises exactly where
# a non-empty body's first advance would raise.
try:
  discard renderStmt("{% for x in [1, 0] if raise_exception('clause ran') %}{% endfor %}")
  doAssert false, "an empty body skipped its filter clause"
except JinjaError as e:
  doAssert "clause ran" in e.what, e.what

# Depth caps:
#   every recursion leg counts toward ExprDepthCap, dry walks and unary chains included.
#   Deep nesting raises located before the C stack runs out.

doAssert render("not not not not not not not not not not 1") == "True",
    "a unary chain within the cap renders"
try:
  discard render("not ".repeat(40000) & "1")
  doAssert false, "a unary chain past the cap rendered instead of raising"
except JinjaError as e:
  doAssert "ExprDepthCap" in e.what, e.what
try:
  discard render("0 and " & "(".repeat(25000) & "1" & ")".repeat(25000))
  doAssert false, "a dry walk over a skipped operand ran uncapped"
except JinjaError as e:
  doAssert "ExprDepthCap" in e.what, e.what

# Ternaries leave no depth behind, only the real nesting depth spending budget:
#   chained and sequenced ternaries render well within the cap.
var chained = ""
for k in 1 .. 20:
  chained.add $k & " if z else "
chained.add "21"
doAssert render(chained) == "21", "20 chained ternaries render within the cap"

var sequenced = ""
for k in 1 .. 25:
  if k > 1:
    sequenced.add " ~ "
  sequenced.add "(9 if z else 8)"
doAssert render(sequenced) == "8".repeat(25),
    "25 ternaries sequenced on one cursor render within the cap"

# Lazy range, one element bound at construction, so no consumer loops, materializes,
# drains or scans past RangeElemCap, and the span arithmetic never wraps:
#   range(0, int64 high) answers its count as a located raise, not an OverflowDefect.

doAssert render("range(0, 1000000) | length") == "1000000",
    "a range at the cap answers its count"
try:
  discard render("range(0, 1000001) | length")
  doAssert false, "a range past the cap answered instead of raising"
except JinjaError as e:
  doAssert "RangeElemCap" in e.what, e.what
try:
  discard render("range(0, 9223372036854775807) | length")
  doAssert false, "an overflow-scale range raised a defect instead of the cap"
except JinjaError as e:
  doAssert "RangeElemCap" in e.what, e.what
try:
  discard renderStmt("{% for x in range(0, 4611686018427387904) %}{{ x }}{% endfor %}")
  doAssert false, "an unbounded range entered a for loop"
except JinjaError as e:
  doAssert "RangeElemCap" in e.what, e.what
try:
  discard render("range(0, 2000001) | list | length")
  doAssert false, "a range past the cap materialized"
except JinjaError as e:
  doAssert "RangeElemCap" in e.what, e.what
try:
  discard render("range(0, 2000001) | tojson")
  doAssert false, "a range past the cap drained to the serializer"
except JinjaError as e:
  doAssert "RangeElemCap" in e.what, e.what

# Range membership and equality answer arithmetically, no scan:
#   direction-aware bounds plus an exact stride split for ints, the element whose
#   rounding matches a float needle, and progression identity for `==`.
doAssert render("5 in range(0, 10)") == "True"
doAssert render("5.0 in range(0, 10)") == "True", "a float needle matches by rounding"
doAssert render("5 in range(0, 10, 2)") == "False", "an off-stride int is absent"
doAssert render("5 in range(9, -1, -1)") == "True", "a backward range contains by its stride"
doAssert render("6.0 in range(9, -1, -1)") == "True",
    "a float needle matches on a backward walk, its monotone direction honored"
doAssert render("5.0 in range(9, -1, -1)") == "True", "a float needle hits a backward element"
doAssert render("2.5 in range(9, -1, -1)") == "False",
    "an off-stride float is absent from a backward range"
try:
  discard render("5 in range(0, 4611686018427387904)")
  doAssert false, "an overflow-scale membership did not raise"
except JinjaError as e:
  doAssert "RangeElemCap" in e.what, e.what
try:
  discard render("range(0, 4611686018427387904) == range(0, 4611686018427387903)")
  doAssert false, "an overflow-scale equality did not raise"
except JinjaError as e:
  doAssert "RangeElemCap" in e.what, e.what
try:
  discard render("range(-9223372036854775807, 0) | length")
  doAssert false, "a negative-extreme span did not raise"
except JinjaError as e:
  doAssert "RangeElemCap" in e.what, e.what
doAssert render("0 in range(0, 0)") == "False", "an empty range contains nothing"
doAssert render("range(0, 6, 2) == range(0, 5, 2)") == "True",
    "equal progressions compare equal past their differing bounds"
doAssert render("range(3, 2, -5) == range(3, 2, -1)") == "True",
    "backward singletons compare equal by element, not stride"
doAssert render("range(0, 6, 2) == range(0, 6, 3)") == "False",
    "same bounds, different strides compare unequal"
doAssert render("range(0, 3) == [0, 1, 2]") == "False",
    "a range never equals a sequence of the same elements"
doAssert render("range(0, 3) | list | length") == "3"
doAssert render("range(0, 3) | tojson") == "[0, 1, 2]"
doAssert renderStmt("{% for x in range(1, 4) %}{{ x }}{% endfor %}") == "123"

# Hostile value depth raises located, from both container walkers:
#   equality and membership count one level per container entered toward
#   ValueDepthCap, so neither runs off the C stack.

proc nestedVal(depth: int): JinjaVal =
  var v = intVal(1)
  for _ in 1 ..< depth:
    v = seqVal(@[v])
  v

proc ctxPair(depth: int): JinjaVal =
  var d = DictVal()
  dictSet(d, "a", nestedVal(depth))
  dictSet(d, "b", nestedVal(depth))
  dictVal(d)

try:
  discard render("a == b", ctxPair(1200))
  doAssert false, "deep data compared instead of raising"
except JinjaError as e:
  doAssert "ValueDepthCap" in e.what, e.what
try:
  discard render("b in a", ctxPair(1200))
  doAssert false, "deep data scanned instead of raising"
except JinjaError as e:
  doAssert "ValueDepthCap" in e.what, e.what
doAssert render("a == b", ctxPair(50)) == "True",
    "data nesting well within the cap compares"

# Cyclic value graphs raise located at both the walkers and the serializer,
# rendering only shapes within the cap:
#   a namespace holding itself is valid Jinja, its comparison and its rendering
#   must terminate, deep-but-acyclic data still renders.

doAssert renderStmt("{% set ns = namespace() %}{% set ns.a = ns %}{{ ns == ns }}") == "True",
    "a namespace equals itself, the shared payload answering by identity"
try:
  discard renderStmt("{% set ns = namespace() %}{% set ns.a = ns %}{{ ns }}")
  doAssert false, "a cyclic namespace rendered unbounded"
except JinjaError as e:
  doAssert "ValueDepthCap" in e.what, e.what
try:
  discard renderStmt("{% set ns = namespace() %}{% set ns.a = ns %}{{ ns | tojson }}")
  doAssert false, "a cyclic namespace serialized unbounded"
except JinjaError as e:
  doAssert "ValueDepthCap" in e.what, e.what
block:
  var v = intVal(1)
  for _ in 1 .. 500:
    v = seqVal(@[v])
  doAssert renderStmt("{{ v }}", (proc: JinjaVal =
    var d = DictVal()
    dictSet(d, "v", v)
    dictVal(d))()) == ("[").repeat(500) & "1" & ("]").repeat(500),
    "acyclic data within the cap renders"

echo "t_expr: expression tier ok"
