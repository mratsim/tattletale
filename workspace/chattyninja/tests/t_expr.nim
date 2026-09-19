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
## Feature-skipped branches are witnessed, not sampled. When `raise_exception`
## sits in a branch the engine must not enter, a re-run or an un-skipped operand
## raises it, which a value comparison cannot see.
##
## Run:
##   $ ./workspace/chattyninja/run_tests.sh t_expr

import std/strutils
import cnj_types, cnj_values, chattyninja

proc render(expr: string, ctx = Value(kind: vkUndefined)): string =
  ## Renders one expression through `renderToString`.
  renderToString("{{ " & expr & " }}", ctx)

proc renderStmt(stmt: string, ctx = Value(kind: vkUndefined)): string =
  ## Renders one statement through `renderToString`.
  renderToString(stmt, ctx)

func ctx(pairs: varargs[(string, Value)]): Value =
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

# Registries: filters and tests dispatch by name
# ---------------------------------------------------------------------------

doAssert render("[1, 2] | length") == "2"
doAssert render("'  pad  ' | trim | length") == "3"
doAssert render("{'a': 1, 'b': 2} | tojson") == "{\"a\": 1, \"b\": 2}"
doAssert render("[1, 2] | tojson") == "[1, 2]"
doAssert render("'x' | tojson") == "\"x\""
doAssert render("'<' | tojson") == "\"\\u003c\"", "tojson applies the Jinja HTML escape"
# Plain `| tojson` renders non-ASCII as raw UTF-8, the recording environment's policy.
doAssert render("'東京' | tojson") == "\"東京\"", "tojson keeps non-ASCII verbatim by default"
doAssert renderStmt("{#- hello -#}\nA{{ name }}",
    ctx(("name", strVal("B")))) == "AB", "comment `-` markers strip the surrounding whitespace runs"
doAssert render("people.tags | join('-')", withPeople) == "x-y"
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
  except NotImplementedError as e:
    reported = e.msg
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
doAssert render("'abcdef'[1:5:2]") == "bd", "string slicing counts codepoints"
block zeroStepIsAnError:
  var reported = ""
  try:
    discard render("[1, 2, 3][::0]")
  except CatchableError as e:
    reported = e.msg
  doAssert reported.len > 0, "a zero slice step must raise, not loop forever"

# Dict literals inside an expression
# ---------------------------------------------------------------------------

doAssert render("{'role': 'user'}['role']") == "user"
doAssert render("{'role': 'user'}") == "{'role': 'user'}"
doAssert render("[{'role': 'user'}] | length") == "1"

echo "t_expr: expression tier ok"
