# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Parse shape of the chattyninja arena, its links, and the whitespace-resolved verbatim spans.
##
## Asserted against `corpus/deepseekv2lite`, the cheapest byte-exact template, plus
## fixtures carrying the whitespace-control and comment forms the corpus uses.
##
## Run:
##   $ nim test_chattyninja

import std/[os, strutils, sequtils]
import cnj_types, jinja_data_model, cnj_parse
import workspace/data_structures/src/small_seqs

const root = currentSourcePath().parentDir
let src = readFile(root / "corpus" / "deepseekv2lite" / "deepseekv2lite.jinja")
let (tmpl, symbols) = parseTemplate(src)
let nodes = tmpl.nodes

# Node count and the kind sequence in index order. A construct is appended after its body, so
# an `if` or `for` node sits above the nodes it dispatches into. Arena index order is source order,
# and a construct node is reserved before its body is walked, so index 0 is the outermost
# `{% if %}` rather than a node from inside one of its bodies.
const wantKinds = [
  nkIf, nkSet, nkEmit, nkFor, nkIf, nkEmit, nkIf, nkEmit, nkIf, nkEmit, nkIf, nkEmit
]
doAssert nodes.len == wantKinds.len,
    "deepseekv2lite node count: " & $nodes.len & " want " & $wantKinds.len
for i, k in wantKinds:
  doAssert nodes[i].kind == k, "node " & $i & " is " & $nodes[i].kind & " want " & $k

# Names are interned rather than stored per node, and a for-header carries its bindings
# inside its own payload slots past position 6.
doAssert symbols.names.len == 3, "interned names: " & $symbols.names.len
doAssert nodes[3].kind == nkFor and int(nodes[3].slots.len) == 8,
    "nkFor carries its seven fixed slots plus one target id"
doAssert symbols.names[nodes[3].loopName] == "loop", "nkFor binds the interned `loop` name"
doAssert symbols.names[nodes[3].slots[7]] == "message", "nkFor carries its target name id"
doAssert src[nodes[3].lo ..< nodes[3].hi].strip == "messages",
    "nkFor keeps the iterable as a text span"

# Spans and links stay inside the artifact that owns them. `succ` is a link on every kind,
# `child` only on the kinds that carry a body.
for i, n in nodes:
  doAssert n.hi >= n.lo and n.hi <= src.len.int32, "node " & $i & " has a bad span"
  doAssert n.succ == NoLink or (n.succ >= 0'i32 and n.succ < nodes.len.int32),
      "node " & $i & " has an out-of-range successor"
  if n.kind in {nkIf, nkFor}:
    doAssert n.child == NoLink or (n.child >= 0'i32 and n.child < nodes.len.int32),
        "node " & $i & " has an out-of-range body link"

# An expression is never a node. Every `nkEmit` payload is one `lo..hi` span and nothing else,
# the three-slot shape with no subgraph below it.
for i, n in nodes:
  if n.kind == nkEmit:
    doAssert int(n.slots.len) == 3, "node " & $i & ": an expression must not become a node"

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

# `nkFor` is the re-entrant case, and the only one in this template:
#   its body terminators come back
# to the for node so iteration advances through the for-row, never through a node field.
doAssert nodes[3].succ == 10'i32, "nkFor exits past endfor"
var reentries = 0
for n in nodes:
  if n.succ == 3'i32:
    inc reentries
doAssert reentries >= 3, "nkFor body must loop back to the for node, saw " & $reentries

# The arena entry must be the outermost construct. Node 0 is a construct, and no body
# node is reachable from index 0 by `succ` alone without going through the construct
# that owns it.
doAssert nodes[0].kind in {nkIf, nkFor, nkEmit, nkVerbatim, nkSet}, "entry is a construct"
doAssert nodes[0].kind == nkIf, "the template's outermost construct is the first `if`"

# Whitespace control and comments are resolved at parse time, so an `nkVerbatim` span is final text:
# it holds no delimiter, and no indentation or newline the policy removes.
const wsFixture = "A   {%- if x -%}\n   hello   \n{%- endif -%}   B\n"
let (wn, _) = parseTemplate(wsFixture)
var verbatim = newSeq[string]()
for n in wn.nodes:
  doAssert n.kind == nkVerbatim or n.kind == nkIf, "unexpected kind " & $n.kind
  if n.kind == nkVerbatim:
    let t = wsFixture[n.lo ..< n.hi]
    verbatim.add t
    doAssert "{%" notin t and "{{" notin t and "{#" notin t,
        "nkVerbatim carries a delimiter: " & t.escape
doAssert verbatim == @["A", "hello", "B"], "resolved verbatim: " & $verbatim
doAssert wn.nodes.filterIt(it.kind == nkIf).len == 1, "one if node for one {% if %}"

# A comment leaves no node of its own, erasure rather than a construct.
# Surrounding runs keep their own borrowed spans, so the arena points into the template
# instead of copying merged text.
const cSrc = "a{# dropped #}b"
let (cn, _) = parseTemplate(cSrc)
doAssert cn.nodes.allIt(it.kind == nkVerbatim), "a comment became a node kind"
doAssert cn.nodes.mapIt(cSrc[it.lo ..< it.hi]).join == "ab",
    "a comment is erased, not preserved: " & $cn.nodes.mapIt(cSrc[it.lo ..< it.hi])

# A comment carries the block tag's whitespace rules. `lstrip_blocks` strips blanks before a plain
# comment on its line, trim_blocks drops one newline after it, and `{#-` / `-#}` strip the whole
# whitespace run before / after the comment.
const cmSrc = "A\n   {# plain #}\nB{#- gone -#}  C"
let (cmn, _) = parseTemplate(cmSrc)
doAssert cmn.nodes.allIt(it.kind == nkVerbatim), "a comment became a node kind"
doAssert cmn.nodes.mapIt(cmSrc[it.lo ..< it.hi]).join == "A\nBC",
    "comment whitespace rules: " & $cmn.nodes.mapIt(cmSrc[it.lo ..< it.hi])

# Only a trailing newline is dropped, so interior and trailing spacing survive byte for byte:
# final does not mean trimmed.
const pSrc = "keep  me  "
let (pn, _) = parseTemplate(pSrc)
doAssert pn.nodes.len == 1 and pn.nodes[0].kind == nkVerbatim
doAssert pSrc[pn.nodes[0].lo ..< pn.nodes[0].hi] == "keep  me  ",
    "spacing must survive: " & pSrc[pn.nodes[0].lo ..< pn.nodes[0].hi].escape

# Declared-gap constructs parse into nodes:
#   the gap raises at render when a row reaches the construct, per row, and the suite's parse
#   stays alive. The step tier carries the gap for `nkBreak`, `nkSetBlock` and `nkGeneration`,
#   the parse tier building the nodes with spans and links so that raise stays located.
const breakSrc = "{% for x in xs %}{% break %}{% endfor %}"
let (bn, _) = parseTemplate(breakSrc)
doAssert bn.nodes.mapIt($it.kind).join(",") == "nkFor,nkBreak",
    "nkBreak builds after its for: " & $bn.nodes.mapIt($it.kind)
doAssert bn.nodes[1].lo.int == 20 and bn.nodes[1].hi.int == 26,
    "nkBreak spans the keyword token"
doAssert bn.nodes[1].succ == 0 and bn.nodes[0].child == 1,
    "the break sits inside the for body, closing back on it"
let (cnt, _) = parseTemplate("{% for x in xs %}{% continue %}{% endfor %}")
doAssert cnt.nodes[1].kind == nkBreak, "`{% continue %}` parses to the same nkBreak node"

# A break outside every macro and every for is malformed and raises located. A macro body
# defers the check to its call site and parses.
try:
  discard parseTemplate("{% break %}")
  doAssert false, "a top-level break parsed instead of raising"
except JinjaError as e:
  doAssert "`{% break %}` is outside any `{% for %}`" in e.what, e.what
  doAssert e.offset == 3 and e.span == 5, "the raise locates the keyword: " &
      $e.offset & "+" & $e.span
let (mo, _) = parseTemplate("{% macro m() %}{% break %}{% endmacro %}{% for x in xs %}{{ m() }}{% endfor %}")
doAssert mo.nodes.anyIt(it.kind == nkBreak),
    "a break inside a macro body parses, the call site deciding the enclosure"

const setBlockSrc = "{% set x %}body{% endset %}"
let (sn, ss) = parseTemplate(setBlockSrc)
doAssert sn.nodes.mapIt($it.kind).join(",") == "nkSetBlock,nkVerbatim",
    "the set-block body is the capture body: " & $sn.nodes.mapIt($it.kind)
doAssert sn.nodes[0].lo.int == 3 and sn.nodes[0].hi.int == 9,
    "the set-block node spans the tag's keyword and target"
doAssert ss.names[sn.nodes[0].setTarget] == "x", "the set-block target is interned"
doAssert sn.nodes[0].child == 1 and sn.nodes[1].succ == 0,
    "the set-block body ends back on the block node"

const genSrc = "{% generation %}A{% endgeneration %}"
let (gn, _) = parseTemplate(genSrc)
doAssert gn.nodes.mapIt($it.kind).join(",") == "nkGeneration,nkVerbatim",
    "the generation body is the span body: " & $gn.nodes.mapIt($it.kind)
doAssert gn.nodes[0].lo.int == 3 and gn.nodes[0].hi.int == 14,
    "the generation node spans the opening tag's keyword"
doAssert gn.nodes[0].child == 1 and gn.nodes[1].succ == 0,
    "the generation body ends back on the generation node"

echo "t_parse: ", nodes.len, " nodes, ", symbols.names.len, " interned names, verbatim spans final"
