# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# The chattyninja parser turns template text into a flat node arena.
#
# Lifecycle:
#
# - tokenise:
#   one scan splits the text into runs and tags, applying the whitespace policy
#   in place, so no leading or trailing whitespace reaches a node and no node carries a whitespace flag
# - emit:
#   a recursive descent over the tag list appends nodes in source order, so arena order
#   equals source order and `succ` stays a link field rather than a program counter
# - backpatch:
#   each construct returns the nodes whose successor is unresolved and the enclosing
#   construct resolves them, which is how an `if` chain's bodies terminate past the whole chain
#
# | Rule                 | Effect                                                                                            |
# | -------------------- | ------------------------------------------------------------------------------------------------- |
# | `trim_blocks`        | one newline directly after a block or comment tag's close is dropped                              |
# | `lstrip_blocks`      | spaces and tabs immediately before `{%` or `{#` are dropped when only they precede it on the line |
# | `{{- ` `{%- ` `{#- ` | every whitespace run before the tag is stripped                                                   |
# | `- }}` `-%}` `-#}`   | every whitespace run after the tag is stripped                                                    |
#
# - template text is literal, with no backslash escaping of delimiters. `{% raw %}` holds its body verbatim
# - an expression stays template text:
#   no expression becomes a node, and each `{{ x }}` leaves an `nkEmit` carrying the `x` span
#
# An unterminated construct is a `TemplateError` naming the construct and its byte offset, so
# a truncated template fails at load instead of rendering short.

import std/[strbasics, strutils]
import cnj_errors, cnj_strbuf, cnj_types, cnj_values
import workspace/data_structures/src/small_seqs

func at(s: string, prefix: string, i: int): bool =
  ## Reports whether `prefix` occurs at `i`.
  s.len - i >= prefix.len and cmpMem(s[i].addr, prefix[0].addr, prefix.len) == 0

type
  TagKind = enum
    tkText, tkVariable, tkBlock

  Tag = object
    kind: TagKind
    lo, hi: int # text run to emit, whitespace-resolved (unused for tag rows)
    tLo, tHi: int # inside of the delimiters, with any `-` marker stripped

  P = object
    ## Parse state:
    ##   the tag list, an index into it, the arena under construction and the tables.
    src: string
    tags: seq[Tag]
    i: int
    tables: Tables
    nodes: seq[Node]

  Head = object
    ## A constructed run of nodes:
    ##   its entry point, and the nodes whose successor is still unresolved.
    head: int32
    tails: seq[int32]

proc mkNode(kind: NodeKind, slots: varargs[int32]): Node =
  ## Builds one node, appending its payload slots in layout order. The enclosing construct
  ## backpatches `succ` and `child` from `noLink` through `patch` or a direct slot write.
  result.kind = kind
  for s in slots:
    result.slots.add s

# Tokenise
# ---------------------------------------------------------------------------

func atLineStart(src: string, at: int): bool =
  ## Reports whether only spaces and tabs separate `at` from the preceding newline, the `lstrip_blocks` test.
  var i = at - 1
  while i >= 0 and (src[i] == ' ' or src[i] == '\t'):
    dec i
  i < 0 or src[i] == '\n'

func findTagClose(src: string, at, stop: int, close: string): int =
  ## Returns the offset of `close` at or after `at`, skipping quoted literals so a closing marker
  ## inside a string does not end the construct. Raises when the construct is never closed.
  var i = at
  var q: char = '\0'
  while i + close.len <= stop:
    if q != '\0':
      if src[i] == '\\':
        inc i
      elif src[i] == q:
        q = '\0'
    elif src[i] in {'\'', '"'}:
      q = src[i]
    elif at(src, close, i):
      return i
    inc i
  -1

func nextOpen(src: string, at, stop: int): int =
  ## Returns the offset of the next `{{`, `{%` or `{#`, or `stop` when there is none.
  var i = at
  while i < stop:
    if at(src, "{{", i) or at(src, "{%", i) or at(src, "{#", i):
      return i
    inc i
  stop

proc tokenize(src: string, stop: int): seq[Tag] =
  ## Splits the template into text runs and tags with every whitespace rule applied.
  var runStart = 0
  var i = 0
  var pendBr = false # trim_blocks drops one newline at the next run's start
  while true:
    let openAt = nextOpen(src, i, stop)
    var lo = runStart
    var hi = openAt
    if pendBr and lo < hi and src[lo] == '\n':
      inc lo
    var stripAfter = false
    var kind = tkText
    var tLo = 0
    var tHi = 0
    var afterTag = stop # offset just past the tag's closing delimiter
    var isRaw = false
    if openAt < stop:
      if at(src, "{#", openAt):
        let c = findTagClose(src, openAt + 2, stop, "#}")
        if c < 0:
          raise err("unclosed comment opened at byte " & $openAt)
        # The comment's own body is erased, but the text run before it is real output and must
        # be emitted under the block tag's whitespace rules:
        #   `{#-` strips the run before the tag, `lstrip_blocks` the blanks preceding it
        #   on its line, `-#}` the run after the tag, and trim_blocks one newline after it
        let afterTag = c + 2
        var innerLo = openAt + 2
        var innerHi = c
        let stripBefore = innerLo < innerHi and src[innerLo] == '-'
        let stripAfter = innerHi > innerLo and src[innerHi - 1] == '-'
        if stripBefore:
          inc innerLo
        if stripAfter:
          dec innerHi
        if stripBefore:
          while lo < hi and src[hi - 1] in wsSpace:
            dec hi
        elif atLineStart(src, openAt):
          while lo < hi and src[hi - 1] in {' ', '\t'}:
            dec hi
        if lo < hi:
          result.add Tag(kind: tkText, lo: int32 lo, hi: int32 hi, tLo: 0, tHi: 0)
        if stripAfter:
          var j = afterTag
          while j < stop and src[j] in wsSpace:
            inc j
          i = j
        else:
          i = afterTag
        runStart = i
        pendBr = true
        continue
      let isVar = at(src, "{{", openAt)
      # The close is the bare delimiter. `-%}` carries no space before `%}`, so a `" %}"` marker
      # would skip every whitespace-controlled tag.
      let close = if isVar: "}}" else: "%}"
      let bodyStart = openAt + 2
      let c = findTagClose(src, bodyStart, stop, close)
      if c < 0:
        raise err("unclosed " & (if isVar: "`{{`" else: "`{%`") & " opened at byte " & $openAt)
      afterTag = c + 2
      var innerLo = bodyStart
      var innerHi = c
      let stripBefore = innerLo < innerHi and src[innerLo] == '-'
      if stripBefore:
        inc innerLo
      stripAfter = innerHi > innerLo and src[innerHi - 1] == '-'
      if stripAfter:
        dec innerHi
      if not isVar:
        var k = innerLo
        while k < innerHi and src[k] in wsSpace:
          inc k
        isRaw = at(src, "raw", k) and (k + 3 >= innerHi or src[k + 3] notin wsNameChars)
      if isRaw:
        # `{% raw %}` holds its body verbatim:
        #   one text run up to `{% endraw %}`.
        let e = findTagClose(src, c + 2, stop, "endraw %}")
        if e < 0:
          raise err("unclosed `{% raw %}` opened at byte " & $openAt)
        var rawLo = c + 2
        var rawHi = e - 2
        if rawHi > rawLo and src[rawHi - 1] == '-':
          dec rawHi
        if rawLo < rawHi and src[rawLo] in {' ', '\t'} and atLineStart(src, rawLo):
          while rawLo < rawHi and src[rawLo] in {' ', '\t'}:
            inc rawLo
        result.add Tag(kind: tkText, lo: int32 rawLo, hi: int32 rawHi, tLo: 0, tHi: 0)
        i = e + "endraw %}".len
        runStart = i
        pendBr = true
        continue
      kind = if isVar: tkVariable else: tkBlock
      tLo = innerLo
      tHi = innerHi
      if stripBefore:
        while lo < hi and src[hi - 1] in wsSpace:
          dec hi
      elif kind == tkBlock and atLineStart(src, openAt):
        while lo < hi and src[hi - 1] in {' ', '\t'}:
          dec hi
    if lo < hi:
      result.add Tag(kind: tkText, lo: int32 lo, hi: int32 hi, tLo: 0, tHi: 0)
    if openAt >= stop:
      break
    result.add Tag(kind: kind, lo: 0, hi: 0, tLo: int32 tLo, tHi: int32 tHi)
    if stripAfter:
      var j = afterTag
      while j < stop and src[j] in wsSpace:
        inc j
      i = j
    else:
      i = afterTag
    runStart = i
    pendBr = kind == tkBlock
  # The final newline of a template is dropped, matching the upstream `rstrip("\n")` before compile.
  var last = result.len - 1
  while last >= 0 and result[last].kind == tkText:
    var t = result[last]
    while t.hi > t.lo and src[t.hi - 1] == '\n':
      dec t.hi
    result[last] = t
    if t.hi > t.lo:
      break
    dec last
  result.setLen(last + 1)

# Emit
# ---------------------------------------------------------------------------

func intern(p: var P, name: openArray[char]): int32 =
  ## Interns `name` in parse order, scope lookup then a byte compare against one interned name
  ## instead of a string compare against a live string. A carried name allocates nothing, a new
  ## one copying exactly once into `Tables.names`.
  let got = findName(p.tables, name)
  if got != noLink:
    return got
  var interned: string
  interned.add name
  result = int32 p.tables.names.len
  p.tables.names.add interned

func addNode(p: var P, n: sink Node): int32 =
  ## Appends one node and returns its arena index.
  result = int32 p.nodes.len
  p.nodes.add n

func patch(nodes: var seq[Node], idx, target: int32) =
  ## Resolves one node's successor, the `slotSucc` position.
  nodes[idx].slots[slotSucc] = target

func tagKeyword(p: P, t: Tag): string =
  ## Returns the leading identifier of a `{% %}` tag.
  var i = t.tLo
  while i < t.tHi and p.src[i] in wsSpace:
    inc i
  let start = i
  while i < t.tHi and p.src[i] in wsNameChars:
    inc i
  spanString(p.src.toOpenArray(start, i - 1))

func afterKeyword(p: P, t: Tag, kwLen: int): int =
  ## Returns the offset just past the tag's keyword and following whitespace, the `tLo` offset
  ## being the inside of the delimiters, so the keyword itself may start after whitespace.
  var i = t.tLo
  while i < t.tHi and p.src[i] in wsSpace:
    inc i
  inc i, kwLen
  while i < t.tHi and p.src[i] in wsSpace:
    inc i
  i

func anyOf(src: openArray[char]; i: int; chars: string): bool =
  ## Returns whether `src[i]` is one of the characters in `chars`.
  for c in chars:
    if src[i] == c: return true
  false

func skipBalanced(p: P, at: int, stop: int, stopAt: string): int =
  ## Returns the offset of any character of `stopAt` at bracket depth zero, skipping quoted literals,
  ## raising when the construct ends first.
  var i = at
  var depth = 0
  var q: char = '\0'
  while i < stop:
    if q != '\0':
      if p.src[i] == '\\':
        inc i
      elif p.src[i] == q:
        q = '\0'
    elif p.src[i] in {'\'', '"'}:
      q = p.src[i]
    elif depth == 0 and anyOf(p.src, i, stopAt):
      return i
    elif p.src[i] in {'(', '['}:
      inc depth
    elif p.src[i] in {')', ']'}:
      dec depth
    inc i
  raise err("expected `" & stopAt & "` before byte " & $stop)

func findKeyword(p: P, at, stop: int, word: string): int =
  ## Returns the offset of the bare word `word` at bracket depth zero, or `stop` when the span holds no such
  ## word. Quoted literals and bracketed subexpressions are skipped, so an `if` inside a string or a call's
  ## argument list is not mistaken for the for-`if` clause.
  var i = at
  var depth = 0
  var q: char = '\0'
  while i < stop:
    if q != '\0':
      if p.src[i] == '\\':
        inc i
      elif p.src[i] == q:
        q = '\0'
    elif p.src[i] in {'\'', '"'}:
      q = p.src[i]
    elif p.src[i] in {'(', '['}:
      inc depth
    elif p.src[i] in {')', ']'}:
      dec depth
    elif depth == 0 and at(p.src, word, i) and
        (i + word.len >= stop or p.src[i + word.len] notin wsNameChars):
      return i
    inc i
  stop

proc parseBody(p: var P, stopKws: openArray[string]): Head
proc parseConstruct(p: var P): Head

proc parseMacroParams(p: var P, t: Tag, at: int, nodeIdx: int32) =
  ## Parses `(a, b = expr, ...)` starting at the open paren and appending one payload triple per
  ## parameter to the `nkMacroDef` node at `nodeIdx`, the interned name then the default span,
  ## `noLink` when absent, defaults staying template text, each evaluated per call after binding.
  ## - evaluation happens after the parameters bind
  var i = at + 1 # past the open paren
  while true:
    while i < t.tHi and p.src[i] in wsSpace:
      inc i
    if i < t.tHi and p.src[i] == ')':
      inc i
      break
    let nameStart = i
    while i < t.tHi and p.src[i] in wsNameChars:
      inc i
    if i == nameStart:
      raise err("macro parameter needs a name at byte " & $nameStart)
    let name = intern(p, p.src.toOpenArray(nameStart, i - 1))
    var k = i
    while k < t.tHi and p.src[k] in wsSpace:
      inc k
    var defLo = noLink
    var defHi = noLink
    if k < t.tHi and p.src[k] == '=':
      inc k
      while k < t.tHi and p.src[k] in wsSpace:
        inc k
      defLo = int32 k
      defHi = int32 skipBalanced(p, k, t.tHi, ",)")
      k = defHi.int
    # One parameter per iteration, the triple `paramNameAt` reads from `macroParamsBase`,
    # name id first, then the default `lo`, then the default `hi`.
    p.nodes[nodeIdx].slots.add name
    p.nodes[nodeIdx].slots.add defLo
    p.nodes[nodeIdx].slots.add defHi
    i = k
    while i < t.tHi and p.src[i] in wsSpace:
      inc i
    if i < t.tHi and p.src[i] == ',':
      inc i
      continue
    if i < t.tHi and p.src[i] == ')':
      inc i
      break
    raise err("macro parameter list is not closed at byte " & $i)

proc parseMacro(p: var P): Head =
  ## `{% macro name(a, b = 1) %} body {% endmacro %}`. The definition binds a value and never
  ## runs the body here. A macro's output is a string, so only a call can run it, to completion.
  ## The body's terminators land back on this node, how a call detects its end.
  let t = p.tags[p.i]
  var i = afterKeyword(p, t, 5) # past the `macro` keyword
  let nameStart = i
  while i < t.tHi and p.src[i] in wsNameChars:
    inc i
  if i == nameStart:
    raise err("`macro` needs a name at byte " & $nameStart)
  let name = intern(p, p.src.toOpenArray(nameStart, i - 1))
  while i < t.tHi and p.src[i] in wsSpace:
    inc i
  if i >= t.tHi or p.src[i] != '(':
    raise err("`macro` parameters are not parenthesised at byte " & $i)
  inc p.i
  let idx = addNode(p, mkNode(nkMacroDef, name, noLink, noLink, noLink))
  parseMacroParams(p, t, i, idx)
  let body = parseBody(p, ["endmacro"])
  if p.i >= p.tags.len or tagKeyword(p, p.tags[p.i]) != "endmacro":
    raise err("`{% macro %}` has no `{% endmacro %}`")
  inc p.i
  p.nodes[idx].slots[slotChild] = body.head
  for x in body.tails:
    patch(p.nodes, x, idx)
  Head(head: idx, tails: @[idx])

proc parseIf(p: var P): Head =
  ## `{% if %} … {% elif %} … {% else %} … {% endif %}`. One node per chain level, every branch body
  ## terminated past the whole chain, which is what makes the node single-entry and single-activation.
  let t = p.tags[p.i]
  let kw = tagKeyword(p, t)
  let condLo = int32 afterKeyword(p, t, kw.len)
  let condHi = t.tHi.int32
  inc p.i
  # The node is reserved before its body is walked, so arena order stays source order and the arena
  # entry stays index 0:
  #   a construct cannot sit below the nodes it dispatches into.
  let idx = addNode(p, mkNode(nkIf, condLo, condHi, noLink, noLink, noLink))
  let body = parseBody(p, ["elif", "else", "endif"])
  p.nodes[idx].slots[slotChild] = body.head
  var tails = @[idx]
  tails.add body.tails
  if p.i >= p.tags.len:
    raise err("unclosed `{% " & kw & " %}`")
  let nxt = p.tags[p.i]
  let nk = tagKeyword(p, nxt)
  case nk
  of "elif":
    let nested = parseIf(p)
    p.nodes[idx].slots[slotAlt] = nested.head
    tails.add nested.tails
  of "else":
    inc p.i
    let eb = parseBody(p, ["endif"])
    p.nodes[idx].slots[slotAlt] = eb.head
    tails.add eb.tails
    if p.i >= p.tags.len or tagKeyword(p, p.tags[p.i]) != "endif":
      raise err("`{% else %}` has no `{% endif %}`")
    inc p.i
  of "endif":
    inc p.i
  else:
    raise err("`{% " & kw & " %}` has no `{% endif %}`")
  Head(head: idx, tails: tails)

proc parseFor(p: var P): Head =
  ## `{% for a, b in expr if cond %} body {% endfor %}`. The header stays one span. Only the target
  ## names and the filter clause are split out, because those are bindings, not computation.
  let t = p.tags[p.i]
  var i = afterKeyword(p, t, 3) # past the `for` keyword
  var names = newSeq[tuple[lo, hi: int]]()
  while true:
    let start = i
    while i < t.tHi and p.src[i] in wsNameChars:
      inc i
    if i == start:
      raise err("`for` needs a target name at byte " & $start)
    names.add (lo: start, hi: i)
    while i < t.tHi and p.src[i] in wsSpace:
      inc i
    if i < t.tHi and p.src[i] == ',':
      inc i
      while i < t.tHi and p.src[i] in wsSpace:
        inc i
      continue
    break
  if not at(p.src, "in", i) or (i + 2 < t.tHi and p.src[i + 2] in wsNameChars):
    raise err("`for` target is not followed by `in` at byte " & $i)
  var j = i + 2
  while j < t.tHi and p.src[j] in wsSpace:
    inc j
  let iterLo = j
  let filterAt = findKeyword(p, iterLo, t.tHi, "if")
  let iterHi = filterAt
  var filterLo = noLink
  var filterHi = noLink
  if filterAt < t.tHi:
    var k = filterAt + 2
    while k < t.tHi and p.src[k] in wsSpace:
      inc k
    filterLo = int32 k
    filterHi = int32 t.tHi
  var targets = newSeq[int32](names.len)
  for k, n in names:
    targets[k] = intern(p, p.src.toOpenArray(n.lo, n.hi - 1))
  let loopId = intern(p, "loop")
  inc p.i
  let idx = addNode(p, mkNode(nkFor, int32 iterLo, int32 iterHi, noLink, noLink, loopId,
      filterLo, filterHi))
  # Target ids append after the fixed prefix, the tail `targetAt` reads from `forTargetsBase`.
  for tg in targets:
    p.nodes[idx].slots.add tg
  let body = parseBody(p, ["endfor"])
  if p.i >= p.tags.len or tagKeyword(p, p.tags[p.i]) != "endfor":
    raise err("`{% for %}` has no `{% endfor %}`")
  inc p.i
  p.nodes[idx].slots[slotChild] = body.head
  for x in body.tails:
    patch(p.nodes, x, idx)
  Head(head: idx, tails: @[idx])

proc parseSet(p: var P): Head =
  ## `{% set target = expr %}` and `{% set ns.field = expr %}` inline, plus `{% set x %} body {% endset %}`
  ## as a capture.
  let t = p.tags[p.i]
  var i = afterKeyword(p, t, 3) # past the `set` keyword
  let nameStart = i
  while i < t.tHi and p.src[i] in wsNameChars:
    inc i
  if i >= t.tHi:
    raise err("`set` needs a target at byte " & $t.tLo)
  var j = i
  while j < t.tHi and p.src[j] in wsSpace:
    inc j
  if j < t.tHi and p.src[j] == '.':
    # `ns.field = expr`:
    #   the namespace and field are recorded as a name pair, and the value span is still template text.
    let nsLo = nameStart
    let nsHi = i
    var k = j + 1
    while k < t.tHi and p.src[k] in wsNameChars:
      inc k
    let fieldLo = j + 1
    let fieldHi = k
    k = skipBalanced(p, k, t.tHi, "=")
    if k >= t.tHi or p.src[k] != '=':
      raise err("`{% set %}` needs a namespace member and `=`")
    var v = k + 1
    while v < t.tHi and p.src[v] in wsSpace:
      inc v
    let targetId = intern(p, p.src.toOpenArray(nsLo, nsHi - 1))
    let fieldId = intern(p, p.src.toOpenArray(fieldLo, fieldHi - 1))
    inc p.i
    let idx = addNode(p, mkNode(nkSetNs, int32 v, t.tHi.int32, noLink, targetId, fieldId))
    return Head(head: idx, tails: @[idx])
  if j >= t.tHi or p.src[j] != '=':
    raise err("`{% set %}` needs a target and `=`")
  var v = j + 1
  while v < t.tHi and p.src[v] in wsSpace:
    inc v
  let target = intern(p, p.src.toOpenArray(nameStart, i - 1))
  inc p.i
  let idx = addNode(p, mkNode(nkSet, int32 v, t.tHi.int32, noLink, target))
  Head(head: idx, tails: @[idx])

proc gap(what, corpusSite: string): Head =
  ## Reports a declared construct that is not implemented, stating what the corpus demands of it
  ## or that no template in the corpus demands it.
  raise newImplementError(what & " is not implemented; " & corpusSite)

proc parseConstruct(p: var P): Head =
  ## Dispatches one `{% %}` tag to its construct parser.
  let t = p.tags[p.i]
  let kw = tagKeyword(p, t)
  case kw
  of "if":
    parseIf(p)
  of "for":
    parseFor(p)
  of "set":
    parseSet(p)
  of "macro":
    parseMacro(p)
  of "break", "continue":
    gap("nkBreak", "corpus demand is 8 sites: 7 in glm53flash.jinja inside the macro " &
        "has_dup_tool_result_id, 1 in northminicode10.jinja")
  of "endfor", "endif", "else", "elif", "endset":
    raise err("`{% " & kw & " %}` has no matching opener")
  of "endmacro", "call", "filter", "block", "extends", "include", "import", "from":
    gap("`{% " & kw & " %}`", "no template in the corpus uses call, filter, block, endmacro " &
        "without a matching macro, extends, include, import or from")
  else:
    raise err("unknown `{% " & kw & " %}` tag")

proc parseBody(p: var P, stopKws: openArray[string]): Head =
  ## Emits nodes until a `{% %}` tag whose keyword is in `stopKws`, leaving the index on that tag.
  ##
  ## `open` holds every node whose successor is still unresolved. Each new entry point closes them,
  ## so a construct's exit links to its following sibling and only the body's last exits stay open
  ## for the enclosing construct to backpatch.
  var head = noLink
  var open = newSeq[int32]()
  while p.i < p.tags.len:
    let t = p.tags[p.i]
    var entry = noLink
    var fresh = newSeq[int32]()
    case t.kind
    of tkText:
      inc p.i
      entry = addNode(p, mkNode(nkVerbatim, int32 t.lo, int32 t.hi, noLink))
      fresh = @[entry]
    of tkVariable:
      inc p.i
      entry = addNode(p, mkNode(nkEmit, int32 t.tLo, int32 t.tHi, noLink))
      fresh = @[entry]
    of tkBlock:
      if tagKeyword(p, t) in stopKws:
        break
      let c = parseConstruct(p)
      entry = c.head
      fresh = c.tails
    if head == noLink:
      head = entry
    for x in open:
      patch(p.nodes, x, entry)
    open = fresh
  Head(head: head, tails: open)

proc parseTemplate*(src: string): (seq[Node], Tables) =
  ## Compiles template text to the arena plus its `Tables`, interned names built in parse order and read-only at render.
  var p = P(src: src, tags: tokenize(src, src.len), i: 0, tables: Tables(), nodes: newSeq[Node]())
  let body = parseBody(p, [])
  for x in body.tails:
    patch(p.nodes, x, noLink)
  (p.nodes, p.tables)
