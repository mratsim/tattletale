# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Chat-template parser.
# Template text in, the shared node arena plus interned-name arena out.
#
# Lifecycle:
#
# - split:
#   a pull stream yields one tag per advance, the whitespace policy resolved at the run
#   boundaries so no node carries a whitespace flag
# - emit:
#   a recursive descent over the tag stream appends nodes in source order, so arena order
#   equals source order and `succ` stays a link field, never a program counter
# - backpatch:
#   each construct returns its unresolved-successor nodes and the enclosing construct
#   resolves them, so an `if` chain's bodies terminate past the whole chain
#
# Termination:
#
# Every parse walk advances against a fixed bound, so no truncated or degenerate template
# spins a loop and no input grows the walk without bound:
#
# - scan-to-bound loops (whitespace runs, name spans, marker finds) advance a byte cursor against a fixed source-span end
# - the split's main loop returns one tag per pass, a tagless pass still moving the scan cursor past the tag it consumed
# - each construct body walk consumes at least one tag per pass, so arena growth stays bounded by the tag count
#
# - a find that misses reports `stop` or raises, never rescans
# - construct nesting is capped at `ParseNestingCap`, the dispatch recursion bounded, a breach raising located at the tag
# - the trailing-newline back-trim walk in `settle` decrements its arena index toward `tagMark`
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
# An unterminated construct is a `JinjaError` naming the construct and its byte offset,
# so a truncated template fails at load with that error, never renders short.

import std/[strbasics, strutils]
import cnj_types, jinja_data_model
import workspace/data_structures/src/small_seqs

func at(s: openArray[char], prefix: openArray[char], i: int): bool =
  ## Reports whether the bytes of `prefix` occur at `i`.
  if s.len - i < prefix.len:
    return false
  for k in 0 ..< prefix.len:
    if s[i + k] != prefix[k]:
      return false
  true

type
  TagKind = enum
    tkEnd, tkText, tkVariable, tkBlock

  Tag = object
    ## One split step:
    ## a text run to emit, a `{%`/`{{` tag row, or the end sentinel.
    kind: TagKind
    lo, hi: int # text run to emit, whitespace-resolved (unused for tag rows)
    tLo, tHi: int # inside of the delimiters, with any `-` marker stripped

  Parser = object
    ## Parse state:
    ##   the current tag with a one-tag lookahead, the scan cursor into `src`,
    ##   the arena under construction and the symbol arena.
    src: string
    cur, nxt: Tag # `nxt` is the lookahead, `tkEnd` there marking the end of the stream
    pending: Tag # a tag row scanned together with the text run before it, delivered next
    i: int # scan cursor, start of the run being split
    pendBr: bool # trim_blocks drops one newline at the next run's start
    done: bool # the scan reached the end of the template
    dropCur: bool # settle emptied `cur`, parseBody emits no node for it
    tagMark: int # arena length at the last tag pull, the back-trim walk stops here
    nesting: int # body-carrying constructs under construction, `parseNested` caps the recursion on it
    loopDepth: int # enclosing `{% for %}` bodies under construction, 0 at top level
    macroDepth: int # enclosing `{% macro %}` bodies under construction, 0 at top level
    symbols: CompiledSymbols
    nodes: seq[Node]

  Head = object
    ## A constructed run of nodes:
    ##   its entry point, and the nodes whose successor is still unresolved.
    head: int32
    tails: seq[int32]

proc mkNode(kind: NodeKind, slots: varargs[int32]): Node =
  ## Builds one node, appending its payload slots in layout order. The enclosing construct
  ## backpatches `succ` and `child` from `NoLink` through `patch` or a direct slot write.
  result.kind = kind
  for s in slots:
    result.slots.add s

# Split:

func atLineStart(src: openArray[char], at: int): bool =
  ## Reports whether only spaces and tabs separate `at` from the preceding newline, the `lstrip_blocks` test.
  var i = at - 1
  while i >= 0 and (src[i] == ' ' or src[i] == '\t'):
    dec i
  i < 0 or src[i] == '\n'

func tagBounds(src: openArray[char], innerLo, innerHi: int): (int, int, bool, bool) =
  ## Splits a tag's inside against its `-` markers, returning the marker-free span
  ## plus the two strip flags:
  ## - `stripBefore` reads a `{%-`-shaped open, `stripAfter` a `-%}`-shaped close
  ## - both flags read the unadjusted span, a degenerate whole-dash interior
  ##   (`{#-#}`) losing both markers at once
  var lo = innerLo
  var hi = innerHi
  let stripBefore = lo < hi and src[lo] == '-'
  let stripAfter = hi > lo and src[hi - 1] == '-'
  if stripBefore:
    inc lo
  if stripAfter:
    dec hi
  (lo, hi, stripBefore, stripAfter)

func runBeforeTag(src: openArray[char], lo, hi, openAt: int, stripBefore, blockTag: bool): int =
  ## Resolves the whitespace of the text run before a tag, returning the run's new end.
  ##
  ## - a `{%-`-shaped open strips every whitespace byte of the run's tail
  ## - otherwise only blanks precede the tag on its line, `lstrip_blocks` stripping
  ##   those blanks of a block or comment tag
  result = hi
  if stripBefore:
    while result > lo and src[result - 1] in cnj_types.Whitespace:
      dec result
  elif blockTag and src.atLineStart(openAt):
    while result > lo and src[result - 1] in {' ', '\t'}:
      dec result

func passTagClose(p: var Parser, afterTag: int, stripAfter, blockTag: bool) =
  ## Steps the cursor past a tag's close and records the trim_blocks flag for the next run:
  ## a `-%}`-shaped close skips the whitespace run after it, and one newline of the next
  ## run is dropped when the tag is a block or comment tag.
  var j = afterTag
  if stripAfter:
    while j < p.src.len and p.src[j] in cnj_types.Whitespace:
      inc j
  p.i = j
  p.pendBr = blockTag

func findTagClose(src: openArray[char], at, stop: int, close: string): int =
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

func findRun(src: openArray[char], at, stop: int, needle: string): int =
  ## Returns the offset of `needle` at or after `at`, a plain byte scan with no
  ## quote or bracket tracking.
  ## - comment and `{% raw %}` bodies are verbatim runs, a `'` or `"` inside them
  ##   must not defer the scan
  ## - `findTagClose` is the quote-tracking form, reserved for expression spans
  var i = at
  while i + needle.len <= stop:
    if at(src, needle, i):
      return i
    inc i
  -1

func nextOpen(src: openArray[char], at, stop: int): int =
  ## Returns the offset of the next `{{`, `{%` or `{#`, or `stop` when there is none.
  var i = at
  while i < stop:
    if at(src, "{{", i) or at(src, "{%", i) or at(src, "{#", i):
      return i
    inc i
  stop

func splitTags(p: var Parser): Tag =
  ## Produces the next tag of the split, one call per tag:
  ## a whitespace-resolved text run or one `{%`/`{{` tag row, scanned from the parser's cursor.
  ##
  ## | Property    | Contract                                                           |
  ## | ----------- | ------------------------------------------------------------------ |
  ## | pending     | a tag row scanned with the text run before it, delivered next call |
  ## | end         | the `tkEnd` sentinel at the template's end                         |
  ## | termination | a tagless pass still moves the scan cursor, an unclosed raise      |
  ## | unclosed    | `JinjaError` on an unterminated comment, tag or raw body           |
  if p.pending.kind != tkEnd:
    result = p.pending
    p.pending = Tag()
    return
  if p.done:
    return Tag()
  let stop = p.src.len
  while true:
    let openAt = p.src.nextOpen(p.i, stop)
    var lo = p.i
    var hi = openAt
    if p.pendBr and lo < hi and p.src[lo] == '\n':
      inc lo
    var kind = tkText
    var tLo = 0
    var tHi = 0
    var afterTag: int # offset just past the tag's closing delimiter, set in every tag branch
    var stripAfter: bool
    var isRaw = false
    if openAt < stop:
      if p.src.at("{#", openAt):
        let c = p.src.findRun(openAt + 2, stop, "#}")
        if c < 0:
          raise jinjaErr("unclosed comment opened at byte " & $openAt, openAt)
        # A comment's body is erased, the run before it is real output under the block
        # tag's whitespace rules:
        #   `{#-` strips the run before the tag
        #   `lstrip_blocks` the blanks preceding it on its line
        #   `-#}` the run after the tag
        #   trim_blocks one newline after it
        afterTag = c + 2
        let (_, _, stripBefore, stripAfter) = p.src.tagBounds(openAt + 2, c)
        hi = p.src.runBeforeTag(lo, hi, openAt, stripBefore, true)
        p.passTagClose(afterTag, stripAfter, true)
        if lo < hi:
          return Tag(kind: tkText, lo: int32 lo, hi: int32 hi, tLo: 0, tHi: 0)
        continue
      let isVar = p.src.at("{{", openAt)
      # A tag's close needle is the bare delimiter:
      #   `-%}` carries no space before `%}`, so a `" %}"` needle would skip
      #   every whitespace-controlled tag
      let close = if isVar: "}}" else: "%}"
      let bodyStart = openAt + 2
      let c = p.src.findTagClose(bodyStart, stop, close)
      if c < 0:
        raise jinjaErr("unclosed " & (if isVar: "`{{`" else: "`{%`") & " opened at byte " & $openAt, openAt)
      afterTag = c + 2
      let (innerLo, innerHi, stripBefore, tagStripAfter) = p.src.tagBounds(bodyStart, c)
      stripAfter = tagStripAfter
      if not isVar:
        var k = innerLo
        while k < innerHi and p.src[k] in cnj_types.Whitespace:
          inc k
        isRaw = p.src.at("raw", k) and (k + 3 >= innerHi or p.src[k + 3] notin WsNameChars)
      if isRaw:
        let openDash = stripAfter
        # `{% raw %}` holds its body verbatim:
        #   one text run up to the next `{% endraw %}` tag, closed by a quote-blind
        #   scan anchored on a tag-shaped `{%`, so a quote or a bare `endraw %}`
        #   inside the body neither defers the scan nor ends the run early
        # - the run's end is the matched tag's `{`, so no byte of the closing tag
        #   joins the body
        var endOpen = -1   # the closing tag's `{`
        var endDash = false # the closing tag carries `{%-`
        var endAfterTag = 0 # offset just past the closing tag's `%}`
        var endStripAfter = false # the closing tag carries `- %}`
        var i = c + 2
        while i < stop:
          if p.src.at("{%", i):
            var j = i + 2
            var dash = false
            if j < stop and p.src[j] == '-':
              dash = true
              inc j
            while j < stop and p.src[j] in cnj_types.Whitespace:
              inc j
            if p.src.at("endraw", j) and (j + 6 >= stop or p.src[j + 6] notin WsNameChars):
              var k = j + 6
              var dashAfter = false
              while k < stop and p.src[k] in cnj_types.Whitespace:
                inc k
              if k < stop and p.src[k] == '-':
                dashAfter = true
                inc k
                while k < stop and p.src[k] in cnj_types.Whitespace:
                  inc k
              if k + 2 <= stop and p.src.at("%}", k):
                endOpen = i
                endDash = dash
                endAfterTag = k + 2
                endStripAfter = dashAfter
                break
            # the `{%` was body text, the scan resumes past it
            inc i, 2
            continue
          inc i
        if endOpen < 0:
          raise jinjaErr("unclosed `{% raw %}` opened at byte " & $openAt, openAt)
        var rawLo = c + 2
        var rawHi = endOpen
        # `{%- endraw %}` strips the body's trailing whitespace run, the same
        # whitespace policy every other tag boundary follows
        if endDash:
          while rawHi > rawLo and p.src[rawHi - 1] in cnj_types.Whitespace:
            dec rawHi
        # `-%}` on the open tag strips the body's leading whitespace run
        if openDash:
          while rawLo < rawHi and p.src[rawLo] in cnj_types.Whitespace:
            inc rawLo
        if rawLo < rawHi and p.src[rawLo] in {' ', '\t'} and p.src.atLineStart(rawLo):
          while rawLo < rawHi and p.src[rawLo] in {' ', '\t'}:
            inc rawLo
        # trim_blocks drops the one newline a plain `{% raw %}` opening carries into
        # the body, the dashed opening's leading-whitespace strip covering it already
        if rawLo < rawHi and p.src[rawLo] == '\n':
          inc rawLo
        p.passTagClose(endAfterTag, endStripAfter, true)
        # One text run before the raw tag is real output under the open tag's
        # whitespace rules, delivered first, the body queueing in `pending` after it.
        hi = p.src.runBeforeTag(lo, hi, openAt, stripBefore, true)
        if lo < hi:
          p.pending = Tag(kind: tkText, lo: int32 rawLo, hi: int32 rawHi, tLo: 0, tHi: 0)
          return Tag(kind: tkText, lo: int32 lo, hi: int32 hi, tLo: 0, tHi: 0)
        return Tag(kind: tkText, lo: int32 rawLo, hi: int32 rawHi, tLo: 0, tHi: 0)
      kind = if isVar: tkVariable else: tkBlock
      tLo = innerLo
      tHi = innerHi
      hi = p.src.runBeforeTag(lo, hi, openAt, stripBefore, kind == tkBlock)
    if openAt >= stop:
      p.done = true
      if lo < hi:
        return Tag(kind: tkText, lo: int32 lo, hi: int32 hi, tLo: 0, tHi: 0)
      return Tag()
    p.passTagClose(afterTag, stripAfter, kind == tkBlock)
    if lo < hi:
      p.pending = Tag(kind: kind, lo: 0, hi: 0, tLo: int32 tLo, tHi: int32 tHi)
      return Tag(kind: tkText, lo: int32 lo, hi: int32 hi, tLo: 0, tHi: 0)
    return Tag(kind: kind, lo: 0, hi: 0, tLo: int32 tLo, tHi: int32 tHi)

func settle(p: var Parser) =
  ## Trailing-newline rule for the tag that just became `cur`:
  ## - the final text run of a template drops its trailing newlines, matching the upstream
  ##   `rstrip("\n")` before compile, and `dropCur` marks it so parseBody emits no node
  ## - a run that empties passes the trim to the verbatim nodes before it, each emptied
  ##   run passing it on, until one keeps bytes, a non-text node ends the walk,
  ##   or the walk reaches `tagMark`, the arena length at the last tag pull
  ## - emptied nodes stay in the arena, they render nothing
  p.dropCur = false
  if p.nxt.kind != tkEnd or p.cur.kind != tkText:
    return
  while p.cur.hi > p.cur.lo and p.src[p.cur.hi - 1] == '\n':
    dec p.cur.hi
  if p.cur.hi > p.cur.lo:
    return
  p.dropCur = true
  var k = p.nodes.len - 1
  while k >= p.tagMark and p.nodes[k].kind == nkVerbatim:
    let lo = p.nodes[k].lo.int
    var hi = p.nodes[k].hi.int
    while hi > lo and p.src[hi - 1] == '\n':
      dec hi
    p.nodes[k].slots[SlotHi] = int32 hi
    if hi > lo:
      break
    dec k

func advance(p: var Parser) =
  ## Pull step of the stream:
  ## `cur` steps to the lookahead, the scan produces the next tag.
  ## The trailing-newline rule applies once the end of the template is the lookahead.
  p.cur = p.nxt
  if p.cur.kind != tkText:
    p.tagMark = p.nodes.len
  p.nxt = p.splitTags()
  p.settle()

func start(p: var Parser) =
  ## Seeds the stream:
  ## the first tag becomes `cur`, the second the lookahead.
  p.nxt = p.splitTags()
  p.advance()

# Emit:

func addNode(p: var Parser, n: sink Node): int32 =
  ## Appends one node and returns its arena index.
  result = int32 p.nodes.len
  p.nodes.add n

func patch(nodes: var seq[Node], idx, target: int32) =
  ## Resolves one node's successor, the `SlotSucc` position.
  nodes[idx].slots[SlotSucc] = target

func keywordStart(src: openArray[char], t: Tag): int =
  ## Offset of a `{% %}` tag's leading identifier, past the tag's leading whitespace.
  var i = t.tLo
  while i < t.tHi and src[i] in cnj_types.Whitespace:
    inc i
  i

func keywordSpan(src: openArray[char], t: Tag): openArray[char] =
  ## View of a `{% %}` tag's leading identifier over `src`, so neither a dispatch
  ## nor a terminator test materializes the keyword.
  ## The view's first byte offset is `keywordStart`, the same leading-whitespace skip.
  var i = t.tLo
  while i < t.tHi and src[i] in cnj_types.Whitespace:
    inc i
  let start = i
  while i < t.tHi and src[i] in WsNameChars:
    inc i
  result = src.toOpenArray(start, i - 1)

func keywordIs(src: openArray[char], t: Tag, kw: string): bool =
  ## Reports whether the tag's leading identifier is `kw`, compared in place.
  let span = src.keywordSpan(t)
  if span.len != kw.len:
    return false
  for k in 0 ..< kw.len:
    if span[k] != kw[k]:
      return false
  true

func keywordIn(src: openArray[char], t: Tag, kws: openArray[string]): bool =
  ## Reports whether the tag's leading identifier is one of `kws`, compared in place.
  for kw in kws:
    if src.keywordIs(t, kw):
      return true
  false

func afterKeyword(p: Parser, t: Tag, kwLen: int): int =
  ## Returns the offset just past the tag's keyword and following whitespace, the `tLo` offset
  ## being the inside of the delimiters, so the keyword itself may start after whitespace.
  var i = t.tLo
  while i < t.tHi and p.src[i] in cnj_types.Whitespace:
    inc i
  inc i, kwLen
  while i < t.tHi and p.src[i] in cnj_types.Whitespace:
    inc i
  i

func scanDepth0(src: openArray[char], at, stop: int, stops: set[char], word: string): int =
  ## Returns the offset of the first position at bracket depth zero, outside quoted literals,
  ## holding a character of `stops` or the bare word `word`, or `stop` when neither appears.
  ##
  ## - word boundaries are checked against `at` and `stop`, the scan span's ends
  ## - quoted literals are skipped and bracketed subexpressions counted, so a match inside
  ##   a string or a call's argument list never fires
  ## - an empty `word` disables the word test
  var i = at
  var depth = 0
  var q: char = '\0'
  while i < stop:
    if q != '\0':
      if src[i] == '\\':
        inc i
      elif src[i] == q:
        q = '\0'
    elif src[i] in {'\'', '"'}:
      q = src[i]
    elif depth == 0 and (src[i] in stops or
        (word.len > 0 and at(src, word, i) and
          (i == at or src[i - 1] notin WsNameChars) and
          (i + word.len >= stop or src[i + word.len] notin WsNameChars))):
      return i
    elif src[i] in {'(', '['}:
      inc depth
    elif src[i] in {')', ']'}:
      dec depth
    inc i
  stop

func skipBalanced(src: openArray[char], at, stop: int, stops: set[char], expected: string): int =
  ## Returns the offset of any character of `stops` at bracket depth zero, skipping quoted
  ## literals and raising when the construct ends first.
  ## `expected` names the stop characters in the error message.
  let i = scanDepth0(src, at, stop, stops, "")
  if i == stop:
    raise jinjaErr("expected `" & expected & "` before byte " & $stop, stop)
  i

func findKeyword(src: openArray[char], at, stop: int, word: string): int =
  ## Returns the offset of the bare word `word` at bracket depth zero, or `stop` on a span
  ## holding no such word.
  ##
  ## - quoted literals and bracketed subexpressions are skipped, so an `if` inside a string
  ##   or a call's argument list is not mistaken for the for-`if` clause
  scanDepth0(src, at, stop, {}, word)

proc parseBody(p: var Parser, stopKws: openArray[string]): Head
proc parseConstruct(p: var Parser): Head

template capNesting(p: var Parser, t: Tag) =
  ## Counts one recursion level of the parse dispatch toward `ParseNestingCap`, a breach
  ## raising located at the tag. A raise aborts the whole parse, the parser value abandoned.
  ##
  ## Counted sites:
  ## - body walks, at `parseNested`
  ## - the `{% elif %}` chain, which recurses `parseIf` outside any body walk
  inc p.nesting
  if p.nesting > ParseNestingCap:
    raise jinjaErr("template nests deeper than ParseNestingCap = " & $ParseNestingCap &
        " at byte " & $t.tLo, t.tLo, t.tHi - t.tLo)

template parseNested(p: var Parser, stopKws: openArray[string], t: Tag): Head =
  ## One body walk of a construct, the recursion's nesting count rising at its entry,
  ## falling once the body returns.
  capNesting(p, t)
  let body = parseBody(p, stopKws)
  dec p.nesting
  body

proc parseMacroParams(p: var Parser, t: Tag, at: int, nodeIdx: int32) =
  ## Parses `(a, b = expr, ...)` starting at the open paren and appending one payload triple per
  ## parameter to the `nkMacroDef` node at `nodeIdx`, the interned name then the default span,
  ## `NoLink` when absent, defaults staying template text, each evaluated per call after binding.
  ## - each pass consumes a parameter name, a nameless position raising, so the walk is bounded
  ##   by the tag's span
  ## - evaluation happens after the parameters bind
  var i = at + 1 # past the open paren
  while true:
    while i < t.tHi and p.src[i] in cnj_types.Whitespace:
      inc i
    if i < t.tHi and p.src[i] == ')':
      inc i
      break
    let nameStart = i
    while i < t.tHi and p.src[i] in WsNameChars:
      inc i
    if i == nameStart:
      raise jinjaErr("macro parameter needs a name at byte " & $nameStart, nameStart)
    let name = p.symbols.internName(p.src.toOpenArray(nameStart, i - 1))
    var k = i
    while k < t.tHi and p.src[k] in cnj_types.Whitespace:
      inc k
    var defLo = NoLink
    var defHi = NoLink
    if k < t.tHi and p.src[k] == '=':
      inc k
      while k < t.tHi and p.src[k] in cnj_types.Whitespace:
        inc k
      defLo = int32 k
      defHi = int32 p.src.skipBalanced(k, t.tHi, {',', ')'}, ",)")
      k = defHi.int
    # One parameter per iteration, appending the triple `paramNameAt` reads:
    # name id first, then the default `lo`, then the default `hi`.
    p.nodes[nodeIdx].slots.add name
    p.nodes[nodeIdx].slots.add defLo
    p.nodes[nodeIdx].slots.add defHi
    i = k
    while i < t.tHi and p.src[i] in cnj_types.Whitespace:
      inc i
    if i < t.tHi and p.src[i] == ',':
      inc i
      continue
    if i < t.tHi and p.src[i] == ')':
      inc i
      break
    raise jinjaErr("macro parameter list is not closed at byte " & $i, i)

proc parseMacro(p: var Parser): Head =
  ## `{% macro name(a, b = 1) %} body {% endmacro %}`. The definition binds a value and never
  ## runs the body here. A macro's output is a string, so only a call can run it, to completion.
  ## The body's terminators land back on this node, how a call detects its end.
  let t = p.cur
  var i = p.afterKeyword(t, 5) # past the `macro` keyword
  let nameStart = i
  while i < t.tHi and p.src[i] in WsNameChars:
    inc i
  if i == nameStart:
    raise jinjaErr("`macro` needs a name at byte " & $nameStart, nameStart)
  let name = p.symbols.internName(p.src.toOpenArray(nameStart, i - 1))
  while i < t.tHi and p.src[i] in cnj_types.Whitespace:
    inc i
  if i >= t.tHi or p.src[i] != '(':
    raise jinjaErr("`macro` parameters are not parenthesised at byte " & $i, i)
  p.advance()
  let idx = addNode(p, mkNode(nkMacroDef, name, NoLink, NoLink, NoLink))
  parseMacroParams(p, t, i, idx)
  inc p.macroDepth
  let body = parseNested(p, ["endmacro"], t)
  dec p.macroDepth
  if p.cur.kind == tkEnd or not p.src.keywordIs(p.cur, "endmacro"):
    raise jinjaErr("`{% macro %}` has no `{% endmacro %}`", t.tLo)
  p.advance()
  p.nodes[idx].slots[SlotChild] = body.head
  for x in body.tails:
    patch(p.nodes, x, idx)
  Head(head: idx, tails: @[idx])

proc parseIf(p: var Parser): Head =
  ## `{% if %} … {% elif %} … {% else %} … {% endif %}`. One node per chain level, every branch body
  ## terminated past the whole chain, which is what makes the node single-entry and single-activation.
  let t = p.cur
  let kw = p.src.keywordSpan(t)
  let condLo = p.afterKeyword(t, kw.len).int32
  let condHi = t.tHi.int32
  p.advance()
  # Reserved before its body is walked:
  #   arena order stays source order, the arena entry stays index 0
  #   a construct cannot sit below the nodes it dispatches into
  let idx = addNode(p, mkNode(nkIf, condLo, condHi, NoLink, NoLink, NoLink))
  let body = parseNested(p, ["elif", "else", "endif"], t)
  p.nodes[idx].slots[SlotChild] = body.head
  var tails = @[idx]
  tails.add body.tails
  if p.cur.kind == tkEnd:
    let kwErr = p.src.keywordSpan(t)
    raise jinjaErr("unclosed `{% " & spanString(kwErr) & " %}`", p.src.keywordStart(t), kwErr.len)
  let term = p.cur
  if p.src.keywordIs(term, "elif"):
    capNesting(p, t)
    let nested = parseIf(p)
    dec p.nesting
    p.nodes[idx].slots[SlotAlt] = nested.head
    tails.add nested.tails
  elif p.src.keywordIs(term, "else"):
    p.advance()
    let eb = parseBody(p, ["endif"])
    p.nodes[idx].slots[SlotAlt] = eb.head
    tails.add eb.tails
    if p.cur.kind == tkEnd or not p.src.keywordIs(p.cur, "endif"):
      raise jinjaErr("`{% else %}` has no `{% endif %}`", term.tLo, term.tHi - term.tLo)
    p.advance()
  elif p.src.keywordIs(term, "endif"):
    p.advance()
  else:
    let kwErr = p.src.keywordSpan(t)
    raise jinjaErr("`{% " & spanString(kwErr) & " %}` has no `{% endif %}`", p.src.keywordStart(t), kwErr.len)
  Head(head: idx, tails: tails)

proc parseFor(p: var Parser): Head =
  ## `{% for a, b in expr if cond %} body {% endfor %}`. The header stays one span,
  ## the target names and the filter clause split out as bindings. The target walk consumes
  ## one name per pass, a nameless position raising, so it is bounded by the tag's span.
  let t = p.cur
  var i = p.afterKeyword(t, 3) # past the `for` keyword
  var targets = newSeq[int32]()
  while true:
    let start = i
    while i < t.tHi and p.src[i] in WsNameChars:
      inc i
    if i == start:
      raise jinjaErr("`for` needs a target name at byte " & $start, start)
    targets.add p.symbols.internName(p.src.toOpenArray(start, i - 1))
    while i < t.tHi and p.src[i] in cnj_types.Whitespace:
      inc i
    if i < t.tHi and p.src[i] == ',':
      inc i
      while i < t.tHi and p.src[i] in cnj_types.Whitespace:
        inc i
      continue
    break
  if not p.src.at("in", i) or (i + 2 < t.tHi and p.src[i + 2] in WsNameChars):
    raise jinjaErr("`for` target is not followed by `in` at byte " & $i, i)
  var j = i + 2
  while j < t.tHi and p.src[j] in cnj_types.Whitespace:
    inc j
  let iterLo = j
  let filterAt = p.src.findKeyword(iterLo, t.tHi, "if")
  let iterHi = filterAt
  var filterLo = NoLink
  var filterHi = NoLink
  if filterAt < t.tHi:
    var k = filterAt + 2
    while k < t.tHi and p.src[k] in cnj_types.Whitespace:
      inc k
    filterLo = int32 k
    filterHi = int32 t.tHi
  let loopId = p.symbols.internName("loop")
  p.advance()
  let idx = addNode(p, mkNode(nkFor, int32 iterLo, int32 iterHi, NoLink, NoLink, loopId,
      filterLo, filterHi))
  # Target ids append after the fixed prefix, forming the tail `targetAt` reads.
  for tg in targets:
    p.nodes[idx].slots.add tg
  inc p.loopDepth
  let body = parseNested(p, ["endfor"], t)
  dec p.loopDepth
  if p.cur.kind == tkEnd or not p.src.keywordIs(p.cur, "endfor"):
    raise jinjaErr("`{% for %}` has no `{% endfor %}`", t.tLo)
  p.advance()
  p.nodes[idx].slots[SlotChild] = body.head
  for x in body.tails:
    patch(p.nodes, x, idx)
  Head(head: idx, tails: @[idx])

proc parseSet(p: var Parser): Head =
  ## `{% set target = expr %}` and `{% set ns.field = expr %}` inline, plus `{% set x %} body {% endset %}`
  ## as a capture.
  let t = p.cur
  var i = p.afterKeyword(t, 3) # past the `set` keyword
  let nameStart = i
  while i < t.tHi and p.src[i] in WsNameChars:
    inc i
  # An empty target name raises, `{% set = 3 %}` and a bare `{% set %}` included.
  if i == nameStart:
    raise jinjaErr("`set` needs a target at byte " & $t.tLo, t.tLo)
  var j = i
  while j < t.tHi and p.src[j] in cnj_types.Whitespace:
    inc j
  if j < t.tHi and p.src[j] == '.':
    # `ns.field = expr`:
    #   the namespace and field are recorded as a name pair, and the value span is still template text.
    let nsLo = nameStart
    let nsHi = i
    var k = j + 1
    while k < t.tHi and p.src[k] in WsNameChars:
      inc k
    let fieldLo = j + 1
    let fieldHi = k
    k = p.src.skipBalanced(k, t.tHi, {'='}, "=")
    if k >= t.tHi or p.src[k] != '=':
      raise jinjaErr("`{% set %}` needs a namespace member and `=`", t.tLo)
    var v = k + 1
    while v < t.tHi and p.src[v] in cnj_types.Whitespace:
      inc v
    let targetId = p.symbols.internName(p.src.toOpenArray(nsLo, nsHi - 1))
    let fieldId = p.symbols.internName(p.src.toOpenArray(fieldLo, fieldHi - 1))
    p.advance()
    let idx = addNode(p, mkNode(nkSetNamespace, int32 v, t.tHi.int32, NoLink, targetId, fieldId))
    return Head(head: idx, tails: @[idx])
  if j >= t.tHi:
    # `{% set target %}` block assignment:
    #   the capture body parses like a construct body and the target binds on close,
    #   the declared `nkSetBlock` gap raising at render when a row reaches the construct.
    let target = p.symbols.internName(p.src.toOpenArray(nameStart, i - 1))
    p.advance()
    let idx = addNode(p, mkNode(nkSetBlock, int32 p.src.keywordStart(t), int32 t.tHi,
        NoLink, NoLink, target))
    let body = parseNested(p, ["endset"], t)
    if p.cur.kind == tkEnd or not p.src.keywordIs(p.cur, "endset"):
      raise jinjaErr("`{% set %}` block assignment has no `{% endset %}`", t.tLo)
    p.advance()
    p.nodes[idx].slots[SlotChild] = body.head
    for x in body.tails:
      patch(p.nodes, x, idx)
    return Head(head: idx, tails: @[idx])
  if p.src[j] != '=':
    raise jinjaErr("`{% set %}` needs a target and `=`", t.tLo)
  var v = j + 1
  while v < t.tHi and p.src[v] in cnj_types.Whitespace:
    inc v
  let target = p.symbols.internName(p.src.toOpenArray(nameStart, i - 1))
  p.advance()
  let idx = addNode(p, mkNode(nkSet, int32 v, t.tHi.int32, NoLink, target))
  Head(head: idx, tails: @[idx])

func gap(what, corpusSite: string): Head =
  ## Reports a declared construct that is not implemented, stating what the corpus demands of it
  ## or that no template in the corpus demands it.
  raise jinjaErr(what & " is not implemented; " & corpusSite, cause = ceUnimplemented)

proc parseConstruct(p: var Parser): Head =
  ## Dispatches one `{% %}` tag to its construct parser.
  let t = p.cur
  if p.src.keywordIs(t, "if"):
    parseIf(p)
  elif p.src.keywordIs(t, "for"):
    parseFor(p)
  elif p.src.keywordIs(t, "set"):
    parseSet(p)
  elif p.src.keywordIs(t, "macro"):
    parseMacro(p)
  elif p.src.keywordIs(t, "break") or p.src.keywordIs(t, "continue"):
    let kw = p.src.keywordSpan(t)
    if p.macroDepth == 0 and p.loopDepth == 0:
      # A macro body defers the enclosure check to its call site, so only a break
      # outside every macro and every for is malformed.
      raise jinjaErr("`{% " & spanString(kw) & " %}` is outside any `{% for %}`",
          p.src.keywordStart(t), kw.len)
    let idx = addNode(p, mkNode(nkBreak, int32 p.src.keywordStart(t), int32 t.tHi, NoLink))
    p.advance()
    Head(head: idx, tails: @[idx])
  elif p.src.keywordIs(t, "generation"):
    p.advance()
    let idx = addNode(p, mkNode(nkGeneration, int32 p.src.keywordStart(t), int32 t.tHi,
        NoLink, NoLink))
    let body = parseNested(p, ["endgeneration"], t)
    if p.cur.kind == tkEnd or not p.src.keywordIs(p.cur, "endgeneration"):
      raise jinjaErr("`{% generation %}` has no `{% endgeneration %}`", t.tLo)
    p.advance()
    p.nodes[idx].slots[SlotChild] = body.head
    for x in body.tails:
      patch(p.nodes, x, idx)
    Head(head: idx, tails: @[idx])
  elif p.src.keywordIn(t, ["endfor", "endif", "else", "elif", "endset", "endgeneration"]):
    let kw = p.src.keywordSpan(t)
    raise jinjaErr("`{% " & spanString(kw) & " %}` has no matching opener", p.src.keywordStart(t), kw.len)
  elif p.src.keywordIn(t, ["endmacro", "call", "filter", "block", "extends", "include",
      "import", "from"]):
    let kw = p.src.keywordSpan(t)
    gap("`{% " & spanString(kw) & " %}`", "no template in the corpus uses " &
        "call, filter, block, endmacro without a matching macro, extends, include, import or from")
  else:
    let kw = p.src.keywordSpan(t)
    raise jinjaErr("unknown `{% " & spanString(kw) & " %}` tag", p.src.keywordStart(t), kw.len)

proc parseBody(p: var Parser, stopKws: openArray[string]): Head =
  ## Emits nodes until a `{% %}` tag whose keyword is in `stopKws`, leaving `cur` on that tag.
  ##
  ## `open` holds every node whose successor is still unresolved. Each new entry point closes them,
  ## so a construct's exit links to its following sibling and only the body's last exits stay open
  ## for the enclosing construct to backpatch.
  ##
  ## Every pass consumes at least one tag. Text runs, emits and construct dispatches all
  ## advance the stream or raise, so the walk is bounded by the tag count and truncation
  ## ends it at the end sentinel or a close check's raise.
  var head = NoLink
  var open = newSeq[int32]()
  while p.cur.kind != tkEnd:
    let t = p.cur
    var entry = NoLink
    var fresh = newSeq[int32]()
    case t.kind
    of tkText:
      if p.dropCur:
        p.advance()
        continue
      entry = addNode(p, mkNode(nkVerbatim, int32 t.lo, int32 t.hi, NoLink))
      fresh = @[entry]
      p.advance()
    of tkVariable:
      entry = addNode(p, mkNode(nkEmit, int32 t.tLo, int32 t.tHi, NoLink))
      fresh = @[entry]
      p.advance()
    of tkBlock:
      if p.src.keywordIn(t, stopKws):
        break
      let c = parseConstruct(p)
      entry = c.head
      fresh = c.tails
    of tkEnd:
      break
    if head == NoLink:
      head = entry
    for x in open:
      patch(p.nodes, x, entry)
    open = fresh
  Head(head: head, tails: open)

proc parseTemplate*(src: string): (CompiledTemplate, CompiledSymbols) =
  ## Compiles template text to the shared artifact plus its `CompiledSymbols`, interned
  ## names built in parse order and read-only at render.
  ## Returns the template borrowing `src`, so it must not outlive the caller's text.
  var p = Parser(src: src, symbols: CompiledSymbols(), nodes: newSeq[Node]())
  p.start()
  let body = parseBody(p, [])
  for x in body.tails:
    patch(p.nodes, x, NoLink)
  (CompiledTemplate(jinja: src, nodes: p.nodes), p.symbols)
