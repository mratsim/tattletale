# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Regex pattern engine: parser, NFA compiler, and
## priority-preserving leftmost-first matcher.
##
## Why a leftmost-first NFA simulation and not a powerset DFA: a
## powerset DFA is a longest-match machine. A PCRE2
## backtracking-priority scan returns the
## first match in pattern-preference order, whose extent differs from
## the longest extent whenever alternatives overlap, e.g. a trailing
## whitespace group `\s++$|\s+(?!\S)|\s` or twin letter alternatives.
## The engine implements the RE2 / Rust-regex-style leftmost-first
## simulation: a Thompson NFA whose threads carry the pattern preference
## order, expanded depth-first per slot (each slot's whole subtree
## contributes before the next slot), with the cut rule (an acceptance
## kills every lower-priority thread, and a surviving higher-priority
## acceptance overrides the recorded match). That reproduces
## backtracking-priority match extents without state explosion and
## without exponential backtracking. Worst-case scan cost is
## O(len x program) when matches are dense. An attempt anchored inside
## a long backtracking-quantifier run costs O(remaining run), so
## adversarial single-rune-class runs of length L cost O(L^2) overall,
## far beyond anything the parity corpora contain (documented, see the
## fuzz suite rows).
##
## Deterministic compile (frontier DFA): the default scan driver
## of this module memoizes the simulation's ordered frontiers
## and per-symbol transition tables, so steady-state matching
## is a table walk instead of per-position epsilon closures.
## A plain powerset DFA would be a longest-match machine and cannot
## carry the leftmost-first extents, while the frontier compile
## preserves them by construction
## (see the frontier DFA section at the bottom). The NFA simulation
## stays as the equivalence oracle of the parity suites
## and the documented fallback when the lazy determinization cap
## is hit.
##
## Supported constructs:
## alternation, concatenation, non-capturing groups, caseless groups
## (?i:...), single-codepoint negative lookaheads (?!\S) and
## (?!\p{Script=Han}) (regular: complement class plus end-of-window
## test), the $ assertion (PCRE2 semantics: end of subject or before a
## final LF, matching the bundled PCRE2 default newline), character
## classes with \p{...} members and complements, and the quantifiers
## * + ? {m,n} each greedy, with the possessive suffix (+) accepted and
## compiled as greedy.
##
## Possessive-as-greedy justification: possessive quantifiers pin the
## extent at the greedy maximum, forbidding back-off. Every possessive
## quantifier in the served patterns is a single-class quantifier whose
## continuation either does not exist (end of alternative), is the $
## assertion (whose before-final-LF position can never lie strictly
## inside a \s run, since a final LF is itself \s and the greedy run
## would have consumed it), or starts with a class provably disjoint
## from the quantified class ([^\r\n\p{L}\p{N}]?+ before \p{L}++).
## Greedy and possessive therefore admit the
## same match extents on every input, and the parity suites prove segment
## equality against the PCRE2 oracle over adversarial rows
## that exercise exactly these zones (trailing whitespace, CRLF runs,
## digit runs).
##
## NOTEMPTY handling: a zero-width total acceptance is discarded
## (neither recorded nor cutting), mirroring PCRE2 NOTEMPTY where an
## empty match is not a success and backtracking continues into the
## lower-priority paths.
##
## Unicode classes come from the regex_unicode_tables module, generated
## by probing the bundled PCRE2 UCD. Callers whose reference engine is
## the Rust regex family resolve \s through the White_Space definition
## instead (the whitespaceIsRust compile flag). Known divergence: U+180E
## sits in the PCRE2 UCP \s table but
## not in White_Space (dropped in Unicode 6.3). Parity fuzz rows
## carry U+180E so a wrong \s variant cannot pass silently.

import std/[strutils, algorithm, tables]

import ./regex_unicode_tables

const
  MaxCodepoint* = 0x10FFFF'u32
  SurrogateLo = 0xD800'u32
  SurrogateHi = 0xDFFF'u32

  RustSpaceRanges* = [
    # Rust regex crate \s == Unicode White_Space (the crate documents
    # \s as \p{White_Space}). Distinct from the PCRE2 UCP table which
    # still carries U+180E.
    (0x0009'u32, 0x000D'u32), (0x0020'u32, 0x0020'u32),
    (0x0085'u32, 0x0085'u32), (0x00A0'u32, 0x00A0'u32),
    (0x1680'u32, 0x1680'u32), (0x2000'u32, 0x200A'u32),
    (0x2028'u32, 0x2029'u32), (0x202F'u32, 0x202F'u32),
    (0x205F'u32, 0x205F'u32), (0x3000'u32, 0x3000'u32),
  ]

type
  CpRanges* = object
    ## Sorted, disjoint, inclusive codepoint ranges.
    rs*: seq[tuple[lo, hi: uint32]]

  ClassId* = int32

  InstrKind* = enum
    iChar     # consume one codepoint of class cls, continue at next
    iSplit    # epsilon branch, a preferred over b
    iLookNeg  # zero-width: pass when no class codepoint at the cursor
    iDollar   # zero-width: end of window or before a final LF
    iMatch    # accept

  Instr* = object
    case kind*: InstrKind
    of iChar, iLookNeg:
      cls*: ClassId
      next*: int32
    of iSplit:
      a*, b*: int32
    of iDollar:
      after*: int32
    of iMatch:
      discard

  DfaState = object
    ## One frontier state of the deterministic compile: the ordered
    ## live-thread list of the simulation at some step, the memoized
    ## transition table over symbol contexts, and the memoized
    ## end-of-window verdict.
    threads: seq[int32]
    trans: Table[uint64, int64]
    eofDone: bool
    eofAcc: bool

  DfaEngine = ref object
    ## Deterministic frontier machine for one pattern: states
    ## are the ordered live-thread frontiers (the simulation's clist contents),
    ## transitions are the simulation's closure plus consume step
    ## memoized per symbol context, so a match is a table walk
    ## reproducing attempt() bit for bit while never re-running
    ## epsilon closures at steady state. Construction is lazy
    ## and memoized (first use compiles each reachable state once),
    ## single-threaded per pattern like the NFA scratch buffers.
    states: seq[DfaState]
    index: Table[seq[int32], int32]
    startTrans: Table[uint64, int64]
    visited: seq[int64]
    vgen: int64
    work: seq[int32]
    frontier: seq[int32]
    # Symbol table compiled once at engine attach: the membership
    # vector is constant between class range boundaries, so one
    # table read per codepoint (ASCII plane) or one binary search
    # (the planes above) replaces the per-class membership probes.
    symAscii: array[256, uint64]
    symBounds: seq[uint32]
    symVecs: seq[uint64]

  PatternCompileError* = object of ValueError

  CompiledPattern* = ref object
    ## Compiled pattern: NFA program plus class table. Scratch
    ## buffers make attempts allocation-free once warm. They are shared
    ## per compiled pattern (single-threaded stage use, no nesting).
    prog*: seq[Instr]
    classes*: seq[CpRanges]
    asciiBits: seq[array[4, uint64]]
    start*: int32
    stamps: seq[int64]
    gen: int64
    clist*, nlist*: seq[int32]
    stack*: seq[int32]
    name*: string
    # First-codepoint prefilter: the union of every iChar class reachable
    # through the start epsilon closure (lookaheads and assertions treated
    # as transparent, an over-approximation). A codepoint outside it can
    # never be consumed by any thread spawned at the start, so no non-empty
    # match can begin there and attempt() would fail anyway. Zero-width
    # acceptances are dropped by the NOTEMPTY rule and contribute nothing.
    firstRanges: CpRanges
    firstAscii: array[256, uint8]
    hasFirst: bool
    # Deterministic compile of this program, an empty shell attached
    # at compilePattern with states and transitions compiled lazily
    # on first use. nil only when the class table exceeds the packed
    # transition key width, in which case the NFA simulation stays
    # the engine.
    dfa*: DfaEngine
    # Permanent fallback latch: once a lazy determinization hits
    # MaxDfaStates the engine is abandoned and every later scan
    # takes the NFA simulation path
    # (output identical, both paths are step-for-step replicas).
    dfaOverflow*: bool

proc dfaBuildSymbols(p: CompiledPattern, e: DfaEngine)
  # forward declaration, the body sits in the frontier DFA section

# ───────────────────────────────────────────────────────────────────────
# Codepoint classes
# ───────────────────────────────────────────────────────────────────────

proc classAdd*(c: var CpRanges, lo, hi: uint32) =
  c.rs.add (lo, hi)

proc classFinish*(c: var CpRanges): CpRanges =
  ## Sorts and merges overlapping or adjacent ranges.
  c.rs.sort()
  var w = 0
  for r in c.rs.items:
    if w > 0 and r.lo <= c.rs[w - 1].hi + 1:
      if r.hi > c.rs[w - 1].hi:
        c.rs[w - 1].hi = r.hi
    else:
      c.rs[w] = r
      inc w
  c.rs.setLen(w)
  c

proc classUnion*(a, b: CpRanges): CpRanges =
  result = a
  for r in b.rs.items:
    result.rs.add r
  result = result.classFinish()

proc addComplementRange(res: var CpRanges, lo, hi: uint32)
  # forward, defined with the table helpers

proc classComplement*(a: CpRanges): CpRanges =
  ## Complement over 0x0000..0x10FFFF minus the UTF-16 surrogate block
  ## (surrogates never appear in valid UTF-8 and no class matches them).
  var cur = 0'u32
  for r in a.rs.items:
    if r.lo > cur:
      result.addComplementRange(cur, r.lo - 1)
    if r.hi >= cur:
      cur = r.hi + 1
  if cur <= MaxCodepoint:
    result.addComplementRange(cur, MaxCodepoint)
  result = result.classFinish()

proc classContains*(c: CpRanges, cp: uint32): bool {.inline.} =
  var lo = 0
  var hi = c.rs.len
  while lo < hi:
    let mid = (lo + hi) div 2
    if cp < c.rs[mid].lo:
      hi = mid
    elif cp > c.rs[mid].hi:
      lo = mid + 1
    else:
      return true
  false

proc buildAsciiBits(c: CpRanges): array[4, uint64] =
  ## ASCII-plane bitmap derived from the finished ranges, compile time
  ## only: bit cp set iff classContains(c, cp.uint32) for cp < 256.
  for r in c.rs.items:
    if r.lo <= 255'u32:
      let top = if r.hi > 255'u32: 255'u32 else: r.hi
      for cp in r.lo .. top:
        let idx = int(cp)
        result[idx shr 6] = result[idx shr 6] or
          (1'u64 shl (idx and 63))

proc classEquals(a, b: CpRanges): bool =
  if a.rs.len != b.rs.len:
    return false
  for i in 0 ..< a.rs.len:
    if a.rs[i].lo != b.rs[i].lo or a.rs[i].hi != b.rs[i].hi:
      return false
  true

proc fromTableRanges(table: openArray[tuple[lo, hi: uint32]]): CpRanges =
  for r in table.items:
    result.rs.add r

proc addComplementRange(res: var CpRanges, lo, hi: uint32) =
  ## Clips one complement range against the surrogate hole.
  if hi < SurrogateLo or lo > SurrogateHi:
    res.rs.add (lo, hi)
    return
  if lo < SurrogateLo:
    res.rs.add (lo, SurrogateLo - 1)
  if hi > SurrogateHi:
    res.rs.add (SurrogateHi + 1, hi)

# ───────────────────────────────────────────────────────────────────────
# Pattern AST
# ───────────────────────────────────────────────────────────────────────

type
  NodeKind = enum
    nAlt, nCat, nClass, nQuant, nLookNeg, nDollar, nEmpty

  Node = ref object
    case kind: NodeKind
    of nAlt, nCat:
      kids: seq[Node]
    of nClass:
      cls: CpRanges
    of nQuant:
      child: Node
      minRep, maxRep: int
      possessive: bool
    of nLookNeg:
      negCls: CpRanges
    of nDollar, nEmpty:
      discard

# ───────────────────────────────────────────────────────────────────────
# Parser (recursive descent over the supported subset)
# ───────────────────────────────────────────────────────────────────────

type
  Parser = object
    pat: string
    pos: int
    caselessDepth: int
    whitespaceIsRust: bool

proc fail(p: Parser, msg: string) {.noreturn.} =
  raise newException(PatternCompileError,
    "pattern compile error at offset " & $p.pos & " in [" & p.pat & "]: " & msg)

proc peek(p: Parser): char {.inline.} =
  if p.pos < p.pat.len: p.pat[p.pos] else: '\0'

proc peekAt(p: Parser, k: int): char {.inline.} =
  if p.pos + k < p.pat.len: p.pat[p.pos + k] else: '\0'

proc eof(p: Parser): bool {.inline.} = p.pos >= p.pat.len

proc foldClassOf(p: Parser, c: CpRanges): CpRanges
  # forward, defined after the fold tables section

proc baseClassRanges(p: Parser, name: string): CpRanges =
  ## Resolves a \p{...} name or the \s / \S builtins to table ranges.
  case name
  of "L": result = fromTableRanges(RangesL)
  of "Lu": result = fromTableRanges(RangesLu)
  of "Lt": result = fromTableRanges(RangesLt)
  of "Lm": result = fromTableRanges(RangesLm)
  of "Lo": result = fromTableRanges(RangesLo)
  of "Ll": result = fromTableRanges(RangesLl)
  of "M": result = fromTableRanges(RangesM)
  of "N": result = fromTableRanges(RangesN)
  of "P": result = fromTableRanges(RangesP)
  of "S": result = fromTableRanges(RangesS)
  of "Script=Han": result = fromTableRanges(RangesHan)
  of "s":
    if p.whitespaceIsRust:
      result = fromTableRanges(RustSpaceRanges)
    else:
      result = fromTableRanges(RangesSpace)
  else:
    p.fail("unsupported property class \\p{" & name & "}")

proc parseClassAtomEscape(p: var Parser): CpRanges =
  ## Class-valued escape right after the backslash, cursor past the text.
  inc p.pos
  let esc = p.peek()
  inc p.pos
  case esc
  of 'p', 'P':
    if p.peek() != '{':
      p.fail("\\p expects {Name}")
    inc p.pos
    var name = ""
    while not p.eof() and p.peek() != '}':
      name.add p.peek()
      inc p.pos
    if p.eof():
      p.fail("unterminated \\p{...}")
    inc p.pos
    result = p.baseClassRanges(name)
    if esc == 'P':
      result = classComplement(result)
  of 's': result = p.baseClassRanges("s")
  of 'S':
    # \S is the \s complement, distinct from \p{S} (symbols)
    if p.whitespaceIsRust:
      result = classComplement(fromTableRanges(RustSpaceRanges))
    else:
      result = classComplement(fromTableRanges(RangesSpace))
  of 'r': result.classAdd(0x0D'u32, 0x0D'u32)
  of 'n': result.classAdd(0x0A'u32, 0x0A'u32)
  of 't': result.classAdd(0x09'u32, 0x09'u32)
  else:
    if esc in {'!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',',
        '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']',
        '^', '_', '`', '{', '|', '}', '~', ' '}:
      result.classAdd(uint32(esc), uint32(esc))
    else:
      p.fail("unsupported class escape \\" & esc)

proc parseLiteralEscape(p: var Parser): uint32 =
  ## Backslash escape used as a literal codepoint.
  inc p.pos
  let esc = p.peek()
  inc p.pos
  case esc
  of 'r': result = 0x0D'u32
  of 'n': result = 0x0A'u32
  of 't': result = 0x09'u32
  of '\0':
    p.fail("trailing backslash")
  else:
    if esc in {'!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',',
        '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']',
        '^', '_', '`', '{', '|', '}', '~', ' '}:
      result = uint32(esc)
    else:
      p.fail("unsupported escape \\" & esc)

proc parseLiteralChar(p: var Parser): uint32 =
  ## One literal codepoint (no backslash), decoding UTF-8 patterns.
  let b0 = uint8(p.peek())
  if b0 < 0x80:
    inc p.pos
    return uint32(b0)
  var cp: uint32
  var width: int
  if b0 >= 0xC2 and b0 <= 0xDF: cp = uint32(b0 and 0x1F); width = 2
  elif b0 >= 0xE0 and b0 <= 0xEF: cp = uint32(b0 and 0x0F); width = 3
  elif b0 >= 0xF0 and b0 <= 0xF4: cp = uint32(b0 and 0x07); width = 4
  else:
    p.fail("invalid UTF-8 lead byte in pattern")
  for k in 1 ..< width:
    let b = uint8(p.peekAt(k))
    if (b and 0xC0) != 0x80:
      p.fail("invalid UTF-8 continuation in pattern")
    cp = (cp shl 6) or uint32(b and 0x3F)
  p.pos += width
  cp

proc foldPartnersOf(c: uint32): seq[uint32] =
  ## Simple-fold partners of one codepoint, from the probed contraction
  ## fold tables plus the ASCII case pair for letters outside them.
  result = @[]
  case c
  of 0x53, 0x73:
    for cp in FoldS.items:
      if cp != c:
        result.add cp
  of 0x44, 0x64:
    for cp in FoldD.items:
      if cp != c:
        result.add cp
  of 0x4D, 0x6D:
    for cp in FoldM.items:
      if cp != c:
        result.add cp
  of 0x54, 0x74:
    for cp in FoldT.items:
      if cp != c:
        result.add cp
  of 0x4C, 0x6C:
    for cp in FoldL.items:
      if cp != c:
        result.add cp
  of 0x56, 0x76:
    for cp in FoldV.items:
      if cp != c:
        result.add cp
  of 0x52, 0x72:
    for cp in FoldR.items:
      if cp != c:
        result.add cp
  of 0x45, 0x65:
    for cp in FoldE.items:
      if cp != c:
        result.add cp
  else:
    if (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A):
      result.add(if c >= 0x61: c - 0x20 else: c + 0x20)

proc literalToClass(p: Parser, c: uint32): CpRanges =
  ## Literal codepoint to class, applying fold partners inside caseless
  ## groups (PCRE2 UCP simple folding, e.g. the s fold class carries
  ## U+017F).
  if p.caselessDepth > 0:
    result.classAdd(c, c)
    for cp in foldPartnersOf(c).items:
      result.classAdd(cp, cp)
    result = result.classFinish()
  else:
    result.classAdd(c, c)

proc parseClassBracket(p: var Parser): CpRanges =
  ## [...] with negation, ranges and \p / \s / escape members.
  doAssert p.peek() == '['
  inc p.pos
  var neg = false
  if p.peek() == '^':
    neg = true
    inc p.pos
  var acc: CpRanges
  while true:
    if p.eof():
      p.fail("unterminated [...]")
    if p.peek() == ']':
      inc p.pos
      break
    var atomLo: uint32
    var atomHi: uint32
    if p.peek() == '\\':
      let cls = p.parseClassAtomEscape()
      if p.peek() == '-' and p.peekAt(1) != ']':
        p.fail("cannot build a range over a multi-codepoint class atom")
      acc = classUnion(acc, cls)
      continue
    else:
      atomLo = p.parseLiteralChar()
      atomHi = atomLo
    if p.peek() == '-' and p.peekAt(1) != ']':
      inc p.pos
      if p.peek() == '\\':
        atomHi = p.parseLiteralEscape()
      else:
        atomHi = p.parseLiteralChar()
      if atomHi < atomLo:
        p.fail("reversed class range")
    acc.classAdd(atomLo, atomHi)
  if p.caselessDepth > 0:
    acc = p.foldClassOf(acc)
  if neg:
    result = classComplement(acc)
  else:
    result = acc.classFinish()

proc parseAltInto(p: var Parser): Node
  # forward, used by the group parsers

proc parseGroup(p: var Parser): Node =
  ## Cursor on '('.
  inc p.pos
  if p.peek() == '?':
    inc p.pos
    case p.peek()
    of ':':
      inc p.pos
      result = p.parseAltInto()
      if p.peek() != ')':
        p.fail("unterminated (?:...)")
      inc p.pos
    of 'i':
      inc p.pos
      if p.peek() != ':':
        p.fail("(?i expects :")
      inc p.pos
      inc p.caselessDepth
      result = p.parseAltInto()
      dec p.caselessDepth
      if p.peek() != ')':
        p.fail("unterminated (?i:...)")
      inc p.pos
    of '!':
      inc p.pos
      # single-codepoint negative lookahead: complement class plus an
      # end-of-window test, fully regular
      var negCls: CpRanges
      if p.peek() == '[':
        negCls = p.parseClassBracket()
      elif p.peek() == '\\':
        negCls = p.parseClassAtomEscape()
      else:
        p.fail("lookahead expects one class atom")
      if p.peek() != ')':
        p.fail("only single-class negative lookaheads are supported")
      inc p.pos
      result = Node(kind: nLookNeg, negCls: negCls)
    else:
      p.fail("unsupported group (?..." & p.peek() & ")")
  else:
    # bare group behaves as non-capturing
    result = p.parseAltInto()
    if p.peek() != ')':
      p.fail("unterminated (...)")
    inc p.pos

proc parseAtom(p: var Parser): Node =
  case p.peek()
  of '(':
    p.parseGroup()
  of '[':
    Node(kind: nClass, cls: p.parseClassBracket())
  of '\\':
    let esc = p.peekAt(1)
    if esc in {'p', 'P', 's', 'S'}:
      Node(kind: nClass, cls: p.parseClassAtomEscape())
    elif esc in {'r', 'n', 't'}:
      Node(kind: nClass, cls: p.parseClassAtomEscape())
    else:
      Node(kind: nClass, cls: p.literalToClass(p.parseLiteralEscape()))
  of '$':
    inc p.pos
    Node(kind: nDollar)
  of '|', ')':
    p.fail("unexpected alternation or group end")
  of '*', '+', '?', '{':
    p.fail("quantifier with nothing to quantify")
  else:
    Node(kind: nClass, cls: p.literalToClass(p.parseLiteralChar()))

proc parseQuant(p: var Parser, atom: Node): Node =
  ## Reads an optional quantifier suffix: greedy by default, possessive
  ## suffix + accepted (compiled as greedy, see module docstring), lazy
  ## ? rejected with a compile error.
  var minRep = 1
  var maxRep = 1
  case p.peek()
  of '*': minRep = 0; maxRep = -1; inc p.pos
  of '+': minRep = 1; maxRep = -1; inc p.pos
  of '?': minRep = 0; maxRep = 1; inc p.pos
  of '{':
    var probe = p.pos
    var m = ""
    var n = ""
    var ok = false
    inc probe
    while probe < p.pat.len and p.pat[probe] in {'0'..'9'}:
      m.add p.pat[probe]
      inc probe
    if probe < p.pat.len and p.pat[probe] == ',':
      inc probe
      while probe < p.pat.len and p.pat[probe] in {'0'..'9'}:
        n.add p.pat[probe]
        inc probe
      if probe < p.pat.len and p.pat[probe] == '}' and m.len > 0:
        ok = true
    elif probe < p.pat.len and p.pat[probe] == '}' and m.len > 0:
      ok = true
      n = m
    if not ok:
      p.fail("unsupported {..} construct")
    minRep = parseInt(m)
    maxRep = if n.len == 0: -1 else: parseInt(n)
    if maxRep != -1 and maxRep < minRep:
      p.fail("reversed counted repetition {m,n}")
    p.pos = probe + 1
  else:
    return atom
  if p.peek() == '?':
    p.fail("lazy quantifiers are not supported")
  var possessive = false
  if p.peek() == '+':
    possessive = true
    inc p.pos
  Node(kind: nQuant, child: atom, minRep: minRep, maxRep: maxRep,
    possessive: possessive)

proc parseCat(p: var Parser): Node =
  var kids: seq[Node]
  while not p.eof() and p.peek() notin {'|', ')'}:
    let atom = p.parseAtom()
    kids.add p.parseQuant(atom)
  if kids.len == 0:
    Node(kind: nEmpty)
  elif kids.len == 1:
    kids[0]
  else:
    Node(kind: nCat, kids: kids)

proc parseAltInto(p: var Parser): Node =
  var kids = @[p.parseCat()]
  while p.peek() == '|':
    inc p.pos
    kids.add p.parseCat()
  if kids.len == 1:
    kids[0]
  else:
    Node(kind: nAlt, kids: kids)

proc foldClassOf(p: Parser, c: CpRanges): CpRanges =
  ## Caseless expansion of a character class: every member is replaced
  ## by its simple-fold class (PCRE2 UCP caseless classes). Members with
  ## no probed fold data fail with a compile error rather than
## silently diverging.
  for r in c.rs.items:
    var cp = r.lo
    while true:
      let partners = foldPartnersOf(cp)
      if partners.len == 0 and cp >= 0x80:
        p.fail("caseless class member without fold data: U+" &
          toHex(int(cp), 4))
      result.classAdd(cp, cp)
      for partner in partners.items:
        result.classAdd(partner, partner)
      if cp == r.hi:
        break
      inc cp
  result = result.classFinish()

# ───────────────────────────────────────────────────────────────────────
# NFA compiler
# ───────────────────────────────────────────────────────────────────────

type
  HoleTarget = tuple[instr: int, field: int]
    ## field 0 = next of iChar / iLookNeg / iDollar, 1 = split a,
    ## 2 = split b

  Frag = object
    entry: int32
    holes: seq[HoleTarget]

proc patchTarget(prog: var seq[Instr], h: HoleTarget, v: int32) =
  case prog[h.instr].kind
  of iChar, iLookNeg:
    prog[h.instr].next = v
  of iDollar:
    prog[h.instr].after = v
  of iSplit:
    if h.field == 1:
      prog[h.instr].a = v
    else:
      prog[h.instr].b = v
  of iMatch:
    doAssert false, "cannot patch an iMatch instruction"

proc addInstr(prog: var seq[Instr], i: Instr): int =
  prog.add i
  prog.len - 1

proc lower(p: var Parser, prog: var seq[Instr], clsTab: var seq[CpRanges],
    n: Node): Frag

proc internClass(p: Parser, clsTab: var seq[CpRanges], c: CpRanges): ClassId =
  for i in 0 ..< clsTab.len:
    if classEquals(clsTab[i], c):
      return int32(i)
  clsTab.add c
  int32(clsTab.len - 1)

proc chainFrag(acc: var Frag, prog: var seq[Instr], f: Frag) =
  ## Concatenates fragment f after acc (acc entry stays, acc holes now
  ## point at f).
  if acc.entry == -1:
    acc = f
    return
  for h in acc.holes.items:
    prog.patchTarget(h, f.entry)
  acc.holes = f.holes

proc lowerStar(p: var Parser, prog: var seq[Instr],
    clsTab: var seq[CpRanges], child: Node): Frag =
  ## Greedy star: split with the body preferred, body loops back.
  let splitIdx = prog.addInstr(Instr(kind: iSplit, a: -1, b: -1))
  let bodyFrag = p.lower(prog, clsTab, child)
  if bodyFrag.entry == -1:
    # an empty body loops forever, unsupported
    p.fail("quantifier over an empty subpattern")
  for h in bodyFrag.holes.items:
    prog.patchTarget(h, int32(splitIdx))
  prog[splitIdx].a = bodyFrag.entry
  Frag(entry: int32(splitIdx), holes: @[(splitIdx, 2)])

proc lowerQuant(p: var Parser, prog: var seq[Instr], clsTab: var seq[CpRanges],
    n: Node): Frag =
  if n.child.kind == nEmpty:
    return Frag(entry: -1)
  if n.maxRep == -1:
    if n.minRep == 0:
      return p.lowerStar(prog, clsTab, n.child)
    # {m,} = m mandatory copies then a greedy star
    var acc = Frag(entry: -1)
    for i in 0 ..< n.minRep:
      let f = p.lower(prog, clsTab, n.child)
      acc.chainFrag(prog, f)
    if acc.entry == -1:
      return p.lowerStar(prog, clsTab, n.child)
    let starFrag = p.lowerStar(prog, clsTab, n.child)
    for h in acc.holes.items:
      prog.patchTarget(h, starFrag.entry)
    return Frag(entry: acc.entry, holes: starFrag.holes)
  # bounded {m,n}: m mandatory copies then (n - m) greedy optionals
  var acc = Frag(entry: -1)
  for i in 0 ..< n.minRep:
    let f = p.lower(prog, clsTab, n.child)
    acc.chainFrag(prog, f)
  for i in 0 ..< (n.maxRep - n.minRep):
    if acc.entry == -1:
      # leading optionals: a ? -shaped fragment
      let splitIdx = prog.addInstr(Instr(kind: iSplit, a: -1, b: -1))
      let bodyFrag = p.lower(prog, clsTab, n.child)
      prog[splitIdx].a = bodyFrag.entry
      acc = Frag(entry: int32(splitIdx),
        holes: bodyFrag.holes & @[(splitIdx, 2)])
    else:
      let splitIdx = prog.addInstr(Instr(kind: iSplit, a: -1, b: -1))
      let bodyFrag = p.lower(prog, clsTab, n.child)
      for h in acc.holes.items:
        prog.patchTarget(h, int32(splitIdx))
      prog[splitIdx].a = bodyFrag.entry
      acc.holes = bodyFrag.holes & @[(splitIdx, 2)]
  result = acc

proc lower(p: var Parser, prog: var seq[Instr], clsTab: var seq[CpRanges],
    n: Node): Frag =
  case n.kind
  of nClass:
    let cid = p.internClass(clsTab, n.cls)
    let idx = prog.addInstr(Instr(kind: iChar, cls: cid, next: -1))
    result = Frag(entry: int32(idx), holes: @[(idx, 0)])
  of nLookNeg:
    let cid = p.internClass(clsTab, n.negCls)
    let idx = prog.addInstr(Instr(kind: iLookNeg, cls: cid, next: -1))
    result = Frag(entry: int32(idx), holes: @[(idx, 0)])
  of nDollar:
    let idx = prog.addInstr(Instr(kind: iDollar, after: -1))
    result = Frag(entry: int32(idx), holes: @[(idx, 0)])
  of nAlt:
    var frags: seq[Frag]
    for kid in n.kids.items:
      let f = p.lower(prog, clsTab, kid)
      if f.entry == -1:
        p.fail("empty alternation branch")
      frags.add f
    if frags.len == 1:
      result = frags[0]
    else:
      var holes: seq[HoleTarget]
      for f in frags.items:
        holes.add f.holes
      # dispatch chain: split preferring each branch in declaration
      # order, the last branch entered by the final split fall-through
      var dispatches: seq[int]
      for i in 0 ..< frags.len - 1:
        dispatches.add prog.addInstr(
          Instr(kind: iSplit, a: frags[i].entry, b: -1))
      for i in 0 ..< dispatches.len - 1:
        prog[dispatches[i]].b = int32(dispatches[i + 1])
      prog[dispatches[^1]].b = frags[^1].entry
      result = Frag(entry: int32(dispatches[0]), holes: holes)
  of nCat:
    result = Frag(entry: -1)
    for kid in n.kids.items:
      let f = p.lower(prog, clsTab, kid)
      result.chainFrag(prog, f)
  of nQuant:
    result = p.lowerQuant(prog, clsTab, n)
  of nEmpty:
    result = Frag(entry: -1)

proc computeFirstRanges(p: CompiledPattern): CpRanges =
  ## Union of the iChar classes in the epsilon closure of the start
  ## state. Compile-time only, runs once per pattern.
  if p.prog.len == 0:
    return
  var visited = newSeq[bool](p.prog.len)
  var stack = @[p.start]
  while stack.len > 0:
    let pc = stack.pop()
    if pc < 0 or visited[pc]:
      continue
    visited[pc] = true
    case p.prog[pc].kind
    of iChar:
      result = classUnion(result, p.classes[p.prog[pc].cls])
    of iSplit:
      stack.add p.prog[pc].a
      stack.add p.prog[pc].b
    of iLookNeg:
      stack.add p.prog[pc].next
    of iDollar:
      stack.add p.prog[pc].after
    of iMatch:
      discard

proc compilePattern*(pattern: string, whitespaceIsRust = false,
    name = "anonymous"): CompiledPattern =
  ## Parses and lowers one pattern to an NFA program.
  var p = Parser(pat: pattern, whitespaceIsRust: whitespaceIsRust)
  var prog: seq[Instr]
  var clsTab: seq[CpRanges]
  let frag = p.lower(prog, clsTab, p.parseAltInto())
  let matchIdx = prog.addInstr(Instr(kind: iMatch))
  var start: int32
  if frag.entry == -1:
    # pattern matching only the empty string: accept immediately
    prog = @[Instr(kind: iMatch)]
    start = 0
  else:
    for h in frag.holes.items:
      prog.patchTarget(h, int32(matchIdx))
    start = frag.entry
  result = CompiledPattern(prog: prog, classes: clsTab, start: start)
  result.asciiBits = newSeq[array[4, uint64]](result.classes.len)
  for i in 0 ..< result.classes.len:
    result.asciiBits[i] = buildAsciiBits(result.classes[i])
  result.stamps = newSeq[int64](result.prog.len)
  result.name = name
  let first = computeFirstRanges(result)
  if first.rs.len > 0:
    result.hasFirst = true
    result.firstRanges = first
    for cp in 0'u32 .. 255'u32:
      result.firstAscii[int(cp)] =
        (if first.classContains(cp): 1'u8 else: 0'u8)
  # Engine shell at compile time (cheap: no state compiled at this point).
  # The packed transition key spends one bit per interned class plus one
  # flag bit in 64 bits, so patterns wider than that keep the NFA
  # simulation (the served patterns stay far below the width).
  if result.classes.len <= 62:
    result.dfa = DfaEngine(visited: newSeq[int64](result.prog.len))
    dfaBuildSymbols(result, result.dfa)

# ───────────────────────────────────────────────────────────────────────
# Leftmost-first matcher
# ───────────────────────────────────────────────────────────────────────

proc decodeCp*(input: string, pos, hi: int): tuple[cp: uint32, width: int] =
  ## Decodes one UTF-8 codepoint at pos. Byte sequences no valid UTF-8
  ## char decodes to yield a sentinel codepoint that matches no class
  ## (callers only feed valid UTF-8, invalid bytes fail all paths).
  if pos >= hi:
    return (0'u32, 0)
  let b0 = uint8(input[pos])
  if b0 < 0x80:
    return (uint32(b0), 1)
  var cp: uint32
  var width: int
  if b0 >= 0xC2 and b0 <= 0xDF: cp = uint32(b0 and 0x1F'u8); width = 2
  elif b0 >= 0xE0 and b0 <= 0xEF: cp = uint32(b0 and 0x0F'u8); width = 3
  elif b0 >= 0xF0 and b0 <= 0xF4: cp = uint32(b0 and 0x07'u8); width = 4
  else:
    return (0xFFFFFFFF'u32, 1)
  if pos + width > hi:
    return (0xFFFFFFFF'u32, 1)
  for k in 1 ..< width:
    let b = uint8(input[pos + k])
    if (b and 0xC0'u8) != 0x80'u8:
      return (0xFFFFFFFF'u32, 1)
    cp = (cp shl 6) or uint32(b and 0x3F'u8)
  (cp, width)

proc classHas*(p: CompiledPattern, cls: ClassId, cp: uint32): bool {.inline.} =
  ## Bitmap-first class membership for the matcher hot path. The bitmap
  ## is derived at compile time from the same finished ranges classContains
  ## binary-searches, so the two predicates agree bit for bit on the whole
  ## ASCII plane (the fuzz suite pins the explicit equivalence).
  if cp < 256:
    let idx = int(cp)
    return (p.asciiBits[cls][idx shr 6] and
      (1'u64 shl (idx and 63))) != 0
  p.classes[cls].classContains(cp)

proc attempt(p: CompiledPattern, input: string, winLo, winHi, s: int): int =
  ## One anchored leftmost-first attempt starting at byte offset s
  ## (a codepoint boundary inside [winLo, winHi)). Returns the match end
  ## or -1 when nothing non-empty matches from s.
  doAssert s >= winLo and s <= winHi
  var matchedEnd = -1
  var pos = s
  p.clist.setLen(0)
  p.clist.add p.start
  while true:
    # single-iChar-thread fast path: with exactly one live waiter, one
    # whose instruction consumes, no epsilon closure can fire and no
    # accept sits upstream, so the generic slot machinery below rebuilds
    # exactly this step: a one-element nlist, the byte filter, a plain
    # advance. Both exits replicate the generic loop. A waiter moving
    # past the window breaks with the recorded extent, an empty nlist.
    # A class miss breaks the same way. A following iMatch or split
    # falls back to the generic path on the next iteration.
    if p.clist.len == 1 and p.prog[p.clist[0]].kind == iChar:
      if pos >= winHi:
        break
      let decoded = decodeCp(input, pos, winHi)
      let pc = p.clist[0]
      if not p.classHas(p.prog[pc].cls, decoded.cp):
        break
      p.clist[0] = p.prog[pc].next
      pos += decoded.width
      continue
    # epsilon closure, slot by slot, each slot's subtree expanded
    # depth-first so pattern preference order is preserved
    p.gen += 1
    let g = p.gen
    p.nlist.setLen(0)
    var cut = false
    for i in 0 ..< p.clist.len:
      if cut:
        break
      p.stack.setLen(0)
      p.stack.add p.clist[i]
      while p.stack.len > 0:
        let pc = p.stack.pop()
        if p.stamps[pc] == g:
          continue
        p.stamps[pc] = g
        case p.prog[pc].kind
        of iSplit:
          p.stack.add p.prog[pc].b
          p.stack.add p.prog[pc].a
        of iLookNeg:
          var passes: bool
          if pos >= winHi:
            passes = true
          else:
            let decoded = decodeCp(input, pos, winHi)
            passes = not p.classHas(p.prog[pc].cls, decoded.cp)
          if passes:
            p.stack.add p.prog[pc].next
        of iDollar:
          if pos == winHi or (pos == winHi - 1 and input[pos] == '\n'):
            p.stack.add p.prog[pc].after
        of iChar:
          if pos < winHi:
            p.nlist.add pc
        of iMatch:
          if pos > s:
            matchedEnd = pos
            cut = true
          # a zero-width acceptance is dropped (NOTEMPTY)
        if cut:
          break
    # an acceptance kills every lower-priority thread (the remaining
    # slots and the rest of this slot subtree) but the attempt keeps
    # running: surviving higher-priority threads can still accept and
    # override the recorded extent (greedy loops keep extending)
    if p.nlist.len == 0:
      break
    # consume one codepoint for the surviving char-waiters
    p.gen += 1
    let g2 = p.gen
    let decoded = decodeCp(input, pos, winHi)
    var newLen = 0
    for pc in p.nlist.items:
      if p.classHas(p.prog[pc].cls, decoded.cp):
        let t = p.prog[pc].next
        if p.stamps[t] != g2:
          p.stamps[t] = g2
          p.nlist[newLen] = t
          inc newLen
    p.nlist.setLen(newLen)
    swap(p.clist, p.nlist)
    pos += decoded.width
    if p.clist.len == 0:
      break
  result = matchedEnd

# ───────────────────────────────────────────────────────────────────────
# Frontier DFA: deterministic compile of the leftmost-first simulation
# ───────────────────────────────────────────────────────────────────────
#
# Why not a powerset DFA: a powerset DFA is a longest-match machine
# and the oracle scan is leftmost-first, whose extent differs
# from the longest extent whenever alternatives overlap
# as in the GPT-2 trailing whitespace group or the twin letter
# alternatives of the o200k / Kimi / Moonlight patterns.
# This compile keeps the simulation's leftmost-first semantics
# by determinizing its ordered frontiers instead of plain instruction sets:
# frontiers instead of plain instruction sets: a DFA state is the exact
# ordered live-thread list
# the simulation carries (slot preference order, first-occurrence dedup),
# and a transition is the simulation's closure-plus-consume step
# for one symbol context, memoized. Matching is a table walk, epsilon
# closures never re-run at steady state, and every recorded extent,
# cut and truncation happens at the same step as in the NFA simulation,
# which makes the two paths step-for-step replicas
# (the fuzz suite pins byte-identical segmentation).
#
# Symbol contexts: one context = the membership vector of the current
# codepoint over the interned class table plus the dollar-window flag
# (the current codepoint is a LF that ends the window). Lookahead
# and end-of-window behavior fold into the context, so the transition
# function is a pure function of (frontier, context) and memoization
# is sound. The end-of-window context is handled by a per-state verdict
# instead of a table entry (no codepoint is consumed there).

const
  MaxDfaStates = 50_000
  ## Lazy determinization cap per pattern. The served patterns compile
  ## to modest frontier counts on ordinary text, so the cap exists only
  ## to bound pathological inputs: hitting it permanently reverts
  ## that pattern to the NFA simulation path
  ## (same output, the NFA path is the step-for-step replica).

proc dfaBuildSymbols(p: CompiledPattern, e: DfaEngine) =
  ## Compiles the symbol table once at engine attach: sorted class
  ## range boundaries partition the codepoint space into cells
  ## with a constant membership vector, one vector per cell. A linear
  ## sweep over per-class range events fills the vectors, no per-cell
  ## searches, vectors derived from the same finished ranges
  ## classHas bitmaps were derived from. The plane above the last
  ## boundary and the invalid-byte sentinel read as the all-zero
  ## vector, exactly like classContains over them.
  type SymEvent = tuple[cp: uint32, cls: int32, start: bool]
  var events: seq[SymEvent]
  for c in 0 ..< p.classes.len:
    for r in p.classes[c].rs.items:
      events.add (r.lo, int32(c), true)
      events.add (r.hi + 1, int32(c), false)
  events.sort(system.cmp)
  var active = newSeq[int32](p.classes.len)
  var vec = 0'u64
  var bounds = @[0'u32]
  var vecs: seq[uint64]
  vecs.add vec
  var i = 0
  while i < events.len:
    let cp = events[i].cp
    while i < events.len and events[i].cp == cp:
      let cls = events[i].cls
      if events[i].start:
        active[cls] += 1
        if active[cls] == 1:
          vec = vec or (1'u64 shl cls)
      else:
        active[cls] -= 1
        if active[cls] == 0:
          vec = vec and (not (1'u64 shl cls))
      inc i
    bounds.add cp
    vecs.add vec
  for c in 0 ..< p.classes.len:
    doAssert active[c] == 0
  # sentinel boundary: the plane above the last event is one more cell
  # with the all-zero vector, so every codepoint up to MaxCodepoint
  # resolves inside a searched cell and the invalid-byte sentinel plane
  # above reads all-zero like classContains
  bounds.add MaxCodepoint + 1
  vecs.add 0'u64
  for i in 0 ..< vecs.len - 1:
    let cellStart = bounds[i]
    let cellEnd = bounds[i + 1]
    let cellVec = vecs[i]
    if cellStart < 256:
      # ASCII plane cells fill the direct table, a cell straddling
      # the 256 boundary also enters the searched table at 256
      var f = cellStart
      let top = min(cellEnd, 256'u32)
      while f < top:
        e.symAscii[int(f)] = cellVec
        inc f
    if cellEnd > 256:
      e.symBounds.add max(cellStart, 256'u32)
      e.symVecs.add cellVec

proc dfaSymbolBits(p: CompiledPattern, cp: uint32): uint64 {.inline.} =
  ## Membership vector of one codepoint over the interned class table,
  ## read off the compiled symbol table
  ## (two codepoints with the same vector are interchangeable symbols for the program).
  if cp < 256:
    return p.dfa.symAscii[int(cp)]
  if cp > MaxCodepoint:
    return 0
  var lo = 0
  var hi = p.dfa.symBounds.len
  while lo < hi:
    let mid = (lo + hi) div 2
    if cp < p.dfa.symBounds[mid]:
      hi = mid
    else:
      lo = mid + 1
  p.dfa.symVecs[lo - 1]

proc dfaStep(p: CompiledPattern, e: DfaEngine, threads: openArray[int32],
    bits: uint64, dollarOk: bool, atEnd: bool, firstStep: bool): bool =
  ## One deterministic replication of the simulation step: epsilon
  ## closure slot by slot in frontier order with first-visit dedup
  ## and the cut rule at an acceptance, then the consume stage
  ## (filter by class membership, map to continuations, first-occurrence dedup).
  ## Writes the next frontier into e.frontier and returns whether
  ## an acceptance fired (the caller records the extent at its own position).
  ## Scratch buffers live in the engine, reused per step.
  e.vgen += 1
  let g = e.vgen
  e.frontier.setLen(0)
  var cut = false
  for i in 0 ..< threads.len:
    if cut:
      break
    e.work.setLen(0)
    e.work.add threads[i]
    while e.work.len > 0:
      let pc = e.work.pop()
      if e.visited[pc] == g:
        continue
      e.visited[pc] = g
      case p.prog[pc].kind
      of iSplit:
        e.work.add p.prog[pc].b
        e.work.add p.prog[pc].a
      of iLookNeg:
        let passes = atEnd or
          ((bits shr int(p.prog[pc].cls)) and 1'u64) == 0
        if passes:
          e.work.add p.prog[pc].next
      of iDollar:
        if atEnd or dollarOk:
          e.work.add p.prog[pc].after
      of iChar:
        if not atEnd:
          e.frontier.add pc
      of iMatch:
        if not firstStep:
          cut = true
      if cut:
        break
  if not atEnd:
    e.vgen += 1
    let g2 = e.vgen
    var newLen = 0
    for k in 0 ..< e.frontier.len:
      let pc = e.frontier[k]
      if ((bits shr int(p.prog[pc].cls)) and 1'u64) != 0:
        let t = p.prog[pc].next
        if e.visited[t] != g2:
          e.visited[t] = g2
          e.frontier[newLen] = t
          inc newLen
    e.frontier.setLen(newLen)
  result = cut

proc dfaIntern(p: CompiledPattern, e: DfaEngine): int32 =
  ## Interns the frontier just computed into the state table. -1 means
  ## dead (empty frontier) or the determinization cap
  ## was reached. The stored copy never aliases the engine scratch
  ## buffers (they are reused and mutated by the next step).
  if e.frontier.len == 0:
    return -1
  result = e.index.getOrDefault(e.frontier, int32(-1))
  if result >= 0:
    return result
  if e.states.len >= MaxDfaStates:
    p.dfaOverflow = true
    return -1
  var copy = newSeq[int32](e.frontier.len)
  for k in 0 ..< e.frontier.len:
    copy[k] = e.frontier[k]
  result = int32(e.states.len)
  e.index[copy] = result
  e.states.add DfaState(threads: copy)

proc dfaStartState(p: CompiledPattern, e: DfaEngine, bits: uint64,
    dollarOk: bool): int32 =
  ## Initial frontier for one anchored attempt: the closure
  ## of the start thread at the attempt position with accepts inert
  ## (NOTEMPTY drops them and no cut fires at the attempt position),
  ## then the consume stage. Memoized per symbol context.
  var key = bits shl 1
  if dollarOk:
    key = key or 1
  let packedV = e.startTrans.getOrDefault(key, low(int64))
  if packedV != low(int64):
    return int32(packedV shr 1) - 1
  var seed: array[1, int32]
  seed[0] = p.start
  discard dfaStep(p, e, seed, bits, dollarOk, atEnd = false,
    firstStep = true)
  let id = dfaIntern(p, e)
  e.startTrans[key] = (int64(id) + 1) shl 1
  result = id

proc dfaStepState(p: CompiledPattern, e: DfaEngine, sid: int32,
    bits: uint64, dollarOk: bool): int64 =
  ## Memoized transition of one state on one symbol context. Packed
  ## result: (state id + 1) shifted left once with the acceptance bit
  ## in the low bit, so a dead continuation packs to 0 or 1
  ## and every present entry is nonnegative.
  var key = bits shl 1
  if dollarOk:
    key = key or 1
  result = e.states[sid].trans.getOrDefault(key, low(int64))
  if result != low(int64):
    return
  let accepted = dfaStep(p, e, e.states[sid].threads, bits, dollarOk,
    atEnd = false, firstStep = false)
  let id = dfaIntern(p, e)
  result = (int64(id) + 1) shl 1
  if accepted:
    result = result or 1
  e.states[sid].trans[key] = result

proc dfaEofAccept(p: CompiledPattern, e: DfaEngine, sid: int32): bool =
  ## End-of-window verdict of one state, memoized: the closure runs
  ## with every lookahead and the dollar assertion passing and no
  ## consuming waiter collected (the window is exhausted),
  ## so the verdict is whether an acceptance fired. Nothing survives
  ## the step, the attempt ends there exactly like the NFA loop.
  if e.states[sid].eofDone:
    return e.states[sid].eofAcc
  let accepted = dfaStep(p, e, e.states[sid].threads, 0'u64, false,
    atEnd = true, firstStep = false)
  e.states[sid].eofDone = true
  e.states[sid].eofAcc = accepted
  result = accepted

proc attemptDfa(p: CompiledPattern, input: string, winLo, winHi, s: int): int =
  ## One anchored leftmost-first attempt as a frontier walk
  ## over the deterministic compile: initial frontier at s, then one
  ## memoized table step per consumed codepoint with the recorded
  ## extent refreshed at every accepted step.
  ## Extents grow with the position, so the last accepted step
  ## carries the final extent, exactly like the cut-and-override
  ## bookkeeping of the NFA simulation. Returns -2 on a determinization
  ## cap hit, which flips the pattern to the permanent NFA fallback.
  doAssert s >= winLo and s <= winHi
  let e = p.dfa
  if s >= winHi:
    return -1
  var matchedEnd = -1
  var pos = s
  let d0 = decodeCp(input, pos, winHi)
  let bits0 = dfaSymbolBits(p, d0.cp)
  let dollar0 = d0.cp == 0x0A'u32 and pos + d0.width == winHi
  var sid = dfaStartState(p, e, bits0, dollar0)
  if p.dfaOverflow:
    return -2
  if sid < 0:
    return -1
  inc pos, d0.width
  while true:
    if pos >= winHi:
      if dfaEofAccept(p, e, sid):
        matchedEnd = pos
      break
    let d = decodeCp(input, pos, winHi)
    let bits = dfaSymbolBits(p, d.cp)
    let dollarOk = d.cp == 0x0A'u32 and pos + d.width == winHi
    let packedOut = dfaStepState(p, e, sid, bits, dollarOk)
    if p.dfaOverflow:
      return -2
    if (packedOut and 1) != 0:
      matchedEnd = pos
    let nid = int32(packedOut shr 1) - 1
    if nid < 0:
      break
    sid = nid
    pos += d.width
  result = matchedEnd

proc matchAt*(p: CompiledPattern, input: string, winLo, winHi, pos: int): int =
  ## One anchored leftmost-first attempt at the codepoint boundary pos:
  ## returns the match end (> pos) or -1 when nothing non-empty matches
  ## from pos. Example: pattern `\s+\s` over "  x" at pos 0 returns 2.
  ## The window [winLo, winHi) binds the assertions exactly like nextMatch:
  ## `$` matches at winHi or before a final LF, and no match extends past
  ## winHi (callers feeding slices of one buffer get per-slice haystack
  ## semantics). Frontier-DFA walk when the deterministic compile is live,
  ## the NFA simulation otherwise; a determinization cap hit mid-call
  ## latches the permanent fallback and returns the NFA verdict.
  doAssert winLo <= pos and pos <= winHi and winHi <= input.len
  if p.dfa != nil and not p.dfaOverflow:
    let e = p.attemptDfa(input, winLo, winHi, pos)
    if e != -2:
      return e
    p.dfaOverflow = true
  p.attempt(input, winLo, winHi, pos)

proc dfaStats*(p: CompiledPattern): tuple[states, edges: int] =
  ## Build receipt of the deterministic compile
  ## (zeros before first use, populated by the scans that follow).
  if p.dfa == nil:
    return (0, 0)
  var edges = 0
  for st in p.dfa.states.items:
    edges += st.trans.len
  (p.dfa.states.len, edges)

proc dfaMemoryEstimate*(p: CompiledPattern): int =
  ## Rough resident estimate of the deterministic compile in bytes:
  ## per state the thread list plus its memo table rows
  ## (key + packed value + load-factor slack), plus the interning
  ## index rows. A bound for receipts, not an exact ledger.
  if p.dfa == nil:
    return 0
  for st in p.dfa.states.items:
    result += st.threads.len * 4 + 96 + st.trans.len * 24
  result += p.dfa.index.len * 48

proc nextMatchImpl(p: CompiledPattern, input: string, fromPos: int,
    winLo = 0, winHi = -1, useDfa = true): tuple[start, stop: int] =
  ## Scan driver shared by the frontier-DFA walk (default) and the NFA
  ## simulation
  ## (equivalence oracle of the parity suites plus the overflow fallback).
  ## Leftmost scan in PCRE2's find-all semantics: the first
  ## position >= fromPos admitting a non-empty match wins,
  ## with the pattern-preference extent. (-1, -1) when no position
  ## matches; a splitting caller emits the gap before each match as
  ## pieces and the unmatched remainder becomes one trailing piece
  ## (the PCRE2 iterator stops at the first unmatched offset).
  let hi = if winHi < 0: input.len else: winHi
  var s = fromPos
  while s < hi:
    if p.hasFirst:
      # sound prefilter: a start codepoint outside the reachable
      # first-class union cannot begin a non-empty match, the attempt
      # would fail anyway
      let d = decodeCp(input, s, hi)
      let can = (d.cp < 256 and p.firstAscii[int(d.cp)] != 0) or
        (d.cp >= 256 and p.firstRanges.classContains(d.cp))
      if not can:
        inc s, d.width
        continue
    let e = if useDfa: p.attemptDfa(input, winLo, hi, s)
            else: p.attempt(input, winLo, hi, s)
    if e == -2:
      # determinization cap reached mid-scan: permanent NFA fallback
      # for this pattern, this call restarts clean
      # (the scan is a pure function of the window, a restart cannot change output)
      p.dfaOverflow = true
      return nextMatchImpl(p, input, fromPos, winLo, winHi,
        useDfa = false)
    if e > s:
      return (s, e)
    inc s
  (-1, -1)

proc nextMatch*(p: CompiledPattern, input: string, fromPos: int,
    winLo = 0, winHi = -1): tuple[start, stop: int] =
  ## DFA-backed scan driver (default after the equivalence proofs of the fuzz suite).
  ## Falls back to the NFA simulation only
  ## through the permanent dfaOverflow latch.
  nextMatchImpl(p, input, fromPos, winLo, winHi, not p.dfaOverflow)

proc nextMatchNfa*(p: CompiledPattern, input: string, fromPos: int,
    winLo = 0, winHi = -1): tuple[start, stop: int] =
  ## NFA-simulation scan driver, kept as the fuzz-suite equivalence
  ## oracle and the overflow fallback path (same window semantics).
  nextMatchImpl(p, input, fromPos, winLo, winHi, useDfa = false)

# ───────────────────────────────────────────────────────────────────────
# Scan machine: object + ctor + ONE items over the scan drivers
# ───────────────────────────────────────────────────────────────────────

type
  RegexScanner* = object
    ## Match-position machine over one compiled pattern: one items
    ## iteration yields the leftmost-first match spans in scan order.
    ## The cursor is the scan position between yields and advances
    ## before each yield, so a consumer that stops mid-stream resumes
    ## exactly there. No allocation: the machine state is the cursor
    ## plus the window bounds, the compiled pattern's scratch buffers
    ## are reused per attempt, the input is held by reference.
    pattern: CompiledPattern
    input: string
    winLo: int
    winHi: int
    cursor: int

proc initRegexScanner*(pattern: CompiledPattern, input: string,
    winLo = 0, winHi = -1): RegexScanner =
  ## Binds one compiled pattern to one input, or to the subwindow
  ## [winLo, winHi) of it (winHi < 0 means input.len). Matches never
  ## extend past the window end. Example: pattern `\p{L}+` over
  ## "ab  cd" yields (0, 2) then (4, 6).
  let hi = if winHi < 0: input.len else: winHi
  doAssert 0 <= winLo and winLo <= hi and hi <= input.len
  RegexScanner(pattern: pattern, input: input, winLo: winLo, winHi: hi,
    cursor: winLo)

iterator items*(m: var RegexScanner): tuple[start, stop: int] =
  ## ONE iteration surface: the next non-empty leftmost-first match
  ## span per yield, byte offsets into the machine's input, until no
  ## further position in the window matches. Post-drain iteration
  ## stays empty.
  while m.cursor < m.winHi:
    let (ms, me) = nextMatch(m.pattern, m.input, m.cursor, m.winLo, m.winHi)
    if ms < 0:
      break
    m.cursor = me
    yield (ms, me)
