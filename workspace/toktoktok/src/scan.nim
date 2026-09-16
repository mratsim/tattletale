# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Text-to-pieces stage of the machine pipeline in one module:
## the special-token scan (step 1) plus the per-family
## pre-tokenization (step 2, with the pattern-split fast path).
##
## Each stage carries its section doc below.

import std/strutils
import std/monotimes
import std/times

import workspace/data_structures/src/aho_corasick
import workspace/regex_engine
import ./machine

# ------------------------------------------------------------------------
# Step 1: special-token scan (SpecialScanner + SpecialScan machine)
# ------------------------------------------------------------------------

## Special-token scan stage, a double-array Aho-Corasick
## over the special-token dictionary (workspace/data_structures),
## yielding (ordinary-text | special-token) decisions as offset pairs.
##
##   input text ─→ AC machine over the dictionary ─→ (ordinary-text | special-token) decisions
##
## Decision semantics, matching toktoktok's special-token scan:
##
## | rule            | decision                                                                                                                                   |
## | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
## | start selection | greedy leftmost-start, the earliest start of any dictionary match wins                                                                     |
## | same-start tie  | longest match wins (longest-prefix-match, matching HF tokenizers LeftmostLongest)                                                          |
## | consume         | a winning special consumes its bytes, the text before it is one ordinary decision                                                          |
## | region split    | an ordinary region is never split, each ordinary region is one ordinary decision, so downstream pre-tokenization sees identical boundaries |
##
## Construction:
## - the scanner always builds the Aho-Corasick automaton at init,
##   the single serving path for every dictionary.
##
## Streaming, the machine walks bytes once, carrying automaton state
## and pending matches across feeds (carry-buffer discipline):
##
## | rule         | behavior                                                                                                                        |
## | ------------ | ------------------------------------------------------------------------------------------------------------------------------- |
## | finality     | a decision at start s is final once the scan has progressed s + maxPatternLen bytes                                             |
## | chunk safety | that finality makes chunk boundaries safe, a special straddling a chunk edge is held and matched after the next feed            |
## | offsets      | decision offsets index the machine window, absolute base `winBase` over `win`, valid until the next feed                        |
## | input mode   | whole-input mode aliases the caller's string (zero copy), chunked mode reuses one growing window with decided-prefix compaction |
##
## Decisions are offsets plus one id, no per-decision strings.
##
## - `items` refills the decision queue on demand, one decision per yield.
## - `feed`/`finish` are state updates, not consumption surfaces.
## - queued decision offsets stay valid only until the next feed.

type
  SpecialScanner* = ref object
    ## Built special-token dictionary, shared across machines, match
    ## length decides same-start ties.
    patterns: seq[string]
    ids: seq[int]
    maxLen: int
    ahoCorasick: AhoCorasick

  PendingMatch = object
    ## One discovered-but-undecided match (absolute stream coordinates).
    start: int
    pat: int32

  SpecialScan* {.final.} = object
    ## Special-token scan machine (chain depth 1), decisions queue in `dq`, the scan machinery refills it on demand.
    scanner: SpecialScanner
    win*: string
    ## Window buffer indexing the decision offsets, valid until the next feed (whole-input mode aliases the caller's string).
    winBase*: int
    ## Absolute stream offset of win[0].
    decided: int
    scanPos: int
    acState: int32
    curMin: int
    pending: seq[PendingMatch]
    dq: seq[SpecialDecision]
    dqHead: int
    streamEnded: bool


proc init*(_: type SpecialScanner, patterns: openArray[string],
    ids: openArray[int]): SpecialScanner =
  ## Builds the scanner at init:
  ## - duplicates keep their first occurrence.
  ## Raises ValueError on an empty pattern, the reference scan would not
  ## terminate on it.
  doAssert patterns.len == ids.len
  new result
  for pat in patterns:
    if pat.len == 0:
      raise newException(ValueError,
        "SpecialScanner: empty special-token pattern (the reference scan would not terminate on it)")
  for i in 0 ..< patterns.len:
    var dup = false
    for existing in result.patterns:
      if existing == patterns[i]:
        dup = true
        break
    if not dup:
      result.patterns.add string(patterns[i])
      result.ids.add ids[i]
  result.maxLen = 0
  for pat in result.patterns:
    if pat.len > result.maxLen:
      result.maxLen = pat.len
  var priorityVals: seq[int] = @[]
  for i in 0 ..< result.patterns.len:
    priorityVals.add i
  result.ahoCorasick = buildAhoCorasick(result.patterns, priorityVals)

# ---------------------------------------------------------------------
# Decision machinery
# ---------------------------------------------------------------------

proc emitOrdinary(c: var SpecialScan, absHi: int) {.inline.} =
  ## Ordinary decision over [decided, absHi), skipped when empty (no ordinary decision between adjacent specials).
  if absHi > c.decided:
    c.dq.add SpecialDecision(
      lo: c.decided - c.winBase, hi: absHi - c.winBase,
      specialId: -1)
    c.decided = absHi

proc emitSpecial(c: var SpecialScan, pat: int) {.inline.} =
  ## Special decision for pattern `pat` starting at the decided frontier.
  let endAbs = c.decided + c.scanner.patterns[pat].len
  c.dq.add SpecialDecision(
    lo: c.decided - c.winBase, hi: endAbs - c.winBase,
    specialId: c.scanner.ids[pat])
  c.decided = endAbs

proc finalize(c: var SpecialScan) =
  ## Emits one pending decision:
  ## - ordinary up to the minimal pending match start, then the winning special
  ##   (the longest match starting there, HF tokenizers LeftmostLongest).
  ## - prunes matches consumed or overlapped by the token.
  doAssert c.pending.len > 0
  var winner = -1
  var winnerLen = 0
  for m in c.pending.items:
    if m.start == c.curMin:
      let plen = c.scanner.patterns[int(m.pat)].len
      if winner < 0 or plen > winnerLen:
        winner = int(m.pat)
        winnerLen = plen
  c.emitOrdinary(c.curMin)
  c.emitSpecial(winner)
  var w = 0
  var newMin = high(int)
  for m in c.pending.items:
    if m.start >= c.decided:
      if m.start < newMin:
        newMin = m.start
      c.pending[w] = m
      inc w
  c.pending.setLen(w)
  c.curMin = newMin

proc scanByte(c: var SpecialScan) =
  ## One automaton step, transition then record every output-chain
  ## match that starts at or after the decided frontier.
  let sc = c.scanner
  let b = uint8(c.win[c.scanPos - c.winBase])
  c.acState = sc.ahoCorasick.nextState(c.acState, b)
  let endAbs = c.scanPos + 1
  var op = sc.ahoCorasick.outputHead(c.acState)
  while op != 0:
    let s0 = endAbs - int(sc.ahoCorasick.outputLength(op))
    if s0 >= c.decided:
      if c.pending.len == 0 or s0 < c.curMin:
        c.curMin = s0
      c.pending.add PendingMatch(start: s0, pat: sc.ahoCorasick.outputValue(op))
    op = sc.ahoCorasick.outputParent(op)
  c.scanPos = endAbs

proc ahoCorasickServe(c: var SpecialScan) =
  ## Automaton decision path:
  ## - finalize whenever no undiscovered match can start before
  ##   the minimal pending start (scan reached curMin + maxPatternLen),
  ##   scanning one byte per step otherwise.
  ## - at stream end every match is discovered, so pending drains
  ##   directly and the tail ordinary region closes the stream.
  let sc = c.scanner
  let winEnd = c.winBase + c.win.len
  while true:
    if c.pending.len > 0 and c.scanPos >= c.curMin + sc.maxLen:
      c.finalize()
      return
    if c.scanPos < winEnd:
      c.scanByte()
      continue
    break
  if c.streamEnded:
    while c.pending.len > 0:
      c.finalize()
    c.emitOrdinary(winEnd)

proc serve(c: var SpecialScan) =
  ## Refills the decision queue:
  ## - empty dictionaries are a pure passthrough, one ordinary
  ##   decision over the whole stream at finish.
  c.dq.setLen(0)
  c.dqHead = 0
  let sc = c.scanner
  if sc.patterns.len == 0:
    if c.streamEnded:
      c.emitOrdinary(c.winBase + c.win.len)
    return
  c.ahoCorasickServe()

# ---------------------------------------------------------------------
# Machine surface
# ---------------------------------------------------------------------

proc decisionsQueued*(c: var SpecialScan): bool {.inline.} =
  ## True while decisions are queued but not yet yielded:
  ## - a caller draining a partially consumed stream consumes (or discards) them before re-binding the machine to a new stream.
  ## - their offsets index the current window.
  c.dqHead < c.dq.len

proc drained*(c: var SpecialScan): bool {.inline.} =
  ## True once the machine is finished:
  ## - every byte scanned and decided, nothing queued.
  ## - an iteration past the last decision ends there.
  c.dqHead >= c.dq.len and c.streamEnded and c.pending.len == 0 and
    c.decided >= c.winBase + c.win.len

iterator items*(c: var SpecialScan): SpecialDecision {.inline.} =
  ## Decision stream contract, yielded by `items`:
  ## - ordinary stretches and special-token occurrences left to right,
  ##   covering the decided stream exactly once.
  ## - offsets into machine `win` (absolute base `winBase`), valid only
  ##   until the next feed.
  while true:
    if c.dqHead < c.dq.len:
      let d = c.dq[c.dqHead]
      inc c.dqHead
      if c.dqHead == c.dq.len:
        c.dq.setLen(0)
        c.dqHead = 0
      yield d
      continue
    c.serve()
    if c.dqHead >= c.dq.len:
      break

# ---------------------------------------------------------------------
# Feeding
# ---------------------------------------------------------------------

proc compact(c: var SpecialScan) =
  ## Drops the decided prefix of the window (carry-buffer reclamation).
  let drop = c.decided - c.winBase
  if drop <= 0:
    return
  for i in 0 ..< c.win.len - drop:
    c.win[i] = c.win[i + drop]
  c.win.setLen(c.win.len - drop)
  c.winBase += drop

proc feed*(c: var SpecialScan, chunk: openArray[char]) =
  ## Appends the next stream chunk:
  ## - queued decision offsets stay valid until this call, consumers
  ##   must drain before further feeding.
  ## - compaction runs only when the queue is empty.
  doAssert not c.streamEnded, "feed after finish"
  if c.dqHead >= c.dq.len:
    c.compact()
  let old = c.win.len
  c.win.setLen(old + chunk.len)
  for i in 0 ..< chunk.len:
    c.win[old + i] = chunk[i]

proc finish*(c: var SpecialScan) =
  ## Marks the stream complete, the tail decision drains on the next iteration, idempotent.
  c.streamEnded = true

proc init*(_: type SpecialScan, scanner: SpecialScanner,
    input: sink string): SpecialScan {.inline.} =
  ## Whole-input mode, the window aliases the caller's string (zero copy) and the stream is complete from the start.
  SpecialScan(scanner: scanner, win: input, winBase: 0, decided: 0,
    scanPos: 0, acState: int32(AhoCorasickRootIdx), curMin: high(int),
    streamEnded: true)

proc init*(_: type SpecialScan, scanner: SpecialScanner): SpecialScan {.inline.} =
  ## Chunked mode:
  ##   feed chunks then finish. The window starts empty and carries undecided bytes across feeds.
  SpecialScan(scanner: scanner, win: "", winBase: 0, decided: 0,
    scanPos: 0, acState: int32(AhoCorasickRootIdx), curMin: high(int),
    streamEnded: false)

# ------------------------------------------------------------------------
# Step 2 fast path, pattern-split lookahead emulation (SplitPattern triple + scanSplit)
# ------------------------------------------------------------------------

## Split triple, the fast path the step-2 chain consumes.
##
## Pattern-split pre-tokenization, emulating the rust-gems bpe-openai
## lookahead:
## - patterns at crates/bpe-openai/src/lib.rs:25-58.
## - the anchored Splits iterator with the drop-last flag at lib.rs:252-280.
## A family pattern carrying `\s+(?!\S)` becomes 3 cooperating anchored
## sub-patterns, scanned in priority order:
##
##   pattern ─→ 3 sub-patterns ─→ scan in priority order
##              ├─→ (pat1, false)   lookahead slot becomes `\s+$`
##              ├─→ (pat2, true)    `\s+\s`, drop-last
##              └─→ (pat3, false)   trailing whitespace alternative
##
## | sub-pattern | carries                                                 | why it cooperates                                                                                                                             |
## | ----------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
## | pat1        | every earlier alternative, `\s+$` in the lookahead slot | `\s+$` answers the end-of-window case, a run reaching the window end is one piece                                                             |
## | pat2        | `\s+\s`, drop-last                                      | an interior run of >= 2 whitespace codepoints has the greedy `\s+(?!\S)` extent minus its last codepoint, matched and dropped, resuming there |
## | pat3        | the trailing whitespace alternative, verbatim           | reached where pat1 and pat2 both fail, a one-codepoint run before a non-whitespace codepoint                                                  |
##
## Trailing whitespace alternative stays out of pat1, so it ranks below pat2
## and a trailing `\s` cannot win the interior-run cursors of the pat2 emulation.
##
## Split scan reproduces the single-pattern leftmost-first scan exactly, an alternation admits the earliest alternative with a non-empty match:
## - pat1 answers the end-of-window case.
## - pat2 with the drop-last flag answers the interior lookahead case.
## - pat3 answers the trailing slot.
##
## Window discipline, engine-owned:
## - `$` binds at winHi (or before a final LF), never past it.
## - pieces cut at special-token or region boundaries segment identically
##   to the per-slice PCRE2 subject.
##
## The equivalence suite proves the split byte-identical to the frontier
## engine on every served family.
##
## Direct whitespace paths, proven by case analysis.
## Two build-time capability checks of pat1 and the equivalence suite
## verify those paths.
##
## At a cursor opening a run of ASCII whitespace bytes (0x09-0x0D, 0x20)
## with no byte >= 0x80, every pat1 alternative that could fire falls into
## exactly three shapes:
## - a leading-char alternative whose lead is one non-CRLF codepoint
##   followed by a non-whitespace continuation, firing only when the run
##   is that single codepoint.
## - a dollar-anchored alternative, firing only when the run reaches the window end.
## - a `\s*[\r\n]`-shaped alternative, firing only when the run
##   contains a carriage return or line feed, the engine attempt
##   deciding the extent.
##
## The capability checks ask pat1 whether it can match strictly inside
## a whitespace run, with `\t\tX` covering the no-CRLF shape and `\t\r\nX`
## covering the CRLF shape.
##
## Where a check comes back negative, the scan resolves the cursor without
## touching the engine:
## - an interior run of >= 2 bytes takes the pat2 drop-last shape.
## - a single CR or LF byte is a one-byte piece.
## - a run reaching the window end without CRLF is the pat1 `$` full-run piece, a one-piece emission.
##
## Any other whitespace cursor defers to the engine path:
## - a multi-byte whitespace codepoint such as U+00A0 or U+3000.
## - a check-positive shape.
## The engine path tries pat1, then the pat2 drop-last shape, then pat3,
## then the one-codepoint gap advance, in the reference leftmost-first order.
## The capability checks only ever make the scan more conservative:
## a pattern whose alternatives fall outside the three shapes must not use the split.
##
## Longest-match contract of the split triple, measured over the first
## lookahead-carrying pattern of every served pre-tokenization family:
## - the split requires the regex engine's ordered live-thread frontier
##   machinery. A plain longest-match powerset DFA diverges on the family
##   patterns the split serves.
## - the exception is r50k/p50k, where leftmost-first segmentation equals
##   a longest-match scan over pat1's alternatives, with disjoint
##   non-whitespace continuation classes and dollar alternatives of equal extent.
## - the other ten families diverge on real rows, in three shapes:
##
##   | divergence shape                                                               | example                                                                                             |
##   | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
##   | contraction vs letter overlap                                                  | `'v'De` splits contraction-first, a longest-match scan reaches `'De` through the letter alternative |
##   | uppercase-led vs lowercase-led letter alternatives carrying Lo in both classes | the uppercase-led one reaches farther (`_文D`, `あTZ`)                                                |
##   | `\s*[\r\n]` and `\s+$` alternatives at a window end                            | a trailing `\n\t` segments differently under the two shapes                                         |
##
## The pattern-split equivalence suite (tests/fuzzing/t_pretok_split.nim)
## checks the split scan against a longest-match scan over the pat1
## alternatives for each family.

type
  SplitPatternError* = object of ValueError
    ## Error raised by the pattern-split compilation preconditions.

  SplitPattern* = ref object
    ## Compiled sub-pattern triple of one lookahead-bearing family pattern, cooperation contract in the section doc above.
    pat1*: CompiledPattern
    pat2*: CompiledPattern
    pat3*: CompiledPattern
    interiorPlainCapable*: bool
    ## Build-time test result, true when pat1 can match strictly inside
    ## a whitespace run without CRLF:
    ## - false for every served family pattern.
    interiorCrlfCapable*: bool
    ## Build-time test result for the `\s*[\r\n]`-shaped alternatives:
    ## - true when pat1 can match strictly inside a whitespace run
    ##   containing a carriage return or line feed.
    ## - true for the cl100k/o200k-shaped and chain families.
    name*: string

const
  WhitespaceLookahead* = r"\s+(?!\S)"
  ## Lookahead alternative the split emulates, appearing verbatim
  ## in every served family pattern.

proc splitAlternatives(pattern: string): seq[string] =
  ## Returns the pattern split on top-level `|` operators, depth-tracked
  ## over groups (...) and classes [...] with backslash escapes honored,
  ## so alternations inside groups or classes stay intact:
  ## splitAlternatives(r"a|b(?:c|d)|[|]") == @["a", r"b(?:c|d)", r"[|]"].
  var depth = 0
  var start = 0
  var i = 0
  while i < pattern.len:
    case pattern[i]
    of '\\':
      inc i # the escaped character never opens, closes or splits
    of '(', '[':
      inc depth
    of ')', ']':
      if depth > 0:
        dec depth
    of '|':
      if depth == 0:
        result.add pattern[start ..< i]
        start = i + 1
    else:
      discard
    inc i
  result.add pattern[start ..< pattern.len]

proc hasWhitespaceLookahead(pattern: string): bool =
  ## True when the pattern's top-level alternation carries
  ## `\s+(?!\S)` exactly (the split's precondition).
  for alt in splitAlternatives(pattern).items:
    if alt == WhitespaceLookahead:
      return true


proc splitLookaheadPattern*(pattern: string, rustWs = false,
    name = "anonymous"): SplitPattern =
  ## Compiles one lookahead-bearing family pattern into the sub-pattern
  ## triple (see the section doc above):
  ## - the lookahead alternative must appear exactly once as a whole top-level alternative.
  ## - the alternative immediately after it must be `\s` or `\s+`
  ##   (pat3's equivalence contract).
  ## - anything else raises SplitPatternError rather than silently
  ##   mis-splitting the pattern.
  let alts = splitAlternatives(pattern)
  var lookIdx = -1
  var lookCount = 0
  for i, alt in alts.pairs:
    if alt == WhitespaceLookahead:
      lookIdx = i
      inc lookCount
  if lookCount == 0:
    raise newException(SplitPatternError,
      "pattern [" & pattern & "] has no " & WhitespaceLookahead &
      " alternative to split")
  if lookCount > 1:
    raise newException(SplitPatternError,
      "pattern [" & pattern & "] repeats the " & WhitespaceLookahead &
      " alternative")
  if lookIdx == alts.len - 1:
    raise newException(SplitPatternError,
      "pattern [" & pattern & "] has no trailing whitespace alternative" &
      " after " & WhitespaceLookahead)
  let tail = alts[lookIdx + 1]
  if tail != r"\s" and tail != r"\s+":
    raise newException(SplitPatternError,
      "pattern [" & pattern & "] trailing alternative [" & tail &
      "] is not \\s or \\s+")
  var pat1Parts: seq[string]
  for i, alt in alts.pairs:
    if i == lookIdx:
      pat1Parts.add r"\s+$" # the lookahead slot becomes the anchored tail
    elif i == lookIdx + 1:
      discard # the trailing alternative moves to pat3 verbatim
    else:
      pat1Parts.add alt
  result = SplitPattern(
    pat1: compilePattern(pat1Parts.join("|"), rustWs, name & "_pat1"),
    pat2: compilePattern(r"\s+\s", rustWs, name & "_pat2"),
    pat3: compilePattern(tail, rustWs, name & "_pat3"),
    name: name)
  # Build-time capability tests of pat1 itself (the engine is the ground truth of its own semantics):
  # strictly inside a whitespace run, before a non-whitespace codepoint, with and without CRLF.
  result.interiorPlainCapable =
    result.pat1.matchAt("\t\tX", 0, 3, 0) >= 0
  result.interiorCrlfCapable =
    result.pat1.matchAt("\t\r\nX", 0, 4, 0) >= 0

proc lastCodepointWidth(input: string, pos: int): int {.inline.} =
  ## Returns the byte width of the codepoint ending at pos (the last codepoint of input[0 ..< pos]), walking back over
  ## UTF-8 continuation bytes.
  doAssert pos > 0 and pos <= input.len
  var i = pos - 1
  while i > 0 and (uint8(input[i]) and 0xC0'u8) == 0x80'u8:
    dec i
  result = pos - i

proc scanSplit*(sp: SplitPattern,
    outPieces: var seq[tuple[lo, hi: int32]], input: string,
    winLo, winHi: int) {.inline.} =
  ## Anchored split scan over the window [winLo, winHi):
  ##   window ─→ gap piece before each match ─→ match ─→ unmatched remainder as one trailing piece
  ## - the sub-pattern priority of the section doc above plus the drop-last resume rule for pat2 matches.
  ## - pieces are byte-offset pairs into the input the caller holds,
  ##   the applyRegexStep surface of the frontier-engine step.
  ##
  ## Expected input:
  ## - valid UTF-8, codepoint-aligned window bounds.
  ## Output:
  ## - consecutive, non-empty, non-overlapping [lo, hi) pieces
  ##   covering [winLo, winHi) exactly once.
  var lastEmit = winLo
  var offset = winLo
  while offset < winHi:
    var pieceEnd = -1
    var resume = -1
    let b0 = uint8(input[offset])
    if b0 < 0x80'u8 and (b0 == 0x20'u8 or (b0 >= 0x09'u8 and b0 <= 0x0D'u8)):
      # Whitespace cursor, the run measured at byte level, every ASCII
      # whitespace byte is `\s` under both the PCRE2-UCP and the Rust
      # White_Space variant, while a byte >= 0x80 may open a multi-byte
      # `\s` codepoint and defers the whole cursor to the engine path.
      var p = offset
      var sawCr = false
      var multiByte = false
      while p < winHi:
        let bb = uint8(input[p])
        if bb >= 0x80'u8:
          multiByte = true
          break
        if bb == 0x0A'u8 or bb == 0x0D'u8:
          sawCr = true
          inc p
        elif bb == 0x20'u8 or bb == 0x09'u8 or bb == 0x0B'u8 or
            bb == 0x0C'u8:
          inc p
        else:
          break
      let runEnd = p
      if not multiByte:
        if runEnd == winHi:
          if not sawCr:
            # pat1's `\s+$` extent is the full run, one piece.
            pieceEnd = runEnd
          else:
            # A CRLF run reaching the window end, the preferred extent
            # depends on the family's alternative order, one engine
            # attempt decides, which always succeeds here.
            pieceEnd = sp.pat1.matchAt(input, winLo, winHi, offset)
        elif runEnd - offset == 1:
          if sawCr:
            # A single CR or LF byte, the `\s*[\r\n]`-shaped
            # alternative (when present) and the trailing alternative
            # both give the one-byte piece, the leading-char
            # alternatives exclude CR and LF. No engine attempt.
            pieceEnd = offset + 1
          else:
            # A single space or tab byte, a leading-char alternative
            # may still absorb it into the following piece.
            pieceEnd = sp.pat1.matchAt(input, winLo, winHi, offset)
            if pieceEnd < 0:
              pieceEnd = offset + 1
        else:
          # Interior run of >= 2 bytes, the byte at runEnd is ASCII
          # non-whitespace so genuinely `\S`, pat1 cannot fire
          # without the CRLF shape, and with it one engine attempt decides,
          # otherwise the pat2 drop-last shape resolves the cursor directly.
          if sawCr and sp.interiorCrlfCapable:
            pieceEnd = sp.pat1.matchAt(input, winLo, winHi, offset)
          elif not sawCr and sp.interiorPlainCapable:
            pieceEnd = sp.pat1.matchAt(input, winLo, winHi, offset)
            if pieceEnd < 0:
              pieceEnd = sp.pat2.matchAt(input, winLo, winHi, offset)
              if pieceEnd >= 0:
                resume = pieceEnd - lastCodepointWidth(input, pieceEnd)
                doAssert resume > offset,
                  "a pat2 match must retain at least one codepoint"
              else:
                pieceEnd = sp.pat3.matchAt(input, winLo, winHi, offset)
          if pieceEnd < 0:
            pieceEnd = runEnd - 1
            resume = runEnd - 1
    if pieceEnd < 0:
      # Engine path:
      #   word cursors and multi-byte \s runs, pat1 then
      # pat2 (drop-last) then pat3, in priority order.
      pieceEnd = sp.pat1.matchAt(input, winLo, winHi, offset)
      if pieceEnd < 0:
        pieceEnd = sp.pat2.matchAt(input, winLo, winHi, offset)
        if pieceEnd >= 0:
          resume = pieceEnd - lastCodepointWidth(input, pieceEnd)
          doAssert resume > offset,
            "a pat2 match must retain at least one codepoint"
        else:
          pieceEnd = sp.pat3.matchAt(input, winLo, winHi, offset)
          if pieceEnd < 0:
            # No sub-pattern matches at this cursor, the gap grows
            # by one codepoint, the same advance the reference leftmost-first scan takes on a miss.
            let d = decodeCp(input, offset, winHi)
            offset += (if d.width > 0: d.width else: 1)
            continue
    # A drop-last match emits the match minus its final codepoint,
    # resuming AT the dropped codepoint (the next cursor re-segments it).
    let emitEnd = if resume >= 0: resume else: pieceEnd
    if offset > lastEmit:
      outPieces.add (int32(lastEmit), int32(offset))
    outPieces.add (int32(offset), int32(emitEnd))
    lastEmit = emitEnd
    offset = emitEnd
  if lastEmit < winHi:
    outPieces.add (int32(lastEmit), int32(winHi))

# ------------------------------------------------------------------------
# Step 2, per-family pre-tokenization (PreTokenizer machine, Isolated Split-chain semantics)
# ------------------------------------------------------------------------

## Per-family pre-tokenization stage. Family patterns compile through
## workspace/regex_engine (table-walk matching, never PCRE2 at runtime)
## and emit byte-offset pieces into the caller-owned input on every hot path.
##
## Issue #22 (mratsim/tattletale#22). `Split`-chain checkpoints do not match
## the flat `|` alternation convertHfToTiktoken builds, so the chain-configured
## families diverge from the reference engine.
##
## Checkpoints configure pre_tokenizer chains of `Split` steps, each step
## carrying its regex pattern and `behavior: Isolated`. The chain splits:
## - apply step 1 to the whole input, then apply step 2 to EACH output
##   piece independently, and so on down the chain.
## - a split isolates, pieces matched by an earlier step are never
##   re-examined by later steps, later patterns only see within-piece content.
## - within one step the scan is the PCRE2 leftmost-first scan,
##   exactly findAllPcre2 + splitTextOrdinary.
##
## One step's leftmost scan:
##   first position admitting a non-empty match ─→ place the match at pattern-preference extent
##        │
##        ├─→ gap before the match ─→ emitted as one piece ─→ scan resumes past the match
##        │
##        └─→ first unmatched position ─→ scan stops ─→ the remainder is one trailing piece
##
## NOT a flat alternation join:
## - chain semantics give step-1 matches absolute priority in their own span
##   and hide those spans from later steps, whereas a flat alternation
##   gives earlier patterns priority at every input position.
## - serialization.nim joins the Split patterns with |, the flattening
##   the convertHfToTiktoken path applies to chain-configured checkpoints.
## - the Step-3.5-Flash and EXAONE checkpoints configure chains.
##
## Chain features implemented:
##
## | feature            | behavior                                                                                                                                                                                                       |
## | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | digit grouping     | `\p{N}{1,3}` runs of digits are cut into groups of at most 3 digits (Step-3.5-Flash Split 1, also the cl100k/o200k patterns)                                                                                   |
## | CJK splitting      | [一-龥぀-ゟ゠-ヿ]+ (U+4E00-U+9FA5 ideographs, U+3040-U+309F hiragana, U+30A0-U+30FF katakana) as an Isolated step, so a CJK run is one pre-token piece (Step-3.5-Flash Split 2)                                      |
## | letter/contraction | GPT-2-style letter/contraction patterns with `\p{L}\p{M}*` combining-mark handling (EXAONE Split 1)                                                                                                            |
## | space delimiter    | Gemma-4 style Split(String " ", MergedWithPrevious), a space delimiter joins the preceding non-space run inside its piece, consecutive or leading spaces stand alone (HF-engine verified, chain fixture suite) |
##
## ByteLevel(use_regex=false) as the chain tail, byte remap of each
## piece with add_prefix_space=false at pre position, the tail is not
## a split step, so it stays out of this stage's scope.
##
## Prefix-space semantics are POST-processing, not pre-tokenization
## (EXAONE post_processor ByteLevel add_prefix_space=true).
##
## Pattern-split lookahead emulation (src/pretok_split.nim), family
## patterns whose top-level alternation carries `\s+(?!\S)` build
## a sub-pattern triple on the same step:
## - pat1 = earlier alternatives plus `\s+$`.
## - pat2 = `\s+\s` drop-last.
## - pat3 = the trailing alternative verbatim.
## The step's default scan is the split scan, and the frontier-engine
## pattern stays compiled as the paired reference scan.
##
## - pieces are byte-offset pairs into the input the machine holds.
## - the Isolated chain applies its steps over two reusable scratch
##   buffers inside the object, so per-input work after the first
##   pass is buffer reuse only (zero-alloc steady state across `reset` calls).
## - no collect proc, no pull overload, consumers iterate.
##
## Pattern provenance per family, all patterns regular, all compiling
## in the workspace/regex_engine pattern engine, no construct outside
## its supported regular subset:
## | family             | pattern source                                                                                                                                                                                                                    |
## | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | r50k / p50k / gpt2 | tiktoken openai_public.py r50k_base, shared by the GPT-2 and P50k checkpoints                                                                                                                                                     |
## | cl100k, o200k      | tiktoken openai_public.py, the o200k 7-alternative form joined exactly like tokenizers_regexps.nim does                                                                                                                           |
## | kimik25            | Moonshot tokenization_kimi.py pat_str, Script=Han form                                                                                                                                                                            |
## | moonlight          | Moonshot tokenization_moonshot.py pat_str, the Rust-regex translation with lookahead-guarded Lo classes (tokenizers_regexps.nim notes)                                                                                            |
## | qwen               | Qwen3-0.6B tokenizer.json pre_tokenizer Split regex                                                                                                                                                                               |
## | qwen35             | Qwen3.5-0.8B tokenizer.json, `\p{M}` added to the letter and punctuation alternative classes                                                                                                                                      |
## | glm47              | GLM-4.7-Flash tokenizer.json, a single Split with a `\p{N}{1,3}` digit group                                                                                                                                                      |
## | ling3              | Ling-3.0-tiny tokenizer.json, possessive class quantifiers compiled greedy per the engine's extent-equivalence proof                                                                                                              |
## | gemma4             | gemma-4-E2B-it tokenizer.json pre_tokenizer (a String split)                                                                                                                                                                      |
## | exaone             | K-EXAONE-236B-A23B tokenizer.json Split 1 plus a ByteLevel tail                                                                                                                                                                   |
## | step35flash        | Step-3.5-Flash tokenizer.json Splits 1..3 plus a ByteLevel tail, its chain steps resolve `\s` through the Rust regex White_Space variant (whitespaceIsRust in the engine), matching the HF reference engine for chain checkpoints |

const
  # Each family pat_str is one verbatim HF/tiktoken constant, wrapped
  # as concatenated raw literals (byte-identical values, line-width rules).
  QwenPat* = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|""" &
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
  Qwen35Pat* = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|""" &
    r""" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
  Glm47Pat* = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|""" &
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
  Ling3Pat* = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}|""" &
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
  ExaoneStepPat* = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|""" &
    r"""[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}|""" &
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+"""
  Step35DigitsPat* = r"""\p{N}{1,3}"""
  Step35CjkPat* = "[一-龥぀-ゟ゠-ヿ]+"
  Step35MainPat* = r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|""" &
    r"""[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|""" &
    r"""\s*[\r\n]+|\s+(?!\S)|\s+"""
  R50kPat* = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|""" &
    r"""\s++$|\s+(?!\S)|\s"""
  Cl100kPat* = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|""" &
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
  O200kPat* = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+""" &
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)?|""" &
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*""" &
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}|""" &
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
  KimiK25Pat* = r"""[\p{Script=Han}]+|""" &
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+""" &
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)?|""" &
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*""" &
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}|""" &
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
  MoonlightPat* = r"""[\p{Script=Han}]+|""" &
    r"""[^\r\n\p{L}\p{N}]?(?:(?!\p{Script=Han})\p{Lo}|[\p{Lt}\p{Lu}\p{Lm}\p{M}])*""" &
    r"""(?:(?!\p{Script=Han})\p{Lo}|[\p{Ll}\p{Lm}\p{M}])+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|""" &
    r"""[^\r\n\p{L}\p{N}]?(?:(?!\p{Script=Han})\p{Lo}|[\p{Lt}\p{Lu}\p{Lm}\p{M}])+""" &
    r"""(?:(?!\p{Script=Han})\p{Lo}|[\p{Ll}\p{Lm}\p{M}])*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|""" &
    r"""\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

type
  Family* {.pure.} = enum
    ## Model families this stage pre-tokenizes. pat_str provenance per
    ## family in the module docstring.
    famR50k
    famP50k
    famCl100k
    famO200k
    famKimiK25
    famMoonlight
    famQwen
    famQwen35
    famGlm47
    famLing3
    famGemma4
    famExaone
    famStep35Flash

  SplitStepKind* = enum
    skRegex          # compiled split pattern (frontier DFA walk), Isolated
    skSpaceMergedPrev # Split(String " ", MergedWithPrevious), Gemma-4

  SplitStep* = ref object
    ## One Isolated chain step, shared and immutable once built, per-kind contract:
    ## - a regex step always carries the frontier-engine pattern `pat`,
    ##   the paired reference scan.
    ## - `split` additionally carries the pattern-split lookahead
    ##   emulation when the pattern's top-level alternation has the `\s+(?!\S)` alternative, nil otherwise.
    name*: string
    case kind*: SplitStepKind
    of skRegex:
      pat*: CompiledPattern
      split*: SplitPattern
    of skSpaceMergedPrev:
      discard

  ChainBuildStats* = object
    ## Build receipt per family.
    built*: bool
    steps*: int
    instrs*: int
    buildMillis*: float64

  PreTokRegexCache* = object
    ## Memoized per-family pre-tokenizer split chains and their build receipts.
    ## Passed in by the caller instead of a module-global, so a chain compiles
    ## once and the receipts stay observable for the equivalence suites.
    chains: array[Family, seq[SplitStep]]
    stats: array[Family, ChainBuildStats]

proc buildFamilyChain(c: var PreTokRegexCache, f: Family): seq[SplitStep] =
  ## Compiles the family's Isolated split chain at first use:
  ## - every family here is a regex chain of at most 3 steps or one string split.
  ## - chainStats records the build duration.
  let t0 = getMonoTime()
  template addRegex(stepName: string, pattern: string,
      rustWs = false) =
    # Appends through the enclosing proc's result directly, template method-call syntax would bind the receiver into
    # the stepName parameter, so call sites pass the two strings only.
    result.add SplitStep(name: stepName, kind: skRegex,
      pat: compilePattern(pattern, rustWs, stepName),
      split: (if hasWhitespaceLookahead(pattern):
        splitLookaheadPattern(pattern, rustWs, stepName) else: nil))
  case f
  of famR50k, famP50k:
    addRegex("r50k_split", R50kPat)
  of famCl100k:
    addRegex("cl100k_split", Cl100kPat)
  of famO200k:
    addRegex("o200k_split", O200kPat)
  of famKimiK25:
    addRegex("kimik25_split", KimiK25Pat)
  of famMoonlight:
    addRegex("moonlight_split", MoonlightPat)
  of famQwen:
    addRegex("qwen_split", QwenPat)
  of famQwen35:
    addRegex("qwen35_split", Qwen35Pat)
  of famGlm47:
    addRegex("glm47_split", Glm47Pat)
  of famLing3:
    addRegex("ling3_split", Ling3Pat)
  of famGemma4:
    result.add SplitStep(name: "gemma4_space_merged_prev",
      kind: skSpaceMergedPrev)
  of famExaone:
    # 1 regex step, the ByteLevel(use_regex=false) tail is a remap
    # carrying no split behavior
    addRegex("exaone_split1", ExaoneStepPat, rustWs = true)
  of famStep35Flash:
    # 3 regex steps, chain depth 3 of the documented maximum, then
    # the ByteLevel tail remap (out of scope here)
    addRegex("step35_split1_digits", Step35DigitsPat, rustWs = true)
    addRegex("step35_split2_cjk", Step35CjkPat, rustWs = true)
    addRegex("step35_split3_main", Step35MainPat, rustWs = true)
  c.stats[f].built = true
  c.stats[f].steps = result.len
  var instrs = 0
  for step in result.items:
    if step.kind == skRegex:
      instrs += step.pat.prog.len
  c.stats[f].instrs = instrs
  c.stats[f].buildMillis =
    (getMonoTime() - t0).inMicroseconds.float64 / 1000.0

proc steps*(c: var PreTokRegexCache, f: Family): seq[SplitStep] =
  ## Compiled split chain of the family (compiled at first use).
  if not c.stats[f].built:
    c.chains[f] = buildFamilyChain(c, f)
  c.chains[f]

proc chainStats*(c: PreTokRegexCache, f: Family): ChainBuildStats =
  ## Build receipt of the family chain (zeros before first use).
  c.stats[f]

proc allStats*(c: PreTokRegexCache): array[Family, ChainBuildStats] =
  ## All family build receipts, for the equivalence-suite instrumentation.
  c.stats

proc applyRegexStep(step: SplitStep,
    outPieces: var seq[tuple[lo, hi: int32]], input: string,
    pieceLo, pieceHi: int) {.inline.} =
  ## One Isolated regex step over one piece, segmented by the PCRE2
  ## leftmost-first scan into matched and gap pieces.
  var lastEmit = pieceLo
  var offset = pieceLo
  while offset < pieceHi:
    let (ms, me) = step.pat.nextMatch(input, offset, pieceLo, pieceHi)
    if ms < 0:
      break
    if ms > lastEmit:
      outPieces.add (int32(lastEmit), int32(ms))
    outPieces.add (int32(ms), int32(me))
    lastEmit = me
    offset = me
  if lastEmit < pieceHi:
    outPieces.add (int32(lastEmit), int32(pieceHi))

proc applySpaceMergedPrevStep(
    outPieces: var seq[tuple[lo, hi: int32]], input: string,
    pieceLo, pieceHi: int) {.inline.} =
  ## Split(String " ", MergedWithPrevious) over one piece, a space
  ## delimiter joins the run of non-space bytes before it, consecutive
  ## or leading spaces stand alone as their own pieces.
  var runStart = pieceLo
  var i = pieceLo
  while i < pieceHi:
    if input[i] == ' ':
      if i == pieceLo or input[i - 1] == ' ':
        if runStart < i:
          outPieces.add (int32(runStart), int32(i))
        outPieces.add (int32(i), int32(i + 1))
      else:
        outPieces.add (int32(runStart), int32(i + 1))
      runStart = i + 1
    inc i
  if runStart < pieceHi:
    outPieces.add (int32(runStart), int32(pieceHi))

type
  PreTokenizer* {.final.} = object
    ## Pre-tokenization machine over one input (chain depth 1):
    ## - `items` yields byte-offset pairs into the machine-held input.
    ## - the Isolated chain applies its steps over the two reusable
    ##   scratch buffers inside this object, built once per input, so
    ##   a `reset` + full pass reuses every buffer.
    input: string
    steps: seq[SplitStep]
    pieces: seq[tuple[lo, hi: int32]]
    head: int
    built: bool
    scratchA, scratchB: seq[tuple[lo, hi: int32]]

proc applyStepDefault(step: SplitStep,
    outPieces: var seq[tuple[lo, hi: int32]], input: string,
    pieceLo, pieceHi: int) {.inline.} =
  ## One Isolated regex step, default scan:
  ## - the pattern-split scan when the step carries one (the lookahead emulation).
  ## - the frontier-engine scan otherwise.
  ## The frontier-engine scan stays available as the paired reference scan.
  if step.split != nil:
    scanSplit(step.split, outPieces, input, pieceLo, pieceHi)
  else:
    applyRegexStep(step, outPieces, input, pieceLo, pieceHi)

proc applyStep*(step: SplitStep,
    outPieces: var seq[tuple[lo, hi: int32]], input: string,
    pieceLo, pieceHi: int) {.inline.} =
  ## One Isolated chain step applied to one piece of the input.
  ##
  ##   input[pieceLo..<pieceHi] ──▶ outPieces[]
  ##
  ## Output:
  ## - cuts `input[pieceLo ..< pieceHi]` into smaller pieces and appends
  ##   each as a byte-offset pair to `outPieces`.
  ## - the output pieces are consecutive, non-empty, and cover the input
  ##   piece exactly once.
  ##
  ## Per step kind:
  ## - a regex step cuts at every match of its pattern. It uses the pattern-split scan when present and the frontier scan otherwise.
  ## - a space-merged-prev step cuts at spaces, grouping each space
  ##   with the non-space bytes before it. Consecutive and leading
  ##   spaces stand alone.
  case step.kind
  of skRegex:
    applyStepDefault(step, outPieces, input, pieceLo, pieceHi)
  of skSpaceMergedPrev:
    applySpaceMergedPrevStep(outPieces, input, pieceLo, pieceHi)

proc buildPieces(r: var PreTokenizer) =
  ## Applies the Isolated chain, level by level, reusing the scratch
  ## buffers (at most 3 levels per the chain-depth rules).
  r.pieces.setLen(0)
  if r.steps.len == 0 or r.input.len == 0:
    if r.input.len > 0:
      r.pieces.add (0'i32, int32(r.input.len))
    r.built = true
    return
  r.scratchA.setLen(0)
  r.scratchA.add (0'i32, int32(r.input.len))
  for level in 0 ..< r.steps.len:
    let step = r.steps[level]
    if level mod 2 == 0:
      r.scratchB.setLen(0)
      for piece in r.scratchA.items:
        applyStep(step, r.scratchB, r.input, piece.lo, piece.hi)
    else:
      r.scratchA.setLen(0)
      for piece in r.scratchB.items:
        applyStep(step, r.scratchA, r.input, piece.lo, piece.hi)
  if r.steps.len mod 2 == 1:
    swap(r.pieces, r.scratchB)
  else:
    swap(r.pieces, r.scratchA)
  r.built = true

proc initPreTokenizer*(cache: var PreTokRegexCache, family: Family,
    input: sink string): PreTokenizer {.inline.} =
  ## Builds one pre-tokenization machine over the whole input, which
  ## aliases the caller's string.
  PreTokenizer(input: input, steps: cache.steps(family))

proc reset*(r: var PreTokenizer, cache: var PreTokRegexCache, family: Family,
    input: sink string) =
  ## Rebinds the machine to a new input, reusing every buffer (zero alloc steady state for repeated encodes over one stream).
  r.input = input
  r.steps = cache.steps(family)
  r.pieces.setLen(0)
  r.head = 0
  r.built = false

iterator items*(r: var PreTokenizer): tuple[lo, hi: int] {.inline.} =
  ## Yields the pre-token pieces as byte-offset pairs, left to right:
  ## - the pieces cover the input exactly once.
  ## - a partially consumed stream resumes at the machine's head on re-iteration.
  if not r.built:
    r.buildPieces()
  while r.head < r.pieces.len:
    let piece = r.pieces[r.head]
    inc r.head
    yield (int(piece.lo), int(piece.hi))
