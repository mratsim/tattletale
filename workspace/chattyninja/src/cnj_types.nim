# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Core data of the chattyninja engine. Covers the compiled artifact, the parse-built side tables, and the render
# driver. See chattyninja.nim for the dispatch table and the `items` pull interface.

import cnj_errors, cnj_strbuf, cnj_values
import workspace/data_structures/src/small_seqs

const
  noLink* = -1'i32
    ## Marks an absent link or absent span in every node payload slot.

type
  NodeKind* {.pure.} = enum
    ## Corpus-derived construct vocabulary, one entry per Step function.
    ##
    ## | Kind           | Payload                                                                     |
    ## | -------------- | --------------------------------------------------------------------------- |
    ## | `nkVerbatim`   | final text run, whitespace already resolved                                 |
    ## | `nkEmit`       | `{{ }}` span, evaluates the `lo..hi` span, stringifies, pending piece       |
    ## | `nkIf`         | condition span, then-body, else-or-elif chain, bodies terminated past endif |
    ## | `nkFor`        | iterable span, body, loop name, filter span, interned target ids            |
    ## | `nkBreak`      | unwinds to the nearest for-frame, stopping at a macro-call boundary         |
    ## | `nkSet`        | single-target binding of an expression                                      |
    ## | `nkSetNs`      | `ns.field = expr`, ns and field as interned name ids                        |
    ## | `nkSetBlock`   | capture body into a sink, bind on close                                     |
    ## | `nkGeneration` | marks the root-output span of the model's turn                              |
    ## | `nkMacroDef`   | binds a macro value, never executes                                         |
    nkVerbatim, nkEmit, nkIf, nkFor, nkBreak, nkSet, nkSetNs, nkSetBlock, nkGeneration,
    nkMacroDef

  Node* = object
    ## POD node in one append-only arena, `kind` naming the construct, a node's executable
    ## meaning a pure function of `kind` through `steps`, so the artifact stays data.
    ## Every payload reference is one int32 slot, `noLink` (-1) marking an absent link or span:
    ## - a span into `Machine.jinja`, an arena index, or an interned name id
    ## - slots `0`-`3` uniform across kinds, `4` and past kind-specific, see the accessors below
    kind*: NodeKind
    slots*: SmallSeq[5, int32]
      ## Payload slots, capacity 5 the measured corpus knee:
      ## - 93% of the 1098 nodes of the 13 corpus templates that parse hold at most 5 slots
      ## - no node holds exactly 6, and the other 5 corpus templates raise declared gaps
      ## - only a variable `nkFor` or `nkMacroDef` payload spills, one heap block at parse time

  Machine* = object
    ## Read-only compiled template, two fields and no mutable state, so one artifact renders concurrently under separate drivers.
    ## `jinja` is borrowed, so the artifact must not outlive the template text it points into, which is why it is built at the caller's scope.
    jinja*: openArray[char]
    nodes*: seq[Node]

  Tables* = object
    ## Parse-built side arena, read-only at render, passed into the driver. Node int32 name
    ## slots index into it, so a `Machine` is only meaningful together with its `Tables`.
    names*: seq[string]

const
  ## Caps measured against the corpus, each a compile-time define.

  # Macro recursion is the engine's only render-time recursion. A macro body's output is
  # a string value, so a call runs to completion synchronously into a capture sink.
  # Static cross-macro chain depth in the corpus is 3, two templates recur cyclically, and the shipped
  # `gemma4/tools_tool_response.json` drives the tools subtree to 6, with real depth set by the input.
  # 16 bounds the stack on bad input, a breach raising instead of an overflow.
  MacroDepthCap* {.intdefine.} = 16

  # The expression walker recurses on nesting. Deepest paren nesting measured over the corpus templates
  # is 3 (`gemma4`, `lfm25`), so 24 clears the observed maximum with margin, far below the C stack overflow depth.
  ExprDepthCap* {.intdefine.} = 24

  # Output pieces reach the consumer in slices of at most this many bytes, which is what lets `cur`
  # compose with chunking.
  ChunkSize* {.intdefine.} = 4096

  wsSpace* = {' ', '\t', '\n', '\r', '\v', '\f'}
  wsNameChars* = {'a' .. 'z', 'A' .. 'Z', '0' .. '9', '_'}

# Payload slot accessors
# ---------------------------------------------------------------------------
#
# One slot per construct role, uniform across the kinds that fill it. Each expands textually
# to a slot read, so `nd.succ` on `m.nodes[n]` never copies the node. The per-accessor docs
# below name each slot's meaning and kind. Slot layouts per kind:
#
# | Kind         | Slots                                                                                           |
# |--------------|-------------------------------------------------------------------------------------------------|
# | `nkVerbatim` | 3 slots, `lo`, `hi`, `succ`                                                                     |
# | `nkEmit`     | 3 slots, `lo`, `hi`, `succ`                                                                     |
# | `nkIf`       | 5 slots, `lo`, `hi`, `succ`, `child`, `alt`                                                     |
# | `nkFor`      | 7 + one per target, `lo`, `hi`, `succ`, `child`, `loopName`, `filterLo`, `filterHi`, target ids |
# | `nkSet`      | 4 slots, `lo`, `hi`, `succ`, interned target name id                                            |
# | `nkSetNs`    | 5 slots, `lo`, `hi`, `succ`, `target`, `field`                                                  |
# | `nkMacroDef` | 4 + 3 per parameter, `macroName`, filler, `succ`, `child` body, name and default-span triples   |
#
# `nkSetBlock` and `nkGeneration` are declared kinds the parser never builds, so they carry
# no slot layout. `succ` shadows `system.succ`, still reachable for ordinal arguments because overload resolution
# only sees a `Node` receiver in this module.

const
  ## One slot position per construct role, plus the two variable-tail bases. Parse and render
  ## index through the same constants, so a slot-layout move is one shared edit.
  slotLo* = 0
  slotHi* = 1
  slotSucc* = 2
  slotChild* = 3
  slotAlt* = 4
  slotLoopName* = 4
  slotFilterLo* = 5
  slotFilterHi* = 6
  slotNsTarget* = 3
  slotNsField* = 4
  slotMacroName* = 0
  forTargetsBase* = 7
    ## First `nkFor` target name id slot, directly after the fixed prefix.
  macroParamsBase* = 4
    ## First `nkMacroDef` parameter slot, each parameter one name id then its default span,
    ## three slots in all.

template lo*(nd: Node): int32 =
  ## Payload span start into `Machine.jinja`, or the `nkMacroDef` macro name id.
  nd.slots[slotLo]

template hi*(nd: Node): int32 =
  ## Payload span end into `Machine.jinja`, exclusive.
  nd.slots[slotHi]

template succ*(nd: Node): int32 =
  ## Next node in program order by arena index, `noLink` once the artifact is exhausted.
  nd.slots[slotSucc]

template child*(nd: Node): int32 =
  ## First node of the body by arena index, or the `nkSet` target name id.
  nd.slots[slotChild]

template alt*(nd: Node): int32 =
  ## Next `nkIf` level in the else/elif chain by arena index, `noLink` when the chain ends.
  nd.slots[slotAlt]

template loopName*(nd: Node): int32 =
  ## Interned `loop` name id of `nkFor`, bound so the driver never interns at render time.
  nd.slots[slotLoopName]

template filterLo*(nd: Node): int32 =
  ## `nkFor` filter clause span start into `Machine.jinja`, `noLink` when the header has no `if`.
  nd.slots[slotFilterLo]

template filterHi*(nd: Node): int32 =
  ## `nkFor` filter clause span end into `Machine.jinja`, exclusive.
  nd.slots[slotFilterHi]

template target*(nd: Node): int32 =
  ## Interned namespace name id of `nkSetNs`.
  nd.slots[slotNsTarget]

template field*(nd: Node): int32 =
  ## Interned member name id of `nkSetNs`.
  nd.slots[slotNsField]

template macroName*(nd: Node): int32 =
  ## Interned macro name id that `nkMacroDef` binds.
  nd.slots[slotMacroName]

template targetCount*(nd: Node): int32 =
  ## Number of `nkFor` target name ids in the payload tail.
  nd.slots.len - forTargetsBase

template targetAt*(nd: Node, i: int): int32 =
  ## `nkFor` target name id `i`, `i` in `0 ..< nd.targetCount`.
  nd.slots[forTargetsBase + i]

template paramCount*(nd: Node): int32 =
  ## Number of `nkMacroDef` parameters in the payload tail.
  (nd.slots.len - macroParamsBase) div 3

template paramNameAt*(nd: Node, k: int): int32 =
  ## `nkMacroDef` parameter `k` interned name id, `k` in `0 ..< nd.paramCount`.
  nd.slots[macroParamsBase + 3 * k]

template paramDefLoAt*(nd: Node, k: int): int32 =
  ## `nkMacroDef` parameter `k` default span start, `noLink` when the parameter has no default.
  nd.slots[macroParamsBase + 3 * k + 1]

template paramDefHiAt*(nd: Node, k: int): int32 =
  ## `nkMacroDef` parameter `k` default span end, exclusive, meaningful only while
  ## `paramDefLoAt` is not `noLink`.
  nd.slots[macroParamsBase + 3 * k + 2]

func findName*(t: Tables, name: openArray[char]): int32 =
  ## Returns the interned id of `name`, or `noLink` when the template never names it, the comparison reading the caller's bytes in place so
  ## an interned name allocates nothing.
  for i, n in t.names:
    if n == name:
      return int32 i
  noLink

type
  FrameKind* = enum
    frFor, frCapture, frGeneration

  Frame* = object
    ## Driver frame, the only place re-entry is discriminated. `node` is the frame's identity,
    ## matched against the node being entered, nothing about resumption living in the node.
    node*: int32
    case kind*: FrameKind
    of frFor:
      loop*: LoopState
        ## cursor over the materialized iterable
      scopeAt*: int
        ## `scopes.len` at entry, the mark popped back to on close
      filterLo*, filterHi*: int32
        ## for-`if` clause span, `noLink` when absent
    of frCapture:
      sink*: int
        ## index into the driver's capture stack
      target*: int32
        ## interned name to bind on close, `noLink` for a macro body
    of frGeneration:
      spanStart*: int
        ## root-output byte position at span entry

  Binding* = object
    ## One scope entry:
    ##   an interned name bound to a value.
    name*: int32
    val*: Value

  Scope* = seq[Binding]

  PieceKind* = enum
    pkNone, pkSpan, pkStr, pkScratch

  Piece* = object
    ## Pending output piece. Span pieces deliver straight out of `Machine.jinja`, string
    ## pieces are materialized values (stringify, tojson, a captured body) held by the driver,
    ## scratch pieces are a derived value rendered into `Driver.scratch` and delivered in place.
    pos*: int
    case kind*: PieceKind
    of pkNone: nil
    of pkSpan:
      lo*, hi*: int32
    of pkStr:
      s*: string
    of pkScratch:
      shi*: int32
        ## end of the scratch window, the window starts at scratch byte 0

  Driver* = object
    ## All render control state, owned by the `items` loop, nothing reachable from `Machine`,
    ## so two drivers over one artifact cannot interfere.
    curNode*: int32
      ## node program counter, `noLink` once the artifact is exhausted
    cur*: int
      ## bytes of root output delivered so far, composed with the chunking window
    pend*: Piece
    frames*: seq[Frame]
      ## re-entry stack holding for, capture and generation frames
    scopes*: seq[Scope]
      ## scope 0 is the render context. For-frames push and pop above it
    root*: Value
      ## the render context dict (`messages`, `tools`, `kwargs`), the outermost lookup scope
    sinks*: seq[string]
      ## capture stack. Empty means output goes to the root stream
    spans*: seq[tuple[start, stop: int]]
      ## generation spans in root-output byte coordinates
    clock*: float64
      ## injected epoch, `strftime_now`'s only time source, never the wall
    macroDepth*: int
    scratch*: ptr UncheckedArray[char]
      ## caller-owned render scratch, borrowed like `Machine.jinja`, nil when unattached, must outlive the render. A scratch-backed piece drains
      ## before the next derived value is built, so one buffer reused from byte 0 serves every derived value.
    scratchCap*: int
      ## writable bytes behind `scratch`

func scratchBuf*(d: Driver): StrBuf =
  ## Returns a scratch buffer positioned at scratch byte 0, ready for one derived value,
  ## every build starting over from byte 0 once the previous scratch piece drained.
  StrBuf(buf: d.scratch, cap: d.scratchCap)

func scratchString*(d: Driver, n: int): string =
  ## Returns scratch[0 ..< n] as a fresh string, one allocation bounded by `n`, the bytes
  ## staying untouched until the copy.
  result = newString(n)
  if n > 0:
    copyMem(addr result[0], d.scratch, n)

proc concatVals*(d: Driver, lhs, rhs: Value): string =
  ## Returns the `~` concatenation of two values as template output text.
  ## - with scratch, both sides stringify into scratch, the result materializing once,
  ##   one allocation bounded by the result size
  ## - without scratch, each side materializes its own string and `&` joins them
  if d.scratch != nil:
    var sb = scratchBuf(d)
    try:
      pyStrInto(lhs, sb)
      pyStrInto(rhs, sb)
    except ScratchError as e:
      e.msg = "`~` concat of a " & $lhs.kind & " value, " & e.msg
      raise e
    scratchString(d, sb.len)
  else:
    pyStr(lhs) & pyStr(rhs)

