# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Core data of the chattyninja engine. Covers the compiled artifact, the parse-built
# interned-name table, the per-instantiation render state, the name-resolution reads
# and the macro-force callable. The dispatch table, the render context bundle
# and the `pullInto`/`pullAll` delivery interface live in cnj_engine.nim.
#
# Dataflow of one render, the record types here shared across the module boundary
#
#   template bytes (borrowed, never copied at parse)
#     │  cnj_parse splits tags, scans keywords, appends nodes, interns names once
#     ▼
#   CompiledTemplate + CompiledSymbols (read-only artifact + heap-shared interned-name table)
#     │  cnj_engine dispatches pullInto() steps over the node list
#     ▼
#   JinjaRenderContext = the heap render context every render call holds (tmpl + symbols + state + force)
#     │  cnj_engine dispatches Steps[c.tmpl.nodes[n].kind](c, n), each step one ref borrow of the context
#     ▼
#   RenderState (rows + scopes + pending Piece), the context's per-instantiation third field
#     │  evalSpan drives jinja_interpolation on the same context borrow, expressions reading
#     │  the render state directly and reaching the statement tier only through `c.force`
#     ▼
#   Piece (pkSpan, pkStr, pkCut or pkLazy), drained into the caller's window
#
# Lifecycle and ownership inside the context
#
#   RenderState, one per render, owned by the caller's context object
#     ├─ rows     pushed by step* entry, popped by closeRow, one close path
#     ├─ scopes   owned by rows (scopeAt marks the base), trimmed on close
#     ├─ pend     one Piece, set by emit steps, drained by pullInto or capturePend, reset to pkNone
#
#   force, the engine's macro-force callable, bound once at `startRender` into the context, stateless,
#     every context carrying the same callable
#
#   A context is a ref by construction, one heap object per render, and cannot be copied.
#   A second render over one artifact opens a second context through `startRender`.
#
#   A forced macro body runs on an explicitly constructed second context, over the same
#   artifact refs and a snapshot of the caller's render state.
#   The `forceMacro` contract in cnj_engine states the snapshot's exact shape.

# Public API:
#   JinjaRenderContext, the compiled artifact (CompiledTemplate, CompiledSymbols)
#   and the depth, nesting and step caps. Everything else is render plumbing,
#   cross-module code importing it through `import x {.all.}`.

import jinja_data_model {.all.}
import jinja_serialize {.all.}
import workspace/data_structures/src/small_seqs

type
  NodeKind* {.pure.} = enum
    ## Corpus-derived construct vocabulary, one entry per engine step, the slot payload
    ## per kind in the accessor table below.
    nkVerbatim, nkEmit, nkIf, nkFor, nkBreak, nkSet, nkSetNamespace, nkSetBlock, nkGeneration,
    nkMacroDef

  Node = object
    ## POD node in one append-only node list, every payload reference one int32 slot,
    ## `NoLink` marking an absent link or span, the slot layout per kind in the table below.
    kind*: NodeKind
    slots*: SmallSeq[5, int32]
      ## Payload slots, capacity 5 the measured corpus knee, a variable `nkFor` or `nkMacroDef`
      ## payload spilling to one heap block at parse time.

  CompiledTemplate* = ref object
    ## Read-only compiled template, shared across renders with two fields and no mutable state,
    ## so one artifact serves any number of render instantiations.
    ##
    ## Borrow contract:
    ## - `jinja` is borrowed, so the artifact must not outlive the caller's template text
    ## - the caller builds `CompiledTemplate` at the scope that owns the text
    jinja*: openArray[char]
    nodes*: seq[Node]

  CompiledSymbols* = ref object
    ## Parse-built interned-name table, read-only at render and shared by every render
    ## over its artifact, node name slots indexing into it. Name resolution is one
    ## linear scan over `names`, parse-time only.
    names*: seq[string]

const
  ## Caps measured against the corpus, each a compile-time define.

  # Macro recursion is the engine's only render-time recursion, a call running synchronously
  # to completion into a capture sink. 16 bounds the stack on bad input.
  TTT_CNJ_MacroDepthCap* {.intdefine.} = 16

  # Expression-walker recursion bound, deepest corpus paren nesting 3, 24 clears it
  # with margin, far below the C stack overflow depth.
  TTT_CNJ_ExprDepthCap* {.intdefine.} = 24

  # Parse-time nesting bound of the parser's dispatch recursion, one level per
  # body-carrying construct. 64 clears the corpus with margin, a breach raising at the tag.
  TTT_CNJ_ParseNestingCap* {.intdefine.} = 64

  # Step-dispatch bound of one `pullInto` call, corpus renders staying below 240, a breach
  # raising located at the node the walk reached.
  TTT_CNJ_StepBudget* {.intdefine.} = 1_000_000

  Whitespace = {' ', '\t', '\n', '\r', '\v', '\f'}
  WsNameChars = {'a' .. 'z', 'A' .. 'Z', '0' .. '9', '_'}

# Payload slot accessors
#
# One slot per construct role, uniform across the kinds that fill it. Each expands textually
# to a slot read, so `nd.succ` on `m.nodes[n]` never copies the node. The per-accessor docs
# below name each slot's meaning and kind. Slot layouts per kind:
#
# | Kind             | Slots                                                                                           | |
# | ---------------- | ------------------------------------------------------------------------------------------------- |
# | `nkVerbatim`     | 3 slots, `lo`, `hi`, `succ`                                                                     | |
# | `nkEmit`         | 3 slots, `lo`, `hi`, `succ`                                                                     | |
# | `nkIf`           | 5 slots, `lo`, `hi`, `succ`, `child`, `alt`                                                     | |
# | `nkFor`          | 7 + one per target, `lo`, `hi`, `succ`, `child`, `loopName`, `filterLo`, `filterHi`, target ids | |
# | `nkSet`          | 4 slots, `lo`, `hi`, `succ`, interned target name id                                            | |
# | `nkSetNamespace` | 5 slots, `lo`, `hi`, `succ`, `target`, `field`                                                  | |
# | `nkMacroDef`     | 4 + 3 per parameter, `macroName`, filler, `succ`, `child` body, name and default-span triples   | |
# | `nkBreak`        | 3 slots, `lo`, `hi`, `succ`, the span naming the gap location                    |                |
# | `nkSetBlock`     | 5 slots, `lo`, `hi`, `succ`, `child` capture body, target name id                |                |
# | `nkGeneration`   | 4 slots, `lo`, `hi`, `succ`, `child` span body                                   |                |
#
# `nkBreak`, `nkSetBlock` and `nkGeneration` are declared-gap kinds, the parser building
# them so the step raises `ceUnimplemented` when a row reaches the construct.
#
# `succ` shadows `system.succ`, reachable for ordinal arguments.
# Overload resolution only sees a `Node` receiver in this module.

const
  ## One slot position per construct role, plus the two variable-tail bases.
  SlotLo = 0
  SlotHi = 1
  SlotSucc = 2
  SlotChild = 3
  SlotAlt = 4
  SlotLoopName = 4
  SlotFilterLo = 5
  SlotFilterHi = 6
  SlotNsTarget = 3
  SlotNsField = 4
  SlotMacroName = 0
  ForTargetsBase = 7
    ## First `nkFor` target name id slot, directly after the fixed prefix.
  MacroParamsBase = 4
    ## First `nkMacroDef` parameter slot, three slots per parameter.

template lo(nd: Node): int32 =
  ## Payload span start into `CompiledTemplate.jinja`, or the `nkMacroDef` macro name id.
  nd.slots[SlotLo]

template hi(nd: Node): int32 =
  ## Payload span end into `CompiledTemplate.jinja`, exclusive.
  nd.slots[SlotHi]

template succ(nd: Node): int32 =
  ## Next node in program order by node index, `NoLink` once the artifact is exhausted.
  nd.slots[SlotSucc]

template child(nd: Node): int32 =
  ## First node of the body by node index, or the `nkSet` target name id.
  nd.slots[SlotChild]

template alt(nd: Node): int32 =
  ## Next `nkIf` level in the else/elif chain by node index, `NoLink` when the chain ends.
  nd.slots[SlotAlt]

template loopName(nd: Node): int32 =
  ## Interned `loop` name id of `nkFor`, bound so render never interns at lookup time.
  nd.slots[SlotLoopName]

template filterLo(nd: Node): int32 =
  ## `nkFor` filter clause span start into `CompiledTemplate.jinja`, `NoLink` when the header has no `if`.
  nd.slots[SlotFilterLo]

template filterHi(nd: Node): int32 =
  ## `nkFor` filter clause span end into `CompiledTemplate.jinja`, exclusive.
  nd.slots[SlotFilterHi]

template target(nd: Node): int32 =
  ## Interned namespace name id of `nkSetNamespace`.
  nd.slots[SlotNsTarget]

template field(nd: Node): int32 =
  ## Interned member name id of `nkSetNamespace`.
  nd.slots[SlotNsField]

template macroName(nd: Node): int32 =
  ## Interned macro name id that `nkMacroDef` binds.
  nd.slots[SlotMacroName]

template targetCount(nd: Node): int32 =
  ## Number of `nkFor` target name ids in the payload tail.
  nd.slots.len - ForTargetsBase

template targetAt(nd: Node, i: int): int32 =
  ## `nkFor` target name id `i`, `i` in `0 ..< nd.targetCount`.
  nd.slots[ForTargetsBase + i]

template paramCount(nd: Node): int32 =
  ## Number of `nkMacroDef` parameters in the payload tail.
  (nd.slots.len - MacroParamsBase) div 3

template paramNameAt(nd: Node, k: int): int32 =
  ## `nkMacroDef` parameter `k` interned name id, `k` in `0 ..< nd.paramCount`.
  nd.slots[MacroParamsBase + 3 * k]

template paramDefLoAt(nd: Node, k: int): int32 =
  ## `nkMacroDef` parameter `k` default span start, `NoLink` when the parameter has no default.
  nd.slots[MacroParamsBase + 3 * k + 1]

template paramDefHiAt(nd: Node, k: int): int32 =
  ## `nkMacroDef` parameter `k` default span end, exclusive, meaningful only while `paramDefLoAt` is not `NoLink`.
  nd.slots[MacroParamsBase + 3 * k + 2]


type
  RowKind* = enum
    frFor, frGeneration, frMacro

  Row = object
    ## Render-state row, the only place re-entry is discriminated, `node` the entered node.
    node*: int32
    scopeAt*: int
      ## Scope state at row entry, a row close truncating scopes to this mark.
    case kind*: RowKind
    of frFor:
      loop*: LoopState
        ## cursor over the materialized iterable
      filterLo*, filterHi*: int32
        ## for-`if` clause span, `NoLink` when absent
    of frGeneration:
      spanStart*: int
        ## root-output byte position at span entry
    of frMacro:
      pc*: int32
        ## body node the render enters next
      retNode*: int32
        ## node control returns to once the body ends

  Binding = object
    ## One scope entry, an interned name bound to a value.
    name*: int32
    val*: JinjaVal

  Scope = seq[Binding]

  PieceKind* = enum
    pkNone, pkSpan, pkStr, pkCut, pkLazy

  Piece = object
    ## Pending output piece, span pieces delivering straight out of `CompiledTemplate.jinja`,
    ## string and cut pieces owned by the render state, lazy pieces the serializer's.
    pos*: int
    case kind*: PieceKind
    of pkNone: nil
    of pkSpan:
      lo*, hi*: int32
    of pkStr:
      s*: string
    of pkCut:
      raw*: string
        ## the cut's input string, moved in, the piece rendering its sub-span
      clo*, chi*: int32
        ## byte bounds of the sub-span, rendering as `raw[clo ..< chi]`
    of pkLazy:
      nil
        ## rendered by the serializer in `RenderState.lazy`, no payload here

  RenderState = object
    ## All per-instantiation render control state, nothing reachable from `CompiledTemplate`,
    ## so two instantiations over one artifact cannot interfere.
    curNode*: int32
      ## node program counter, `NoLink` once the artifact is exhausted
    cur*: int
      ## bytes of root output delivered so far, composed with the caller's delivery window
    pend*: Piece
    rows*: seq[Row]
      ## re-entry stack holding for, capture and generation rows
    scopes*: seq[Scope]
      ## scope 0 is the render context. For-rows push and pop above it
    root*: JinjaVal
      ## the render context dict (`messages`, `tools`, `kwargs`), the outermost lookup scope
    spans*: seq[tuple[start, stop: int]]
      ## generation spans in root-output byte coordinates
    clock*: float64
      ## injected epoch, `strftime_now`'s only time source, never the wall
    macroDepth*: int
    lazy*: Ser
      ## serializer state machine of a pending lazy piece, repositioned from byte 0 per value

  JinjaRenderContext* = ref object
    ## One heap render context per render, opened by `startRender` and owned by the caller,
    ## holding the shared artifact, the shared interned-name table ref, one per-instantiation
    ## render state and the engine's macro-force callable.
    ## - a context is a ref and is never copied, borrows are ref borrows, and there
    ##   are no threads in the engine, so a field write through one borrow is
    ##   visible to every other borrow of the same context
    ## - a consumer that stops mid-render resumes through the same context object.
    ##   A second render over the artifact opens a second context through `startRender`
    tmpl*: CompiledTemplate
    symbols*: CompiledSymbols
      ## the parse-built interned-name table, shared by ref with the parse caller, no lifetime contract
    state*: RenderState
    force*: MacroForcer
      ## the engine's macro-force callable, bound once at `startRender`, stateless

  MacroForcer = proc (c: JinjaRenderContext, mc: MacroVal, args: var Args): JinjaVal {.nimcall.}
    ## Runs one macro body to completion on a second render context built over the caller's
    ## artifact refs, the captured text returned as a string value,
    ## the caller's render context untouched.

func findName(t: CompiledSymbols, name: openArray[char]): int32 =
  ## Returns the interned id of `name`, or `NoLink` when the template never names it.
  for i, n in t.names:
    if n == name:
      return int32 i
  NoLink

func scopeHas(st: var RenderState, id: int32, val: var JinjaVal): bool =
  ## Scope scan innermost first, returning true with `val` set when `id` is bound.
  for si in countdown(st.scopes.len - 1, 0):
    for b in st.scopes[si]:
      if b.name == id:
        val = b.val
        return true
  false

func lookupName(c: JinjaRenderContext, name: openArray[char]): JinjaVal =
  ## Returns the binding of `name` in one render, scopes innermost first, then the root dict, then undefined.
  ## Absence is a value, never an error.
  let id = c.symbols.findName(name)
  var got: JinjaVal
  if id != NoLink and c.state.scopeHas(id, got):
    return got
  if c.state.root.kind == vkDict:
    return c.state.root.d.dictGet(name)
  undefinedVal()


proc internName(t: var CompiledSymbols, name: openArray[char]): int32 =
  ## Returns the interned id of `name`, inserting the one copy into `CompiledSymbols.names` when absent.
  result = t.findName(name)
  if result != NoLink:
    return
  result = int32 t.names.len
  t.names.add spanString(name)
