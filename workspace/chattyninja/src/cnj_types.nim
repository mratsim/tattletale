# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Core data of the chattyninja engine. Covers the compiled artifact, the parse-built
# symbol arena, the per-instantiation render state, the name-resolution reads
# and the macro-force handle. The dispatch table, the render context bundle
# and the `items` pull interface live in cnj_engine.nim.
#
# Dataflow of one render, the record types here shared across the module boundary
#
#   template bytes (borrowed, never copied at parse)
#     │  cnj_parse splits tags, scans keywords, appends arena nodes, interns names once
#     ▼
#   CompiledTemplate + CompiledSymbols (read-only artifact + interned-name arena)
#     │  cnj_engine dispatches pull() steps over the arena
#     ▼
#   RenderState (rows + scopes + pending Piece)
#     │  evalSpan drives jinja_interpolation, expressions reading the render state
#     │  directly and reaching the statement tier only through the MacroForcer handle
#     ▼
#   Piece (pkSpan, pkStr, pkCut or pkLazy), drained into the caller's window
#
# Lifecycle and ownership of the render state and its forcer handle
#
#   RenderState, one per render, owned by the caller's Context
#     ├─ rows     pushed by step* entry, popped by closeRow, one close path
#     ├─ scopes   owned by rows (scopeAt marks the base), trimmed on close
#     ├─ pend     one Piece, set by emit steps, drained by pull or capturePend, reset to pkNone
#     └─ force    the engine's macro forcer, bound per dispatch, borrowed by expressions

import jinja_data_model, jinja_serialize
import workspace/data_structures/src/small_seqs

type
  NodeKind* {.pure.} = enum
    ## Corpus-derived construct vocabulary, one entry per engine step.
    ##
    ## | Kind             | Payload                                                                     | |
    ## | ---------------- | ----------------------------------------------------------------------------- |
    ## | `nkVerbatim`     | final text run, whitespace already resolved                                 | |
    ## | `nkEmit`         | `{{ }}` span, evaluates the `lo..hi` span, stringifies, pending piece       | |
    ## | `nkIf`           | condition span, then-body, else-or-elif chain, bodies terminated past endif | |
    ## | `nkFor`          | iterable span, body, loop name, filter span, interned target ids            | |
    ## | `nkBreak`        | unwinds to the nearest for-row, stopping at a macro-call boundary          |  |
    ## | `nkSet`          | single-target binding of an expression                                      | |
    ## | `nkSetNamespace` | `ns.field = expr`, ns and field as interned name ids                        | |
    ## | `nkSetBlock`     | capture body into a sink, bind on close                                     | |
    ## | `nkGeneration`   | marks the root-output span of the model's turn                              | |
    ## | `nkMacroDef`     | binds a macro value, never executes                                         | |
    nkVerbatim, nkEmit, nkIf, nkFor, nkBreak, nkSet, nkSetNamespace, nkSetBlock, nkGeneration,
    nkMacroDef

  Node* = object
    ## POD node in one append-only arena, `kind` naming the construct, a node's executable
    ## meaning a pure function of `kind` through `steps`, so the artifact stays data.
    ## Every payload reference is one int32 slot, `NoLink` (-1) marking an absent link or span:
    ## - a span into `CompiledTemplate.jinja`, an arena index, or an interned name id
    ## - slots `0`-`3` uniform across kinds, `4` and past kind-specific, see the accessors below
    kind*: NodeKind
    slots*: SmallSeq[5, int32]
      ## Payload slots, capacity 5 the measured corpus knee:
      ## - 93% of the 1098 nodes of the 13 corpus templates that parse hold at most 5 slots
      ## - no node holds exactly 6, and the other 5 corpus templates raise declared gaps
      ## - only a variable `nkFor` or `nkMacroDef` payload spills, one heap block at parse time

  CompiledTemplate* = ref object
    ## Read-only compiled template, shared across renders with two fields and no mutable state,
    ## so one artifact serves any number of render instantiations.
    ##
    ## Borrow contract:
    ## - `jinja` is borrowed, so the artifact must not outlive the caller's template text
    ## - the caller builds `CompiledTemplate` at the scope that owns the text
    jinja*: openArray[char]
    nodes*: seq[Node]

  CompiledSymbols* = object
    ## Parse-built interned-name arena, read-only at render and shared by every render
    ## over its artifact. Node int32 name slots index into it, so `CompiledTemplate` is
    ## only meaningful together with the matching `CompiledSymbols`.
    ##
    ## Name resolution is one linear scan over `names`, parse-time only, the corpus
    ## topping out at 33 interned names, render lookups carrying interned ids.
    names*: seq[string]

const
  ## Caps measured against the corpus, each a compile-time define.

  # Macro recursion is the engine's only render-time recursion. A macro body's output is
  # a string value, so a call runs to completion synchronously into a capture sink.
  # Static cross-macro chain depth in the corpus is 3, two templates recur cyclically, and the shipped
  # `gemma4/tools_tool_response.json` drives the tools subtree to 6, with real depth set by the input.
  # 16 bounds the stack on bad input, a breach raising.
  MacroDepthCap* {.intdefine.} = 16

  # Expression-walker recursion bound:
  # deepest paren nesting measured over the corpus templates is 3 (`gemma4`, `lfm25`),
  # 24 clears it with margin, far below the C stack overflow depth.
  ExprDepthCap* {.intdefine.} = 24

  # Parse-time nesting bound of the parser's dispatch recursion, one level per body-carrying
  # construct. Corpus nesting tops out at 5 (`glm53flash`'s macro-in-for chain), so 64 clears it
  # with margin and bounds the walk on adversarial input, a breach raising located at the tag.
  ParseNestingCap* {.intdefine.} = 64

  # Output pieces reach the consumer in slices of at most this many bytes, which is what lets `cur`
  # compose with chunking.
  ChunkSize* {.intdefine.} = 4096

  # Step-dispatch bound of one `pull` call. One call dispatches at most this many steps,
  # a breach raising located at the node the walk reached. The corpus suite completes
  # with 240 and fails with 230, so no corpus pull dispatches past 240, and a 200x200
  # nested loop test lands in the low thousands. 1_000_000 keeps ample headroom.
  StepBudget* {.intdefine.} = 1_000_000

  Whitespace* = {' ', '\t', '\n', '\r', '\v', '\f'}
  WsNameChars* = {'a' .. 'z', 'A' .. 'Z', '0' .. '9', '_'}

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
  ## One slot position per construct role, plus the two variable-tail bases. Parse and render
  ## index through the same constants, so a slot-layout move is one shared edit.
  SlotLo = 0
  SlotHi* = 1
  SlotSucc* = 2
  SlotChild* = 3
  SlotAlt* = 4
  SlotLoopName = 4
  SlotSetTarget = 4
    ## `nkSetBlock` interned target name id, the capture body binding it on close.
  SlotFilterLo* = 5
  SlotFilterHi* = 6
  SlotNsTarget = 3
  SlotNsField = 4
  SlotMacroName = 0
  ForTargetsBase = 7
    ## First `nkFor` target name id slot, directly after the fixed prefix.
  MacroParamsBase = 4
    ## First `nkMacroDef` parameter slot, each parameter one name id then its default span,
    ## three slots in all.

template lo*(nd: Node): int32 =
  ## Payload span start into `CompiledTemplate.jinja`, or the `nkMacroDef` macro name id.
  nd.slots[SlotLo]

template hi*(nd: Node): int32 =
  ## Payload span end into `CompiledTemplate.jinja`, exclusive.
  nd.slots[SlotHi]

template succ*(nd: Node): int32 =
  ## Next node in program order by arena index, `NoLink` once the artifact is exhausted.
  nd.slots[SlotSucc]

template child*(nd: Node): int32 =
  ## First node of the body by arena index, or the `nkSet` target name id.
  nd.slots[SlotChild]

template alt*(nd: Node): int32 =
  ## Next `nkIf` level in the else/elif chain by arena index, `NoLink` when the chain ends.
  nd.slots[SlotAlt]

template loopName*(nd: Node): int32 =
  ## Interned `loop` name id of `nkFor`, bound so render never interns at lookup time.
  nd.slots[SlotLoopName]

template filterLo*(nd: Node): int32 =
  ## `nkFor` filter clause span start into `CompiledTemplate.jinja`, `NoLink` when the header has no `if`.
  nd.slots[SlotFilterLo]

template filterHi*(nd: Node): int32 =
  ## `nkFor` filter clause span end into `CompiledTemplate.jinja`, exclusive.
  nd.slots[SlotFilterHi]

template target*(nd: Node): int32 =
  ## Interned namespace name id of `nkSetNamespace`.
  nd.slots[SlotNsTarget]

template field*(nd: Node): int32 =
  ## Interned member name id of `nkSetNamespace`.
  nd.slots[SlotNsField]

template setTarget*(nd: Node): int32 =
  ## Interned target name id of `nkSetBlock`, the name the capture body binds on close.
  nd.slots[SlotSetTarget]

template macroName*(nd: Node): int32 =
  ## Interned macro name id that `nkMacroDef` binds.
  nd.slots[SlotMacroName]

template targetCount*(nd: Node): int32 =
  ## Number of `nkFor` target name ids in the payload tail.
  nd.slots.len - ForTargetsBase

template targetAt*(nd: Node, i: int): int32 =
  ## `nkFor` target name id `i`, `i` in `0 ..< nd.targetCount`.
  nd.slots[ForTargetsBase + i]

template paramCount*(nd: Node): int32 =
  ## Number of `nkMacroDef` parameters in the payload tail.
  (nd.slots.len - MacroParamsBase) div 3

template paramNameAt*(nd: Node, k: int): int32 =
  ## `nkMacroDef` parameter `k` interned name id, `k` in `0 ..< nd.paramCount`.
  nd.slots[MacroParamsBase + 3 * k]

template paramDefLoAt*(nd: Node, k: int): int32 =
  ## `nkMacroDef` parameter `k` default span start, `NoLink` when the parameter has no default.
  nd.slots[MacroParamsBase + 3 * k + 1]

template paramDefHiAt*(nd: Node, k: int): int32 =
  ## `nkMacroDef` parameter `k` default span end, exclusive, meaningful only while
  ## `paramDefLoAt` is not `NoLink`.
  nd.slots[MacroParamsBase + 3 * k + 2]


type
  RowKind* = enum
    frFor, frGeneration, frMacro

  Row* = object
    ## Render-state row, the only place re-entry is discriminated. `node` is the row's
    ## identity and matches the node being entered, nothing about resumption living in the node.
    node*: int32
    scopeAt*: int
      ## Scope state at row entry. A row close truncates scopes to this mark.
      ## Bindings the row pushed or mutated live above it
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

  Binding* = object
    ## One scope entry:
    ##   an interned name bound to a value.
    name*: int32
    val*: JinjaVal

  Scope* = seq[Binding]

  PieceKind* = enum
    pkNone, pkSpan, pkStr, pkCut, pkLazy

  Piece* = object
    ## Pending output piece:
    ## - span pieces deliver straight out of `CompiledTemplate.jinja`
    ## - string and cut pieces are owned by the render state, the string materialized,
    ##   the cut rendering its sub-span bytes
    ## - lazy pieces are a derived value the serializer in `RenderState.lazy` renders
    ##   straight into the delivery window
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

  RenderState* = object
    ## All per-instantiation render control state, owned by the pull consumer, nothing
    ## reachable from `CompiledTemplate`, so two instantiations over one artifact cannot
    ## interfere. Lives only at the step tier, the expression evaluator never seeing it.
    curNode*: int32
      ## node program counter, `NoLink` once the artifact is exhausted
    cur*: int
      ## bytes of root output delivered so far, composed with the chunking window
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

  MacroForcer* = proc (tmpl: CompiledTemplate, sym: ptr CompiledSymbols,
      st: var RenderState, mc: MacroVal, args: Args): JinjaVal {.nimcall, noSideEffect.}
    ## Runs one macro body to completion on the render state given, returning the captured
    ## text as a string value.
    ## Contract:
    ## - bound by the statement tier's engine at its dispatch sites, the expression tier
    ##   receiving it as a plain stateless handle
    ## - every state a call serves arrives as a typed borrow on the call itself, so no
    ##   adapter value and no erased pointer exists
    ## - this handle is the one edge the tier split keeps, no import cycle crossing it

  Context* = object
    ## Object the caller holds, bundling the shared artifact, a borrowed symbol-arena pointer,
    ## and one per-instantiation render state. Copies render independently.
    ## A consumer that stops mid-render resumes through its own copy only.
    tmpl*: CompiledTemplate
    symbols*: ptr CompiledSymbols
      ## borrowed, must not outlive the binding it was taken from, the same contract
      ## class as the `jinja` borrow of the template text
    state*: RenderState

func findName*(t: CompiledSymbols, name: openArray[char]): int32 =
  ## Returns the interned id of `name`, or `NoLink` when the template never names it.
  ## One linear scan over the interned arena, allocation-free and parse-time only,
  ## the corpus topping out at 33 names.
  for i, n in t.names:
    if n == name:
      return int32 i
  NoLink

func scopeHas*(st: var RenderState, id: int32, val: var JinjaVal): bool =
  ## Scope scan innermost first, returning true with `val` set when `id` is bound.
  ## A binding to an undefined value is still a binding, so the root lookup never sees it.
  for si in countdown(st.scopes.len - 1, 0):
    for b in st.scopes[si]:
      if b.name == id:
        val = b.val
        return true
  false

func lookupName*(sym: CompiledSymbols, st: var RenderState, name: openArray[char]): JinjaVal =
  ## Returns the binding of `name` in one render, resolving the scopes innermost first,
  ## then the render context root dict, then undefined.
  ## Absence is a value, never an error, `is defined` testing for exactly that shape.
  let id = sym.findName(name)
  var got: JinjaVal
  if id != NoLink and st.scopeHas(id, got):
    return got
  if st.root.kind == vkDict:
    return st.root.d.dictGet(name)
  undefinedVal()


func internName*(t: var CompiledSymbols, name: openArray[char]): int32 =
  ## Returns the interned id of `name`, inserting the one arena copy when absent.
  ## - a carried name allocates nothing
  ## - a new name copies exactly once into `CompiledSymbols.names`
  result = t.findName(name)
  if result != NoLink:
    return
  result = int32 t.names.len
  t.names.add spanString(name)
