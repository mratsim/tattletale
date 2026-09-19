# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Core data of the chattyninja engine. Covers the compiled artifact, the parse-built
# symbol arena, the per-instantiation render state, and the injected render ports
# the expression tier reads the render through. See cnj_engine.nim for the dispatch table,
# the render context bundle, the port adapter, and the `items` pull interface.

import jinja_data_model, jinja_serialize
import workspace/data_structures/src/small_seqs


type
  NodeKind* {.pure.} = enum
    ## Corpus-derived construct vocabulary, one entry per Step function.
    ##
    ## | Kind             | Payload                                                                     |
    ## | ---------------- | --------------------------------------------------------------------------- |
    ## | `nkVerbatim`     | final text run, whitespace already resolved                                 |
    ## | `nkEmit`         | `{{ }}` span, evaluates the `lo..hi` span, stringifies, pending piece       |
    ## | `nkIf`           | condition span, then-body, else-or-elif chain, bodies terminated past endif |
    ## | `nkFor`          | iterable span, body, loop name, filter span, interned target ids            |
    ## | `nkBreak`        | unwinds to the nearest for-frame, stopping at a macro-call boundary         |
    ## | `nkSet`          | single-target binding of an expression                                      |
    ## | `nkSetNamespace` | `ns.field = expr`, ns and field as interned name ids                        |
    ## | `nkSetBlock`     | capture body into a sink, bind on close                                     |
    ## | `nkGeneration`   | marks the root-output span of the model's turn                              |
    ## | `nkMacroDef`     | binds a macro value, never executes                                         |
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

  # Output pieces reach the consumer in slices of at most this many bytes, which is what lets `cur`
  # compose with chunking.
  ChunkSize* {.intdefine.} = 4096

  Whitespace* = {' ', '\t', '\n', '\r', '\v', '\f'}
  wsNameChars* = {'a' .. 'z', 'A' .. 'Z', '0' .. '9', '_'}

# Payload slot accessors
#
# One slot per construct role, uniform across the kinds that fill it. Each expands textually
# to a slot read, so `nd.succ` on `m.nodes[n]` never copies the node. The per-accessor docs
# below name each slot's meaning and kind. Slot layouts per kind:
#
# | Kind             | Slots                                                                                           |
# |------------------|-------------------------------------------------------------------------------------------------|
# | `nkVerbatim`     | 3 slots, `lo`, `hi`, `succ`                                                                     |
# | `nkEmit`         | 3 slots, `lo`, `hi`, `succ`                                                                     |
# | `nkIf`           | 5 slots, `lo`, `hi`, `succ`, `child`, `alt`                                                     |
# | `nkFor`          | 7 + one per target, `lo`, `hi`, `succ`, `child`, `loopName`, `filterLo`, `filterHi`, target ids |
# | `nkSet`          | 4 slots, `lo`, `hi`, `succ`, interned target name id                                            |
# | `nkSetNamespace` | 5 slots, `lo`, `hi`, `succ`, `target`, `field`                                                  |
# | `nkMacroDef`     | 4 + 3 per parameter, `macroName`, filler, `succ`, `child` body, name and default-span triples   |
#
# `nkSetBlock` and `nkGeneration` are declared kinds the parser never builds, so they carry
# no slot layout. `succ` shadows `system.succ`, still reachable for ordinal arguments,
# overload resolution only seeing a `Node` receiver in this module.

const
  ## One slot position per construct role, plus the two variable-tail bases. Parse and render
  ## index through the same constants, so a slot-layout move is one shared edit.
  SlotLo* = 0
  SlotHi* = 1
  SlotSucc* = 2
  SlotChild* = 3
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
  nd.slots[slotAlt]

template loopName*(nd: Node): int32 =
  ## Interned `loop` name id of `nkFor`, bound so render never interns at lookup time.
  nd.slots[slotLoopName]

template filterLo*(nd: Node): int32 =
  ## `nkFor` filter clause span start into `CompiledTemplate.jinja`, `NoLink` when the header has no `if`.
  nd.slots[slotFilterLo]

template filterHi*(nd: Node): int32 =
  ## `nkFor` filter clause span end into `CompiledTemplate.jinja`, exclusive.
  nd.slots[slotFilterHi]

template target*(nd: Node): int32 =
  ## Interned namespace name id of `nkSetNamespace`.
  nd.slots[slotNsTarget]

template field*(nd: Node): int32 =
  ## Interned member name id of `nkSetNamespace`.
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
  ## `nkMacroDef` parameter `k` default span start, `NoLink` when the parameter has no default.
  nd.slots[macroParamsBase + 3 * k + 1]

template paramDefHiAt*(nd: Node, k: int): int32 =
  ## `nkMacroDef` parameter `k` default span end, exclusive, meaningful only while
  ## `paramDefLoAt` is not `NoLink`.
  nd.slots[macroParamsBase + 3 * k + 2]


type
  FrameKind* = enum
    frFor, frCapture, frGeneration, frMacro

  Frame* = object
    ## Render-state frame, the only place re-entry is discriminated. `node` is the frame's
    ## identity and matches the node being entered, nothing about resumption living in the node.
    node*: int32
    scopeAt*: int
      ## One past the mark popped back to on close.
      ## The entry scope occupies `scopes[scopeAt - 1]`, and close truncates to that mark
    case kind*: FrameKind
    of frFor:
      loop*: LoopState
        ## cursor over the materialized iterable
      filterLo*, filterHi*: int32
        ## for-`if` clause span, `NoLink` when absent
    of frCapture:
      target*: int32
        ## interned name to bind on close, never built while `nkSetBlock` is a declared gap
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
    pkNone, pkSpan, pkStr, pkLazy

  Piece* = object
    ## Pending output piece:
    ## - span pieces deliver straight out of `CompiledTemplate.jinja`
    ## - string pieces are materialized strings held by the render state
    ## - lazy pieces are a derived value the serializer in `RenderState.lazy` renders
    ##   straight into the delivery window
    pos*: int
    case kind*: PieceKind
    of pkNone: nil
    of pkSpan:
      lo*, hi*: int32
    of pkStr:
      s*: string
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
    frames*: seq[Frame]
      ## re-entry stack holding for, capture and generation frames
    scopes*: seq[Scope]
      ## scope 0 is the render context. For-frames push and pop above it
    root*: JinjaVal
      ## the render context dict (`messages`, `tools`, `kwargs`), the outermost lookup scope
    spans*: seq[tuple[start, stop: int]]
      ## generation spans in root-output byte coordinates
    clock*: float64
      ## injected epoch, `strftime_now`'s only time source, never the wall
    macroDepth*: int
    lazy*: Ser
      ## serializer state machine of a pending lazy piece, repositioned from byte 0 per value

  LookupPort* = proc (env: pointer, name: openArray[char]): JinjaVal {.nimcall, noSideEffect.}
    ## Resolves one name of the enclosing render to its binding, undefined when absent.
    ## `env` carries the adapter state the trampoline reads, owned by the pull consumer.

  ClockPort* = proc (env: pointer): float64 {.nimcall, noSideEffect.}
    ## Returns the render's injected epoch, `strftime_now`'s only time source.

  MacroForcer* = proc (env: pointer, mc: MacroVal, args: Args): JinjaVal {.nimcall, noSideEffect.}
    ## Runs one macro body to completion and returns the captured text as a string value.
    ## The pull consumer injects the forcer, so the expression tier never reaches the statement
    ## tier and no import cycle forms.

  Ports* = object
    ## Render services the expression tier reads, injected per dispatch. The adapter behind
    ## `env` lives with the pull consumer and is rebuilt there, so a value a step carries
    ## never outlives the call that built it.
    lookup*: LookupPort
    clock*: ClockPort
    force*: MacroForcer
    env*: pointer
      ## adapter state the trampolines cast back, opaque here by construction

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
  ## Returns the interned id of `name`, or `NoLink` when the template never names it, the comparison reading the caller's bytes in place so
  ## an interned name allocates nothing.
  for i, n in t.names:
    if n == name:
      return int32 i
  NoLink
