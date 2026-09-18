# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Core data of the chattyninja engine. Covers the compiled artifact, the parse-built side tables, and the render
# driver. See chattyninja.nim for the dispatch table and the `items` pull interface.

import cjn_errors, cjn_values

export cjn_errors, cjn_values

const
  noLink* = -1'i32
    ## Marks an absent link in every int32 node field.

type
  NodeKind* {.pure.} = enum
    ## Corpus-derived construct vocabulary, one entry per Step function.
    ##
    ## | Kind           | Payload                                                                     |
    ## | -------------- | --------------------------------------------------------------------------- |
    ## | `nkVerbatim`   | final text run, whitespace already resolved                                 |
    ## | `nkEmit`       | `{{ }}` span, evaluates the `lo..hi` span, stringifies, pending piece       |
    ## | `nkIf`         | condition span, then-body, else-or-elif chain, bodies terminated past endif |
    ## | `nkFor`        | iterable span, body, aux record for targets and the filter clause           |
    ## | `nkBreak`      | unwinds to the nearest for-frame, stopping at a macro-call boundary         |
    ## | `nkSet`        | single-target binding of an expression                                      |
    ## | `nkSetNs`      | `ns.field = expr`, ns and field as interned name ids                        |
    ## | `nkSetBlock`   | capture body into a sink, bind on close                                     |
    ## | `nkGeneration` | marks the root-output span of the model's turn                              |
    ## | `nkMacroDef`   | binds a macro value, never executes                                         |
    nkVerbatim
    nkEmit
    nkIf
    nkFor
    nkBreak
    nkSet
    nkSetNs
    nkSetBlock
    nkGeneration
    nkMacroDef

  Node* = object
    ## POD node in one append-only arena. Six fields, no proc field. A node's executable meaning
    ## is a pure function of `kind` through `steps`, so the artifact stays data.
    ##
    ## `noLink` (-1) marks an absent link in every int32 field.
    kind*: NodeKind
    lo*, hi*: int32
      ## payload span into `Machine.jinja`:
      ##   the expression or text this node owns
    succ*, child*, alt*: int32
      ## graph links by arena index

  Machine* = object
    ## Read-only compiled template. Two fields and no mutable state, so one artifact renders
    ## concurrently under separate drivers. `jinja` is borrowed. The artifact must not outlive
    ## the template text it points into, which is why it is built at the caller's scope.
    jinja*: openArray[char]
    nodes*: seq[Node]

  AuxKind* = enum
    axFor
    axSet
    axMacro

  MacroParam* = object
    ## One macro parameter:
    ##   an interned name plus the default expression span, `noLink` when absent.
    name*: int32
    defLo*, defHi*: int32

  Aux* = object
    ## Variable-length payloads the six node fields cannot hold. Parse-built and read-only at render, so
    ## `nkFor` and `nkMacroDef` keep the six-field budget.
    case kind*: AuxKind
    of axFor:
      targets*: seq[int32]
        ## interned loop target names, more than one means tuple unpacking
      loopName*: int32
        ## interned `loop`, bound alongside the targets so the driver never interns at render time
      filterLo*, filterHi*: int32
        ## the `if` clause of a for header, `noLink` when absent
    of axSet:
      target*: int32
        ## interned namespace name
      field*: int32
        ## interned member name
    of axMacro:
      name*: int32
        ## interned macro name, the binding `nkMacroDef` writes
      params*: seq[MacroParam]

  Tables* = object
    ## Parse-built side arenas, read-only at render, passed into the driver. Node int32 name and aux
    ## fields index these, so a `Machine` is only meaningful together with its `Tables`.
    names*: seq[string]
    aux*: seq[Aux]

const
  ## Caps measured against the corpus, each a compile-time define.

  # Macro recursion is the engine's only render-time recursion. A macro body's output is
  # a string value, so a call cannot stream and runs to completion synchronously into a capture
  # sink. Static cross-macro chain depth in the corpus is 3, and two templates recur cyclically
  # (`gptoss20b` `render_typescript_type` on schema `items` /`variant`, `gemma4` `format_parameters` with `format_argument`),
  # so real depth is set by the input. The shipped `gemma4/tools_tool_response.json` drives
  # the tools subtree to 6. 16 bounds the stack on pathological input, and a breach raises
  # instead of overflowing.
  MacroDepthCap* {.intdefine.} = 16

  # The expression walker recurses on nesting. Deepest paren nesting measured over the corpus
  # templates is 3 (`gemma4`, `lfm25`), so 24 clears the observed maximum with margin
  # while staying far below the depth a C stack would overflow on.
  ExprDepthCap* {.intdefine.} = 24

  # Output pieces reach the consumer in slices of at most this many bytes, which is what lets `cur`
  # compose with chunking.
  ChunkSize* {.intdefine.} = 4096

  wsSpace* = {' ', '\t', '\n', '\r', '\v', '\f'}
  wsNameChars* = {'a' .. 'z', 'A' .. 'Z', '0' .. '9', '_'}

func findName*(t: Tables, name: openArray[char]): int32 =
  ## Returns the interned id of `name`, or `noLink` when the template never names it.
  ## Comparison reads the caller's bytes in place, so an already interned name allocates nothing.
  for i, n in t.names:
    if n == name:
      return int32 i
  noLink

type
  FrameKind* = enum
    frFor
    frCapture
    frGeneration

  Frame* = object
    ## Driver frame:
    ##   the only place re-entry is discriminated. `node` is the frame's identity,
    ## matched against the node being entered. Nothing about resumption lives in the node.
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
    pkNone
    pkSpan
    pkStr

  Piece* = object
    ## Pending output piece. Span pieces are delivered straight out of `Machine.jinja`, string
    ## pieces are materialized values (stringify, tojson, a captured body) held by the driver.
    pos*: int
    case kind*: PieceKind
    of pkNone: nil
    of pkSpan:
      lo*, hi*: int32
    of pkStr:
      s*: string

  Driver* = object
    ## All render control state, owned by the `items` loop. Nothing here is reachable from `Machine`, so
    ## two drivers over one artifact cannot interfere.
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

