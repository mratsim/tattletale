# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## lean_pragmas: verification pragmas and their payload language (P3, poc-local)
##
## Lean properties cannot be written in Nim, so each pragma payload is a string holding
## a Lean property expression over the kernel's parameter names,
## plus the model binders `tid` (thread id) and `n` (buffer bound).
##
## - `{.precondition: "<payload>".}` - assumed at launch, never proved
## - `{.invariant: "<payload>".}` - holds whenever the guarded branch is entered, implied by the guard `tid < n`
## - `{.postcondition: "<payload>".}` - holds for the written buffer after the thread ran
##
## - `prop := cmp ('∧' cmp)*`
## - `cmp  := sum ('=' | '≠' | '<' | '≤') sum`
## - `sum  := term ('+' term)*`
## - `term := ident | ident 'tid' | uint literal`
##
## Example: `{.postcondition: "output tid = a tid + b tid".}` on the add kernel.
## A well-formed payload stating a false property is not judged here.
## The emitter downgrades its failed Lean proof to a sorry-backed scaffold.
## Anything the grammar cannot parse is refused loudly.

import std/[strutils, tables]

template precondition*(expr: static string) {.pragma.}
  ## Payload: a Lean property assumed true at kernel launch, like `"tid < n"`.

template invariant*(expr: static string) {.pragma.}
  ## Payload: a Lean property implied by the guard of the guarded branch, `tid < n` in the tracer model, like `"tid < n"`.

template postcondition*(expr: static string) {.pragma.}
  ## Payload: a Lean property over the kernel parameters, `tid` and `n`,
  ## stating the written buffer's state after the thread ran, like `"output tid = a tid + b tid"`.

type
  RawSpec* = object
    ## One pragma captured off the kernel proc, before the payload is parsed
    name*: string    ## one of precondition, invariant, postcondition
    payload*: string ## the string payload verbatim

  SpecTokenKind* = enum
    tkIdent  ## `[A-Za-z_][A-Za-z0-9_]*`
    tkUint   ## `[0-9]+`
    tkPlus   ## `+`
    tkEq     ## `=`
    tkNeq    ## `≠`
    tkLt     ## `<`
    tkLe     ## `≤`
    tkAnd    ## `∧`
    tkEof

  SpecToken* = object
    kind*: SpecTokenKind
    text*: string ## identifier name or literal digits

  SpecExprKind* = enum
    seIdent ## bare binder: `tid` or `n`
    seApp   ## buffer read: `<buffer> tid`
    seUint  ## unsigned integer literal
    seBin   ## binary operator: `+`, `=`, `≠`, `<`, `≤`, `∧`

  SpecExpr* = ref object
    ## Parsed payload. The grammar fixes precedence (`∧` < cmp < `+`).
    ## `seBin` chains are left-associative, so rendering needs no parentheses.
    case kind*: SpecExprKind
    of seIdent:
      name*: string
    of seApp:
      fn*: string ## the buffer parameter name, applied to `tid`
    of seUint:
      value*: string
    of seBin:
      op*: string
      left*, right*: SpecExpr

proc tokenizeSpec(payload: string): seq[SpecToken] =
  ## Splits a payload into spec tokens, refusing characters outside the grammar with a `nim_to_lean: invalid spec payload` ValueError
  var i = 0
  while i < payload.len:
    let c = payload[i]
    if c in {' ', '\t'}:
      inc i
    elif c in IdentStartChars:
      let start = i
      while i < payload.len and payload[i] in IdentChars:
        inc i
      result.add SpecToken(kind: tkIdent, text: payload[start ..< i])
    elif c in {'0' .. '9'}:
      let start = i
      while i < payload.len and payload[i] in {'0' .. '9'}:
        inc i
      result.add SpecToken(kind: tkUint, text: payload[start ..< i])
    elif c == '+':
      result.add SpecToken(kind: tkPlus, text: "+")
      inc i
    elif c == '=':
      result.add SpecToken(kind: tkEq, text: "=")
      inc i
    elif c == '<':
      result.add SpecToken(kind: tkLt, text: "<")
      inc i
    elif payload[i ..< min(i + 3, payload.len)] == "≠":
      result.add SpecToken(kind: tkNeq, text: "≠")
      inc i, 3
    elif payload[i ..< min(i + 3, payload.len)] == "≤":
      result.add SpecToken(kind: tkLe, text: "≤")
      inc i, 3
    elif payload[i ..< min(i + 3, payload.len)] == "∧":
      result.add SpecToken(kind: tkAnd, text: "∧")
      inc i, 3
    else:
      raise newException(ValueError, "nim_to_lean: invalid spec payload '" &
        payload & "' - unexpected character '" & $c & "' at byte " & $i)

proc peek(toks: seq[SpecToken], i: int): SpecToken =
  ## Token at `i`, or `tkEof` at the end of input
  if i < toks.len: toks[i]
  else: SpecToken(kind: tkEof)

proc parseSum(toks: seq[SpecToken], i: var int): SpecExpr
proc parseCmp(toks: seq[SpecToken], i: var int): SpecExpr
proc parseProp(toks: seq[SpecToken], i: var int): SpecExpr

proc parseTerm(toks: seq[SpecToken], i: var int): SpecExpr =
  ## term := ident | ident 'tid' | uint literal
  let tok = toks.peek(i)
  case tok.kind
  of tkIdent:
    inc i
    if toks.peek(i).kind == tkIdent and toks.peek(i).text == "tid":
      inc i
      result = SpecExpr(kind: seApp, fn: tok.text)
    else:
      result = SpecExpr(kind: seIdent, name: tok.text)
  of tkUint:
    inc i
    result = SpecExpr(kind: seUint, value: tok.text)
  else:
    raise newException(ValueError, "nim_to_lean: invalid spec payload - " &
      "expected an identifier or a uint literal, got '" & tok.text & "'")

proc parseSum(toks: seq[SpecToken], i: var int): SpecExpr =
  ## sum := term ('+' term)*
  result = parseTerm(toks, i)
  while toks.peek(i).kind == tkPlus:
    inc i
    let r = parseTerm(toks, i)
    result = SpecExpr(kind: seBin, op: "+", left: result, right: r)

proc parseCmp(toks: seq[SpecToken], i: var int): SpecExpr =
  ## cmp := sum ('=' | '≠' | '<' | '≤') sum
  result = parseSum(toks, i)
  let tok = toks.peek(i)
  if tok.kind in {tkEq, tkNeq, tkLt, tkLe}:
    inc i
    let r = parseSum(toks, i)
    result = SpecExpr(kind: seBin, op: tok.text, left: result, right: r)

proc parseProp(toks: seq[SpecToken], i: var int): SpecExpr =
  ## prop := cmp ('∧' cmp)*
  result = parseCmp(toks, i)
  while toks.peek(i).kind == tkAnd:
    inc i
    let r = parseCmp(toks, i)
    result = SpecExpr(kind: seBin, op: "∧", left: result, right: r)

proc parseSpecPayload*(payload: string): SpecExpr =
  ## Parses one payload into a `SpecExpr`, refusing every construct outside
  ## the grammar with a `nim_to_lean: invalid spec payload` ValueError
  let toks = tokenizeSpec(payload)
  var i = 0
  result = parseProp(toks, i)
  if toks.peek(i).kind != tkEof:
    raise newException(ValueError, "nim_to_lean: invalid spec payload '" &
      payload & "' - trailing input at token '" & toks.peek(i).text & "'")

proc specRefs*(e: SpecExpr): seq[string] =
  ## Names referenced by the payload: bare binders (`tid`, `n`) and applied buffers, deduplicated in first-occurrence order
  case e.kind
  of seIdent:
    if e.name notin result: result.add e.name
  of seApp:
    if e.fn notin result: result.add e.fn
  of seUint:
    discard
  of seBin:
    result = specRefs(e.left)
    for r in specRefs(e.right):
      if r notin result: result.add r

proc renderSpec*(e: SpecExpr, nameMap: Table[string, string]): string =
  ## Renders the payload as Lean text over the model's names.
  ##
  ## `nameMap` maps every kernel buffer parameter to its Lean name in the emitted module.
  ## The written buffer maps to the post-state application `addBody tid n a b out` of the per-thread body.
  ##
  ## - bare binders render as themselves (`tid`, `n`)
  ## - a buffer read renders as `<mapped name> tid`
  ## - everything else is refused loudly: neither a bare buffer name nor an unknown identifier is a Lean Prop over the model
  case e.kind
  of seIdent:
    if e.name in ["tid", "n"]:
      return e.name
    if e.name in nameMap:
      raise newException(ValueError, "nim_to_lean: invalid spec payload - '" &
        e.name & "' must be applied to `tid`")
    raise newException(ValueError, "nim_to_lean: invalid spec payload - '" &
      e.name & "' is neither `tid`, `n` nor a kernel buffer parameter")
  of seApp:
    if e.fn notin nameMap:
      raise newException(ValueError, "nim_to_lean: invalid spec payload - '" &
        e.fn & " tid' is not a read of a kernel buffer parameter")
    result = nameMap[e.fn] & " tid"
  of seUint:
    result = e.value
  of seBin:
    result = renderSpec(e.left, nameMap) & " " & e.op & " " &
      renderSpec(e.right, nameMap)
