# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Subset parser: Tile IR text to the tattletile object model.
## Tokenizer plus recursive descent over exactly the constructs the fixtures
## use (per-construct grammar citations in `fixtures/GRAMMAR-EVIDENCE.md`).
## Anything outside the subset refuses loudly through SubsetError.
##
## Every refusal names:
## - the construct (`make_strided_view`, `f16`, ...)
## - the kernel where the construct appears
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_tile_ir_parse.nim

import std/os
import std/strutils

import ./tile_ir

const PunctChars = {'{', '}', '(', ')', '<', '>', '=', ',', ':', '[', ']', '?', '*', '+'}
  ## Single-character tokens. `*` and `+` appear in official ops outside
  ## the subset: the tokenizer recognizes them, the parser refuses by name.

proc stripLeading(s: string, c: char): string =
  var i = 0
  while i < s.len and s[i] == c:
    inc i
  s.substr(i)

proc isIntToken(tok: string): bool =
  ## True for the number tokens the tokenizer produces without a `.`:
  ## an optional `-` then one or more digits.
  var i = 0
  if i < tok.len and tok[i] == '-':
    inc i
  if i >= tok.len:
    return false
  while i < tok.len and tok[i] in {'0' .. '9'}:
    inc i
  i == tok.len

proc isIdentStart(c: char): bool =
  c in {'a' .. 'z', 'A' .. 'Z', '_'}

proc isIdentChar(c: char): bool =
  c in {'a' .. 'z', 'A' .. 'Z', '_', '0' .. '9', '.'}

proc isDigit(c: char): bool =
  c in {'0' .. '9'}

proc tokenizeError(text: string, i: int): ref SubsetError =
  subsetError("cannot tokenize Tile IR at: '" &
    text.substr(i, min(i + 39, text.len - 1)) & "'")

proc tokenize*(text: string): seq[string] =
  ## Splits Tile IR text into tokens, dropping whitespace and `//` comments.
  ## Token forms: `@name`, `%ssa`, numbers (`-1`, `3.0`, `3.000000e+00`),
  ## `->`, quoted strings (`"name"`), the `x` dimension separator, keywords,
  ## and `PunctChars`.
  ##
  ## Raises:
  ## - SubsetError with a context excerpt on any other character
  var i = 0
  while i < text.len:
    let c = text[i]
    if c in {' ', '\t', '\n', '\r'}:
      inc i
    elif c == '/' and i + 1 < text.len and text[i + 1] == '/':
      while i < text.len and text[i] != '\n':
        inc i
    elif c == '"':
      # quoted attribute value, kept as one token including the quotes
      var j = i + 1
      while j < text.len and text[j] != '"':
        inc j
      if j >= text.len:
        raise tokenizeError(text, i)
      result.add(text.substr(i, j))
      i = j + 1
    elif c == '@':
      if i + 1 >= text.len or not isIdentStart(text[i + 1]):
        raise tokenizeError(text, i)
      var j = i + 1
      while j < text.len and isIdentChar(text[j]):
        inc j
      result.add(text.substr(i, j - 1))
      i = j
    elif c == '%':
      if i + 1 >= text.len or not isIdentChar(text[i + 1]):
        raise tokenizeError(text, i)
      var j = i + 1
      while j < text.len and isIdentChar(text[j]):
        inc j
      result.add(text.substr(i, j - 1))
      i = j
    elif c == '-' and i + 1 < text.len and text[i + 1] == '>':
      result.add("->")
      inc i, 2
    elif isDigit(c) or (c == '-' and i + 1 < text.len and isDigit(text[i + 1])):
      var j = i + 1
      while j < text.len and isDigit(text[j]):
        inc j
      if j < text.len and text[j] == '.':
        inc j
        while j < text.len and isDigit(text[j]):
          inc j
        if j < text.len and text[j] in {'e', 'E'}:
          inc j
          if j < text.len and text[j] in {'+', '-'}:
            inc j
          while j < text.len and isDigit(text[j]):
            inc j
      result.add(text.substr(i, j - 1))
      i = j
    elif c == 'x':
      # `x` separates dims only when a dim or an element type comes next
      # (`16x16`, `?xf32`). Otherwise `x` opens a keyword (`xfoo`) whose
      # element name the parser refuses.
      let looksLikeSep = i + 1 < text.len and
        (isDigit(text[i + 1]) or text[i + 1] == '?' or
         text.substr(i + 1, i + 3) == "f32" or text.substr(i + 1, i + 3) == "i32")
      if looksLikeSep:
        result.add("x")
        inc i
      else:
        var j = i + 1
        while j < text.len and isIdentChar(text[j]):
          inc j
        result.add(text.substr(i, j - 1))
        i = j
    elif isIdentStart(c):
      var j = i + 1
      while j < text.len and isIdentChar(text[j]):
        inc j
      result.add(text.substr(i, j - 1))
      i = j
    elif c in PunctChars:
      result.add($c)
      inc i
    else:
      raise tokenizeError(text, i)

type
  SubsetParser = object
    ## Recursive-descent parser state over the token stream of one module.
    toks: seq[string]
    pos: int
    kernel: string  ## innermost entry name, for refusal messages
    inFor: bool     ## inside a for region: `continue` closes it, the parser refuses `yield`/`return`

proc peek(p: SubsetParser, ahead = 0): string =
  let i = p.pos + ahead
  if i < p.toks.len:
    p.toks[i]
  else:
    ""

proc next(p: var SubsetParser): string =
  let tok = peek(p)
  if tok == "":
    raise subsetError("kernel '" & p.kernel & "': unexpected end of Tile IR")
  inc p.pos
  tok

proc expect(p: var SubsetParser, tok: string): string =
  let got = next(p)
  if got != tok:
    raise subsetError("kernel '" & p.kernel & "': expected '" & tok &
      "', got '" & got & "'")
  got

proc refuse(p: SubsetParser, construct: string, detail = ""): ref SubsetError =
  let suffix = if detail != "":
    ": " & detail
  else:
    ""
  subsetError("kernel '" & p.kernel & "': construct '" & construct &
    "' is outside the supported tattletile subset" & suffix)

proc refuseArity(p: SubsetParser, construct: string, got: int, expected: string): ref SubsetError =
  p.refuse(construct, "expects " & expected & ", got " & $got)

proc parseScalarElem(p: SubsetParser, tok: string): ScalarElem =
  ## Converts an element type token to a ScalarElem.
  ##
  ## Raises:
  ## - SubsetError naming the unsupported element type otherwise
  case tok
  of "i32":
    eI32
  of "f32":
    eF32
  else:
    raise p.refuse("element type '" & tok & "'", "supported element types are f32 and i32")

proc parseType(p: var SubsetParser): TileIrType
proc parseOp(p: var SubsetParser): Op

proc parseDimsAndElem(p: var SubsetParser): tuple[shape: seq[int], elem: ScalarElem] =
  ## Parses `16x16xf32`, `?xf32` or a 0-d `f32`. Dims are joined by
  ## `x`, the element type comes last.
  ##
  ## Example:
  ##   `tile<8xf32>`  ->  shape @[8], elem eF32
  ##   `tile<?xf32>`  ->  shape @[DynamicDim], elem eF32
  ##   `tile<f32>`    ->  shape @[], elem eF32
  var shape: seq[int] = @[]
  while true:
    let tok = next(p)
    if tok == "f32" or tok == "i32":
      return (shape, parseScalarElem(p, tok))
    if tok == "?":
      shape.add(DynamicDim)
    elif tok.len > 1 and tok[0] == 'x':
      # keyword token like `xbf16` after a dim: the element type name
      return (shape, parseScalarElem(p, tok.substr(1)))
    elif isIntToken(tok):
      shape.add(parseInt(tok))
    else:
      raise p.refuse("dimension token '" & tok & "'", "expected an int or element type")
    let sep = next(p)
    if sep == "x":
      continue
    if sep.len > 1 and sep[0] == 'x':
      return (shape, parseScalarElem(p, sep.substr(1)))
    raise p.refuse("shape separator '" & sep & "'", "expected 'x' between dimensions")

proc parseDimList(p: var SubsetParser): seq[int] =
  ## Parses the `?`/int values of a tensor_view strides list, joined by `,`.
  ## The caller consumes the brackets.
  ##
  ## Example:
  ##   `strides = [16, 1]`  ->  @[16, 1]
  ##   `strides = [?]`      ->  @[DynamicDim]
  while true:
    let tok = next(p)
    if tok == "?":
      result.add(DynamicDim)
    elif isIntToken(tok):
      result.add(parseInt(tok))
    else:
      raise p.refuse("stride token '" & tok & "'", "expected an int or ?")
    if peek(p) == ",":
      discard next(p)
    else:
      return

proc parseValueList(p: var SubsetParser): seq[ConstScalar] =
  ## Parses the values of an explicit constant list joined by `,`.
  ## One int or float per tile element. Unlike strides, `?` is not a value.
  ##
  ## Example:
  ##   `<f32: [1.0, 2.0]>`  ->  @[floatScalar(1.0), floatScalar(2.0)]
  while true:
    let tok = next(p)
    if tok.contains('.'):
      result.add(floatScalar(parseFloat(tok)))
    elif isIntToken(tok):
      result.add(intScalar(parseInt(tok)))
    else:
      raise p.refuse("constant list value token '" & tok & "'", "expected an int or float")
    if peek(p) == ",":
      discard next(p)
    else:
      return

proc parseTensorViewBody(p: var SubsetParser): TileIrType =
  ## Parses the inside of `tensor_view<...>`: dims and element first.
  ## The strides list comes next. The caller consumes the angle brackets.
  let (shape, elem) = parseDimsAndElem(p)
  discard p.expect(",")
  discard p.expect("strides")
  discard p.expect("=")
  discard p.expect("[")
  let strides = parseDimList(p)
  discard p.expect("]")
  TileIrType(kind: tkTensorView, viewShape: shape, viewElem: elem,
             viewStrides: strides)

proc parseType(p: var SubsetParser): TileIrType =
  ## Parses one type of the subset.
  ##
  ## Raises:
  ## - SubsetError naming the type construct for anything else
  let tok = next(p)
  case tok
  of "token":
    TokenIrType
  of "tile":
    discard p.expect("<")
    if peek(p) == "ptr":
      discard next(p)
      discard p.expect("<")
      let pointee = parseScalarElem(p, next(p))
      discard p.expect(">")
      discard p.expect(">")
      ptrTile(pointee)
    else:
      let (shape, elem) = parseDimsAndElem(p)
      discard p.expect(">")
      TileIrType(kind: tkTile, tileShape: shape,
                 tileElem: TypeElem(kind: teScalar, scalar: elem))
  of "tensor_view":
    discard p.expect("<")
    let ty = parseTensorViewBody(p)
    discard p.expect(">")
    ty
  of "partition_view":
    discard p.expect("<")
    discard p.expect("tile")
    discard p.expect("=")
    discard p.expect("(")
    var tileShape: seq[int] = @[]
    while true:
      let dimTok = next(p)
      if not isIntToken(dimTok):
        raise p.refuse("partition_view with dynamic tile dims", "partition tile dims are statically shaped, got '" & dimTok & "'")
      tileShape.add(parseInt(dimTok))
      if peek(p) == "x":
        discard next(p)
      else:
        break
    discard p.expect(")")
    var pvPadding = false
    if peek(p) == "," and p.peek(1) == "padding_value":
      discard next(p)
      discard next(p)
      discard p.expect("=")
      let pad = next(p)
      if pad != "zero":
        raise p.refuse("padding_value '" & pad & "'", "the subset supports padding_value = zero only")
      pvPadding = true
    discard p.expect(",")
    discard p.expect("tensor_view")
    discard p.expect("<")
    let (shape, elem) = parseDimsAndElem(p)
    discard p.expect(",")
    discard p.expect("strides")
    discard p.expect("=")
    discard p.expect("[")
    let strides = parseDimList(p)
    discard p.expect("]")
    discard p.expect(">")
    discard p.expect(">")
    TileIrType(kind: tkPartitionView, pvTile: tileShape, pvShape: shape,
               pvElem: elem, pvStrides: strides, pvPadding: pvPadding)
  else:
    raise p.refuse(tok, "unsupported type")

proc parseDynEntries(p: var SubsetParser): seq[ShapeEntry] =
  ## Parses the shape/stride list entries of `make_tensor_view`.
  ## Entries are ints or SSA names, joined by `,`.
  ## The caller consumes the brackets.
  ##
  ## Example:
  ##   `shape = [%shape_a]`    ->  @[ssaEntry("shape_a")]
  ##   `shape = [16, 16]`      ->  @[staticEntry(16), staticEntry(16)]
  while true:
    let tok = next(p)
    if tok.startsWith("%"):
      result.add(ssaEntry(stripLeading(tok, '%')))
    elif isIntToken(tok):
      result.add(staticEntry(parseInt(tok)))
    else:
      raise p.refuse("shape/stride entry '" & tok & "'", "expected an int or an SSA name")
    if peek(p) == ",":
      discard next(p)
    else:
      return

proc checkArity(p: SubsetParser, construct: string, results: seq[string], expected: int) =
  ## Refuses ops with a result count the subset grammar does not allow.
  ## The official ops have fixed arities, so a wrong count is malformed text.
  if results.len != expected:
    let plural = if expected != 1:
      "s"
    else:
      ""
    raise refuseArity(p, construct, results.len, $expected & " result" & plural)

proc parseConstant(p: var SubsetParser, results: seq[string]): Op =
  ## Parses `%r = constant <elem: v | [v, ...]> : tile<...>`.
  ## A scalar value splats into a shaped tile, the CudaTile.cpp splat rule
  ## (see `fixtures/GRAMMAR-EVIDENCE.md`). An explicit list must fill
  ## the tile shape exactly and needs static dims.
  checkArity(p, "constant", results, 1)
  discard p.expect("<")
  let elem = parseScalarElem(p, next(p))
  discard p.expect(":")
  var payload: ConstPayload
  if peek(p) == "[":
    discard p.expect("[")
    let values = parseValueList(p)
    discard p.expect("]")
    payload = ConstPayload(isList: true, values: values)
  else:
    let tok = next(p)
    if not isIntToken(tok) and not tok.contains('.'):
      raise p.refuse("constant value token '" & tok & "'", "expected an int or float")
    let scalar = if tok.contains('.'):
      floatScalar(parseFloat(tok))
    else:
      intScalar(parseInt(tok))
    payload = ConstPayload(isList: false, scalar: scalar)
  discard p.expect(">")
  discard p.expect(":")
  let ty = parseType(p)
  if ty.kind != tkTile:
    raise p.refuse("constant", "constants produce tile values, got a " & $ty.kind & " type")
  if payload.isList:
    for d in ty.tileShape:
      if isDynamic(d):
        raise p.refuse("constant", "an explicit value list cannot fill a tile with dynamic dims, use a scalar splat")
    var count = 1
    for d in ty.tileShape:
      count *= d
    if payload.values.len != count:
      raise p.refuse("constant", $payload.values.len & " values do not fill tile shape (" & ty.tileShape.join("x") & ")")
  Op(results: results, resultTypes: @[ty], kind: okConstant,
     constValue: payload, constElem: elem)

proc parseMakeTensorView(p: var SubsetParser, results: seq[string]): Op =
  ## Parses `%r = make_tensor_view %base, shape = [...], strides = [...] : ...`.
  ## Dynamic shape or strides entries force the `tile<i32> -> ` operand-type
  ## prefix (MakeTensorViewOp mlirExamples, dynamic case).
  checkArity(p, "make_tensor_view", results, 1)
  let base = stripLeading(next(p), '%')
  discard p.expect(",")
  discard p.expect("shape")
  discard p.expect("=")
  discard p.expect("[")
  let shape = parseDynEntries(p)
  discard p.expect("]")
  discard p.expect(",")
  discard p.expect("strides")
  discard p.expect("=")
  discard p.expect("[")
  let strides = parseDynEntries(p)
  discard p.expect("]")
  discard p.expect(":")
  if peek(p) == "tile":
    discard p.expect("tile")
    discard p.expect("<")
    discard p.expect("i32")
    discard p.expect(">")
    discard p.expect("->")
  let ty = parseType(p)
  if ty.kind != tkTensorView:
    raise p.refuse("make_tensor_view", "result must be a tensor_view")
  Op(results: results, operands: @[base], resultTypes: @[ty],
     kind: okMakeTensorView, dynShape: shape, dynStrides: strides)

proc parseViewTko(p: var SubsetParser, construct: string, isLoad: bool, results: seq[string]): Op =
  ## Parses `load_view_tko` / `store_view_tko`.
  ## Tokens are results with an optional `token =` input:
  ##
  ##   %tile, %tok = load_view_tko weak %view[%i, ...] token = %tok : vty, ity -> tilety, token
  ##   %tok = store_view_tko weak %value, %view[%i, ...] token = %tok : vty, wty, ity -> token
  let expectedResults = if isLoad:
    2
  else:
    1
  let detail = if isLoad:
    "2 results (the loaded tile and a token)"
  else:
    "1 result (the token)"
  if results.len != expectedResults:
    raise refuseArity(p, construct, results.len, detail)
  discard p.expect("weak")
  var operands = @[stripLeading(next(p), '%')]
  if not isLoad:
    discard p.expect(",")
    operands.add(stripLeading(next(p), '%'))
  discard p.expect("[")
  var indices: seq[string] = @[]
  while peek(p) != "]":
    indices.add(stripLeading(next(p), '%'))
    if peek(p) == ",":
      discard next(p)
  discard p.expect("]")
  var token = ""
  if peek(p) == "token":
    discard next(p)
    discard p.expect("=")
    token = stripLeading(next(p), '%')
  discard p.expect(":")
  var operandTypes: seq[TileIrType] = @[]
  for i in 0 ..< operands.len + 1:
    operandTypes.add(parseType(p))
    if i + 1 < operands.len + 1:
      discard p.expect(",")
  discard p.expect("->")
  var resultTypes: seq[TileIrType] = @[]
  for i in 0 ..< expectedResults:
    resultTypes.add(parseType(p))
    if i + 1 < expectedResults:
      discard p.expect(",")
  let opKind = if isLoad:
    okLoadViewTko
  else:
    okStoreViewTko
  var op = Op(results: results, operands: operands, operandTypes: operandTypes,
              resultTypes: resultTypes, kind: opKind)
  op.indices = indices
  op.token = token
  op

proc parseBinary(p: var SubsetParser, construct: string, isAdd: bool, results: seq[string]): Op =
  ## Parses `%r = addf|mulf %a, %b : tile<...>`.
  checkArity(p, construct, results, 1)
  let a = stripLeading(next(p), '%')
  discard p.expect(",")
  let b = stripLeading(next(p), '%')
  discard p.expect(":")
  let ty = parseType(p)
  let opKind = if isAdd:
    okAddf
  else:
    okMulf
  Op(results: results, operands: @[a, b], operandTypes: @[ty, ty],
     resultTypes: @[ty], kind: opKind)

proc parseMmaf(p: var SubsetParser, results: seq[string]): Op =
  ## Parses `%r = mmaf %a, %b, %acc : tileA, tileB, tileAcc`.
  ## MmaFOp: C = A @ B + C with the accumulator as third operand.
  checkArity(p, "mmaf", results, 1)
  var operands = @[stripLeading(next(p), '%')]
  for _ in 0 ..< 2:
    discard p.expect(",")
    operands.add(stripLeading(next(p), '%'))
  discard p.expect(":")
  var types: seq[TileIrType] = @[]
  for i in 0 ..< 3:
    types.add(parseType(p))
    if i + 1 < 3:
      discard p.expect(",")
  Op(results: results, operands: operands, operandTypes: types,
     resultTypes: types, kind: okMmaf)

proc parseAssumeInt(p: var SubsetParser, what: string): int =
  ## Parses one int-valued `assume` attribute slot.
  ##
  ## Raises:
  ## - SubsetError when the token is not an int
  let tok = next(p)
  if not isIntToken(tok):
    raise p.refuse(what & " '" & tok & "'", "expected an int")
  parseInt(tok)

proc parseAssume(p: var SubsetParser, results: seq[string]): Op =
  ## Parses `%r = assume bounded<lo, hi>, %v : ty` or
  ## `%r = assume div_by<n>, %v : ty`. The result has the value's type.
  ##
  ## Example:
  ##   `%17 = assume bounded<0, ?>, %1 : tile<i32>`  ->
  ##     okAssume, akBounded, lo 0, hi DynamicDim
  checkArity(p, "assume", results, 1)
  let predicate = next(p)
  var kind: AssumeKind
  var lo = 0
  var hi = 0
  var divN = 0
  case predicate
  of "bounded":
    kind = akBounded
    discard p.expect("<")
    lo = parseAssumeInt(p, "bounded lower bound")
    discard p.expect(",")
    let hiTok = next(p)
    if hiTok == "?":
      hi = DynamicDim
    elif isIntToken(hiTok):
      hi = parseInt(hiTok)
    else:
      raise p.refuse("bounded upper bound '" & hiTok & "'", "expected an int or ?")
    discard p.expect(">")
  of "div_by":
    kind = akDivBy
    discard p.expect("<")
    divN = parseAssumeInt(p, "div_by divisor")
    discard p.expect(">")
  else:
    raise p.refuse("assume predicate '" & predicate & "'", "supported predicates are bounded and div_by")
  discard p.expect(",")
  let value = stripLeading(next(p), '%')
  discard p.expect(":")
  let ty = parseType(p)
  Op(results: results, operands: @[value], operandTypes: @[ty],
     resultTypes: @[ty], kind: okAssume, assumeKind: kind,
     assumeLo: lo, assumeHi: hi, assumeDiv: divN)

proc parseReshapeBroadcast(p: var SubsetParser, construct: string, results: seq[string]): Op =
  ## Parses `%r = reshape|broadcast %v : tyIn -> tyOut`, one operand and one result.
  ## A scalar enters a tile like this: reshape to `tile<1x1x...>`, then broadcast.
  checkArity(p, construct, results, 1)
  let value = stripLeading(next(p), '%')
  discard p.expect(":")
  let inTy = parseType(p)
  discard p.expect("->")
  let outTy = parseType(p)
  let kind = if construct == "reshape":
    okReshape
  else:
    okBroadcast
  Op(results: results, operands: @[value], operandTypes: @[inTy],
     resultTypes: @[outTy], kind: kind)

proc parseFor(p: var SubsetParser, results: seq[string]): Op =
  ## Parses the for region op:
  ##
  ##   %r = for %i in (%lb to %ub, step %s) : ty
  ##     iter_values(%c = %init, ...) -> (tyC, ...) { ops, continue ... }
  ##
  ## `iter_values` is absent when the loop has no init values. A `continue` with one value per iter name closes the region.
  ##
  ## Raises:
  ## - SubsetError for a region without a continue terminator
  checkArity(p, "for", results, 1)
  let loopVar = stripLeading(next(p), '%')
  discard p.expect("in")
  discard p.expect("(")
  let lb = stripLeading(next(p), '%')
  discard p.expect("to")
  let ub = stripLeading(next(p), '%')
  discard p.expect(",")
  discard p.expect("step")
  let step = stripLeading(next(p), '%')
  discard p.expect(")")
  discard p.expect(":")
  let ivTy = parseType(p)
  var iterNames: seq[string] = @[]
  var iterInits: seq[string] = @[]
  var iterTypes: seq[TileIrType] = @[]
  if peek(p) == "iter_values":
    discard next(p)
    discard p.expect("(")
    while peek(p) != ")":
      iterNames.add(stripLeading(next(p), '%'))
      discard p.expect("=")
      iterInits.add(stripLeading(next(p), '%'))
      if peek(p) == ",":
        discard next(p)
    discard p.expect(")")
    discard p.expect("->")
    discard p.expect("(")
    if peek(p) != ")":
      iterTypes.add(parseType(p))
      while peek(p) == ",":
        discard next(p)
        iterTypes.add(parseType(p))
    discard p.expect(")")
    if iterNames.len != iterTypes.len:
      raise refuseArity(p, "for", iterTypes.len, "one carried type per iter_values entry, got " & $iterNames.len)
  discard p.expect("{")
  let savedInFor = p.inFor
  p.inFor = true
  var body: seq[Op] = @[]
  while peek(p) != "}":
    body.add(parseOp(p))
  discard p.expect("}")
  p.inFor = savedInFor
  if body.len == 0 or body[^1].kind != okContinue:
    raise p.refuse("for region without continue", "a for region closes with a continue carrying the iter values")
  Op(results: results, operands: @[lb, ub, step], resultTypes: iterTypes,
     kind: okFor, loopVar: loopVar, loopVarType: ivTy,
     iterNames: iterNames, iterInits: iterInits, body: body)

proc parseOp(p: var SubsetParser): Op =
  ## Parses one op of the subset.
  ##
  ## Raises:
  ## - SubsetError naming the construct and the kernel for anything else
  case peek(p)
  of "yield":
    discard next(p)
    if p.inFor:
      raise p.refuse("yield", "a for region closes with continue")
    return Op(kind: okYield)
  of "continue":
    discard next(p)
    if not p.inFor:
      raise p.refuse("continue", "continue closes a for region, found outside one")
    var operands: seq[string] = @[]
    if peek(p).startsWith("%"):
      operands.add(stripLeading(next(p), '%'))
      while peek(p) == ",":
        discard next(p)
        operands.add(stripLeading(next(p), '%'))
    var types: seq[TileIrType] = @[]
    if operands.len > 0:
      discard p.expect(":")
      types.add(parseType(p))
      while peek(p) == ",":
        discard next(p)
        types.add(parseType(p))
      if types.len != operands.len:
        raise refuseArity(p, "continue", types.len, "one type per carried value, got " & $operands.len & " values")
    return Op(results: @[], operands: operands, operandTypes: types,
              resultTypes: types, kind: okContinue)
  of "return":
    discard next(p)
    if p.inFor:
      raise p.refuse("return", "return closes an entry, found inside a for region")
    if peek(p).startsWith("%"):
      raise p.refuse("return with operands", "the subset supports a bare return only")
    return Op(kind: okReturn)
  else:
    discard
  var results: seq[string] = @[stripLeading(next(p), '%')]
  while peek(p) == ",":
    discard next(p)
    results.add(stripLeading(next(p), '%'))
  discard p.expect("=")
  let construct = next(p)
  case construct
  of "constant":
    parseConstant(p, results)
  of "make_token":
    checkArity(p, construct, results, 1)
    discard p.expect(":")
    discard p.expect("token")
    Op(results: results, resultTypes: @[TokenIrType], kind: okMakeToken)
  of "make_tensor_view":
    parseMakeTensorView(p, results)
  of "make_partition_view":
    checkArity(p, construct, results, 1)
    let operand = stripLeading(next(p), '%')
    discard p.expect(":")
    let ty = parseType(p)
    if ty.kind != tkPartitionView:
      raise p.refuse(construct, "result must be a partition_view")
    Op(results: results, operands: @[operand], resultTypes: @[ty],
       kind: okMakePartitionView)
  of "get_tile_block_id":
    if results.len != 3:
      raise refuseArity(p, construct, results.len, "3 results (the x/y/z block coordinates)")
    discard p.expect(":")
    discard p.expect("tile")
    discard p.expect("<")
    discard p.expect("i32")
    discard p.expect(">")
    Op(results: results,
       resultTypes: @[scalarTile(eI32), scalarTile(eI32), scalarTile(eI32)],
       kind: okGetTileBlockId)
  of "load_view_tko":
    parseViewTko(p, construct, isLoad = true, results)
  of "store_view_tko":
    parseViewTko(p, construct, isLoad = false, results)
  of "addf":
    parseBinary(p, construct, isAdd = true, results)
  of "mulf":
    parseBinary(p, construct, isAdd = false, results)
  of "mmaf":
    parseMmaf(p, results)
  of "assume":
    parseAssume(p, results)
  of "reshape":
    parseReshapeBroadcast(p, "reshape", results)
  of "broadcast":
    parseReshapeBroadcast(p, "broadcast", results)
  of "for":
    parseFor(p, results)
  else:
    raise p.refuse(construct, "unsupported operation")

proc parseEntry(p: var SubsetParser): Entry =
  ## Parses `entry @name(%arg: type, ...) { ops }` and attributes refusals
  ## inside the body to the entry name.
  discard p.expect("entry")
  p.kernel = stripLeading(next(p), '@')
  discard p.expect("(")
  var args: seq[EntryArg] = @[]
  while peek(p) != ")":
    let argName = stripLeading(next(p), '%')
    discard p.expect(":")
    args.add((argName, parseType(p)))
    if peek(p) == ",":
      discard next(p)
  discard p.expect(")")
  var diName = ""
  while peek(p) != "{":
    let attr = next(p)
    if attr != "di_name":
      raise p.refuse("entry attribute '" & attr & "'", "the subset supports the di_name attribute only")
    discard p.expect("=")
    let strTok = next(p)
    if strTok.len < 2 or strTok[0] != '"' or strTok[^1] != '"':
      raise p.refuse("entry attribute value '" & strTok & "'", "di_name takes a quoted string")
    diName = strTok[1 ..< strTok.len - 1]
  discard p.expect("{")
  result = Entry(name: p.kernel, args: args, diName: diName)
  while peek(p) != "}":
    result.ops.add(parseOp(p))
  discard p.expect("}")
  p.kernel = "<module>"

proc parseModule(p: var SubsetParser): TileIrModule =
  ## Parses `cuda_tile.module @name { entry ... }`.
  discard p.expect("cuda_tile.module")
  p.kernel = "<module>"
  let nameTok = next(p)
  discard p.expect("{")
  result = TileIrModule(name: stripLeading(nameTok, '@'))
  while peek(p) != "}":
    result.entries.add(parseEntry(p))
  discard p.expect("}")

proc parseTileIr*(text: string): TileIrModule =
  ## Parses Tile IR text into a TileIrModule. Runs in the Nim VM and at runtime.
  ##
  ## Raises:
  ## - SubsetError naming the construct and the kernel for anything outside
  ##   the supported subset
  var p = SubsetParser(toks: tokenize(text), kernel: "<module>")
  result = parseModule(p)
  if p.pos != p.toks.len:
    raise p.refuse("trailing text '" & peek(p) & "'", "expected end of module")

template parseTileIrFile*(path: static string): TileIrModule =
  ## Compile-time ingestion of a Tile IR file through `staticRead`.
  ## Nim VM only. The path resolves relative to the instantiating module.
  ##
  ## Examples:
  ##   const m = parseTileIrFile("fixtures/vec_add_axpy.tileir")
  parseTileIr(staticRead(
      if path.isAbsolute: path
      else: instantiationInfo(-1, true).filename.parentDir / path))
