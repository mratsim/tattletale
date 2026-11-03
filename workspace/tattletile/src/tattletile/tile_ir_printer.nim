# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile IR printer: the object model back to Tile IR text.
## Emits exactly the fixture form, so `parse o print` reproduces the parsed
## module (round-trip fixed point, checked by the parse test).
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_tile_ir_parse.nim

import std/math
import std/sequtils
import std/strutils

import ./tile_ir

proc elemToText(elem: TypeElem): string =
  ## Element slot of a `tile<...>` type: `f32` or `ptr<f32>`.
  case elem.kind
  of teScalar:
    $elem.scalar
  of tePointer:
    "ptr<" & $elem.pointee & ">"

proc dimsToText(dims: seq[int]): string =
  ## Dims joined by `x`, `?` for a dynamic dim.
  ##
  ## Examples:
  ##   @[8, 8]          ->  "8x8"
  ##   @[DynamicDim]    ->  "?"
  for i, d in dims:
    if i > 0:
      result.add("x")
    if isDynamic(d):
      result.add("?")
    else:
      result.add($d)

proc strideListToText(strides: seq[int]): string =
  ## Strides joined by `,` as written inside `strides=[...]`.
  strides.mapIt(if isDynamic(it): "?" else: $it).join(",")

proc typeToText*(ty: TileIrType): string =
  ## Prints a type in the shorthand Tile IR form (no `!cuda_tile.` prefix).
  ##
  ## Examples:
  ##   scalarTile(eI32)                    ->  "tile<i32>"
  ##   ptrTile(eF32)                       ->  "tile<ptr<f32>>"
  ##   a dynamic 1-d tensor view of f32    ->  "tensor_view<?xf32, strides=[?]>"
  case ty.kind
  of tkToken:
    "token"
  of tkTile:
    let dims = dimsToText(ty.tileShape)
    let prefix = if dims != "":
      dims & "x"
    else:
      ""
    "tile<" & prefix & elemToText(ty.tileElem) & ">"
  of tkTensorView:
    "tensor_view<" & dimsToText(ty.viewShape) & "x" & $ty.viewElem &
      ", strides=[" & strideListToText(ty.viewStrides) & "]>"
  of tkPartitionView:
    let padding = if ty.pvPadding:
      ", padding_value = zero"
    else:
      ""
    "partition_view<tile=(" & dimsToText(ty.pvTile) & ")" & padding &
      ", tensor_view<" & dimsToText(ty.pvShape) & "x" & $ty.pvElem &
      ", strides=[" & strideListToText(ty.pvStrides) & "]>>"

proc numToText(v: ConstScalar): string =
  ## One constant element. Integral floats print in the fixture form
  ## (`3.0` becomes `3.000000e+00`), other floats in their shortest
  ## round-trip form, so printing never loses a parsed value.
  if v.isInt:
    $v.intVal
  elif v.floatVal == floor(v.floatVal):
    formatFloat(v.floatVal, ffDecimal, 6) & "e+00"
  else:
    $v.floatVal

proc constToText*(payload: ConstPayload, elem: ScalarElem): string =
  ## Prints the elements literal of a constant op.
  ##
  ## Examples:
  ##   scalar splat 3.0 of f32   ->  "<f32: 3.000000e+00>"
  ##   list [1, 2] of i32        ->  "<i32: [1, 2]>"
  if payload.isList:
    var parts: seq[string] = @[]
    for v in payload.values:
      parts.add(numToText(v))
    "<" & $elem & ": [" & parts.join(", ") & "]>"
  else:
    "<" & $elem & ": " & numToText(payload.scalar) & ">"

proc dynListToText(entries: seq[ShapeEntry]): string =
  ## A `make_tensor_view` shape or strides list: `%name` for SSA entries,
  ## the int for static entries.
  ##
  ## Examples:
  ##   @[ssaEntry("shape_a")]                 ->  "[%shape_a]"
  ##   @[staticEntry(16), staticEntry(1)]     ->  "[16, 1]"
  var parts: seq[string] = @[]
  for e in entries:
    if e.isSsa:
      parts.add("%" & e.name)
    else:
      parts.add($e.value)
  "[" & parts.join(", ") & "]"

proc joinPrefixed(names: seq[string], prefix: string): string =
  ## Names joined by `, `, each carrying `prefix` (`%` for SSA names).
  for i, name in names:
    if i > 0:
      result.add(", ")
    result.add(prefix & name)

proc tokenInputToText(token: string): string =
  ## Optional `token = %tok` input of load/store ops, "" when absent.
  if token != "":
    " token = %" & token
  else:
    ""

proc opToText*(op: Op): string =
  ## Prints one op of the supported subset in the fixture text form.
  case op.kind
  of okYield:
    "yield"
  of okConstant:
    "%" & op.results[0] & " = constant " & constToText(op.constValue, op.constElem) &
      " : " & typeToText(op.resultTypes[0])
  of okMakeToken:
    "%" & op.results[0] & " = make_token : " & typeToText(op.resultTypes[0])
  of okMakeTensorView:
    var dyn = false
    for e in op.dynShape:
      if e.isSsa:
        dyn = true
    for e in op.dynStrides:
      if e.isSsa:
        dyn = true
    let prefix = if dyn:
      "tile<i32> -> "
    else:
      ""
    "%" & op.results[0] & " = make_tensor_view %" & op.operands[0] &
      ", shape = " & dynListToText(op.dynShape) &
      ", strides = " & dynListToText(op.dynStrides) &
      " : " & prefix & typeToText(op.resultTypes[0])
  of okMakePartitionView:
    "%" & op.results[0] & " = make_partition_view %" & op.operands[0] &
      " : " & typeToText(op.resultTypes[0])
  of okGetTileBlockId:
    joinPrefixed(op.results, "%") & " = get_tile_block_id : tile<i32>"
  of okLoadViewTko:
    "%" & op.results[0] & ", %" & op.results[1] & " = load_view_tko weak %" &
      op.operands[0] & "[" & joinPrefixed(op.indices, "%") & "]" &
      tokenInputToText(op.token) & " : " & typeToText(op.operandTypes[0]) &
      ", " & typeToText(op.operandTypes[1]) & " -> " &
      typeToText(op.resultTypes[0]) & ", token"
  of okStoreViewTko:
    "%" & op.results[0] & " = store_view_tko weak %" & op.operands[0] & ", %" &
      op.operands[1] & "[" & joinPrefixed(op.indices, "%") & "]" &
      tokenInputToText(op.token) & " : " & typeToText(op.operandTypes[0]) &
      ", " & typeToText(op.operandTypes[1]) & ", " &
      typeToText(op.operandTypes[2]) & " -> token"
  of okAddf, okMulf:
    let name = case op.kind
               of okAddf:
                 "addf"
               of okMulf:
                 "mulf"
               else:
                 ""
    "%" & op.results[0] & " = " & name & " %" & op.operands[0] & ", %" &
      op.operands[1] & " : " & typeToText(op.resultTypes[0])
  of okMmaf:
    var types = ""
    for i, ty in op.resultTypes:
      if i > 0:
        types.add(", ")
      types.add(typeToText(ty))
    "%" & op.results[0] & " = mmaf %" & op.operands[0] & ", %" & op.operands[1] &
      ", %" & op.operands[2] & " : " & types
  of okAssume:
    let predicate = case op.assumeKind
                    of akBounded:
                      "bounded<" & $op.assumeLo & ", " &
                        (if isDynamic(op.assumeHi): "?" else: $op.assumeHi) & ">"
                    of akDivBy:
                      "div_by<" & $op.assumeDiv & ">"
    "%" & op.results[0] & " = assume " & predicate & ", %" & op.operands[0] &
      " : " & typeToText(op.resultTypes[0])
  of okReshape, okBroadcast:
    "%" & op.results[0] & " = " & opSpelling(op.kind) & " %" & op.operands[0] &
      " : " & typeToText(op.operandTypes[0]) & " -> " &
      typeToText(op.resultTypes[0])
  of okFor:
    var head = "%" & op.results[0] & " = for %" & op.loopVar & " in (%" &
      op.operands[0] & " to %" & op.operands[1] & ", step %" & op.operands[2] &
      ") : " & typeToText(op.loopVarType)
    if op.iterNames.len > 0:
      var pairs = ""
      var types = ""
      for i, name in op.iterNames:
        if i > 0:
          pairs.add(", ")
          types.add(", ")
        pairs.add("%" & name & " = %" & op.iterInits[i])
        types.add(typeToText(op.resultTypes[i]))
      head.add(" iter_values(" & pairs & ") -> (" & types & ")")
    head.add(" {")
    var lines = @[head]
    for bodyOp in op.body:
      lines.add("      " & opToText(bodyOp))
    lines.add("    }")
    lines.join("\n")
  of okContinue:
    if op.operands.len == 0:
      "continue"
    else:
      var names = ""
      var types = ""
      for i, name in op.operands:
        if i > 0:
          names.add(", ")
          types.add(", ")
        names.add("%" & name)
        types.add(typeToText(op.operandTypes[i]))
      "continue " & names & " : " & types
  of okReturn:
    "return"

proc printTileIr*(m: TileIrModule): string =
  ## Prints a TileIrModule back to Tile IR text in the fixture form.
  ## The output reparses to an equal module (round-trip fixed point).
  result = "cuda_tile.module @" & m.name & " {\n"
  for entry in m.entries:
    var args = ""
    for i, arg in entry.args:
      if i > 0:
        args.add(", ")
      args.add("%" & arg.name & ": " & typeToText(arg.typ))
    let diAttr = if entry.diName != "":
      " di_name = \"" & entry.diName & "\""
    else:
      ""
    result.add("  entry @" & entry.name & "(" & args & ")" & diAttr & " {\n")
    for op in entry.ops:
      result.add("    " & opToText(op) & "\n")
    result.add("  }\n")
  result.add("}\n")
