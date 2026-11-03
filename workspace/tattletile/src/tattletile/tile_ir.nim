# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile IR object model: types, ops, entries, module for the supported subset
## of official NVIDIA Tile IR text. Value objects throughout, usable in the
## Nim VM and at runtime. Constructs outside the subset raise SubsetError.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_tile_ir_parse.nim

type
  SubsetError* = object of ValueError
    ## Refusal raised for a Tile IR construct outside the supported subset.
    ## The message names the construct and the kernel where it appears.

  ScalarElem* = enum
    ## Element types of the supported subset. Enum values carry the Tile IR
    ## text form (`i32`, `f32`), so `$` produces it directly.
    eI32 = "i32"
    eF32 = "f32"

  TypeKind* = enum
    ## Type forms of the supported subset:
    tkToken          ## `token`: ordering witness produced and consumed by tko ops
    tkTile           ## `tile<SxE>`: shaped value, static or dynamic dims
    tkTensorView     ## `tensor_view<SxE, strides=[...]>` over global memory
    tkPartitionView  ## `partition_view<tile=(T...), tensor_view<...>>`

  TypeElemKind* = enum
    ## Element slot of a `tile<...>` type: a scalar, or a pointer
    ## (`tile<ptr<E>>`).
    teScalar
    tePointer

  TypeElem* = object
    case kind*: TypeElemKind
    of teScalar:
      scalar*: ScalarElem
    of tePointer:
      pointee*: ScalarElem

  TileIrType* = object
    ## One Tile IR type of the supported subset.
    ##
    ## Contract:
    ## - shape and stride dims use `DynamicDim` (-1) for a dynamic `?`,
    ##   resolved at launch time
    ## - a `partition_view` tile shape is always static
    ## - the inner view of a `partition_view` is always a tensor view, stored
    ##   inline in the `tkPartitionView` branch
    case kind*: TypeKind
    of tkToken:
      discard
    of tkTile:
      tileShape*: seq[int]
      tileElem*: TypeElem
    of tkTensorView:
      viewShape*: seq[int]
      viewElem*: ScalarElem
      viewStrides*: seq[int]
    of tkPartitionView:
      pvTile*: seq[int]
      pvShape*: seq[int]
      pvElem*: ScalarElem
      pvStrides*: seq[int]
      pvPadding*: bool  ## the type lists `padding_value = zero`

  ConstScalar* = object
    ## One element of a constant value, kept as int or float so the printer
    ## can emit `1` vs `1.0` as written in the source text.
    case isInt*: bool
    of true:
      intVal*: int
    of false:
      floatVal*: float

  ConstPayload* = object
    ## Elements literal of a constant op: a scalar splat, or an explicit
    ## list that fills the result tile.
    case isList*: bool
    of true:
      values*: seq[ConstScalar]
    of false:
      scalar*: ConstScalar

  ShapeEntry* = object
    ## One entry of a `make_tensor_view` shape or strides list: a static int,
    ## or an SSA name resolved at launch time.
    case isSsa*: bool
    of true:
      name*: string
    of false:
      value*: int

  OpKind* = enum
    ## Ops of the supported subset.
    okConstant
    okMakeToken
    okMakeTensorView
    okMakePartitionView
    okGetTileBlockId
    okLoadViewTko
    okStoreViewTko
    okAddf
    okMulf
    okMmaf
    okAssume
    okReshape
    okBroadcast
    okFor
    okContinue
    okReturn
    okYield

  AssumeKind* = enum
    ## Attr form of an `assume` op.
    akBounded   ## `bounded<lo, hi>`: value range, `hi` may be `?`
    akDivBy     ## `div_by<n>`: value divisible by n

  Op* = object
    ## One Tile IR operation of the supported subset.
    ##
    ## `results` are SSA names in result order, `operands` in operand order.
    ## `operandTypes` and `resultTypes` are the parsed types, one per operand
    ## and result. Only the payload fields of the active kind branch are set.
    results*: seq[string]
    operands*: seq[string]
    operandTypes*: seq[TileIrType]
    resultTypes*: seq[TileIrType]
    case kind*: OpKind
    of okConstant:
      constValue*: ConstPayload
      constElem*: ScalarElem
    of okMakeTensorView:
      dynShape*: seq[ShapeEntry]
      dynStrides*: seq[ShapeEntry]
    of okLoadViewTko, okStoreViewTko:
      indices*: seq[string]  ## view index operand names
      token*: string         ## optional `token =` input, "" when absent
    of okAssume:
      assumeKind*: AssumeKind
      assumeLo*: int  ## bounded lower bound, ignored for div_by
      assumeHi*: int  ## bounded upper bound, DynamicDim for `?`, ignored for div_by
      assumeDiv*: int ## div_by divisor, ignored for bounded
    of okFor:
      loopVar*: string         ## induction value name inside the region
      loopVarType*: TileIrType ## type of the induction value
      iterNames*: seq[string]  ## loop-carried names inside the region
      iterInits*: seq[string]  ## SSA init names for the iter values
      body*: seq[Op]           ## region body ops ending in `continue`
    else:
      discard

  EntryArg* = tuple[name: string, typ: TileIrType]

  Entry* = object
    ## Kernel entry point: `entry @name(args) { ops }`.
    name*: string
    args*: seq[EntryArg]
    ops*: seq[Op]
    diName*: string  ## `di_name` entry attribute (source fn name), "" when absent

  TileIrModule* = object
    ## Parsed Tile IR module: `cuda_tile.module @name { entries }`.
    name*: string
    entries*: seq[Entry]

const DynamicDim* = -1
  ## Sentinel for a dynamic `?` dim or stride in shape and stride lists.

proc isDynamic*(d: int): bool {.inline.} =
  d == DynamicDim

const TokenIrType* = TileIrType(kind: tkToken)
  ## `token` type singleton.

proc subsetError*(msg: string): ref SubsetError =
  ## Builds a SubsetError refusal with message `msg`.
  ## Call sites raise it, so refusals stay one construction site per module.
  newException(SubsetError, msg)

proc `==`*(a, b: TypeElem): bool =
  if a.kind != b.kind:
    return false
  case a.kind
  of teScalar:
    a.scalar == b.scalar
  of tePointer:
    a.pointee == b.pointee

proc `==`*(a, b: TileIrType): bool =
  if a.kind != b.kind:
    return false
  case a.kind
  of tkToken:
    true
  of tkTile:
    a.tileShape == b.tileShape and a.tileElem == b.tileElem
  of tkTensorView:
    a.viewShape == b.viewShape and a.viewElem == b.viewElem and
      a.viewStrides == b.viewStrides
  of tkPartitionView:
    a.pvTile == b.pvTile and a.pvShape == b.pvShape and a.pvElem == b.pvElem and
      a.pvStrides == b.pvStrides and a.pvPadding == b.pvPadding

proc `==`*(a, b: ConstScalar): bool =
  if a.isInt != b.isInt:
    return false
  result = if a.isInt:
    a.intVal == b.intVal
  else:
    a.floatVal == b.floatVal

proc `==`*(a, b: ConstPayload): bool =
  if a.isList != b.isList:
    return false
  case a.isList
  of true:
    a.values == b.values
  of false:
    a.scalar == b.scalar

proc `==`*(a, b: ShapeEntry): bool =
  if a.isSsa != b.isSsa:
    return false
  case a.isSsa
  of true:
    a.name == b.name
  of false:
    a.value == b.value

# noSideEffect is declared: the proc recurses through the for-region body
# (seq[Op]) and the compiler cannot prove a recursive proc effect-free.
proc `==`*(a, b: Op): bool {.noSideEffect.} =
  if a.kind != b.kind:
    return false
  if a.results != b.results:
    return false
  if a.operands != b.operands:
    return false
  if a.operandTypes != b.operandTypes:
    return false
  if a.resultTypes != b.resultTypes:
    return false
  case a.kind
  of okConstant:
    a.constValue == b.constValue and a.constElem == b.constElem
  of okMakeTensorView:
    a.dynShape == b.dynShape and a.dynStrides == b.dynStrides
  of okLoadViewTko, okStoreViewTko:
    a.indices == b.indices and a.token == b.token
  of okAssume:
    a.assumeKind == b.assumeKind and a.assumeLo == b.assumeLo and
      a.assumeHi == b.assumeHi and a.assumeDiv == b.assumeDiv
  of okFor:
    a.loopVar == b.loopVar and a.loopVarType == b.loopVarType and
      a.iterNames == b.iterNames and a.iterInits == b.iterInits and
      a.body == b.body
  else:
    true

proc intScalar*(v: int): ConstScalar {.inline.} =
  ConstScalar(isInt: true, intVal: v)

proc floatScalar*(v: float): ConstScalar {.inline.} =
  ConstScalar(isInt: false, floatVal: v)

proc ssaEntry*(name: string): ShapeEntry {.inline.} =
  ShapeEntry(isSsa: true, name: name)

proc staticEntry*(value: int): ShapeEntry {.inline.} =
  ShapeEntry(isSsa: false, value: value)

proc scalarTile*(e: ScalarElem): TileIrType =
  ## 0-d tile of element type `e`, the Tile IR text `tile<i32>` or `tile<f32>`.
  TileIrType(kind: tkTile, tileShape: @[], tileElem: TypeElem(kind: teScalar, scalar: e))

proc ptrTile*(pointee: ScalarElem): TileIrType =
  ## Tile IR type `tile<ptr<E>>` for element type `pointee`.
  TileIrType(kind: tkTile, tileShape: @[], tileElem: TypeElem(kind: tePointer, pointee: pointee))

proc getEntry*(m: TileIrModule, name: string): Entry =
  ## Returns the entry named `name` of module `m`.
  ##
  ## Raises:
  ## - SubsetError when the module has no such entry
  for e in m.entries:
    if e.name == name:
      return e
  raise subsetError("module @" & m.name & ": no entry @" & name)

proc argType*(e: Entry, name: string): TileIrType =
  ## Returns the type of entry argument `name`.
  ##
  ## Raises:
  ## - SubsetError when the entry has no such argument
  for arg in e.args:
    if arg.name == name:
      return arg.typ
  raise subsetError("kernel '" & e.name & "': unknown entry argument %" & name)

proc opSpelling*(kind: OpKind): string =
  ## Tile IR text form of an op kind, so refusals read like the input text.
  case kind
  of okConstant: "constant"
  of okMakeToken: "make_token"
  of okMakeTensorView: "make_tensor_view"
  of okMakePartitionView: "make_partition_view"
  of okGetTileBlockId: "get_tile_block_id"
  of okLoadViewTko: "load_view_tko"
  of okStoreViewTko: "store_view_tko"
  of okAddf: "addf"
  of okMulf: "mulf"
  of okMmaf: "mmaf"
  of okAssume: "assume"
  of okReshape: "reshape"
  of okBroadcast: "broadcast"
  of okFor: "for"
  of okContinue: "continue"
  of okReturn: "return"
  of okYield: "yield"

proc readNames*(op: Op): seq[string] =
  ## All SSA names this op reads in a stable order. Operands come first,
  ## dynamic shape and stride names next, view indices and input token last.
  result = op.operands
  case op.kind
  of okMakeTensorView:
    for e in op.dynShape:
      if e.isSsa:
        result.add(e.name)
    for e in op.dynStrides:
      if e.isSsa:
        result.add(e.name)
  of okLoadViewTko, okStoreViewTko:
    result.add(op.indices)
    if op.token != "":
      result.add(op.token)
  of okFor:
    result.add(op.iterInits)
  else:
    discard
