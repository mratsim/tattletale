# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Nim types to CirType: the frontend boundary translation.
##
## Each Nim type translates to a `CirType` over the closed set of type kinds.
##
## Nim `int`/`uint` lower to 32 bits.
## Rejections go into `err` as the first refusal message.

import ../ir/cirv2_lang
import ../ir/cirv2_build
import std/[macros, tables, strutils, hashes]

type
  CirTypeResolver* = object
    ## Nim-to-CirType translation state: struct memo, nominal struct set,
    ## first refusal message.
    structMemo: Table[string, CirType]
    ## Struct memo keyed on the canonical identity text, consulted before any field derivation.
    ## Deriving one struct's fields re-resolves every field type, so a repeated identity pays the whole field walk again without it.
    structs*: seq[CirType]
    ## Nominal struct types declared by the module, in discovery order.
    err*: string
    ## First refusal message, "" while the walk is clean.

proc resolveCirType*(r: var CirTypeResolver, n: NimNode, ptrSpace = caDevice): CirType

proc fail*(r: var CirTypeResolver, detail: string) =
  if r.err.len == 0:
    r.err = "cirv2 frontend: " & detail

func scalarShortName(n: NimNode): string =
  ## Short name of a builtin scalar type node used as a generic argument.
  ## `float32` renders `f32`, Nim `int` renders `i32`.
  ## Empty when `n` is not a builtin scalar.
  case n.typeKind
  of ntyBool: result = "bool"
  of ntyInt8: result = "i8"
  of ntyInt16: result = "i16"
  of ntyInt, ntyInt32: result = "i32"
  of ntyInt64: result = "i64"
  of ntyUInt8: result = "u8"
  of ntyUInt16: result = "u16"
  of ntyUInt, ntyUInt32: result = "u32"
  of ntyUInt64: result = "u64"
  of ntyFloat32: result = "f32"
  of ntyFloat, ntyFloat64: result = "f64"
  else: result = ""

func deriveStructName*(n: NimNode): string =
  ## Short base of a struct name: the head symbol's name, plus a short scalar
  ## name for one scalar argument.
  ##
  ## Returns the base alone, not a unique name:
  ## - uniqueness comes from the base58 hash `resolveStructType` appends
  ## - two or more generic arguments and tuple member types stay out of the base
  ##
  ## TensorView[float32, Shape, Stride] -> "TensorView"
  ## Meter[float32]                     -> "Meter_f32"
  ## FixMe[8]                           -> "FixMe8"
  ## tuple[a: int32, b: int32]          -> "Tuple"
  case n.kind
  of nnkSym, nnkIdent:
    result = sanitizeIdent(n.strVal)
    if result != n.strVal:
      # structural type text (`tuple[a: int, b: int]`) and gensym'd symbols
      # collapse to a kind word, uniqueness comes from the hash
      result = if n.kind == nnkSym and n.typeKind == ntyTuple: "Tuple" else: "Object"
  of nnkBracketExpr:
    result = deriveStructName(n[0])
    if n.len == 2:
      # a single argument contributes its short form, a static integer joins
      # without a separator and any other argument stays out of the base
      let short = scalarShortName(n[1])
      if short.len > 0:
        result.add("_" & short)
      elif n[1].kind in nnkIntLit .. nnkUInt64Lit:
        result.add($n[1].intVal)
  of nnkTupleTy, nnkTupleConstr:
    result = "Tuple"
  else:
    result = "Object"

proc evalConstInt*(r: var CirTypeResolver, n: NimNode): int64 =
  ## Returns the compile-time integer value of `n`.
  ##
  ## - `..` reads as the array length `hi - lo + 1`
  ## - typed infix nodes carry exactly two operands after semantic analysis,
  ##   no length guard on the infix branch
  template refuse(detail: string) =
    r.fail(detail)
    return 0
  case n.kind
  of nnkIntLit .. nnkUInt64Lit:
    result = n.intVal
  of nnkSym:
    let impl = n.getImpl()
    if impl.kind == nnkConstDef and impl[2].kind in nnkIntLit .. nnkUInt64Lit:
      result = impl[2].intVal
    else:
      refuse("compile-time value expected, got symbol " & n.repr &
        " whose value is not a literal")
  of nnkPar:
    if n.len == 1:
      result = r.evalConstInt(n[0])
    else:
      refuse("compile-time value expected, got a parenthesized list")
  of nnkHiddenStdConv, nnkHiddenSubConv:
    if n.len >= 2:
      result = r.evalConstInt(n[1])
    else:
      refuse("compile-time value expected, got a bare conversion node")
  of nnkInfix:
    let a = r.evalConstInt(n[1])
    let b = r.evalConstInt(n[2])
    if r.err.len > 0:
      return 0
    case n[0].strVal
    of "..":
      # an array index range, the array length is hi - lo + 1
      result = b - a + 1
    of "+": result = a + b
    of "-": result = a - b
    of "*": result = a * b
    of "div": result = a div b
    of "mod": result = a mod b
    of "shl": result = a shl b
    of "shr": result = a shr b
    of "and": result = a and b
    of "or": result = a or b
    of "xor": result = a xor b
    else:
      refuse("compile-time arithmetic expected, got operator " & n[0].strVal)
  else:
    refuse("compile-time value expected, got node kind " & $n.kind)

func fieldSpaceAt(valueSpaces: Table[string, AddressSpace], name: string): AddressSpace =
  ## Pointer space a construction proves for the field `name`, `caDevice` when the construction names no space for it.
  valueSpaces.getOrDefault(name, caDevice)

proc deriveStructFields(r: var CirTypeResolver, impl: NimNode,
    valueSpaces: Table[string, AddressSpace] = initTable[string, AddressSpace]()): seq[CirField] =
  ## Field list of an object or tuple implementation node.
  ## `valueSpaces` maps each field name to the pointer space its construction value proves,
  ## every other field resolves under `caDevice`.
  case impl.kind
  of nnkObjectTy:
    if impl.len > 2 and impl[2].kind == nnkRecList:
      for ch in impl[2]:
        if ch.kind == nnkRecCase:
          r.fail("case object: variant records have no C representation")
          return @[]
        for j in 0 ..< ch.len - 2:
          result.add(fld(sanitizeIdent(ch[j].strVal),
            r.resolveCirType(ch[ch.len - 2],
              fieldSpaceAt(valueSpaces, sanitizeIdent(ch[j].strVal)))))
        if r.err.len > 0:
          return @[]
  of nnkTupleTy, nnkTupleConstr:
    for i, ch in impl:
      case ch.kind
      of nnkIdentDefs:
        for j in 0 ..< ch.len - 2:
          result.add(fld(sanitizeIdent(ch[j].strVal),
            r.resolveCirType(ch[ch.len - 2],
              fieldSpaceAt(valueSpaces, sanitizeIdent(ch[j].strVal)))))
      of nnkSym:
        result.add(fld("F" & $i, r.resolveCirType(ch,
          fieldSpaceAt(valueSpaces, "F" & $i))))
      else:
        result.add(fld("F" & $i, r.resolveCirType(ch.getTypeInst(),
          fieldSpaceAt(valueSpaces, "F" & $i))))
      if r.err.len > 0:
        return @[]
  of nnkBracketExpr:
    # `tuple[A, B]` is the bracket form of an anonymous tuple type
    for i in 1 ..< impl.len:
      result.add(fld("F" & $(i - 1), r.resolveCirType(impl[i].getTypeInst(),
        fieldSpaceAt(valueSpaces, "F" & $(i - 1)))))
  else:
    r.fail("struct fields: unexpected type implementation " & $impl.kind)
    return @[]

func followAliasHead(n: NimNode): NimNode =
  ## Type node with a type-alias head replaced by the alias's target:
  ## `Tile[float32, Sh, St]` where `Tile[T, Sh, St] = TensorOwned[T, Sh, St]`
  ## reads `TensorOwned[float32, Sh, St]`.
  ##
  ## The CIRv2 struct keeps the aliased name so one alias and its target name
  ## the same struct. Only the arity-matched form rewrites.
  ## Anything else returns the input.
  if n.kind == nnkBracketExpr and n.len > 1 and n[0].kind == nnkSym:
    let rhs = n[0].getTypeImpl()
    if rhs.kind == nnkBracketExpr and rhs.len == n.len and rhs[0].kind == nnkSym:
      result = nnkBracketExpr.newTree(rhs[0])
      for i in 1 ..< n.len:
        result.add(n[i])
      return
  return n

func canonicalTypeNode(n: NimNode): NimNode =
  ## Canonical type node: alias symbols replaced by the types they name, so one type
  ## identity derives one struct name. `V2F` aliasing `Vec2[float32]`
  ## reads back as `Vec2[float32]`.
  ##
  ## Bracket arguments canonicalize recursively: the head symbol stays.
  ## An alias head with unbound generic parameters rewrites through
  ## `followAliasHead`.
  case n.kind
  of nnkSym:
    if n.typeKind == ntyGenericInst and n.symKind == nskType:
      let impl = n.getImpl()
      if impl.kind == nnkTypeDef and impl[2].kind in {nnkSym, nnkBracketExpr}:
        return canonicalTypeNode(impl[2])
    result = n
  of nnkBracketExpr:
    if n.len > 1:
      result = nnkBracketExpr.newTree(n[0])
      for i in 1 ..< n.len:
        result.add(canonicalTypeNode(n[i]))
    else:
      result = n
  else:
    result = n

proc resolveStructType*(r: var CirTypeResolver, n: NimNode,
    valueSpaces: Table[string, AddressSpace] = initTable[string, AddressSpace]()): CirType =
  ## Nominal struct translation for an object, tuple, or generic instantiation type node.
  ##
  ## Identity key: the canonical text of the type node plus every non-device
  ## field address space.
  ##
  ## - an alias and the instantiation it names share one struct
  ## - two type nodes with different canonical text resolve to two structs, even
  ##   when their field lists agree, and each emits under its own name
  ## - the struct name hashes over the identity key
  let canon = followAliasHead(canonicalTypeNode(n))
  # one instantiation per pointer-space combination over the fields:
  # threadgroup-pointer constructions instantiate a struct that is separate
  # from the device instantiation. The canonical text never decides identity,
  # it names the struct the first registration emits.
  var nameKey = canon.repr
  for name, spc in valueSpaces:
    if spc != caDevice:
      nameKey &= "/" & name & "=" & $spc
  if r.structMemo.hasKey(nameKey):
    return r.structMemo[nameKey]
  let impl = n.getTypeImpl()
  let fields = r.deriveStructFields(impl, valueSpaces)
  if r.err.len > 0:
    return nil
  result = tStruct(deriveStructName(canon) & "_" &
    base58Short(typeIdentityHash(nameKey)), fields)
  r.structs.add(result)
  r.structMemo[nameKey] = result

func constrHasTypeElem(t: NimNode): bool =
  ## True when an instantiated tuple/object constructor type node has a type element anywhere.
  ##
  ## A runtime tuple-typed formal instantiates to its member types.
  ## A bound static value instantiates to the value's own elements.
  ## An object constructor's child 0 names the constructed type, never a value,
  ## so an `ObjConstr` scan starts after child 0.
  case t.kind
  of nnkEmpty:
    result = false
  of nnkSym:
    result = t.symKind == nskType
  of nnkObjConstr, nnkTupleConstr, nnkPar, nnkExprColonExpr:
    let first = if t.kind == nnkObjConstr: 1 else: 0
    for i in first ..< t.len:
      let e = t[i]
      # a named field's name is constructor vocabulary, the value decides
      let v = if e.kind == nnkExprColonExpr: e[1] else: e
      if constrHasTypeElem(v):
        return true
    result = false
  else:
    # literals are values, any other shape (bracket exprs, static ty) is a type
    result = t.kind notin nnkLiterals

func isCompileTimeFormal*(t: NimNode): bool =
  ## `static T` (either node layout) or `typedesc`: the argument is compile-time only,
  ## erased from the closure.
  ##
  ## A `static` formal bound to a value shows that value as its instantiated
  ## type node:
  ## - a literal value
  ## - an enum field symbol
  ## - a constructor over the value's elements
  ##
  ## A runtime tuple-typed formal instantiates to a constructor over its member types,
  ## its parameters are runtime values.
  if t.kind in nnkIntLit .. nnkUInt64Lit:
    return true
  if t.kind == nnkSym and t.symKind == nskEnumField:
    return true
  if t.kind in {nnkStaticTy, nnkBracketExpr, nnkCommand}:
    if t.len > 0 and t[0].repr == "static":
      return true
  if t.kind in {nnkIdent, nnkSym} and t.strVal == "typedesc":
    return true
  if t.kind in {nnkObjConstr, nnkTupleConstr}:
    # a bound static record or tuple formal instantiates to its value,
    # such an argument is compile-time only like any bound static
    return not constrHasTypeElem(t)
  if t.kind == nnkBracketExpr and t.len > 0 and t[0].strVal == "typeDesc":
    # an instantiated `typeDesc[T]` formal of a generic callee
    return true
  return false

proc resolveProcType(r: var CirTypeResolver, procTy: NimNode): CirType =
  ## Proc type from a `ProcTy` node. Compile-time (`static`/`typedesc`) parameters are erased,
  ## never passed to the closure.
  procTy.expectKind(nnkProcTy)
  let formals = procTy[0]
  formals.expectKind(nnkFormalParams)
  var paramTypes: seq[CirType]
  for i in 1 ..< formals.len:
    let formal = formals[i]
    let ftype = formal[formal.len - 2]
    if isCompileTimeFormal(ftype):
      continue
    let ptyp = r.resolveCirType(ftype)
    if r.err.len > 0:
      return nil
    if ptyp.isNil:
      continue
    paramTypes.add(ptyp)
  let ret = r.resolveCirType(formals[0])
  if r.err.len > 0:
    return nil
  result = tProcT(paramTypes, ret)

proc resolveCirType*(r: var CirTypeResolver, n: NimNode, ptrSpace = caDevice): CirType =
  ## Translates one Nim type node to CirType.
  ##
  ## `ptrSpace` names the memory class of the pointer kinds this type node contains:
  ## kernel parameters pass `caDevice`, a body-side `addr` result constructs
  ## its own `caThread` pointer.
  if r.err.len > 0:
    return nil
  if n.kind == nnkEmpty:
    # a missing return type is void
    return tVoid()
  if n.kind in {nnkStmtListType, nnkStmtListExpr}:
    # an inline type or value block, the resolved node is its last child
    return r.resolveCirType(n[^1], ptrSpace)
  if n.kind == nnkCall:
    # a macro call in type position (`tile(T, layout)`), the type is the macro's instantiated result
    return r.resolveCirType(n.getTypeInst(), ptrSpace)
  case n.typeKind
  of ntyBool:
    result = tBool()
  of ntyInt, ntyInt32:
    result = tInt(32)
  of ntyInt8:
    result = tInt(8)
  of ntyInt16:
    result = tInt(16)
  of ntyInt64:
    result = tInt(64)
  of ntyUInt, ntyUInt32:
    result = tUint(32)
  of ntyUInt8:
    result = tUint(8)
  of ntyUInt16:
    result = tUint(16)
  of ntyUInt64:
    result = tUint(64)
  of ntyFloat32:
    result = tFloat(32)
  of ntyFloat, ntyFloat64:
    result = tFloat(64)
  of ntyPtr:
    let pointee = n[0]
    if pointee.typeKind == ntyUncheckedArray:
      # a typed `ptr UncheckedArray[T]` pointee varies:
      # - a direct bracket `UncheckedArray[T]`
      # - a call node `UncheckedArray(T)` from a template instantiation
      # Resolve the element type as the instantiation's type argument.
      var inst = pointee.getTypeInst()
      if inst.kind != nnkBracketExpr:
        inst = inst.getTypeImpl()
      inst.expectKind(nnkBracketExpr)
      result = tAptr(ptrSpace, r.resolveCirType(inst[1], ptrSpace))
    else:
      result = tPtr(ptrSpace, r.resolveCirType(pointee, ptrSpace))
  of ntyVar:
    # `var T` parameters travel as thread pointers, the closure never
    # passes them by reference across the device boundary
    result = tPtr(caThread, r.resolveCirType(n[0], caThread))
  of ntyAlias:
    # an alias IS the aliased type: the CIRv2 struct keeps the aliased name,
    # `Tile[T, Sh, St] = TensorOwned[T, Sh, St]` must not derive two structs
    result = r.resolveCirType(n.getTypeImpl(), ptrSpace)
  of ntyObject, ntyTuple, ntyGenericInst:
    result = r.resolveStructType(n)
  of ntyArray:
    var arr: NimNode = n
    if n.kind == nnkSym:
      arr = n.getTypeImpl()
    arr.expectKind(nnkBracketExpr)
    if arr[1].kind == nnkIdent:
      # a named const length: the implementation tree keeps the name, the instantiation tree gives the value
      arr = arr.getTypeInst()
      arr.expectKind(nnkBracketExpr)
    let len = r.evalConstInt(arr[1])
    if r.err.len > 0:
      return nil
    result = tArray(int(len), r.resolveCirType(arr[2], ptrSpace))
  of ntyEnum:
    # enums lower as their 32-bit storage, the type kinds have no enum entry
    result = tUint(32)
  of ntyDistinct:
    result = r.resolveCirType(n.getTypeImpl()[0], ptrSpace)
  of ntyProc:
    result = r.resolveProcType(n.getTypeInst())
  of ntyTypeDesc:
    result = r.resolveCirType(n.getType()[1], ptrSpace)
  of ntyStatic:
    result = r.resolveCirType(n.getTypeInst(), ptrSpace)
  of ntyOpenArray:
    # openArray[T] travels as a span over the element type in the pointer space.
    # Span split later decomposes a span into an element pointer
    # plus an int32 length, no span reaches the printers
    var span = n
    if span.kind == nnkSym:
      span = span.getTypeImpl()
    span.expectKind(nnkBracketExpr)
    result = tSpan(ptrSpace, r.resolveCirType(span[1], ptrSpace))
  of ntyRange:
    # A range type has no runtime representation. Sem wraps index literals
    # and `range` aliases in `range[lo .. hi]` around the real type: its base type.
    var rng = n
    if n.kind == nnkSym:
      rng = n.getTypeImpl()
    rng.expectKind(nnkBracketExpr)
    result = r.resolveCirType(rng[1][1], ptrSpace)
  else:
    r.fail("type not supported: " & n.repr & " (" & $n.typeKind & ")")
    return nil
