# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../ir/cirv2_lang
import ../ir/cirv2_build
import std/[macros, tables]

# ############################################################
#
#            Nim type resolution
#
# ############################################################

type
  CirTypeResolver* = object
    structIdentityCache: Table[string, CirType]
    resolvedStructs*: seq[CirType]

func resolveCirType*(r: var CirTypeResolver, n: NimNode, ptrSpace = caDevice): CirType

func getBuiltinDistinctScalar(n: NimNode): CirType =
  if n.kind != nnkSym or n.symKind != nskType:
    return nil
  let impl = n.getImpl()
  if impl.kind != nnkTypeDef or impl[0].kind != nnkPragmaExpr or
      impl[0][1].kind != nnkPragma:
    return nil
  var marked = false
  for p in impl[0][1]:
    if p.kind in {nnkSym, nnkIdent} and p.strVal == "builtin":
      marked = true
    if p.kind in {nnkExprColonExpr, nnkCall} and p.len >= 1 and
        p[0].kind in {nnkIdent, nnkSym} and p[0].strVal == "builtin":
      marked = true
  if not marked:
    return nil
  case n.strVal
  of "float16": tFloat(16)
  of "bfloat16": tBf16()
  of "fp8e4m3", "fp8e5m2": tUint(8)
  else: nil

func getScalarShortName(n: NimNode): string =
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
      let short = getScalarShortName(n[1])
      if short.len > 0:
        result.add("_" & short)
      elif n[1].kind in nnkIntLit .. nnkUInt64Lit:
        result.add($n[1].intVal)
  of nnkTupleTy, nnkTupleConstr:
    result = "Tuple"
  else:
    result = "Object"

func evalConstInt*(r: var CirTypeResolver, n: NimNode): int64 =
  ## Returns the compile-time integer value of `n`.
  ##
  ## - `..` yields the array length `hi - lo + 1`
  template refuse(detail: string) =
    raiseAssert(detail)
  case n.kind
  of nnkIntLit .. nnkUInt64Lit:
    result = n.intVal
  of nnkSym:
    let impl = n.getImpl()
    if impl.kind == nnkConstDef and impl[2].kind in nnkIntLit .. nnkUInt64Lit:
      result = impl[2].intVal
    else:
      refuse("compile-time value expected, got symbol " & n.repr & " whose value is not a literal")
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
    case n[0].strVal
    of "..":
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

func deriveStructFields(r: var CirTypeResolver, impl: NimNode,
    valueSpaces: Table[string, AddressSpace] = initTable[string, AddressSpace]()): seq[CirField] =
  ## `valueSpaces` maps each field name to the pointer space that the field's
  ## construction value proves, every other field resolves under `caDevice`.
  case impl.kind
  of nnkObjectTy:
    if impl.len > 2 and impl[2].kind == nnkRecList:
      for ch in impl[2]:
        if ch.kind == nnkRecCase:
          raiseAssert("case object: variant records have no C representation")
        for j in 0 ..< ch.len - 2:
          result.add(fld(sanitizeIdent(ch[j].strVal),
            r.resolveCirType(ch[ch.len - 2],
              valueSpaces.getOrDefault(sanitizeIdent(ch[j].strVal), caDevice))))
  of nnkTupleTy, nnkTupleConstr:
    for i, ch in impl:
      case ch.kind
      of nnkIdentDefs:
        for j in 0 ..< ch.len - 2:
          result.add(fld(sanitizeIdent(ch[j].strVal),
            r.resolveCirType(ch[ch.len - 2],
              valueSpaces.getOrDefault(sanitizeIdent(ch[j].strVal), caDevice))))
      of nnkSym:
        result.add(fld("F" & $i, r.resolveCirType(ch,
          valueSpaces.getOrDefault("F" & $i, caDevice))))
      else:
        result.add(fld("F" & $i, r.resolveCirType(ch.getTypeInst(),
          valueSpaces.getOrDefault("F" & $i, caDevice))))
  of nnkBracketExpr:
    # `tuple[A, B]` is the bracket form of an anonymous tuple type
    for i in 1 ..< impl.len:
      result.add(fld("F" & $(i - 1), r.resolveCirType(impl[i].getTypeInst(),
        valueSpaces.getOrDefault("F" & $(i - 1), caDevice))))
  else:
    raiseAssert("struct fields: unexpected type implementation " & $impl.kind)

func followAliasHead(n: NimNode): NimNode =
  ## `Tile[float32, Sh, St]` where `Tile[T, Sh, St] = TensorOwned[T, Sh, St]`
  ## reads `TensorOwned[float32, Sh, St]`. The CIRv2 struct is named after the alias,
  ## so an alias and its target resolve to one struct.
  if n.kind == nnkBracketExpr and n.len > 1 and n[0].kind == nnkSym:
    let rhs = n[0].getTypeImpl()
    if rhs.kind == nnkBracketExpr and rhs.len == n.len and rhs[0].kind == nnkSym:
      result = nnkBracketExpr.newTree(rhs[0])
      for i in 1 ..< n.len:
        result.add(n[i])
      return
  return n

func canonicalizeTypeNode(n: NimNode): NimNode =
  ## Alias symbols are replaced by the types behind them, so one type identity
  ## derives one struct name. `V2F` aliasing `Vec2[float32]` reads back as `Vec2[float32]`.
  ## Alias heads with unbound generic parameters rewrite later, in `resolveStructType`.
  case n.kind
  of nnkSym:
    if n.typeKind == ntyGenericInst and n.symKind == nskType:
      let impl = n.getImpl()
      if impl.kind == nnkTypeDef and impl[2].kind in {nnkSym, nnkBracketExpr}:
        return canonicalizeTypeNode(impl[2])
    result = n
  of nnkBracketExpr:
    if n.len > 1:
      result = nnkBracketExpr.newTree(n[0])
      for i in 1 ..< n.len:
        result.add(canonicalizeTypeNode(n[i]))
    else:
      result = n
  else:
    result = n

func resolveStructType*(r: var CirTypeResolver, n: NimNode,
    valueSpaces: Table[string, AddressSpace] = initTable[string, AddressSpace]()): CirType =
  ## Identity key: the canonical text of the type node plus every non-device
  ## field address space.
  ##
  ## - an alias and the instantiation it names share one struct
  ## - two type nodes with different canonical text resolve to two structs, even
  ##   when their field lists agree, and each emits under its own name
  let canon = followAliasHead(canonicalizeTypeNode(n))
  # one instantiation exists per pointer-space combination:
  # threadgroup-pointer constructions stay separate from the device instantiation
  var nameKey = canon.repr
  for name, spc in valueSpaces:
    if spc != caDevice:
      nameKey &= "/" & name & "=" & $spc
  if r.structIdentityCache.hasKey(nameKey):
    return r.structIdentityCache[nameKey]
  let impl = n.getTypeImpl()
  let fields = r.deriveStructFields(impl, valueSpaces)
  let candidate = tStruct(deriveStructName(canon) & "_" &
    base58Short(typeIdentityHash(nameKey)), fields)
  # A second dedup runs over structure: an alias instantiation
  # can reach the resolver through the generic body's own object text,
  # so two nodes with different canonical text derive one struct.
  # An equal structure is returned under the identifier
  # its first registration chose.
  for registered in r.resolvedStructs:
    if structuralEqual(registered, candidate):
      result = registered
      r.structIdentityCache[nameKey] = result
      return
  result = candidate
  r.resolvedStructs.add(result)
  r.structIdentityCache[nameKey] = result

func constrHasTypeElem(t: NimNode): bool =
  ## A runtime tuple-typed formal instantiates to its member types.
  ## A bound static value instantiates to the value's own elements.
  ## An object constructor's child 0 is the constructed type, never a value,
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
      let v = if e.kind == nnkExprColonExpr: e[1] else: e
      if constrHasTypeElem(v):
        return true
    result = false
  else:
    result = t.kind notin nnkLiterals

func isCompileTimeFormal*(t: NimNode): bool =
  ## Returns whether the formal is compile-time only and erased from the closure:
  ## `static T` (either node layout) or `typedesc`.
  ##
  ## A `static` formal bound to a value shows that value as its instantiated type node,
  ## so a literal, an enum field symbol or a constructor is compile-time only.
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
    return not constrHasTypeElem(t)
  if t.kind == nnkBracketExpr and t.len > 0 and t[0].strVal == "typeDesc":
    # an instantiated `typeDesc[T]` formal of a generic callee
    return true
  return false

func getPhantomStaticArg*(n: NimNode): NimNode =
  ## Static argument of a phantom value type, nil when absent.
  ##
  ## - a phantom value type is an object with no fields whose value is
  ##   its static argument `V`, instantiated as `Phantom[V]`
  ##   (the phantom-value convention, Ceramic's `Int[V]`)
  ## - the type resolves to the argument's integer type and a value
  ##   of it lowers to the argument, so the value survives into
  ##   runtime layout arithmetic
  ## Returns:
  ## - the static argument node, or nil for every other type
  var t = n
  if t.kind == nnkSym and t.typeKind == ntyGenericInst:
    # an instantiation symbol reads back as its bracket form `Phantom[V]`
    let impl = t.getImpl()
    if impl.kind != nnkTypeDef or impl[2].kind != nnkSym:
      return nil
    t = impl[2]
  if t.kind != nnkBracketExpr or t.len != 2:
    return nil
  if t[1].kind notin nnkIntLit .. nnkUInt64Lit:
    return nil
  let head = t[0]
  if head.kind != nnkSym or head.symKind != nskType:
    return nil
  let impl = head.getImpl()
  if impl.kind != nnkTypeDef or impl[2].kind != nnkObjectTy:
    return nil
  let recList = impl[2][2]
  if recList.kind == nnkRecList and recList.len > 0:
    return nil
  return t[1]

func resolveProcType(r: var CirTypeResolver, procTy: NimNode): CirType =
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
    if ptyp.isNil:
      continue
    paramTypes.add(ptyp)
  let ret = r.resolveCirType(formals[0])
  result = tProcT(paramTypes, ret)

func resolveCirType*(r: var CirTypeResolver, n: NimNode, ptrSpace = caDevice): CirType =
  ## Translates one Nim type node to CirType.
  ##
  ## `ptrSpace` is the memory class of the pointer kinds this type node contains:
  ## kernel parameters pass `caDevice`, a body-side `addr` result constructs
  ## its own `caThread` pointer.
  if n.kind == nnkEmpty:
    # a missing return type is void
    return tVoid()
  if n.kind in {nnkStmtListType, nnkStmtListExpr}:
    # an inline type or value block, with the resolved node as its last child
    return r.resolveCirType(n[^1], ptrSpace)
  if n.kind == nnkCall:
    # a macro call in type position (`tile(T, layout)`), resolving to the macro's instantiated result
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
  of ntyString:
    # a string literal's type. Only the C-family targets carry a type name
    # for it (`const char*`), the wgsl and hlsl printers refuse the type name.
    result = tString()
  of ntyPtr:
    let pointee = n[0]
    if pointee.typeKind == ntyUncheckedArray:
      # a `ptr UncheckedArray[T]` pointee varies by provenance: a direct bracket `UncheckedArray[T]`
      # or a call node `UncheckedArray(T)` from a template instantiation
      var inst = pointee.getTypeInst()
      if inst.kind != nnkBracketExpr:
        inst = inst.getTypeImpl()
      inst.expectKind(nnkBracketExpr)
      result = tAptr(ptrSpace, r.resolveCirType(inst[1], ptrSpace))
    else:
      result = tPtr(ptrSpace, r.resolveCirType(pointee, ptrSpace))
  of ntyVar:
    # `var T` parameters travel as thread pointers, and the closure never
    # passes them by reference across the device boundary
    result = tPtr(caThread, r.resolveCirType(n[0], caThread))
  of ntyAlias:
    result = r.resolveCirType(n.getTypeImpl(), ptrSpace)
  of ntyObject, ntyTuple, ntyGenericInst:
    if getPhantomStaticArg(n).isNil:
      result = r.resolveStructType(n)
    else:
      # a phantom value type's value is its static argument, an int
      result = tInt(32)
  of ntyArray:
    var arr: NimNode = n
    if n.kind == nnkSym:
      arr = n.getTypeImpl()
    arr.expectKind(nnkBracketExpr)
    if arr[1].kind == nnkIdent:
      # a named const length: the implementation tree has the identifier
      # and the instantiation tree has the value
      arr = arr.getTypeInst()
      arr.expectKind(nnkBracketExpr)
    let len = r.evalConstInt(arr[1])
    result = tArray(int(len), r.resolveCirType(arr[2], ptrSpace))
  of ntyEnum:
    # C enums are int
    result = tUint(32)
  of ntyDistinct:
    let marked = getBuiltinDistinctScalar(n)
    if marked != nil:
      result = marked
    else:
      result = r.resolveCirType(n.getTypeImpl()[0], ptrSpace)
  of ntyProc:
    result = r.resolveProcType(n.getTypeInst())
  of ntyTypeDesc:
    result = r.resolveCirType(n.getType()[1], ptrSpace)
  of ntyStatic:
    result = r.resolveCirType(n.getTypeInst(), ptrSpace)
  of ntyOpenArray:
    # openArray[T] passes through the frontend as a span over the element type in the pointer space.
    # Span split decomposes a span into an element pointer
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
    raiseAssert("type not supported: " & n.repr & " (" & $n.typeKind & ")")
