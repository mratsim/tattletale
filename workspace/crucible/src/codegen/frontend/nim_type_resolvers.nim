# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Nim AST → GpuType resolvers

import std / [macros, strutils, sequtils, tables, sets]
import ../ir/ast_datatypes
import ../ir/ast_type_constructors
import ../../builtins/builtins_gpu_types

# ═══════════════════════════════════════════════════════════════════════
#  Generic type name utilities
# ═══════════════════════════════════════════════════════════════════════

func unpackGenericInst*(t: NimNode): NimNode =
  let tKind = t.typeKind
  if tKind == ntyGenericInst:
    let impl = t.getTypeImpl()
    case impl.kind
    of nnkDistinctTy: # just skip the distinct
      result = impl[0]
    of nnkObjectTy, nnkEnumTy:
      result = t # keep object/enum types as-is
    else:
      error "Unsupport type so far: " & $t.treerepr & " of impl: " & $impl.treerepr
  else:
    result = t

func toGpuTypeKind*(t: NimNode): GpuTypeKind =
  result = t.unpackGenericInst().typeKind.toGpuTypeKind()

# ═══════════════════════════════════════════════════════════════════════
#  Array length resolution
# ═══════════════════════════════════════════════════════════════════════

proc evalConstInt(n: NimNode): int =
  ## Handles the array-length forms `getTypeInst` produces, incl. infix `32 * 16` and `0 .. 15` ranges.
  case n.kind
  of nnkIntLit, nnkUIntLit:
    result = n.intVal
  of nnkSym:
    result = n.getImpl.intVal
  of nnkInfix:
    let a = evalConstInt(n[1])
    let b = evalConstInt(n[2])
    case n[0].strVal
    of "*":  result = a * b
    of "+":  result = a + b
    of "-":  result = a - b
    of "div": result = a div b
    of "mod": result = a mod b
    of "shl": result = a shl b
    of "shr": result = a shr b
    of "and": result = a and b
    of "..":  result = b - a + 1
    of "..<": result = b - a
    else:
      error "Unsupported operator in array length: " & n[0].strVal & " in " & n.treerepr
  else:
    error "Unsupported node in array length: " & n.treerepr

func resolveArrayLength*(n: NimNode): int =
  ## `n` must be `getTypeInst()` output, idents are unsupported.
  result = evalConstInt(n[1])

# ═══════════════════════════════════════════════════════════════════════
#  Forward declarations
# ═══════════════════════════════════════════════════════════════════════

proc resolveGpuArrayType*(reg: var TypeRegistry, aTyp: NimNode, len: int): GpuType
proc resolveType*(reg: var TypeRegistry, n: NimNode): GpuType
proc resolveInnerPointerType*(reg: var TypeRegistry, n: NimNode): GpuType
proc resolveStructuralType*(reg: var TypeRegistry, impl: NimNode, t: NimNode): GpuType
proc resolveInstantiatedType*(reg: var TypeRegistry, t: NimNode): GpuType
proc resolveRecordFields*(reg: var TypeRegistry, node: NimNode): seq[GpuTypeField]
func assignTypeName*(n: NimNode, recursedSym: bool = false): string

type
  FieldInfo = object
    name: string
    typeNode: NimNode

proc resolveTupleFields(node: NimNode): seq[FieldInfo] =
  ## Handles nnkTupleTy, nnkTupleConstr and the nnkBracketExpr `tuple[...]` form.
  case node.kind
  of nnkTupleTy, nnkTupleConstr:
    for i, ch in node:
      case ch.kind
      of nnkIdentDefs:
        for j in 0 ..< ch.len - 2:
          result.add FieldInfo(name: ch[j].strVal, typeNode: ch[ch.len - 2])
      of nnkSym:
        result.add FieldInfo(name: "F" & $i, typeNode: ch)
      of nnkExprColonExpr:
        result.add FieldInfo(name: ch[0].strVal, typeNode: ch[1])
      of nnkBracketExpr:
        # resolve via getTypeInst() to avoid leaking ObjectTy repr into struct name
        result.add FieldInfo(name: "F" & $i, typeNode: ch.getTypeInst())
      else:
        # resolve via getTypeInst() to avoid leaking ObjectTy repr into struct name
        result.add FieldInfo(name: "F" & $i, typeNode: ch.getTypeInst())
  of nnkBracketExpr:
    # `tuple[Int[1], Int[2]]` is the full AST form of an anonymous tuple type
    # (getTypeInst output, e.g. generic args of an object construction).
    doAssert node[0].kind in {nnkSym, nnkIdent} and node[0].strVal == "tuple",
      "unexpected tuple bracket: " & $node.treerepr
    for i in 1 ..< node.len:
      result.add FieldInfo(name: "F" & $(i - 1), typeNode: node[i].getTypeInst())
  else:
    error "Unsupported node kind for tuple fields: " & $node.kind

proc tupleTypeName(fields: seq[FieldInfo]): string =
  ## `Tuple_` + one `Name_TypeName` segment per field, shared by every AST form.
  result = "Tuple_"
  for i, fi in fields:
    if i > 0:
      result.add "_"
    result.add fi.name & "_" & fi.typeNode.assignTypeName()

# ═══════════════════════════════════════════════════════════════════════
#  Generic type argument / implementation resolution
# ═══════════════════════════════════════════════════════════════════════


proc resolveStructuralType*(reg: var TypeRegistry, impl: NimNode, t: NimNode): GpuType =
  case impl.kind
  of nnkDistinctTy:
    ## XXX: assumes distinct of inbuilt type, not object!
    result = resolveType(reg, impl[0])
  of nnkObjectTy:
    doAssert impl.kind == nnkObjectTy, "Unexpected node kind for generic inst: " & $impl.treerepr
    ## XXX: use signature hash for type name? Otherwise will produce duplicates
    result = initGpuGenericInstType(t.repr)
    result.fields = resolveRecordFields(reg, impl)
  of nnkStaticTy:
    # Static int, the resolver preserves its value for struct naming.
    result = GpuType(kind: gtStatic, builtin: true, sValue: int(impl[0].intVal))
  else:
    error "Unexpected node kind in for genericInst: " & $impl.treerepr & " kind=" & $impl.kind

# ═══════════════════════════════════════════════════════════════════════
#  Type-alias canonicalization
# ═══════════════════════════════════════════════════════════════════════

proc isAliasBase(n: NimNode): bool =
  ## True when the type-def RHS is a type application or template call,
  ## never a direct object/enum/distinct/tuple/ref definition.
  if n.kind != nnkSym: return false
  let impl = n.getImpl()
  if impl.kind != nnkTypeDef: return false
  case impl[2].kind
  of nnkObjectTy, nnkEnumTy, nnkDistinctTy, nnkTupleTy, nnkRefTy, nnkEmpty:
    false
  else:
    true

proc substituteTypeSyms(n, fromN, toN: NimNode): NimNode =
  if n.kind == nnkSym:
    return if n.eqIdent(fromN): toN else: n
  result = n.copyNimTree()
  for i in 0 ..< result.len:
    result[i] = substituteTypeSyms(result[i], fromN, toN)

proc substituteAliasArgs(typeDef, app: NimNode): NimNode =
  ## Missing application args fall back to the param's declared default.
  result = typeDef[2].copyNimTree()
  if app.kind != nnkBracketExpr: return
  var argIdx = 1
  for gp in typeDef[1]:
    case gp.kind
    of nnkSym:
      if argIdx < app.len:
        result = substituteTypeSyms(result, gp, app[argIdx])
        inc argIdx
      else:
        # the default substitutes as its value (int literal) or declared
        # type (bool, enum, object), an ObjConstr value is not a type
        let defaultNode = gp.getImpl()
        if defaultNode.kind != nnkNilLit:
          let defaultTy = defaultNode.getTypeInst()
          let defaultArg =
            if defaultNode.kind in {nnkIntLit, nnkUIntLit} and
               defaultTy.typeKind notin {ntyBool, ntyEnum}:
              defaultNode
            else:
              defaultTy.getTypeInst()
          result = substituteTypeSyms(result, gp, defaultArg)
    of nnkIdent:
      if argIdx < app.len:
        result = substituteTypeSyms(result, gp, app[argIdx])
        inc argIdx
    else:
      discard

proc resolveAliasExpansion(reg: var TypeRegistry, app, rhs: NimNode): GpuType =
  ## `app.getTypeImpl` supplies the expanded object for the fields.
  case rhs.kind
  of nnkBracketExpr:
    if isAliasBase(rhs[0]):
      let nested = substituteAliasArgs(rhs[0].getImpl(), rhs)
      return resolveAliasExpansion(reg, app, nested)
    let impl = app.getTypeImpl()
    if impl.kind notin {nnkObjectTy, nnkTupleTy, nnkTupleConstr}:
      # A non-object RHS would emit a fieldless generic instantiation.
      error "Alias RHS '" & $rhs.repr &
        "' does not expand to an object or tuple type. Use the canonical type application"
    result = initGpuGenericInstType(getGenericTypeName(rhs))
    for i in 1 ..< rhs.len:
      result.gArgs.add resolveType(reg, rhs[i])
    result.fields = resolveRecordFields(reg, impl)
  of nnkSym:
    if isAliasBase(rhs):
      let nested = substituteAliasArgs(rhs.getImpl(), rhs)
      return resolveAliasExpansion(reg, app, nested)
    # Alias to a named type: resolve the sym in the alias's own scope.
    result = resolveType(reg, rhs)
  of nnkCall, nnkCommand:
    # Expansion needs Nim's semantic analysis, unavailable at resolve time.
    error "Alias RHS is a template call '" & $rhs[0].repr &
      "': use the canonical type application, " &
      "e.g. RtLeft[float32, R, C, TileConfigFor(float32)]"
  else:
    result = resolveType(reg, rhs)

proc resolveInstantiatedType*(reg: var TypeRegistry, t: NimNode): GpuType =
  if t.typeKind notin {ntyGenericInst}:
    if t.typeKind != ntyNone:
      return resolveType(reg, t.getTypeInst())
    else:
      return initGpuGenericInstType(t.repr)
  # Value expressions are canonicalized at the resolveType entry.
  case t.kind
  of nnkBracketExpr:
    if isAliasBase(t[0]):
      # keeps the emitted type name equal to the canonical instantiation's
      return resolveAliasExpansion(reg, t, substituteAliasArgs(t[0].getImpl(), t))
    result = initGpuGenericInstType(getGenericTypeName(t))
    for i in 1 ..< t.len:
      result.gArgs.add resolveType(reg, t[i])
    let impl = t.getTypeImpl() # resolves the `fields` members
    result.fields = resolveRecordFields(reg, impl)
  of nnkObjectTy:
    result = resolveStructuralType(reg, t, t)
  of nnkSym:
    # getImpl() can return local variable IdentDefs from template expansions.
    if isAliasBase(t):
      return resolveAliasExpansion(reg, t, substituteAliasArgs(t.getImpl(), t))
    let inst = t.getTypeInst()
    if inst.kind == nnkBracketExpr:
      result = resolveInstantiatedType(reg, inst)
    else:
      let impl = inst.getTypeImpl()
      result = resolveStructuralType(reg, impl, t)
  else:
    error "Unexpected t.kind for genericInst: " & $t.kind & " treerepr=" & $t.treerepr

# ═══════════════════════════════════════════════════════════════════════
#  Pointer / inner type resolution
# ═══════════════════════════════════════════════════════════════════════

proc resolveInnerPointerType*(reg: var TypeRegistry, n: NimNode): GpuType =
  doAssert n.typeKind in {ntyPtr, ntyPointer, ntyUncheckedArray, ntyVar} or n.kind == nnkPtrTy, "But was: " & $n.treerepr & " of typeKind " & $n.typeKind
  if n.typeKind in {ntyPointer, ntyUncheckedArray}:
    let typ = n.getTypeInst()
    doAssert typ.kind == nnkBracketExpr, "No, was: " & $typ.treerepr
    doAssert typ[0].kind in {nnkIdent, nnkSym}
    doAssert typ[0].strVal in ["ptr", "UncheckedArray"]
    result = resolveType(reg, typ[1])
  elif n.kind == nnkPtrTy:
    result = resolveType(reg, n[0])
  elif n.kind == nnkAddr:
    let typ = n.getTypeInst()
    result = resolveInnerPointerType(reg, typ)
  elif n.kind == nnkVarTy:
    # VarTy
    #   Sym "BigInt"
    result = resolveType(reg, n[0])
  elif n.kind == nnkSym: # symbol of e.g. `ntyVar`
    result = resolveType(reg, n.getTypeInst())
  else:
    error "Found what: " & $n.treerepr

# ═══════════════════════════════════════════════════════════════════════
#  Tuple type naming
# ═══════════════════════════════════════════════════════════════════════


func assignTypeName*(n: NimNode, recursedSym: bool = false): string =
  case n.kind
  of nnkIdent: result = n.strVal
  of nnkSym:
    if recursedSym:
      result = n.strVal
    else:
      result = n.getTypeInst.assignTypeName(true)
  of nnkObjConstr:
    if n[0].kind == nnkEmpty:
      result = n.getTypeInst.strVal
    else:
      result = n[0].strVal # type is the first node
  of nnkObjectTy:
    # Anonymous object type, use its repr as a fallback name.
    result = n.repr
  of nnkTupleTy, nnkTupleConstr:
    result = tupleTypeName(resolveTupleFields(n))
  of nnkBracketExpr:
    # must name identically to the nnkTupleConstr form
    if n.len >= 2 and n[0].kind in {nnkSym, nnkIdent} and n[0].strVal == "tuple":
      result = tupleTypeName(resolveTupleFields(n))
    elif n[0].kind == nnkSym and n[0].strVal == "array":
      # never recurse through the `array` symbol, its type inst loops
      result = "Array_" & $resolveArrayLength(n) & "_" & n[2].assignTypeName()
    else:
      for i, ch in n:
        result.add ch.assignTypeName()
  of nnkIntLit:
    result = $n.intVal
  of nnkUIntLit:
    result = $n.intVal
  else: error "Unexpected node in `assignTypeName`: " & $n.treerepr

# ═══════════════════════════════════════════════════════════════════════
#  Main type resolver
# ═══════════════════════════════════════════════════════════════════════

proc resolveType*(reg: var TypeRegistry, n: NimNode): GpuType =
  case n.kind
  of nnkIdentDefs: # extract type for let / var based on explicit or implicit type
    if n[n.len - 2].kind != nnkEmpty:
      result = resolveType(reg, n[n.len - 2])
    else:
      result = resolveType(reg, n[n.len - 1].getTypeInst())
  of nnkConstDef:
    if n[1].kind != nnkEmpty: # has an explicit type
      result = resolveType(reg, n[1])
    else:
      result = resolveType(reg, n[2]) # derive from the RHS literal
  of nnkIntLit, nnkUIntLit:
    result = GpuType(kind: gtStatic, builtin: true, sValue: int(n.intVal))
    reg.registerObjectType(result)
    return result
  of nnkStmtListExpr:
    # a template expansion left in a type-bracket arg, the expression is the arg
    if n.len == 2 and n[0].kind == nnkEmpty:
      result = resolveType(reg, n[1])
    else:
      error "Unsupported StmtListExpr in type resolution: " & $n.treerepr
  else:
    if n.kind == nnkEmpty: return initGpuType(gtVoid)
    # a value expression like `w.layout.stride[0]` leaked in, canonicalize it to its type node via getTypeInst()
    let n = block:
      if n.kind notin {nnkSym, nnkIdent, nnkTupleTy, nnkObjectTy, nnkPtrTy, nnkTupleConstr} and
         (n.kind != nnkBracketExpr or n[0].kind notin {nnkSym, nnkBracketExpr}) and
         n.typeKind != ntyNone:
        n.getTypeInst()
      else:
        n
    # builtin GPU types (float16, bfloat16): distinct uint16, no nty* mapping
    if n.kind == nnkSym and builtinGpuTypeKind(n) != gtVoid:
      return initGpuType(builtinGpuTypeKind(n))
    case n.typeKind
    of ntyDistinct:
      # peel the nnkDistinctTy wrapper, recursing on it loops forever
      result = resolveType(reg, n.getTypeImpl()[0])
    of ntyBool, ntyInt .. ntyUint64: # includes all float types
      result = initGpuType(toGpuTypeKind n.typeKind)
    of ntyString: # only supported on some backends!
      result = initGpuType(toGpuTypeKind n.typeKind)
    of ntyPtr:
      result = initGpuPtrType(resolveInnerPointerType(reg, n), implicitPtr = false)
    of ntyVar:
      result = initGpuPtrType(resolveInnerPointerType(reg, n), implicitPtr = true)
    of ntyPointer:
      result = initGpuVoidPtr()
    of ntyUncheckedArray:
      # only a pointer through `ptr UncheckedArray[T]`, resolve the element type
      result = initGpuUAType(resolveInnerPointerType(reg, n))
    of ntyOpenArray:
      ## OpenArray[T] → gtSpan(T)
      doAssert n.kind == nnkBracketExpr and n[0].strVal == "openArray",
        "ntyOpenArray: expected BracketExpr(openArray, T), got " & $n.treerepr
      let innerTyp = resolveType(reg, n[1])
      result = initGpuSpanType(innerTyp)
    of ntyObject, ntyAlias, ntyTuple:
      # For aliases (type F = int), resolve to the underlying type.
      # Don't call resolveRecordFields on aliases of primitive types.
      if n.typeKind == ntyAlias and n.kind == nnkSym:
        return resolveType(reg, n.getTypeImpl())
      let impl = if n.kind == nnkTupleConstr: n # might actually _lose_ information if used getTypeImpl
                 else: n.getTypeImpl
      let flds = resolveRecordFields(reg, impl)
      let typName = assignTypeName(n) # might be an object construction
      result = initGpuObjectType(typName, flds)
    of ntyEnum:
      # atom records only flow as dropped static args, but the const type must resolve
      result = initGpuType(gtUint32)
    of ntyArray:
      if n.kind == nnkSym:
        let typ = resolveType(reg, getTypeImpl(n))
        reg.registerObjectType(typ)
        return typ
      if n.len == 3:
        doAssert n.len == 3, "Length was not 3, but: " & $n.len & " for node: " & n.treerepr
        doAssert n[0].strVal == "array"
        let len = resolveArrayLength(n)
        result = resolveGpuArrayType(reg, n[2], len)
      else: # an array literal
        let len = n.len
        result = resolveGpuArrayType(reg, n[0], len)
    of ntyGenericInvocation:
      error "Generic invocations are not supported in the GPU compiler"
    of ntyGenericInst:
      result = resolveInstantiatedType(reg, n)
    of ntyTypeDesc:
      # `getType` returns a BracketExpr(Sym "typeDesc", Sym "float32")
      result = resolveType(reg, n.getType[1])
    of ntyUnused2: # lent[T]
      doAssert n.kind == nnkBracketExpr and n[0].strVal == "lent", "ntyUnused2: " & $n.treerepr
      result = initGpuPtrType(resolveType(reg, n[1]), implicitPtr = false)
    of ntyProc:
      # a proc symbol processed as a type, fall back to the type instance
      let inst = n.getTypeInst()
      if inst.typeKind == ntyGenericInst:
        result = resolveInstantiatedType(reg, inst)
      else:
        result = initGpuGenericInstType(n.repr)
    of ntyStatic:
      # compile-time int values or type descriptors
      let inst = n.getTypeInst()
      if inst.typeKind == ntyGenericInst:
        result = resolveInstantiatedType(reg, inst)
      elif inst.typeKind in {ntyInt .. ntyUint64}:
        result = initGpuType(toGpuTypeKind inst.typeKind)
      else:
        result = resolveType(reg, inst)
    else:
      error "Type : " & $n.typeKind & " not supported yet: " & $n.treerepr

  # now add this type if not known
  reg.registerObjectType(result)

# ═══════════════════════════════════════════════════════════════════════
#  Type field resolution
# ═══════════════════════════════════════════════════════════════════════

proc resolveRecordFields*(reg: var TypeRegistry, node: NimNode): seq[GpuTypeField] =
  node.expectKind({nnkObjectTy, nnkTupleTy, nnkTupleConstr})
  case node.kind
  of nnkObjectTy:
    # empty objects (e.g. `type Int[V] = object`) have no recList
    if node.len > 2 and node[2].kind == nnkRecList:
      for ch in node[2]:
        if ch.kind == nnkRecCase:
          # variant section: branch fields only, the records flow as dropped static args
          for branch in ch[1 .. ^1]:
            if branch.kind == nnkOfBranch:
              for f in branch[1 .. ^1]:
                if f.kind == nnkIdentDefs and f.len == 3:
                  result.add GpuTypeField(name: f[0].strVal,
                                          typ: resolveType(reg, f[1]))
        else:
          doAssert ch.kind == nnkIdentDefs and ch.len == 3,
            "resolveRecordFields: unexpected recList child " & $ch.kind & ": " & $ch.treerepr
          result.add GpuTypeField(name: ch[0].strVal,
                                  typ: resolveType(reg, ch[1]))
  of nnkTupleTy, nnkTupleConstr:
    for fi in resolveTupleFields(node):
      result.add GpuTypeField(name: fi.name, typ: resolveType(reg, fi.typeNode))
  else:
    error "Unsupported type to parse fields from: " & $node.kind

# ═══════════════════════════════════════════════════════════════════════
#  Return type resolution
# ═══════════════════════════════════════════════════════════════════════

proc resolveProcReturnType*(reg: var TypeRegistry, params: NimNode): GpuType =
  params.expectKind(nnkFormalParams)
  let retType = params[0]
  if retType.kind == nnkEmpty:
    result = GpuType(kind: gtVoid)
  else:
    # may reference constants outside the macro, the generics logic resolves at call time
    result = resolveType(reg, retType)

# ═══════════════════════════════════════════════════════════════════════
#  Array type constructor
# ═══════════════════════════════════════════════════════════════════════

proc resolveGpuArrayType*(reg: var TypeRegistry, aTyp: NimNode, len: int): GpuType =
  result = GpuType(kind: gtArray, aTyp: resolveType(reg, aTyp), aLen: len)
