# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Type resolvers — Nim AST → GpuType / GpuAst metadata.
##
## These functions need access to Nim compiler APIs (getTypeInst, getTypeImpl, etc.)
## but do NOT call `toGpuAst`. They are called from the construction switch in `nim_to_gpu.nim`.

import std / [macros, strutils, sequtils, tables, sets]
import ./gpu_types
import ./gpu_type_constructors

# ═══════════════════════════════════════════════════════════════════════
#  Generic type name utilities
# ═══════════════════════════════════════════════════════════════════════

proc getGenericTypeName*(t: NimNode): string =
  ## Returns the base name of the generic type, i.e. for
  ## `Foo[Bar, Baz]` returns `Foo`.
  case t.kind
  of nnkSym: result = t.strVal
  of nnkBracketExpr: result = t[0].getGenericTypeName()
  else: raiseAssert "Unexpected node kind for generic instantiation type: " & $t.treerepr

proc unpackGenericInst*(t: NimNode): NimNode =
  let tKind = t.typeKind
  if tKind == ntyGenericInst:
    let impl = t.getTypeImpl()
    case impl.kind
    of nnkDistinctTy: # just skip the distinct
      result = impl[0]
    of nnkObjectTy, nnkEnumTy:
      result = t # keep object/enum types as-is
    else:
      raiseAssert "Unsupport type so far: " & $t.treerepr & " of impl: " & $impl.treerepr
  else:
    result = t

proc toGpuTypeKind*(t: NimNode): GpuTypeKind =
  result = t.unpackGenericInst().typeKind.toGpuTypeKind()

# ═══════════════════════════════════════════════════════════════════════
#  Array length resolution
# ═══════════════════════════════════════════════════════════════════════

proc resolveArrayLength*(n: NimNode): int =
  ## Returns the length of a static array type.
  ## Callers must pass `getTypeInst()` output — idents are never expected.
  case n[1].kind
  of nnkSym:
    # resolved symbol — get the constant int value from its implementation
    result = n[1].getImpl.intVal
  of nnkIdent:
    # Should never happen with getTypeInst() output
    raiseAssert "Unresolved ident in array length: " & $n[1].strVal
  else:
    case n[1].kind
    of nnkIntLit: result = n[1].intVal
    else:
      # E.g.
      # BracketExpr
      #   Sym "array"
      #   Infix
      #     Ident ".."
      #     IntLit 0
      #     IntLit 11
      #   Sym "BigInt"
      doAssert n[1].kind == nnkInfix, "No is: " & $n.treerepr
      doAssert n[1][1].kind == nnkIntLit, "No is: " & $n.treerepr
      doAssert n[1][1].intVal == 0, "No is: " & $n.treerepr
      result = n[1][2].intVal + 1

# ═══════════════════════════════════════════════════════════════════════
#  Forward declarations (defined below)
# ═══════════════════════════════════════════════════════════════════════

proc resolveGpuArrayType*(reg: var TypeRegistry, aTyp: NimNode, len: int): GpuType
proc resolveType*(reg: var TypeRegistry, n: NimNode): GpuType
proc resolveInnerPointerType*(reg: var TypeRegistry, n: NimNode): GpuType
proc resolveStructuralType*(reg: var TypeRegistry, impl: NimNode, t: NimNode): GpuType
proc resolveInstantiatedType*(reg: var TypeRegistry, t: NimNode): GpuType
proc resolveRecordFields*(reg: var TypeRegistry, node: NimNode): seq[GpuTypeField]
proc assignTypeName*(n: NimNode, recursedSym: bool = false): string

type
  FieldInfo = object
    name: string
    typeNode: NimNode

proc resolveTupleFields(node: NimNode): seq[FieldInfo] =
  ## Shared traversal of nnkTupleTy / nnkTupleConstr.
  ## Called by both resolveRecordFields and assignTypeName.
  node.expectKind({nnkTupleTy, nnkTupleConstr})
  for i, ch in node:
    case ch.kind
    of nnkIdentDefs:
      for j in 0 ..< ch.len - 2:
        result.add FieldInfo(name: ch[j].strVal, typeNode: ch[ch.len - 2])
    of nnkSym:
      result.add FieldInfo(name: "Field" & $i, typeNode: ch)
    of nnkExprColonExpr:
      result.add FieldInfo(name: ch[0].strVal, typeNode: ch[1])
    of nnkBracketExpr:
      # resolve via getTypeInst() to avoid leaking ObjectTy repr into struct name
      result.add FieldInfo(name: "Field" & $i, typeNode: ch.getTypeInst())
    else:
      # resolve via getTypeInst() to avoid leaking ObjectTy repr into struct name
      result.add FieldInfo(name: "Field" & $i, typeNode: ch.getTypeInst())

# ═══════════════════════════════════════════════════════════════════════
#  Generic type argument / implementation resolution
# ═══════════════════════════════════════════════════════════════════════


proc resolveStructuralType*(reg: var TypeRegistry, impl: NimNode, t: NimNode): GpuType =
  ## Given a type implementation (ObjectTy, DistinctTy, etc.), parse it as a generic instance.
  case impl.kind
  of nnkDistinctTy:
    ## XXX: assumes distinct of inbuilt type, not object!
    result = resolveType(reg, impl[0])
  of nnkObjectTy:
    doAssert impl.kind == nnkObjectTy, "Unexpected node kind for generic inst: " & $impl.treerepr
    ## XXX: use signature hash for type name? Otherwise will produce duplicates
    result = GpuType(kind: gtGenericInst, gName: t.repr)
    result.gFields = resolveRecordFields(reg, impl)
  of nnkStaticTy:
    # Static int — preserve the value for struct naming.
    result = GpuType(kind: gtStatic, builtin: true, sValue: int(impl[0].intVal))
  else:
    raiseAssert "Unexpected node kind in for genericInst: " & $impl.treerepr & " kind=" & $impl.kind

proc resolveInstantiatedType*(reg: var TypeRegistry, t: NimNode): GpuType =
  if t.typeKind notin {ntyGenericInst}:
    if t.typeKind != ntyNone:
      return resolveType(reg, t.getTypeInst())
    else:
      return GpuType(kind: gtGenericInst, gName: t.repr)
  # Note: nnkCall and nnkObjConstr branches were removed —
  # value expressions are now canonicalized at the resolveType entry.
  case t.kind
  of nnkBracketExpr:
    result = GpuType(kind: gtGenericInst, gName: getGenericTypeName(t))
    for i in 1 ..< t.len:
      result.gArgs.add resolveType(reg, t[i])
    let impl = t.getTypeImpl() # impl for the `gFields`
    result.gFields = resolveRecordFields(reg, impl)
  of nnkObjectTy:
    result = resolveStructuralType(reg, t, t)
  of nnkSym:
    # All callers have already verified typeKind == ntyGenericInst.
    # Use getTypeInst() to resolve the full generic type (BracketExpr).
    # Avoid getImpl() which can return local variable IdentDefs from template expansions.
    let inst = t.getTypeInst()
    if inst.kind == nnkBracketExpr:
      result = resolveInstantiatedType(reg, inst)
    else:
      let impl = inst.getTypeImpl()
      result = resolveStructuralType(reg, impl, t)
  else:
    raiseAssert "Unexpected t.kind for genericInst: " & $t.kind & " treerepr=" & $t.treerepr

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
    raiseAssert "Found what: " & $n.treerepr

# ═══════════════════════════════════════════════════════════════════════
#  Tuple type naming
# ═══════════════════════════════════════════════════════════════════════


proc assignTypeName*(n: NimNode, recursedSym: bool = false): string =
  ## Returns the name of the type
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
    # Anonymous object type — use its repr as a fallback name.
    result = n.repr
  of nnkTupleTy, nnkTupleConstr:
    result = "Tuple_"
    for i, fi in resolveTupleFields(n):
      if i > 0: result.add "_"
      let typName = fi.typeNode.assignTypeName()
      result.add fi.name & "_" & typName
  of nnkBracketExpr:
    # construct a type name `Foo_Bar_Baz`
    for i, ch in n:
      result.add ch.assignTypeName()
      if i < n.len - 1:
        result.add "_"
  of nnkIntLit:
    result = $n.intVal
  of nnkUIntLit:
    result = $n.intVal
  else: raiseAssert "Unexpected node in `assignTypeName`: " & $n.treerepr

# ═══════════════════════════════════════════════════════════════════════
#  Main type resolver
# ═══════════════════════════════════════════════════════════════════════

proc resolveType*(reg: var TypeRegistry, n: NimNode): GpuType =
  ## Maps a Nim type to a type on the GPU

  case n.kind
  of nnkIdentDefs: # extract type for let / var based on explicit or implicit type
    if n[n.len - 2].kind != nnkEmpty: # explicit type
      result = resolveType(reg, n[n.len - 2])
    else: # take from last element
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
  else:
    if n.kind == nnkEmpty: return initGpuType(gtVoid)
    # ── Type canonicalization ──
    # If n is a value expression (not a type node) that has a recognizable type,
    # resolve it to its actual type node. This prevents crashes when e.g.
    # w.layout.stride[0] (BracketExpr with DotExpr base) leaks into type resolution;
    # getTypeInst() returns Int[1], a clean type node.
    let n = block:
      # Canonicalize value expressions to their actual type nodes.
      # nnkBracketExpr with a Sym/BracketExpr base (e.g. Int[8], tuple types)
      # are legitimate type expressions — only DotExpr-base bracket expressions
      # are value expressions that need resolution.
      if n.kind notin {nnkSym, nnkIdent, nnkTupleTy, nnkObjectTy, nnkPtrTy, nnkTupleConstr} and
         (n.kind != nnkBracketExpr or n[0].kind notin {nnkSym, nnkBracketExpr}) and
         n.typeKind != ntyNone:
        n.getTypeInst()
      else:
        n
    # ── end canonicalization ──
    case n.typeKind
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
      ## Note: this is just the internal type of the array. It is only a pointer due to
      ## `ptr UncheckedArray[T]`. We simply remove the `UncheckedArray` part.
      result = initGpuUAType(resolveInnerPointerType(reg, n))
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
    of ntyArray:
      # For a generic, static array type, e.g.:
      if n.kind == nnkSym:
        let typ = resolveType(reg, getTypeImpl(n))
        reg.registerObjectType(typ)
        return typ
      if n.len == 3:
        # BracketExpr
        #   Sym "array"
        #   Ident "N"
        #   Sym "uint32"
        doAssert n.len == 3, "Length was not 3, but: " & $n.len & " for node: " & n.treerepr
        doAssert n[0].strVal == "array"
        let len = resolveArrayLength(n)
        result = resolveGpuArrayType(reg, n[2], len)
      else:
        # just an array literal
        # Bracket
        #   UIntLit 2013265921
        let len = n.len
        result = resolveGpuArrayType(reg, n[0], len)
    of ntyGenericInvocation:
      error "Generic invocations are not supported in the GPU compiler"
    of ntyGenericInst:
      result = resolveInstantiatedType(reg, n)
    of ntyTypeDesc:
      # `getType` returns a `BracketExpr` of eg:
      # BracketExpr
      #   Sym "typeDesc"
      #   Sym "float32"
      result = resolveType(reg, n.getType[1]) # for a type desc we need to recurse using the type of it
    of ntyUnused2:
      # BracketExpr
      #   Sym "lent"
      #   Sym "BigInt"
      doAssert n.kind == nnkBracketExpr and n[0].strVal == "lent", "ntyUnused2: " & $n.treerepr
      result = initGpuPtrType(resolveType(reg, n[1]), implicitPtr = false)
    of ntyProc:
      # Procedure types can't be translated to CUDA C directly.
      # They appear when Crucible encounters a proc definition-symbol
      # being processed as a type. Fall back to the type instance.
      let inst = n.getTypeInst()
      if inst.typeKind == ntyGenericInst:
        result = resolveInstantiatedType(reg, inst)
      else:
        result = GpuType(kind: gtGenericInst, gName: n.repr)
    of ntyStatic:
      # Static types — might be compile-time int values or type descriptors.
      # Fall back to type instance to resolve.
      let inst = n.getTypeInst()
      if inst.typeKind == ntyGenericInst:
        result = resolveInstantiatedType(reg, inst)
      elif inst.typeKind in {ntyInt .. ntyUint64}:
        result = initGpuType(toGpuTypeKind inst.typeKind)
      else:
        result = resolveType(reg, inst)
    else:
      raiseAssert "Type : " & $n.typeKind & " not supported yet: " & $n.treerepr

  # now add this type if not known
  reg.registerObjectType(result)

# ═══════════════════════════════════════════════════════════════════════
#  Type field resolution
# ═══════════════════════════════════════════════════════════════════════

proc resolveRecordFields*(reg: var TypeRegistry, node: NimNode): seq[GpuTypeField] =
  case node.kind
  of nnkObjectTy:
    # Empty objects (e.g., `type Int[V] = object`) have no recList.
    if node.len > 2 and node[2].kind == nnkRecList:
      for ch in node[2]:
        doAssert ch.kind == nnkIdentDefs and ch.len == 3
        result.add GpuTypeField(name: ch[0].strVal,
                                typ: resolveType(reg, ch[1]))
  of nnkTupleTy, nnkTupleConstr:
    for fi in resolveTupleFields(node):
      result.add GpuTypeField(name: fi.name, typ: resolveType(reg, fi.typeNode))
  else:
    raiseAssert "Unsupported type to parse fields from: " & $node.kind

# ═══════════════════════════════════════════════════════════════════════
#  Return type resolution
# ═══════════════════════════════════════════════════════════════════════

proc resolveProcReturnType*(reg: var TypeRegistry, params: NimNode): GpuType =
  ## Returns the return type of the given procedure from the `params` node
  ## of type `nnkFormalParams`.
  params.expectKind(nnkFormalParams)
  let retType = params[0] # arg 0 is return type
  if retType.kind == nnkEmpty:
    result = GpuType(kind: gtVoid) # actual void return
  else:
    # attempt to get type. If fails, we need to wait for a caller to this function to get types
    # (e.g. returns something like `array[FOO, BigInt]` where `FOO` is a constant defined outside
    # the macro. We then rely on our generics logic to later look this up when called
    result = resolveType(reg, retType)

# ═══════════════════════════════════════════════════════════════════════
#  Array type constructor (needs resolveType, stays here)
# ═══════════════════════════════════════════════════════════════════════

proc resolveGpuArrayType*(reg: var TypeRegistry, aTyp: NimNode, len: int): GpuType =
  ## Construct an statically sized array type
  result = GpuType(kind: gtArray, aTyp: resolveType(reg, aTyp), aLen: len)
