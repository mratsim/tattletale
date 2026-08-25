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
import ../builtins/builtins_gpu_types

# ═══════════════════════════════════════════════════════════════════════
#  Generic type name utilities
# ═══════════════════════════════════════════════════════════════════════

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
      error "Unsupport type so far: " & $t.treerepr & " of impl: " & $impl.treerepr
  else:
    result = t

proc toGpuTypeKind*(t: NimNode): GpuTypeKind =
  result = t.unpackGenericInst().typeKind.toGpuTypeKind()

# ═══════════════════════════════════════════════════════════════════════
#  Array length resolution
# ═══════════════════════════════════════════════════════════════════════

proc evalConstInt(n: NimNode): int =
  ## Evaluates a compile-time integer expression (the array-length forms
  ## `getTypeInst` produces: literals, const syms, and infix arithmetic
  ## like `32 * 16` or the `0 .. 15` range).
  case n.kind
  of nnkIntLit, nnkUIntLit:
    result = n.intVal
  of nnkSym:
    # const symbol — value from its implementation
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

proc resolveArrayLength*(n: NimNode): int =
  ## Returns the length of a static array type.
  ## Callers must pass `getTypeInst()` output — idents are never expected.
  result = evalConstInt(n[1])

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
  ## Shared traversal of the anonymous tuple type in any of its AST spellings:
  ## nnkTupleTy / nnkTupleConstr and the nnkBracketExpr `tuple[...]` full form.
  ## Called by both resolveRecordFields and assignTypeName.
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
  ## Canonical struct name for an anonymous tuple type: `Tuple_` + one
  ## `Name_TypeName` segment per field. Every AST spelling of the same tuple
  ## type names through this single function.
  result = "Tuple_"
  for i, fi in fields:
    if i > 0:
      result.add "_"
    result.add fi.name & "_" & fi.typeNode.assignTypeName()

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
    error "Unexpected node kind in for genericInst: " & $impl.treerepr & " kind=" & $impl.kind

# ═══════════════════════════════════════════════════════════════════════
#  Type-alias canonicalization
# ═══════════════════════════════════════════════════════════════════════
#
# A generic alias over a type application
# (`type RTileF32[R, C] = RtLeft[float32, R, C, APPLE_8x8x8_F32]`) keeps
# the alias's base name in the typed AST, since `getGenericTypeName`
# returns `RTileF32` for `RTileF32[32, 32]`. The emitted struct name
# would then differ from the canonical instantiation's
# (`RtLeft` plus its args), and a declaration and its use would disagree.
# The alias's own args also differ from the canonical ones, since the RHS
# fixes the element type and the atom. This resolution expands the alias,
# substituting the application args into the definition RHS, resolving
# that RHS as the canonical type application.

proc isAliasBase(n: NimNode): bool =
  ## True when the symbol is a type alias: its definition RHS is an expression
  ## (a type application, a template call, a named type), not a direct
  ## object/enum/distinct/tuple/ref definition and not a magic builtin
  ## (an empty RHS, e.g. `array`). Generic aliases over type applications
  ## canonicalize at resolve time. The alias name never reaches the emitted
  ## type name. A template-call RHS cannot be expanded without sem, so it
  ## errors loudly instead.
  if n.kind != nnkSym: return false
  let impl = n.getImpl()
  if impl.kind != nnkTypeDef: return false
  case impl[2].kind
  of nnkObjectTy, nnkEnumTy, nnkDistinctTy, nnkTupleTy, nnkRefTy, nnkEmpty:
    false
  else:
    true

proc substituteTypeSyms(n, fromN, toN: NimNode): NimNode =
  ## Returns a copy of `n` with every occurrence of the `fromN` sym
  ## replaced by `toN` (the alias's args substituting its params).
  if n.kind == nnkSym:
    return if n.eqIdent(fromN): toN else: n
  result = n.copyNimTree()
  for i in 0 ..< result.len:
    result[i] = substituteTypeSyms(result[i], fromN, toN)

proc substituteAliasArgs(typeDef, app: NimNode): NimNode =
  ## Returns the alias's RHS with the generic param syms replaced
  ## by the application's args, or by the param's declared default
  ## when the application omits the arg. A param with neither stays
  ## unbound, and the RHS resolution fails loudly below.
  ## Typed generic params arrive as syms. The untyped IdentDefs shape
  ## never reaches this proc.
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
        # The application omits this arg: substitute the param's
        # declared default (its impl in the typed AST). A param
        # with no default stays unbound, and the RHS resolution
        # fails loudly below. An integer literal default (an int length)
        # substitutes as its value. A bool or enum default substitutes
        # as its declared type, matching the explicit form (bool, u32).
        # An object default substitutes as its type (an atom const).
        # The resolver cannot take an ObjConstr value as a type argument.
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
  ## Resolves an alias's RHS (args substituted) as a type expression.
  ## `app` is the alias application or alias sym the RHS came from.
  ## `app.getTypeImpl` is the fully expanded object, the same object
  ## the canonical instantiation resolves its fields from.
  case rhs.kind
  of nnkBracketExpr:
    if isAliasBase(rhs[0]):
      # Nested alias: substitute its own args and recurse.
      let nested = substituteAliasArgs(rhs[0].getImpl(), rhs)
      return resolveAliasExpansion(reg, app, nested)
    let impl = app.getTypeImpl()
    if impl.kind notin {nnkObjectTy, nnkTupleTy, nnkTupleConstr}:
      # A non-object RHS (an array, a template call, a bare named type)
      # cannot be canonicalized to a named struct: error loudly instead
      # of emitting a fieldless generic instantiation.
      error "Alias RHS '" & $rhs.repr &
        "' does not expand to an object or tuple type. Spell the alias with the canonical type application"
    result = GpuType(kind: gtGenericInst, gName: getGenericTypeName(rhs))
    for i in 1 ..< rhs.len:
      result.gArgs.add resolveType(reg, rhs[i])
    result.gFields = resolveRecordFields(reg, impl)
  of nnkSym:
    if isAliasBase(rhs):
      # Nested alias through a named type: substitute its own args and recurse.
      let nested = substituteAliasArgs(rhs.getImpl(), rhs)
      return resolveAliasExpansion(reg, app, nested)
    # Alias to a named type: resolve the sym in the alias's own scope.
    result = resolveType(reg, rhs)
  of nnkCall, nnkCommand:
    # A template call cannot be expanded at resolve time: the expansion
    # needs the compiler's sem. Spell the alias with the canonical type
    # application instead, e.g. `RtLeft[float32, R, C, TileConfigFor(float32)]`
    # rather than `rt_l(float32, R, C)`.
    error "Alias RHS is a template call '" & $rhs[0].repr &
      "': spell the alias with the canonical type application, " &
      "e.g. RtLeft[float32, R, C, TileConfigFor(float32)]"
  else:
    result = resolveType(reg, rhs)

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
    if isAliasBase(t[0]):
      # An alias application spells the type under the alias's own name
      # and args. Resolve the alias's RHS with the args substituted.
      # The emitted type name then matches the canonical instantiation's.
      return resolveAliasExpansion(reg, t, substituteAliasArgs(t[0].getImpl(), t))
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
    if isAliasBase(t):
      # An alias sym spells the type with the alias's own name. Resolve
      # the alias's RHS (no args to substitute) for the canonical name.
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
    result = tupleTypeName(resolveTupleFields(n))
  of nnkBracketExpr:
    # `tuple[Int[1], Int[2]]` is the full AST form of an anonymous tuple type
    # (getTypeInst output, e.g. generic args of an object construction).
    # It must name identically to the nnkTupleConstr form.
    if n.len >= 2 and n[0].kind in {nnkSym, nnkIdent} and n[0].strVal == "tuple":
      result = tupleTypeName(resolveTupleFields(n))
    elif n[0].kind == nnkSym and n[0].strVal == "array":
      # `array[N, T]`: name from the length and the element type. Never
      # recurse through the `array` symbol — its type inst is the array
      # itself, which would loop.
      result = "Array_" & $resolveArrayLength(n) & "_" & n[2].assignTypeName()
    else:
      # construct a type name `Foo_Bar_Baz`
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
  of nnkStmtListExpr:
    # A template expansion left in a type-bracket arg
    # (e.g. the tile layer's `TileConfigFor(T)` defaulted atom) arrives
    # as a single expression wrapped in a StmtListExpr. The expression IS the arg.
    if n.len == 2 and n[0].kind == nnkEmpty:
      result = resolveType(reg, n[1])
    else:
      error "Unsupported StmtListExpr in type resolution: " & $n.treerepr
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
    # Builtin GPU types (float16, bfloat16): distinct uint16 tagged `{.builtin.}`,
    # resolved by name before the typeKind switch, since distinct types
    # have no nty* GPU mapping of their own.
    if n.kind == nnkSym and builtinGpuTypeKind(n) != gtVoid:
      return initGpuType(builtinGpuTypeKind(n))
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
    of ntyOpenArray:
      ## openArray[T] → gtSpan(kOpenArray, T)
      doAssert n.kind == nnkBracketExpr and n[0].strVal == "openArray",
        "ntyOpenArray: expected BracketExpr(openArray, T), got " & $n.treerepr
      let innerTyp = resolveType(reg, n[1])
      result = initGpuSpanType(kOpenArray, innerTyp)
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
      # Enum types resolve to their underlying integer storage. The atom
      # records carry enum fields (MmaDType) that never reach the emitted
      # C++ (the record only flows as a dropped static arg), but the const's
      # type must still resolve for the codegen to fold it.
      result = initGpuType(gtUint32)
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
    # Empty objects (e.g., `type Int[V] = object`) have no recList.
    if node.len > 2 and node[2].kind == nnkRecList:
      for ch in node[2]:
        if ch.kind == nnkRecCase:
          # A variant section (`case kind: ... of ...`): resolve the branch
          # fields, skip the discriminator plumbing. The atom records carry
          # variants but only flow as dropped static args.
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
