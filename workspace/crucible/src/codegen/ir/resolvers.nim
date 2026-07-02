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

proc determineArrayLength*(n: NimNode, allowArrayIdent: bool): int =
  ## If `allowArrayIdent` is true, we do not emit the error message when
  ## encountering an ident. This is the case for procs taking arrays
  ## with a static array where the constant comes from outside the
  ## macro. In that case we return `-1` indicating
  ##  `proc mdsRowShfNaive(r: int, v: array[SPONGE_WIDTH, BigInt]): BigInt {.device.} =`
  case n[1].kind
  of nnkSym:
    # resolved symbol — get the constant int value from its implementation
    if not allowArrayIdent:
      result = n[1].getImpl.intVal
      let msg = """Found array with length given by identifier: $#!
  You might want to create a typed template taking a typed parameter for this
  constant to force the Nim compiler to bind the symbol. In theory though this
  error should not appear anymore though, as we don't try to parse generic
  functions.""" % n[1].strVal
      raiseAssert msg
    else:
      result = -1
  of nnkIdent:
    # constant from outside the macro — let Nim inline at generation time
    if not allowArrayIdent:
      let msg = """Found array with length given by identifier: $#!
  You might want to create a typed template taking a typed parameter for this
  constant to force the Nim compiler to bind the symbol.""" % n[1].strVal
      raiseAssert msg
    else:
      result = -1
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

proc initGpuArrayType*(ctx: var GpuContext, aTyp: NimNode, len: int): GpuType
proc nimToGpuType*(ctx: var GpuContext, n: NimNode, allowToFail: bool = false, allowArrayIdent: bool = false): GpuType
proc getInnerPointerType*(ctx: var GpuContext, n: NimNode, allowToFail: bool = false, allowArrayIdent: bool = false): GpuType
proc parseGenericImpl*(ctx: var GpuContext, impl: NimNode, t: NimNode): GpuType
proc initGpuGenericInst*(ctx: var GpuContext, t: NimNode): GpuType
proc gpuTypeMaybeFromSymbol*(ctx: var GpuContext, t: NimNode, n: NimNode): GpuType
proc parseTypeFields*(ctx: var GpuContext, node: NimNode): seq[GpuTypeField]
proc getTypeName*(n: NimNode, recursedSym: bool = false): string
proc constructTupleTypeName*(n: NimNode): string

# ═══════════════════════════════════════════════════════════════════════
#  Generic type argument / implementation resolution
# ═══════════════════════════════════════════════════════════════════════

proc parseGenericArgs*(ctx: var GpuContext, t: NimNode): seq[GpuType] =
  case t.kind
  of nnkSym: return # no generic arguments
  of nnkBracketExpr:
    for i in 1 ..< t.len:
      result.add ctx.nimToGpuType(t[i])
  else:
    raiseAssert "Unexpected node kind in parseGenericArgs: " & $t.treerepr

proc parseGenericImpl*(ctx: var GpuContext, impl: NimNode, t: NimNode): GpuType =
  ## Given a type implementation (ObjectTy, DistinctTy, etc.), parse it as a generic instance.
  case impl.kind
  of nnkDistinctTy:
    ## XXX: assumes distinct of inbuilt type, not object!
    result = ctx.nimToGpuType(impl[0])
  of nnkObjectTy:
    doAssert impl.kind == nnkObjectTy, "Unexpected node kind for generic inst: " & $impl.treerepr
    ## XXX: use signature hash for type name? Otherwise will produce duplicates
    result = GpuType(kind: gtGenericInst, gName: t.repr)
    result.gFields = ctx.parseTypeFields(impl)
  of nnkStaticTy:
    # Static int — preserve the value for struct naming.
    result = GpuType(kind: gtStatic, builtin: true, sValue: int(impl[0].intVal))
  else:
    raiseAssert "Unexpected node kind in for genericInst: " & $impl.treerepr & " kind=" & $impl.kind

proc initGpuGenericInst*(ctx: var GpuContext, t: NimNode): GpuType =
  if t.typeKind notin {ntyGenericInst}:
    if t.typeKind != ntyNone:
      return ctx.nimToGpuType(t.getTypeInst())
    else:
      return GpuType(kind: gtGenericInst, gName: t.repr)
  # Note: nnkCall and nnkObjConstr branches were removed —
  # value expressions are now canonicalized at the nimToGpuType entry.
  case t.kind
  of nnkBracketExpr:
    result = GpuType(kind: gtGenericInst, gName: getGenericTypeName(t))
    result.gArgs = ctx.parseGenericArgs(t)
    let impl = t.getTypeImpl() # impl for the `gFields`
    result.gFields = ctx.parseTypeFields(impl)
  of nnkObjectTy:
    result = ctx.parseGenericImpl(t, t)
  of nnkSym:
    # All callers have already verified typeKind == ntyGenericInst.
    # Use getTypeInst() to resolve the full generic type (BracketExpr).
    # Avoid getImpl() which can return local variable IdentDefs from template expansions.
    let inst = t.getTypeInst()
    if inst.kind == nnkBracketExpr:
      result = ctx.initGpuGenericInst(inst)
    else:
      let impl = inst.getTypeImpl()
      result = ctx.parseGenericImpl(impl, t)
  else:
    raiseAssert "Unexpected t.kind for genericInst: " & $t.kind & " treerepr=" & $t.treerepr

# ═══════════════════════════════════════════════════════════════════════
#  Pointer / inner type resolution
# ═══════════════════════════════════════════════════════════════════════

proc getInnerPointerType*(ctx: var GpuContext, n: NimNode, allowToFail: bool = false, allowArrayIdent: bool = false): GpuType =
  doAssert n.typeKind in {ntyPtr, ntyPointer, ntyUncheckedArray, ntyVar} or n.kind == nnkPtrTy, "But was: " & $n.treerepr & " of typeKind " & $n.typeKind
  if n.typeKind in {ntyPointer, ntyUncheckedArray}:
    let typ = n.getTypeInst()
    doAssert typ.kind == nnkBracketExpr, "No, was: " & $typ.treerepr
    doAssert typ[0].kind in {nnkIdent, nnkSym}
    doAssert typ[0].strVal in ["ptr", "UncheckedArray"]
    result = ctx.nimToGpuType(typ[1], allowToFail, allowArrayIdent)
  elif n.kind == nnkPtrTy:
    result = ctx.nimToGpuType(n[0], allowToFail, allowArrayIdent)
  elif n.kind == nnkAddr:
    let typ = n.getTypeInst()
    result = ctx.getInnerPointerType(typ, allowToFail, allowArrayIdent)
  elif n.kind == nnkVarTy:
    # VarTy
    #   Sym "BigInt"
    result = ctx.nimToGpuType(n[0], allowToFail, allowArrayIdent)
  elif n.kind == nnkSym: # symbol of e.g. `ntyVar`
    result = ctx.nimToGpuType(n.getTypeInst(), allowToFail, allowArrayIdent)
  else:
    raiseAssert "Found what: " & $n.treerepr

# ═══════════════════════════════════════════════════════════════════════
#  Tuple type naming
# ═══════════════════════════════════════════════════════════════════════

proc constructTupleTypeName*(n: NimNode): string =
  ## XXX: overthink if this should really be here and not somewhere else
  ##
  ## Given a tuple, generate a name from the field names and types, e.g.
  ## `Tuple_lo_BaseType_hi_BaseType`
  ##
  ## XXX: `getTypeImpl.repr` is a hacky way to get a string name of the underlying
  ## type, e.g. for `BaseType`. Aliases would lead to duplicate tuple types.
  ## UPDATE: I changed the implementation to recurse into `getTypeName`
  ## TODO: verify that this did not break the tuple test & specifically check for aliases
  result = "Tuple_"
  doAssert n.kind in [nnkTupleTy, nnkTupleConstr]
  for i, ch in n:
    case ch.kind
    of nnkIdentDefs:
      let typName = ch[ch.len - 2].getTypeName() # second to last is type name of field(s)
      for j in 0 ..< ch.len - 2:
        # Example:
        # IdentDefs
        #   Ident "hi"
        #   Ident "lo"      `..< ch.len - 2 `
        #   Sym "BaseType"  `..< ch.len - 1`
        #   Empty           `..< ch.len`
        result.add ch[j].strVal & "_" & typName
        if j < ch.len - 3:
          result.add "_"
      if i < n.len - 1:
        result.add "_"
    of nnkExprColonExpr:
      # ExprColonExpr — two sub-cases:
      # ── Static tuple literal (value embedded in name):
      #   Sym "s0"
      #   IntLit 4
      # ── Type expression (type name embedded):
      #   Sym "hi"
      #   Infix
      #     Sym "shr"
      #     Sym "n"
      #     IntLit 16
      doAssert ch[0].kind == nnkSym, "Not a symbol, but: " & $ch.treerepr
      if ch[1].kind in {nnkIntLit, nnkUIntLit}:
        # Use the actual integer value in the name
        result.add ch[0].strVal & "_" & $ch[1].intVal
      else:
        let typName = ch[1].getTypeName()
        result.add ch[0].strVal & "_" & typName
      if i < n.len - 1:
        result.add "_"
    of nnkSym:
      # TupleConstr
      #   Sym "BaseType" <-- e.g. here
      #   Sym "BaseType"
      let typName = ch.getTypeName()
      result.add "Field" & $i & "_" & typName
      if i < n.len - 1:
        result.add "_"
    else:
      # An object constructor child inside the tuple — e.g.
      #   ObjConstr
      #     BracketExpr
      #       Sym "MyInt"
      #       IntLit 4
      #     ExprColonExpr
      #       Sym "data"
      #       Bracket
      #         ...
      # -> resolve via getTypeInst() instead of getTypeImpl()
      #    to avoid leaking the ObjectTy repr into the C struct name.
      let childInst = ch.getTypeInst()
      let typName = childInst.getTypeName()
      result.add "Field" & $i & "_" & typName
      if i < n.len - 1:
        result.add "_"

proc getTypeName*(n: NimNode, recursedSym: bool = false): string =
  ## Returns the name of the type
  case n.kind
  of nnkIdent: result = n.strVal
  of nnkSym:
    if recursedSym:
      result = n.strVal
    else:
      result = n.getTypeInst.getTypeName(true)
  of nnkObjConstr:
    if n[0].kind == nnkEmpty:
      result = n.getTypeInst.strVal
    else:
      result = n[0].strVal # type is the first node
  of nnkObjectTy:
    # Anonymous object type — use its repr as a fallback name.
    result = n.repr
  of nnkTupleTy, nnkTupleConstr:
    result = constructTupleTypeName(n)
  of nnkBracketExpr:
    # construct a type name `Foo_Bar_Baz`
    for i, ch in n:
      result.add ch.getTypeName()
      if i < n.len - 1:
        result.add "_"
  of nnkIntLit:
    result = $n.intVal
  of nnkUIntLit:
    result = $n.intVal
  else: raiseAssert "Unexpected node in `getTypeName`: " & $n.treerepr

# ═══════════════════════════════════════════════════════════════════════
#  Main type resolver
# ═══════════════════════════════════════════════════════════════════════

proc nimToGpuType*(ctx: var GpuContext, n: NimNode, allowToFail: bool = false, allowArrayIdent: bool = false): GpuType =
  ## Maps a Nim type to a type on the GPU
  ##
  ## If `allowToFail` is `true`, we return `GpuType(kind: gtVoid)` in cases
  ## where we would otherwise raise. This is so that in some cases where
  ## we only _attempt_ to determine a type, we can do so safely.
  template addAndReturn(arg: untyped): untyped =
    ctx.maybeAddType(arg)
    return arg

  case n.kind
  of nnkIdentDefs: # extract type for let / var based on explicit or implicit type
    if n[n.len - 2].kind != nnkEmpty: # explicit type
      result = ctx.nimToGpuType(n[n.len - 2], allowToFail, allowArrayIdent)
    else: # take from last element
      result = ctx.nimToGpuType(n[n.len - 1].getTypeInst(), allowToFail, allowArrayIdent)
  of nnkConstDef:
    if n[1].kind != nnkEmpty: # has an explicit type
      result = ctx.nimToGpuType(n[1], allowToFail, allowArrayIdent)
    else:
      result = ctx.nimToGpuType(n[2], allowToFail, allowArrayIdent) # derive from the RHS literal
  of nnkIntLit, nnkUIntLit:
    result = GpuType(kind: gtStatic, builtin: true, sValue: int(n.intVal))
    ctx.maybeAddType(result)
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
      result = initGpuPtrType(ctx.getInnerPointerType(n, allowToFail, allowArrayIdent), implicitPtr = false)
    of ntyVar:
      result = initGpuPtrType(ctx.getInnerPointerType(n, allowToFail, allowArrayIdent), implicitPtr = true)
    of ntyPointer:
      result = initGpuVoidPtr()
    of ntyUncheckedArray:
      ## Note: this is just the internal type of the array. It is only a pointer due to
      ## `ptr UncheckedArray[T]`. We simply remove the `UncheckedArray` part.
      result = initGpuUAType(ctx.getInnerPointerType(n, allowToFail, allowArrayIdent))
    of ntyObject, ntyAlias, ntyTuple:
      # For aliases (type F = int), resolve to the underlying type.
      # Don't call parseTypeFields on aliases of primitive types.
      if n.typeKind == ntyAlias and n.kind == nnkSym:
        return ctx.nimToGpuType(n.getTypeImpl())
      let impl = if n.kind == nnkTupleConstr: n # might actually _lose_ information if used getTypeImpl
                 else: n.getTypeImpl
      let flds = ctx.parseTypeFields(impl)
      let typName = getTypeName(n) # might be an object construction
      result = initGpuObjectType(typName, flds)
    of ntyArray:
      # For a generic, static array type, e.g.:
      if n.kind == nnkSym:
        addAndReturn ctx.nimToGpuType(getTypeImpl(n), allowToFail, allowArrayIdent)
      if n.len == 3:
        # BracketExpr
        #   Sym "array"
        #   Ident "N"
        #   Sym "uint32"
        doAssert n.len == 3, "Length was not 3, but: " & $n.len & " for node: " & n.treerepr
        doAssert n[0].strVal == "array"
        let len = determineArrayLength(n, allowArrayIdent)
        if len < 0:
          # indicates we found an array with an ident, e.g.
          # BracketExpr
          #   Sym "array"
          #   Ident "SPONGE_WIDTH"
          #   Sym "BigInt"
          return GpuType(kind: gtInvalid)
        else:
          result = ctx.initGpuArrayType(n[2], len)
      else:
        # just an array literal
        # Bracket
        #   UIntLit 2013265921
        let len = n.len
        result = ctx.initGpuArrayType(n[0], len)
    of ntyGenericInvocation:
      result = initGpuType(gtInvalid)
      error("Generic invocations are not supported in the GPU compiler")
    of ntyGenericInst:
      result = ctx.initGpuGenericInst(n)
    of ntyTypeDesc:
      # `getType` returns a `BracketExpr` of eg:
      # BracketExpr
      #   Sym "typeDesc"
      #   Sym "float32"
      result = ctx.nimToGpuType(n.getType[1], allowToFail, allowArrayIdent) # for a type desc we need to recurse using the type of it
    of ntyUnused2:
      # BracketExpr
      #   Sym "lent"
      #   Sym "BigInt"
      doAssert n.kind == nnkBracketExpr and n[0].strVal == "lent", "ntyUnused2: " & $n.treerepr
      result = initGpuPtrType(ctx.nimToGpuType(n[1]), implicitPtr = false)
    of ntyProc:
      # Procedure types can't be translated to CUDA C directly.
      # They appear when Crucible encounters a proc definition-symbol
      # being processed as a type. Fall back to the type instance.
      let inst = n.getTypeInst()
      if inst.typeKind == ntyGenericInst:
        result = ctx.initGpuGenericInst(inst)
      else:
        result = GpuType(kind: gtGenericInst, gName: n.repr)
    of ntyStatic:
      # Static types — might be compile-time int values or type descriptors.
      # Fall back to type instance to resolve.
      let inst = n.getTypeInst()
      if inst.typeKind == ntyGenericInst:
        result = ctx.initGpuGenericInst(inst)
      elif inst.typeKind in {ntyInt .. ntyUint64}:
        result = initGpuType(toGpuTypeKind inst.typeKind)
      else:
        result = ctx.nimToGpuType(inst)
    else:
      if allowToFail:
        result = GpuType(kind: gtVoid)
      else:
        raiseAssert "Type : " & $n.typeKind & " not supported yet: " & $n.treerepr

  # now add this type if not known
  ctx.maybeAddType(result)

# ═══════════════════════════════════════════════════════════════════════
#  Type field resolution
# ═══════════════════════════════════════════════════════════════════════

proc parseTypeFields*(ctx: var GpuContext, node: NimNode): seq[GpuTypeField] =
  case node.kind
  of nnkObjectTy:
    # Empty objects (e.g., `type Int[V] = object`) have no recList.
    if node.len > 2 and node[2].kind == nnkRecList:
      for ch in node[2]:
        doAssert ch.kind == nnkIdentDefs and ch.len == 3
        result.add GpuTypeField(name: ch[0].strVal,
                                typ: ctx.nimToGpuType(ch[1]))
  of nnkTupleTy:
    for ch in node:
      doAssert ch.kind == nnkIdentDefs and ch.len == 3
      result.add GpuTypeField(name: ch[0].strVal,
                              typ: ctx.nimToGpuType(ch[1]))
  of nnkTupleConstr:
    # TupleConstr
    #   Sym "BaseType"
    #   Sym "BaseType"
    for i, ch in node:
      case ch.kind
      of nnkSym:
        result.add GpuTypeField(name: "Field" & $i,
                                typ: ctx.nimToGpuType(ch))
      of nnkExprColonExpr:
        result.add GpuTypeField(name: ch[0].strVal,
                                typ: ctx.nimToGpuType(ch[1]))
      of nnkBracketExpr:
        # E.g. `Int[128]` inside a tuple type constructor.
        # Resolve the type directly from the bracket expression.
        result.add GpuTypeField(name: "Field" & $i,
                                typ: ctx.nimToGpuType(ch))
      else:
        # Unexpected child in tuple type constructor.
        # Resolve the type directly from the child node.
        result.add GpuTypeField(name: "Field" & $i,
                                typ: ctx.nimToGpuType(ch))
  else:
    raiseAssert "Unsupported type to parse fields from: " & $node.kind

# ═══════════════════════════════════════════════════════════════════════
#  Symbol-based type resolution helpers
# ═══════════════════════════════════════════════════════════════════════

proc gpuTypeMaybeFromSymbol*(ctx: var GpuContext, t: NimNode, n: NimNode): GpuType =
  ## Returns the type from a given Nim node `t` representing a type.
  ## If that fails due to an identifier in the type, we instead try
  ## to look up the type from the associated symbol, `n`.
  result = ctx.nimToGpuType(t, allowArrayIdent = true)
  if result.kind == gtInvalid:
    # an existing symbol cannot be `void` by definition, then it wouldn't be a symbol. Means
    # `allowArrayIdent` triggered due to an ident in the type. Use symbol for type instead
    result = ctx.nimToGpuType(n.getTypeInst)

proc parseProcReturnType*(ctx: var GpuContext, params: NimNode): GpuType =
  ## Returns the return type of the given procedure from the `params` node
  ## of type `nnkFormalParams`.
  doAssert params.kind == nnkFormalParams, "Argument is not FormalParams, but: " & $params.treerepr
  let retType = params[0] # arg 0 is return type
  if retType.kind == nnkEmpty:
    result = GpuType(kind: gtVoid) # actual void return
  else:
    # attempt to get type. If fails, we need to wait for a caller to this function to get types
    # (e.g. returns something like `array[FOO, BigInt]` where `FOO` is a constant defined outside
    # the macro. We then rely on our generics logic to later look this up when called
    result = ctx.nimToGpuType(retType, allowArrayIdent = true)
    if result.kind == gtVoid:
      result = GpuType(kind: gtInvalid)

# ═══════════════════════════════════════════════════════════════════════
#  Array type constructor (needs nimToGpuType, stays here)
# ═══════════════════════════════════════════════════════════════════════

proc initGpuArrayType*(ctx: var GpuContext, aTyp: NimNode, len: int): GpuType =
  ## Construct an statically sized array type
  result = GpuType(kind: gtArray, aTyp: ctx.nimToGpuType(aTyp), aLen: len)
