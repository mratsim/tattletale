# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Nim types to CirType: the frontend boundary translation.
##
## Each Nim type is translated exactly once to the closed CirType lattice
## (ADR 0001), cached by the type's `getTypeInst` representation.
##
## Struct names derive from the canonical type identity:
##
## - readable base: the canonical generic instantiation's name
## - suffix: a 7-character base58 hash of the canonical type text
## - a type alias and the instantiation it names share one struct
## Widths use the crucible convention: Nim `int`/`uint` lower to 32 bits.
##
## Rejections go into `err` as the first refusal message, the walk unwinds
## through the err guards, matching the `cirv2_verify` rejection messages.

import ../ir/cirv2_lang
import ../ir/cirv2_build
import std/[macros, tables, strutils, hashes]

type
  CirTypeResolver* = object
    ## Nim-to-CirType translation state: the cache, the nominal struct set,
    ## and the first refusal message.
    cache: Table[string, CirType]
    ## Cached types keyed by the Nim type's `getTypeInst` representation.
    structs*: seq[CirType]
    ## Nominal struct types declared by the module, in discovery order.
    err*: string
    ## First refusal message, "" while the walk is clean.

proc resolveCirType*(r: var CirTypeResolver, n: NimNode, ptrSpace = caDevice): CirType
  ## Forward declaration, the field walkers recurse into field types.

proc fail*(r: var CirTypeResolver, detail: string) =
  ## Records the first refusal. Later refusals keep the first.
  if r.err.len == 0:
    r.err = "cirv2 frontend: " & detail

func sanitizeIdent*(s: string): string =
  ## Returns `s` with every character invalid in a target identifier replaced
  ## by an underscore: Nim gensym backticks, backslashes, and the punctuation
  ## of structural type text (`(int, int)` derives `_int__int_`).
  result = newStringOfCap(s.len)
  for c in s:
    result.add(if c in {'a'..'z', 'A'..'Z', '0'..'9', '_'}: c else: '_')

func mangleOperatorName*(s: string): string =
  ## Returns `s` with operator punctuation replaced by words:
  ## `*=` becomes `star_eq`, `+` becomes `plus`.
  ##
  ## Identifier characters pass through unchanged, any other character
  ## becomes `X` plus its two-digit hex code.
  ##
  ## Callee names use this mapping: an operator is not a legal identifier
  ## in any target language, and distinct operator overloads over the same
  ## operand types keep distinct callee names.
  result = newStringOfCap(s.len + 1)
  if s.len > 0 and s[0] in Digits:
    result.add "X"
  template special(word: string) =
    if result.len > 0 and result[^1] != '_':
      result.add '_'
    result.add word
  for c in s:
    case c
    of 'a'..'z', '0'..'9', 'A'..'Z', '_':
      result.add c
    of '$': special "dollar"
    of '%': special "percent"
    of '&': special "amp"
    of '^': special "roof"
    of '!': special "emark"
    of '?': special "qmark"
    of '*': special "star"
    of '+': special "plus"
    of '-': special "minus"
    of '/': special "slash"
    of '\\': special "backslash"
    of '=': special "eq"
    of '<': special "lt"
    of '>': special "gt"
    of '~': special "tilde"
    of ':': special "colon"
    of '.': special "dot"
    of '@': special "at"
    of '|': special "bar"
    else:
      special "X" & toHex(ord(c), 2)

func deriveStructName*(n: NimNode): string =
  ## Returns the readable base of a struct name: a generic instantiation
  ## contributes its base name and each argument's name.
  ##
  ## `resolveStructType` appends the base58 hash of the canonical type text.
  ##
  ## Example:
  ##   TensorView[float32, Shape, Stride] -> "TensorView_float32_Shape_Stride"
  ##
  ## Example:
  ##   TensorView[float32, Shape, Stride] -> "TensorView_float32_Shape_Stride"
  case n.kind
  of nnkSym, nnkIdent:
    result = sanitizeIdent(n.strVal)
  of nnkBracketExpr:
    result = deriveStructName(n[0])
    for i in 1 ..< n.len:
      if n[i].kind in nnkIntLit .. nnkUInt64Lit:
        # v1 name compatibility: static integer arguments join without a separator, `FixMe[8]` derives `FixMe8`
        result.add(deriveStructName(n[i]))
      else:
        result.add("_" & deriveStructName(n[i]))
  of nnkObjectTy, nnkTupleTy, nnkTupleConstr:
    result = sanitizeIdent(n.repr)
  else:
    result = sanitizeIdent(n.repr)

func evalConstInt*(r: var CirTypeResolver, n: NimNode): int64 =
  ## Returns the compile-time integer value of a typed node: a literal, a const symbol, or a static arithmetic expression over those.
  ## Example:
  ##   a node for `128'i32` or for `chunk * 4` over consts evaluates to 128 or the product
  case n.kind
  of nnkIntLit .. nnkUInt64Lit:
    result = n.intVal
  of nnkSym:
    let impl = n.getImpl()
    if impl.kind == nnkConstDef and impl[2].kind in nnkIntLit .. nnkUInt64Lit:
      result = impl[2].intVal
    else:
      r.fail("compile-time value expected, got symbol " & n.repr &
        " whose value is not a literal")
      result = 0
  of nnkPar:
    if n.len == 1:
      result = r.evalConstInt(n[0])
    else:
      r.fail("compile-time value expected, got a parenthesized list")
      result = 0
  of nnkHiddenStdConv, nnkHiddenSubConv:
    if n.len >= 2:
      result = r.evalConstInt(n[1])
    else:
      r.fail("compile-time value expected, got a bare conversion node")
      result = 0
  of nnkInfix:
    if n.len != 3:
      r.fail("compile-time value expected, got the infix node " & n.repr)
      return 0
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
      r.fail("compile-time arithmetic expected, got operator " & n[0].strVal)
      result = 0
  else:
    r.fail("compile-time value expected, got node kind " & $n.kind)
    result = 0

func fieldSpaceAt(valueSpaces: Table[string, CirAddrSpace], name: string): CirAddrSpace =
  ## Pointer space a construction proves for the field `name`, `caDevice` when the construction names no space for it.
  valueSpaces.getOrDefault(name, caDevice)

proc deriveStructFields(r: var CirTypeResolver, impl: NimNode,
    valueSpaces: Table[string, CirAddrSpace] = initTable[string, CirAddrSpace]()): seq[CirField] =
  ## Field list of an object or tuple implementation node.
  ##
  ## `valueSpaces` maps a field name to the pointer space its construction
  ## value proves: a pointer field instantiates under that space.
  ## Every other field resolves under `caDevice`.
  case impl.kind
  of nnkObjectTy:
    if impl.len > 2 and impl[2].kind == nnkRecList:
      for ch in impl[2]:
        if ch.kind == nnkRecCase:
          r.fail("case object: variant records have no C representation")
          return @[]
        if ch.kind != nnkIdentDefs:
          r.fail("struct field: unexpected record child " & $ch.kind)
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
  ## reads `TensorOwned[float32, Sh, St]`. The CIRv2 struct keeps the aliased
  ## name so one alias and its target name the same struct.
  ##
  ## Only the arity-matched form rewrites. Anything else returns the input.
  if n.kind == nnkBracketExpr and n.len > 1 and n[0].kind == nnkSym:
    let rhs = n[0].getTypeImpl()
    if rhs.kind == nnkBracketExpr and rhs.len == n.len and rhs[0].kind == nnkSym:
      result = nnkBracketExpr.newTree(rhs[0])
      for i in 1 ..< n.len:
        result.add(n[i])
      return
  return n

func canonicalTypeNode(n: NimNode): NimNode =
  ## Returns the canonical type node: alias symbols replaced by the types they name, so one type identity derives one struct name.
  ##
  ## Example:
  ##   `V2F` aliasing `Vec2[float32]` reads back as `Vec2[float32]`
  ##
  ## Contract:
  ## - a symbol unwraps when its instantiated kind is a generic instantiation and its definition names a type on its right-hand side
  ## - a directly named instantiation, a plain object type, and a builtin scalar keep their own nodes
  ## - bracket arguments canonicalize recursively
  ## - the head symbol stays: an alias head with unbound generic parameters rewrites through `followAliasHead`
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

const Base58Alphabet* = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  ## Base58 alphabet, no 0/O/I/l for readability in emitted device code.

func base58Short*(h: uint64): string =
  ## Returns the 7-character base58 encoding of `h`, left-padded with `1` while `h` is small.
  ## 58^7 is about 2.2e12: enough collision avoidance for one compilation unit's struct names.
  var n = h
  result = newString(7)
  for i in countdown(6, 0):
    result[i] = Base58Alphabet[int(n mod 58)]
    n = n div 58
    if n == 0 and i == 0:
      break
    if n == 0:
      for j in 0 ..< i:
        result[j] = '1'
      break

func typeIdentityHash*(s: string): uint64 =
  ## Returns the 64-bit hash of a canonical type text: FNV-1a accumulation plus a splitmix64 finalizer.
  ## The value distributes uniformly over all 64 bits so the base58 digits never cluster on short inputs.
  var h: uint64 = 0xcbf29ce484222325'u64
  for c in s:
    h = (h xor uint64(ord(c))) * 0x100000001b3'u64
  h = (h xor (h shr 30)) * 0xbf58476d1ce4e5b9'u64
  h = (h xor (h shr 27)) * 0x94d049bb133111eb'u64
  h = h xor (h shr 31)
  if h == 0:
    h = 1
  result = h

proc resolveStructType*(r: var CirTypeResolver, n: NimNode,
    valueSpaces: Table[string, CirAddrSpace] = initTable[string, CirAddrSpace]()): CirType =
  ## Nominal struct translation for an object, tuple, or generic instantiation
  ## type node, deduplicated by canonical type identity.
  ##
  ## Struct name:
  ##
  ## - base: `deriveStructName` of the canonical type node
  ## - suffix: base58 hash of the canonical type text
  ##
  ## An alias and the instantiation it names produce one struct, distinct instantiations produce distinct names.
  ##
  ## Identity key is the canonical type text, already the resolver's dedup key.
  ## Nim symbol ids cannot serve as identity:
  ##
  ## - an alias and the instantiation it names carry different ids
  ## - distinct instantiations of one generic can share one id
  let canon = followAliasHead(canonicalTypeNode(n))
  # one instantiation per pointer-space combination over the fields:
  # threadgroup-pointer constructions instantiate a struct that is separate
  # from the device instantiation
  var cacheKey = canon.repr
  for name, spc in valueSpaces:
    if spc != caDevice:
      cacheKey &= "/" & name & "=" & $spc
  if r.cache.hasKey(cacheKey):
    return r.cache[cacheKey]
  let impl = n.getTypeImpl()
  let fields = r.deriveStructFields(impl, valueSpaces)
  if r.err.len > 0:
    return nil
  let name = deriveStructName(canon) & "_" &
    base58Short(typeIdentityHash(cacheKey))
  result = tStruct(name, fields).withOrigin(canon)
  r.cache[cacheKey] = result
  r.structs.add(result)

func isCompileTimeFormal(t: NimNode): bool =
  ## True for `static T` (either node layout), for `typedesc`, and for any
  ## `static` formal whose instantiated type shows the bound value.
  ##
  ## The instantiated type node shows that value, a literal or an enum member.
  ## The parameter is compile-time only.
  if t.kind in nnkIntLit .. nnkUInt64Lit:
    return true
  if t.kind == nnkSym and t.symKind == nskEnumField:
    return true
  if t.kind in {nnkStaticTy, nnkBracketExpr, nnkCommand}:
    if t.len > 0 and t[0].repr in ["static", "typeDesc"]:
      return true
  if t.kind in {nnkIdent, nnkSym} and t.strVal == "typedesc":
    return true
  if t.kind == nnkBracketExpr and t.len > 0 and t[0].strVal == "typeDesc":
    return true
  return false

proc resolveProcType(r: var CirTypeResolver, procTy: NimNode): CirType =
  ## Proc type from a `ProcTy` node: `params[0]` is the return type,
  ## `params[1..^1]` are the parameters. Compile-time (`static`/`typedesc`)
  ## parameters are erased here, the closure never passes them.
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
  ## Translates one Nim type node to CirType, cached.
  ##
  ## `ptrSpace` names the memory class of the pointer kinds this type node
  ## may contain: kernel parameters pass caDevice, body-side `addr` results
  ## construct their own caThread pointer.
  ##
  ## Example:
  ##   var r: CirTypeResolver
  ##   let f32 = r.resolveCirType(newSymNode(...))  # float32 -> (f +32)
  if r.err.len > 0:
    return nil
  if n.isNil:
    r.fail("type node is missing")
    return nil
  if n.kind == nnkEmpty:
    # a missing return type is void
    return tVoid()
  if n.kind in {nnkStmtListType, nnkStmtListExpr}:
    # an inline type or value block, the resolved node is its last child
    return r.resolveCirType(n[^1], ptrSpace)
  let cacheKey = if n.typeKind == ntyProc:
    # proc type's instantiation view erases the static formals and would share one cache entry across distinct instantiations
    block:
      var acc = "proc/"
      for f in n[0]:
        if f.kind == nnkIdentDefs and f.len >= 2:
          acc.add(f[f.len - 2].getTypeInst().repr)
        else:
          acc.add(f.repr)
      acc & "/" & $ptrSpace
  else:
    n.getTypeInst().repr & "/" & $ptrSpace
  if r.cache.hasKey(cacheKey):
    return r.cache[cacheKey]
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
    let len = r.evalConstInt(arr[1])
    if r.err.len > 0:
      return nil
    result = tArray(int(len), r.resolveCirType(arr[2], ptrSpace))
  of ntyEnum:
    # enums lower as their 32-bit storage, no enum type in the lattice
    result = tUint(32)
  of ntyDistinct:
    result = r.resolveCirType(n.getTypeImpl()[0], ptrSpace)
  of ntyProc:
    result = r.resolveProcType(n.getTypeInst())
  of ntyTypeDesc:
    result = r.resolveCirType(n.getType()[1], ptrSpace)
  of ntyStatic:
    result = r.resolveCirType(n.getTypeInst(), ptrSpace)
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
  if result != nil:
    r.cache[cacheKey] = result
