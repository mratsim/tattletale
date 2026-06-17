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


proc initGpuType*(kind: GpuTypeKind): GpuType =
  ## If `kind` is `gtPtr` `to` must be the type we point to
  if kind in [gtObject, gtPtr, gtArray]: raiseAssert "Objects/Pointers/Arrays must be constructed using `initGpuPtr/Object/ArrayType` "
  result = GpuType(kind: kind)

proc initGpuPtrType*(to: GpuType, implicitPtr: bool): GpuType =
  ## If `kind` is `gtPtr` `to` must be the type we point to
  if to.kind == gtInvalid: # this is not a valid type
    result = GpuType(kind: gtInvalid)
  else:
    result = GpuType(kind: gtPtr, to: to, implicit: implicitPtr)

proc initGpuUAType*(to: GpuType): GpuType =
  ## Initializes a GPU type for an unchecked array (ptr wraps this)
  if to.kind == gtInvalid: # this is not a valid type
    result = GpuType(kind: gtInvalid)
  else:
    result = GpuType(kind: gtUA, uaTo: to)

proc initGpuVoidPtr*(): GpuType =
  result = GpuType(kind: gtVoidPtr)

proc initGpuObjectType*(name: string, flds: seq[GpuTypeField]): GpuType =
  ## If `kind` is `gtPtr` `to` must be the type we point to
  result = GpuType(kind: gtObject, name: name, oFields: flds)

proc toTypeDef*(typ: GpuType): GpuAst =
  ## Converts a given object or generic instantiation type into an AST of a
  ## corresponding type def.
  # store the type instantiation
  result = GpuAst(kind: gpuTypeDef, tTyp: typ)
  case typ.kind
  of gtObject:      result.tFields = typ.oFields
  of gtGenericInst: result.tFields = typ.gFields
  else:
    raiseAssert "Type: " & $pretty(typ) & " is neither object type nor generic instantiation."

proc toGpuTypeKind*(t: NimTypeKind): GpuTypeKind =
  case t
  #of ntyBool, ntyChar:
    # , ntyEmpty, ntyAlias, ntyNil, ntyExpr, ntyStmt, ntyTypeDesc, ntyGenericInvocation, ntyGenericBody, ntyGenericInst, ntyGenericParam, ntyDistinct, ntyEnum, ntyOrdinal, ntyArray, ntyObject, ntyTuple, ntySet, ntyRange, ntyPtr, ntyRef, ntyVar, ntySequence, ntyProc,
  #of ntyPointer, ntyUncheckedArray, ntyOpenArray, ntyString, ntyCString
  # , ntyForward, ntyInt, ntyInt8,
  of ntyBool: gtBool
  of ntyInt16: gtInt16
  of ntyInt32: gtInt32
  of ntyInt64: gtInt64
  of ntyInt:   gtInt32 # `int` is always mapped to `int32` as that is the more "native" type on GPUs
  of ntyFloat: gtFloat64
  of ntyFloat32: gtFloat32
  of ntyFloat64: gtFloat64
  #of ntyFloat128: gtFloat128
  of ntyUInt: gtUint64
  of ntyUInt8: gtUint8
  of ntyUInt16: gtUint16
  of ntyUInt32: gtUint32
  of ntyUInt64: gtUint64
  of ntyString: gtString
  else:
    raiseAssert "Not supported yet: " & $t

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


proc determineArrayLength*(n: NimNode, allowArrayIdent: bool): int =
  ## If `allowArrayIdent` is true, we do not emit the error message when
  ## encountering an ident. This is the case for procs taking arrays
  ## with a static array where the constant comes from outside the
  ## macro. In that case we return `-1` indicating
  ##  `proc mdsRowShfNaive(r: int, v: array[SPONGE_WIDTH, BigInt]): BigInt {.device.} =`
  case n[1].kind
  of nnkSym:
    # resolved symbol — get the constant int value from its implementation
    result = n[1].getImpl.intVal
    if not allowArrayIdent:
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
      #doAssert n[1].kind == nnkIntLit, "No is: " & $n.treerepr
      doAssert n[1].kind == nnkInfix, "No is: " & $n.treerepr
      doAssert n[1][1].kind == nnkIntLit, "No is: " & $n.treerepr
      doAssert n[1][1].intVal == 0, "No is: " & $n.treerepr
      result = n[1][2].intVal + 1


proc assignOp*(op: string, isBoolean: bool): string =
  ## Returns the correct CUDA operation given the Nim operator.
  ## This is to replace things like `shl`, `div` or `mod`
  case op
  of "div": result = "/"
  of "+%": result = "+"
  of "-%": result = "-"
  of "*%": result = "*"
  of "mod": result = "%"
  of "shl": result = "<<"
  of "shr": result = ">>"
  of "and": result = if isBoolean: "&&" else: "&" # bitwise OR
  of "or":  result = if isBoolean: "||" else: "|" # bitwise OR
  of "xor": result = "^"
  else: result = op

proc assignPrefixOp*(op: string): string =
  ## Returns the correct CUDA operation given the Nim operator.
  case op
  of "not": result = "!"
  else: result = op


proc requiresMemcpy*(n: NimNode): bool =
  ## At the moment we only emit a `memcpy` statement for array types
  result = n.typeKind == ntyArray and n.kind != nnkBracket # need to emit a memcpy


proc isBuiltIn*(n: NimNode): bool =
  ## Checks if the given proc is a `{.builtin.}` (or if it is a Nim "built in"
  ## proc that uses `importc`, as we cannot emit those; they _need_ to have a
  ## WGSL / CUDA equivalent built in)
  doAssert n.kind in [nnkProcDef, nnkFuncDef], "Argument is not a proc: " & $n.treerepr
  for pragma in n.pragma:
    doAssert pragma.kind in [nnkIdent, nnkSym, nnkCall, nnkExprColonExpr], "Unexpected node kind: " & $pragma.treerepr
    let pragma = if pragma.kind in [nnkCall, nnkExprColonExpr]: pragma[0] else: pragma
    if pragma.strVal in ["builtin", "importc"]:
      return true


proc collectProcAttributes*(n: NimNode): set[GpuAttribute] =
  doAssert n.kind in [nnkPragma, nnkEmpty]
  if n.kind == nnkEmpty: return # no pragmas
  for pragma in n:
    doAssert pragma.kind in [nnkIdent, nnkSym, nnkCall, nnkExprColonExpr], "Unexpected node kind: " & $pragma.treerepr
    let pragma = if pragma.kind in [nnkCall, nnkExprColonExpr]: pragma[0] else: pragma
    case pragma.strVal
    of "device": result.incl attDevice
    of "global": result.incl attGlobal
    of "inline", "forceinline": result.incl attForceInline
    of "nimonly", "builtin":
      # used to fully ignore functions!
      return
    of "importc": # encountered if we analyze a proc from outside `cuda` scope
      return # this _should_ be a builtin function that has a counterpart in Nim, e.g. `math.ceil`
    of "varargs": # attached to some builtins, e.g. `printf` on CUDA backend
      continue
    of "magic":
      return
    of "noinit", "noInit":
      discard
    of "cudaName":
      continue # provides alternative name, not an attribute
    of "raises":
      discard
    # Common Nim pragmas that are not relevant for CUDA C codegen:
    of "noSideEffect", "nimcall", "closure", "inlne", "shallow":
      discard
    else:
      raiseAssert "Unexpected pragma for procs: " & $pragma.treerepr


proc collectAttributes*(n: NimNode): seq[GpuVarAttribute] =
  ## Collects all pragmas associated with the given variable.
  ## Takes the `nnkPragma` node of the `nnkIdentDefs` associated with it.
  # Example AST with multiple pragmas
  # IdentDefs
  #   PragmaExpr
  #     Sym "sharedMem"
  #     Pragma
  #       Sym "cuExtern"
  #       Sym "shared"
  #   BracketExpr
  #     Sym "array"
  #     IntLit 0
  #     Sym "BigInt"
  #   Empty
  doAssert n.kind == nnkPragma
  for pragma in n:
    doAssert pragma.kind in [nnkIdent, nnkSym], "Unexpected node kind: " & $pragma.treerepr
    # NOTE: We don't use `parseEnum`, because on the Nim side some of the attributes
    # do not match the CUDA string we need to emit, which is what the string value of
    # the `GpuVarAttribute` enum stores
    case pragma.strVal.normalize
    of "cuextern", "extern": result.add atvExtern
    of "shared": result.add atvShared
    of "private": result.add atvPrivate
    of "volatile": result.add atvVolatile
    of "constant": result.add atvConstant
    of "noinit": discard # XXX: ignore for now
    of "inject": discard # injected symbols from templates/macros
    of "gensym": discard # template-generated symbol
    else:
      raiseAssert "Unexpected pragma: " & $pragma.treerepr


proc stripPtrOrArrayType*(t: GpuType): GpuType =
  ## Strips any pointer or array type to return any struct / generic instantiation
  ## it might contain
  case t.kind
  of gtPtr:    result = stripPtrOrArrayType t.to
  of gtUA:     result = stripPtrOrArrayType t.uaTo
  of gtArray:  result = stripPtrOrArrayType t.aTyp
  else:        result = t


proc maybeAddType*(ctx: var GpuContext, typ: GpuType) =
  ## Adds the given type to the table of known types, if it is some kind of
  ## object type.
  ##
  ## XXX: What about aliases and distincts?
  let typ = typ.stripPtrOrArrayType() # get any underlying type
  if typ.kind in [gtObject, gtGenericInst] and typ notin ctx.types:
    ctx.types[typ] = toTypeDef(typ)

