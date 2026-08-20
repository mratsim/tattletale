# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GpuType constructors and helpers — no NimNode type resolution.
## These functions create or inspect GpuType/Ast values.
## They do NOT call getTypeInst / getTypeImpl / getImpl.

import std / [macros, strutils, tables]
import ./gpu_types

# ═══════════════════════════════════════════════════════════════════════
#  GpuType constructors
# ═══════════════════════════════════════════════════════════════════════

proc initGpuType*(kind: GpuTypeKind): GpuType =
  ## If `kind` is `gtPtr` `to` must be the type we point to
  if kind in [gtObject, gtPtr, gtArray, gtSpan]:
    raiseAssert "Objects/Pointers/Arrays/Spans must be constructed using dedicated constructors"
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
  ## Initializes an object/struct type
  result = GpuType(kind: gtObject, name: name, oFields: flds)

proc initGpuSpanType*(kind: GpuSpanKind, elemTyp: GpuType): GpuType =
  ## Initializes a span type (openArray/varargs) with the given element type
  if elemTyp.kind == gtInvalid:
    result = GpuType(kind: gtInvalid)
  else:
    result = GpuType(kind: gtSpan, sKind: kind, sElemTyp: elemTyp)

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

# ═══════════════════════════════════════════════════════════════════════
#  Type-kind mapping (Nim → GPU)
# ═══════════════════════════════════════════════════════════════════════

proc toGpuTypeKind*(t: NimTypeKind): GpuTypeKind =
  case t
  of ntyBool: gtBool
  of ntyInt16: gtInt16
  of ntyInt32: gtInt32
  of ntyInt64: gtInt64
  of ntyInt:   gtInt32 # `int` is always mapped to `int32` as that is the more "native" type on GPUs
  of ntyFloat: gtFloat64
  of ntyFloat32: gtFloat32
  of ntyFloat64: gtFloat64
  of ntyUInt: gtUint32 # `uint` is always mapped to `uint32` as that is the more "native" type on GPUs
  of ntyUInt8: gtUint8
  of ntyUInt16: gtUint16
  of ntyUInt32: gtUint32
  of ntyUInt64: gtUint64
  of ntyString: gtString
  else:
    raiseAssert "Not supported yet: " & $t

# ═══════════════════════════════════════════════════════════════════════
#  GpuType utilities
# ═══════════════════════════════════════════════════════════════════════

proc stripPtrOrArrayType*(t: GpuType): GpuType =
  ## Strips any pointer or array type to return any struct / generic instantiation
  ## it might contain
  case t.kind
  of gtPtr:    result = stripPtrOrArrayType t.to
  of gtUA:     result = stripPtrOrArrayType t.uaTo
  of gtArray:  result = stripPtrOrArrayType t.aTyp
  of gtSpan:   result = stripPtrOrArrayType t.sElemTyp
  else:        result = t


proc registerObjectType*(reg: var TypeRegistry, typ: GpuType) =
  ## Adds the given type to the table of known types, if it is some kind of
  ## object type.
  ##
  ## XXX: What about aliases and distincts?
  let typ = typ.stripPtrOrArrayType() # get any underlying type
  if typ.kind in [gtObject, gtGenericInst] and typ notin reg.types:
    reg.types[typ] = toTypeDef(typ)

# ═══════════════════════════════════════════════════════════════════════
#  Nim AST utilities
# ═══════════════════════════════════════════════════════════════════════


proc getGenericTypeName*(t: NimNode): string =
  ## Returns the base name of the generic type, i.e. for
  ## `Foo[Bar, Baz]` returns `Foo`.
  # Recursion handles nested bracket expressions by peeling outer layers
  # until it hits the root Sym, e.g.:
  # ```
  # BracketExpr         # Foo[Bar][Baz]
  #   BracketExpr       #   Foo[Bar]
  #     Sym "Foo"
  #     Sym "Bar"
  #   Sym "Baz"
  # ```
  case t.kind
  of nnkSym: result = t.strVal
  of nnkBracketExpr: result = t[0].getGenericTypeName()
  else: raiseAssert "Unexpected node kind for generic instantiation type: " & $t.treerepr

proc requiresMemcpy*(n: NimNode): bool =
  ## At the moment we only emit a `memcpy` statement for array types
  result = n.typeKind == ntyArray and n.kind != nnkBracket # need to emit a memcpy

proc isBuiltIn*(n: NimNode): bool =
  ## Checks if the given proc is a `{.builtin.}` (or if it is a Nim "built in"
  ## proc that uses `importc`, as we cannot emit those. They _need_ to have a
  ## WGSL / CUDA equivalent built in)
  doAssert n.kind in [nnkProcDef, nnkFuncDef], "Argument is not a proc: " & $n.treerepr
  for pragma in n.pragma:
    doAssert pragma.kind in [nnkIdent, nnkSym, nnkCall, nnkExprColonExpr], "Unexpected node kind: " & $pragma.treerepr
    let pragma = if pragma.kind in [nnkCall, nnkExprColonExpr]: pragma[0] else: pragma
    if pragma.strVal in ["builtin", "importc"]:
      return true

proc collectRawPragmas*(n: NimNode): seq[string] =
  ## Collect ALL pragma names from a pragma node as raw strings.
  ## Preserves Nim-specific pragmas that are filtered out by `filterPragmas` pass.
  if n.kind == nnkEmpty: return
  for pragma in n:
    doAssert pragma.kind in [nnkIdent, nnkSym, nnkCall, nnkExprColonExpr], "Unexpected node kind: " & $pragma.treerepr
    let key = if pragma.kind in [nnkCall, nnkExprColonExpr]: pragma[0] else: pragma
    result.add key.strVal


proc parseWorkgroupSize*(n: NimNode): tuple[x, y, z: int] =
  ## Extract the `{.workgroup: (X, Y, Z).}` / `{.workgroup: N.}` annotation
  ## from a proc pragma node. Returns (0, 0, 0) when absent — targets then
  ## fall back to their per-backend default workgroup size.
  if n.kind == nnkEmpty: return
  for pragma in n:
    if pragma.kind != nnkExprColonExpr: continue
    if pragma[0].strVal != "workgroup": continue
    let v = pragma[1]
    if v.kind == nnkIntLit:
      result.x = v.intVal.int
    elif v.kind == nnkTupleConstr:
      for i, e in v.pairs:
        if e.kind != nnkIntLit: continue
        case i
        of 0: result.x = e.intVal.int
        of 1: result.y = e.intVal.int
        of 2: result.z = e.intVal.int
        else: discard
    break

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
    of "workgroup": # workgroup size annotation, handled by parseWorkgroupSize
      discard
    of "raises":
      discard
    # Common Nim pragmas that are not relevant for CUDA C codegen:
    of "noSideEffect", "nimcall", "closure", "shallow":
      discard
    else:
      raiseAssert "Unexpected pragma for procs: " & $pragma.treerepr

proc hasPragma*(n: NimNode, pragmaName: string): bool =
  ## True when `n` carries the pragma `pragmaName`.
  ## Covers procdefs and symbol references to procs (the magic
  ## system.abs/min/max/operators) and to `{.builtin.}`-tagged lets
  ## (the cuda/wgsl/metal index dummies).
  var pragmaNode: NimNode
  case n.kind
  of nnkProcDef, nnkFuncDef:
    pragmaNode = n.pragma
  of nnkSym:
    let impl = n.getImpl()
    case impl.kind
    of nnkProcDef, nnkFuncDef:
      pragmaNode = impl.pragma
    of nnkIdentDefs, nnkConstDef:
      if impl.len > 0 and impl[0].kind == nnkPragmaExpr and
         impl[0][1].kind == nnkPragma:
        pragmaNode = impl[0][1]
      else:
        return false
    else:
      return false
  else:
    return false
  if pragmaNode.kind == nnkEmpty:
    return false
  for p in pragmaNode:
    let key = if p.kind in {nnkCall, nnkExprColonExpr}: p[0] else: p
    if key.strVal == pragmaName:
      return true
  false

proc collectAddressSpace*(n: NimNode): AddressSpace =
  doAssert n.kind == nnkPragma
  result = asDevice
  for pragma in n:
    doAssert pragma.kind in [nnkIdent, nnkSym], "Unexpected node kind: " & $pragma.treerepr
    case pragma.strVal.toLowerAscii()
    of "smem":
      doAssert result == asDevice, "Multiple address-space pragmas on one variable: " & $n.treerepr
      result = asSMEM
    of "rmem":
      doAssert result == asDevice, "Multiple address-space pragmas on one variable: " & $n.treerepr
      result = asRMEM
    of "const_mem":
      doAssert result == asDevice, "Multiple address-space pragmas on one variable: " & $n.treerepr
      result = asConstant
    of "noinit": discard # XXX: ignore for now
    of "inject": discard # injected symbols from templates/macros
    of "gensym": discard # template-generated symbol
    else:
      raiseAssert "Unexpected pragma: " & $pragma.treerepr

template findIdx*(col, el): untyped =
  var res = -1
  for i, it in col:
    if it.name == el:
      res = i
      break
  res
