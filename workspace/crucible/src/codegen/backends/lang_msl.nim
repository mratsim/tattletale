# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, sequtils, sets, tables, math]
import ./print_common
import ./targets_registry
import ../../builtins/builtins_gpu_types
import ../../crucible_codegen
import ../../builtins/builtins_decl_helpers

# ############################################################
#
#            MSL printer
#
# ############################################################
{.experimental: "codeReordering".}

const printerTarget = "msl"

type PrintCtx = object of WalkCtx
  ## Per-target walk context. The shared walk procs instantiate against
  ## the derived type and the overlay procs below resolve per target.

# ═══════════════════════════════════════════════════════════════════════
#  Target tables
# ═══════════════════════════════════════════════════════════════════════

const mslVocab = BuiltinVocab(
  synchroText: makeTexts([
    (cbkThreadgroupBarrier, "threadgroup_barrier(mem_flags::mem_threadgroup)"),
    (cbkThreadgroupBarrierDevice, "threadgroup_barrier(mem_flags::mem_device)")]),
  reductionText: makeTexts([
    (cbkReduceAdd, "simd_sum($1)"),
    (cbkReduceMax, "simd_max($1)")]))

const mslEmit = TargetEmit(
  vocab: mslVocab,
  functionalCtor: false,
  ctorOpen: "{",
  ctorClose: "}",
  ctorCast: false,
  ctorZero: "",
  aconstrCasts: true,
  dotDerefParens: true,
  vectorDots: false,
  noPointerValues: (spaces: {}, text: ""),
  noPointerRebind: "",
  prefixKernel: "kernel ",
  prefixInline: "inline ",
  prefixDevice: "")

const mslModuleShape = ModuleShape(
  spaces: {caConstant},
  structTypedef: false,
  structPadding: "char _",
  structFieldTerm: ';',
  structClose: "};",
  fwdDecls: false,
  moduleOrderProcs: true,
  kernelSep: "\n\n",
  customProcHead: false,
  printsRefBlocks: false,
  printsParamBindings: false,
  printsReduceTemplates: false)

# ═══════════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════════

# Shared-walk mixin targets, declared so the overlay procs are module-scope
# symbols when the walk's generics resolve (mixin ignores codeReordering).
proc emitTypeName(ctx: var PrintCtx, t: CirType): string
proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string
proc coordRef(ctx: var PrintCtx, e: CirExpr): string
proc coordDot(ctx: var PrintCtx, e: CirExpr): string
proc printConv(ctx: var PrintCtx, e: CirExpr): string
proc getAtomicCallText(ctx: var PrintCtx, callee: CirExpr, argExprs: seq[CirExpr], args: seq[string]): string
proc getStmtBuiltinText(ctx: var PrintCtx, s: CirStmt, pad: string, args: seq[string]): string
proc printAsm(ctx: var PrintCtx, s: CirStmt): string
proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool
proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string
proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string]
proc printPrelude(ctx: var PrintCtx, m: CirModule): string
proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string

proc emitMslScalar(t: CirType): string =
  ## MSL name of a non-array leaf type, "" when the caller builds the type name.
  getFullTypeName(TargetVocabs[cgMetal].scalarNames, mslVectorRule, t)

proc emitMslType(t: CirType, ident: string, allowEmptyIdent: bool): string =
  ## MSL declaration text of `t` with `ident` appended, the shared C declarator
  ## over the MSL tables. Arrays put the identifier into the declarator
  ## (`T ident[N]`, pointer-to-array `T (*ident)[N]`), the printer refuses an array without an identifier.
  printCDecl(TargetVocabs[cgMetal].scalarNames, mslVectorRule, printerTarget, t, ident, allowEmptyIdent)

proc getMslPtrQualifier(addressSpace: AddressSpace): string =
  ## MSL qualifier before a pointer declaration or cast, one word per space.
  case addressSpace
  of caThread: "thread"
  of caThreadgroup: "threadgroup"
  of caConstant: "constant"
  of caDevice: "device"

proc emitMslPtr(t: CirType, ident: string): string =
  ## MSL declaration text of a pointer or array pointer with its address space:
  ## `device float* p`, pointer-to-array `device float (*p)[N]`.
  let qualifier = getMslPtrQualifier(t.addressSpace)
  let sep = if qualifier.len > 0: " " else: ""
  result = qualifier & sep &
    printCDecl(TargetVocabs[cgMetal].scalarNames, mslVectorRule, printerTarget, t, ident, true)

proc emitTypeName(ctx: var PrintCtx, t: CirType): string =
  ## MSL name of `t` with no identifier, as casts and sizeof print it.
  emitMslType(t, "", true)

proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  ## MSL declaration text of one struct member: a pointer member names
  ## the address space of its pointee (`device float* p`).
  if f.typ.kind in {ctPtr, ctAptr}:
    return emitMslPtr(f.typ, f.name)
  return emitMslType(f.typ, f.name, false)

# ═══════════════════════════════════════════════════════════════════════
#  Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

proc printBuiltinRef(ctx: var PrintCtx, k: CirCoordBuiltinKind, whole: bool): string =
  ## MSL token of a builtin reference, the attributed parameter name.
  ## A whole vector use constructs `int3`, a component use casts to `int`,
  ## the flat thread index is a scalar `uint`.
  ctx.builtins.incl(k)
  let name = $k
  if whole and k != cbkThreadIndexInThreadgroup:
    return "int3((int)" & name & ".x, (int)" & name & ".y, (int)" & name & ".z)"
  return name


proc coordRef(ctx: var PrintCtx, e: CirExpr): string =
  ctx.printBuiltinRef(e.coordBuiltin, true)

proc coordDot(ctx: var PrintCtx, e: CirExpr): string =
  ## One component of an attributed `uint3` builtin parameter, cast to `int`.
  ctx.builtins.incl(e.dotBase.coordBuiltin)
  "(int)(" & $e.dotBase.coordBuiltin & '.' & e.fieldName & ")"

# ═══════════════════════════════════════════════════════════════════════
#  Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## Pointer casts name their address space.
  let t = e.castType
  var castTyp = emitMslType(t, "", true)
  if t.kind in {ctPtr, ctAptr}:
    castTyp = getMslPtrQualifier(t.addressSpace) & ' ' & castTyp
  return "(" & castTyp & ")" & ctx.printExpr(e.castOperand)

proc getAtomicCallText(ctx: var PrintCtx, callee: CirExpr, argExprs: seq[CirExpr], args: seq[string]): string =
  ## MSL text of the atomic fetch-and-add call.
  ## Pointer casts to the atomic pointee type, relaxed order only,
  ## fp32 needs Metal 3+, device address space implies the scope.
  var atomicTy = ""
  let p0 = if callee.typ != nil and callee.typ.kind == ctProc and
      callee.typ.params.len > 0: callee.typ.params[0]
  else: nil
  if p0 != nil and p0.kind in {ctPtr, ctAptr} and p0.pElem != nil:
    let pointee = p0.pElem
    if pointee.kind == ctFloat and pointee.bits == 32:
      atomicTy = "atomic_float"
    elif pointee.kind == ctInt and pointee.bits == 32:
      atomicTy = "atomic_int"
    elif pointee.kind == ctUint and pointee.bits == 32:
      atomicTy = "atomic_uint"
  if atomicTy.len == 0:
    error("atomic_fetch_add over a pointee without an MSL atomic type")
  return "atomic_fetch_add_explicit((volatile device " & atomicTy & "*)" &
    "(" & args[0] & "), " & args[1] & ", memory_order_relaxed)"

proc getStmtBuiltinText(ctx: var PrintCtx, s: CirStmt, pad: string, args: seq[string]): string =
  ## MSL resolves statement-position builtins through the value vocabulary.
  let txt = ctx.emitBuiltinCall(s.callCallee, args)
  if txt.len == 0:
    return ""
  pad & txt & ";"

# ═══════════════════════════════════════════════════════════════════════
#  Statement declarations
# ═══════════════════════════════════════════════════════════════════════
proc printAsm(ctx: var PrintCtx, s: CirStmt): string =
  ## One inline assembly statement: the spliced text.
  let pad = spaces(ctx.indent * 2)
  pad & s.asmSpliceText()


proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool =
  ## Only a kernel's initializer-less smem local and ordinary locals reach here.
  ## The lowering pass refuses a device function's smem local and hoists const_mem locals to program scope.
  if s.addressSpace == caConstant:
    ctx.fail("local " & s.name & " in constant addressSpace, the MSL lowering pass hoists const_mem locals to program scope")
    return false
  true

proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string =
  if s.addressSpace == caThreadgroup:
    return "threadgroup " & emitMslType(s.declType, s.name, false)
  if s.declType.kind in {ctPtr, ctAptr}:
    return emitMslPtr(s.declType, s.name)
  emitMslType(s.declType, s.name, false)

# ═══════════════════════════════════════════════════════════════════════
#  Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc emitMslSmemParam(p: CirParam): string =
  ## MSL parameter of a relocated smem local.
  ## An array local reads `threadgroup T* name`, any other local
  ## reads `threadgroup T& name`, trailing the regular parameters
  ## with no binding.
  let inner = p.typ.pElem
  if inner.kind == ctArray:
    return "threadgroup " & emitMslType(inner.aElem, "", true) & "* " & p.name
  return "threadgroup " & emitMslScalar(inner) & "& " & p.name

proc emitBuiltinParam(k: CirCoordBuiltinKind): string =
  ## Attributed MSL parameter of a coordinate builtin reference:
  ## `uint3 name [[name]]` for the coordinate vectors, a scalar `uint` for the flat thread index.
  ## No binding, no buffer slot.
  let name = $k
  if k == cbkThreadIndexInThreadgroup:
    return "uint " & name & " [[" & name & "]]"
  return "uint3 " & name & " [[" & name & "]]"

proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string] =
  ## MSL parameter texts of one proc or kernel, in declaration order.
  ##
  ## Kernel parameters:
  ## - a relocated smem local binds as threadgroup storage, no buffer slot
  ## - the printer assigns the next `[[buffer(n)]]` slot to a device pointer
  ## - a value parameter binds as `constant T&`, bool retypes to `int`
  ##   (Metal has no bool constant storage)
  ## - the coordinate builtins that the body references append as attributed parameters, no slot
  ##
  ## Device-function parameters:
  ## - a threadgroup `ptr` parameter is a relocated smem arena, caller-owned
  ##   storage (Metal rejects threadgroup allocation inside device functions)
  var slot = 0
  if fn.isKernel:
    for p in fn.params:
      if p.typ.kind == ctPtr and p.typ.addressSpace == caThreadgroup:
        result.add emitMslSmemParam(p)
        continue
      case p.typ.kind
      of ctAptr, ctPtr:
        result.add "device " &
          printCDecl(TargetVocabs[cgMetal].scalarNames, mslVectorRule, printerTarget, p.typ, p.name, true) &
          " [[buffer(" & $slot & ")]]"
      of ctArray:
        # an array kernel argument binds in constant storage like a scalar one.
        # xcrun metal rejects an unqualified `const T name[N]` kernel argument
        result.add "constant " & emitMslType(p.typ, p.name, false)
      else:
        var elem = emitMslScalar(p.typ)
        if p.typ.kind == ctBool:
          elem = "int"
        result.add "constant " & elem & "& " & p.name
      inc slot
    for k in ctx.builtins:
      result.add emitBuiltinParam(k)
  else:
    for p in fn.params:
      case p.typ.kind
      of ctAptr, ctPtr:
        if p.typ.addressSpace == caThreadgroup and p.typ.kind == ctPtr:
          result.add emitMslSmemParam(p)
        else:
          result.add emitMslPtr(p.typ, p.name)
      else:
        result.add emitMslType(p.typ, p.name, true)

proc printPrelude(ctx: var PrintCtx, m: CirModule): string =
  ## Translation-unit head of every MSL module: the metal headers.
  result = "#include <metal_stdlib>\n"
  result &= "#include <metal_atomic>\n"
  result &= "using namespace metal;\n\n"

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope `constant` declaration of one relocated const_mem local.
  ## MSL forbids declaring constant storage without an initializer.
  if s.init == nil:
    ctx.refuseUninitializedConstMem(s.name)
    return ""
  var decl = emitMslType(s.declType, s.name, false)
  if s.declType.kind in {ctPtr, ctAptr}:
    decl = emitMslPtr(s.declType, s.name)
  return "constant " & decl & " = " & ctx.printExpr(s.init) & ";"

proc printMslModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full MSL translation unit through the shared module skeleton, returning "" as the error when clean, else the first refusal.
  var ctx = initPrintCtx(mslWalkPolicy, mslVectorRule, mslEmit, cgMetal)
  printModule(ctx, m, getModuleKernels(m), mslModuleShape)

proc codegen*(m: CirModule): string =
  codegenModule(printMslModule(m))
