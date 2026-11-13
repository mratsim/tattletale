# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, sequtils, sets, tables]
import ./print_common
import ./targets_registry
import ../../builtins/builtins_gpu_types
import ../../crucible_codegen
import ../../builtins/builtins_decl_helpers

# ############################################################
#
#            OpenCL C printer
#
# ############################################################
{.experimental: "codeReordering".}

const printerTarget = "opencl"

type PrintCtx = object of WalkCtx
  ## Per-target walk context. The shared walk procs instantiate against
  ## the derived type and the overlay procs below resolve per target.

# ═══════════════════════════════════════════════════════════════════════
#  Target tables
# ═══════════════════════════════════════════════════════════════════════

const openclVocab = BuiltinVocab(
  nilText: makeTexts([(cbkIsNil, "($1 == 0)")]),
  reductionText: makeTexts([
    (cbkReduceAdd, "sub_group_reduce_add($1)"),
    (cbkReduceMax, "sub_group_reduce_max($1)")]),
  shuffleText: makeTexts([
    # subgroup-relative indices cast to `uint` for the intrinsic's parameter type
    (cbkSimdShuffle, "sub_group_shuffle($1, (uint)($2))"),
    (cbkSimdShuffleXor, "sub_group_shuffle_xor($1, (uint)($2))")]),
  stmtText: makeTexts([
    (cbkThreadgroupBarrier, "barrier(CLK_LOCAL_MEM_FENCE)"),
    # global fence bit orders device-address-space accesses
    (cbkThreadgroupBarrierDevice, "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)")]))

const openclEmit = TargetEmit(
  vocab: openclVocab,
  functionalCtor: false,
  ctorOpen: "{",
  ctorClose: "}",
  ctorCast: true,
  ctorZero: "($1){0}",
  aconstrCasts: false,
  dotDerefParens: false,
  vectorDots: true,
  noPointerValues: (spaces: {}, text: ""),
  noPointerRebind: "",
  prefixGlobalFn: "__kernel ",
  prefixInline: "static inline ",
  prefixDevice: "static inline ")

const openclModuleShape = ModuleShape(
  spaces: {caConstant},
  structTypedef: true,
  structPadding: "char _padding_",
  structFieldTerm: ';',
  structClose: "};",
  fwdDecls: true,
  moduleOrderProcs: false,
  globalFnSep: "\n",
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
proc printSizeof(ctx: var PrintCtx, e: CirExpr): string
proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool
proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string
proc printPrelude(ctx: var PrintCtx, m: CirModule): string
proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string]
proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string

proc emitOpenclScalar(t: CirType): string =
  getTypeName(TargetVocabs[cgOpenCL].scalarNames, openclVectorRule, t)

proc emitTypeName(ctx: var PrintCtx, t: CirType): string =
  getFullTypeName(TargetVocabs[cgOpenCL].scalarNames, openclVectorRule, t)

proc emitOpenclArrayDecl(t: CirType, ident: string): string =
  var dims = ""
  var leaf = t
  while leaf.kind == ctArray:
    dims &= "[" & $leaf.aN & "]"
    leaf = leaf.aElem
  emitOpenclScalar(leaf) & ' ' & ident & dims

proc emitOpenclPtrField(t: CirType, ident: string): string =
  # `restrict` is a function-parameter qualifier in OpenCL C.
  let qualifier = case t.addressSpace
    of caThread: ""
    of caThreadgroup: "__local"
    of caConstant: "__constant"
    of caDevice: "__global"
  let sep = if qualifier.len > 0: " " else: ""
  result = qualifier & sep &
    printCDecl(TargetVocabs[cgOpenCL].scalarNames, openclVectorRule, printerTarget, t, ident, true)

proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  if f.typ.kind == ctArray:
    return emitOpenclArrayDecl(f.typ, f.name)
  if f.typ.kind in {ctAptr, ctPtr}:
    if f.typ.kind == ctAptr and f.typ.addressSpace != caDevice:
      raiseAssert("struct field " & t.name & "." & f.name & " in " & $f.typ.addressSpace & " addressSpace, only device pointers nest in structs")
    return emitOpenclPtrField(f.typ, f.name)
  return ctx.emitTypeName(f.typ) & ' ' & f.name

# ═══════════════════════════════════════════════════════════════════════
#  Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

proc getOpenclCoordComponent(k: CirCoordBuiltinKind, field: string): string =
  ## OpenCL C call of one canonical coordinate builtin component, the dimension as a literal.
  ## "" when the field is not a component.
  let d = case field
    of "x": "0"
    of "y": "1"
    of "z": "2"
    else: ""
  if d.len == 0:
    return ""
  case k
  of cbkThreadPositionInGrid: "(int)get_global_id(" & d & ")"
  of cbkThreadgroupPositionInGrid: "(int)get_group_id(" & d & ")"
  of cbkThreadPositionInThreadgroup: "(int)get_local_id(" & d & ")"
  of cbkThreadsPerThreadgroup: "(int)get_local_size(" & d & ")"
  of cbkThreadgroupsPerGrid: "(int)get_num_groups(" & d & ")"
  else: ""

## x-major flat thread index, parenthesized so a trailing `* k` cannot
## mis-associate into `+ get_local_id(0) * k`. OpenCL index builtins are
## size_t, the canonical Nim type is int, the printer casts.
const OpenclFlatThreadIndex =
  "(int)(get_local_id(2)*get_local_size(0)*get_local_size(1) + " &
  "get_local_id(1)*get_local_size(0) + get_local_id(0))"


proc coordRef(ctx: var PrintCtx, e: CirExpr): string =
  if e.coordBuiltin == cbkThreadIndexInThreadgroup:
    return OpenclFlatThreadIndex
  raiseAssert("whole-vector coordinate read " & $e.coordBuiltin & " has no OpenCL C vector form, read the components")

proc coordDot(ctx: var PrintCtx, e: CirExpr): string =
  let mapped = getOpenclCoordComponent(e.dotBase.coordBuiltin, e.fieldName)
  if mapped.len == 0:
    raiseAssert("no coordinate component " & e.fieldName)
  mapped

proc getAtomicOperandPtr(e: CirExpr): CirType =
  if e.kind == ePat:
    let t = getExprType(e.patBase)
    if t != nil and t.kind in {ctPtr, ctAptr}:
      return t
  let t = getExprType(e)
  if t != nil and t.kind in {ctPtr, ctAptr}:
    return t

# ═══════════════════════════════════════════════════════════════════════
#  Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## C compilers reject a pointer cast that silently changes the pointee's address space.
  let target = if e.castType.kind in {ctPtr, ctAptr}:
      emitOpenclPtrField(e.castType, "").strip()
    else:
      ctx.emitTypeName(e.castType)
  return "(" & target & ")" & ctx.printExpr(e.castOperand)

proc getAtomicCallText(ctx: var PrintCtx, callee: CirExpr, argExprs: seq[CirExpr], args: seq[string]): string =
  ## OpenCL C text of the atomic fetch-and-add call.
  ## atomic_fetch_add requires `atomic_*` pointees, the raw qualified
  ## pointer casts at the call site.
  let pt = getAtomicOperandPtr(argExprs[0])
  if pt == nil or pt.pElem == nil:
    raiseAssert("atomicFetchAdd on an operand with no pointee type")
  let qualifier = case pt.addressSpace
    of caDevice: "__global"
    of caThreadgroup: "__local"
    of caConstant: "__constant"
    of caThread:
      raiseAssert("atomicFetchAdd on thread-private storage")
  let atomicTy = case pt.pElem.kind
    of ctFloat:
      if pt.pElem.bits == 32: "atomic_float"
      else: ""
    of ctInt:
      if pt.pElem.bits == 32: "atomic_int"
      else: ""
    of ctUint:
      if pt.pElem.bits == 32: "atomic_uint"
      else: ""
    else: ""
  if atomicTy.len == 0:
    raiseAssert("atomicFetchAdd on " & emitOpenclScalar(pt.pElem) & ", only 32-bit int/uint/float pointees")
  "atomic_fetch_add((volatile " & qualifier & " " & atomicTy & "*)" &
    ctx.printExpr(argExprs[0]) & ", " & ctx.printExpr(argExprs[1]) & ")"

proc printSizeof(ctx: var PrintCtx, e: CirExpr): string =
  if e.ofType.kind == ctArray:
    return "sizeof(" & emitOpenclArrayDecl(e.ofType, "") & ")"
  return "sizeof(" & ctx.emitTypeName(e.ofType) & ")"

# ═══════════════════════════════════════════════════════════════════════
#  Statement declarations
# ═══════════════════════════════════════════════════════════════════════

proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool =
  ## Kernel-only storage classes: `__local` and `__constant`. Relocation moves const_mem locals out and rejects a device function's smem local.
  if s.addressSpace == caThreadgroup:
    if not ctx.inGlobalFn:
      raiseAssert("smem local " & s.name & " in a device function, __local is kernel storage")
    return true
  if s.addressSpace == caConstant:
    raiseAssert("local " & s.name & " in constant addressSpace, the OpenCL lowering pass already relocated it")
  if s.addressSpace == caDevice:
    raiseAssert(s.name & " needs a function/workgroup home in OpenCL")
  true

proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string =
  ## OpenCL C rejects a doubled address-space qualifier.
  if s.declType.kind in {ctPtr, ctAptr}:
    return emitOpenclPtrField(s.declType, s.name)
  let prefix = case s.addressSpace
    of caThread: "__private "
    of caThreadgroup: "__local "
    else: ""
  if s.declType.kind == ctArray:
    return prefix & emitOpenclArrayDecl(s.declType, s.name)
  prefix & ctx.emitTypeName(s.declType) & ' ' & s.name

# ═══════════════════════════════════════════════════════════════════════
#  Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc printPrelude(ctx: var PrintCtx, m: CirModule): string =
  declareExtensions(m.features, [
    (m.features.fp64, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"),
    (m.features.fp16, "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"),
    (m.features.atomicFloat, "#pragma OPENCL EXTENSION cl_ext_float_atomics : enable\n"),
    (m.features.subgroup, "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"),
  ])

proc emitParam(ctx: var PrintCtx, p: CirParam, isGlobalFn: bool): string =
  ## Parameter declaration of one kernel or device-function parameter.
  ## A kernel's `ptr T` parameter binds a device buffer, the engine serializes
  ## it through the same pointer-leaf form as a `ptr UncheckedArray[T]` parameter.
  ## Array parameter values reach the kernel packed by the engine.
  if p.passByRef and not isGlobalFn:
    # const Type* _p_name — pointer to const for large structs (no C++ references)
    # Note: openclLowerByrefParams already renamed the param to _p_
    return "const " & ctx.emitTypeName(p.typ) & "* " & p.name
  case p.typ.kind
  of ctAptr, ctPtr:
    let (qualifier, ident) = case p.typ.addressSpace
      of caDevice: ("__global ", "restrict " & p.name)
      of caThreadgroup: ("__local ", p.name)
      of caConstant: ("__constant ", p.name)
      of caThread: ("", p.name)
    result = qualifier &
      printCDecl(TargetVocabs[cgOpenCL].scalarNames, openclVectorRule, printerTarget, p.typ, ident, true)
  of ctArray:
    raiseAssert("array parameter " & p.name & ", OpenCL C array parameters decay to pointers")
  else:
    result = ctx.emitTypeName(p.typ) & ' ' & p.name

proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string] =
  for p in fn.params:
    result.add ctx.emitParam(p, fn.isGlobalFn)

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope declaration text of one relocated const_mem local.
  ## OpenCL C forbids declaring `__constant` storage without an initializer:
  ## the printer refuses a const_mem local that lacks one.
  if s.init == nil:
    ctx.refuseUninitializedConstMem(s.name)
    return ""
  if s.declType.kind == ctArray:
    result = "__constant " & emitOpenclArrayDecl(s.declType, s.name)
    if s.init != nil and s.init.kind == eAconstr:
      var elems: seq[string]
      for el in s.init.elems:
        elems.add ctx.printExpr(el)
      result &= " = {" & elems.join(", ") & "};"
      return
  else:
    result = "__constant " & ctx.emitTypeName(s.declType) & ' ' & s.name
  result &= " = " & ctx.printExpr(s.init) & ";"

proc printOpenclModule*(m: CirModule): string =
  var ctx = initPrintCtx(openclWalkPolicy, openclVectorRule, openclEmit, cgOpenCL)
  ctx.mathNames = openclMathBuiltinNames
  let kernels = getModuleGlobalFns(m)
  if kernels.len == 0:
    raiseAssert("no kernel in the module")
  printModule(ctx, m, kernels, openclModuleShape)

proc codegen*(m: CirModule): string =
  codegenModule(printOpenclModule(m))
