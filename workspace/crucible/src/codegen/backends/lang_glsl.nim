# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, tables]
import ./print_common
import ./targets_registry
import ../../builtins/builtins_gpu_types
import ../../crucible_codegen
import ../../builtins/builtins_decl_helpers

# ############################################################
#
#            GLSL printer
#
# ############################################################
{.experimental: "codeReordering".}

const printerTarget = "glsl"

type PrintCtx = object of WalkCtx
  ## Per-target walk context. The shared walk procs instantiate against
  ## the derived type and the overlay procs below resolve per target.



# ═══════════════════════════════════════════════════════════════════════
#  Target tables
# ═══════════════════════════════════════════════════════════════════════

const glslVocab = BuiltinVocab(
  reductionText: makeTexts([
    (cbkReduceAdd, "subgroupAdd($1)"),
    (cbkReduceMax, "subgroupMax($1)")]),
  shuffleText: makeTexts([
    # subgroupShuffle's index parameter is `uint`, a direct cast
    # compiles without GL_NV_explicit_typecast
    (cbkSimdShuffle, "subgroupShuffle($1, uint($2))"),
    (cbkSimdShuffleXor, "subgroupShuffleXor($1, uint($2))")]))

const glslEmit = TargetEmit(
  vocab: glslVocab,
  functionalCtor: true,
  ctorOpen: "(",
  ctorClose: ")",
  ctorCast: false,
  ctorZero: "$1(0)",
  aconstrCasts: false,
  dotDerefParens: false,
  vectorDots: true,
  noPointerValues: (spaces: {caThread, caThreadgroup, caConstant},
    text: "pointer local $1 used as a value, GLSL has no pointer values"),
  noPointerRebind: "pointer local $1 rebound, GLSL has no pointer values, " &
    "the GLSL lowering pass rewrites pointer locals to their pointees",
  prefixKernel: "",
  prefixInline: "",
  prefixDevice: "")

const glslModuleShape = ModuleShape(
  spaces: {caThreadgroup, caConstant},
  structTypedef: false,
  structPadding: "uint _padding",
  structFieldTerm: ';',
  structClose: "};",
  fwdDecls: true,
  moduleOrderProcs: false,
  kernelSep: "\n\n",
  customProcHead: true,
  printsRefBlocks: true,
  printsParamBindings: true,
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
proc printExtraExpr(ctx: var PrintCtx, e: CirExpr): string
proc getStmtBuiltinText(ctx: var PrintCtx, s: CirStmt, pad: string, args: seq[string]): string
proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool
proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string
proc checkConstDecl(ctx: var PrintCtx, s: CirStmt): bool
proc constDecl(ctx: var PrintCtx, s: CirStmt): string
proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string]
proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string
proc printPrelude(ctx: var PrintCtx, m: CirModule): string
proc printRefBlocks(ctx: var PrintCtx, m: CirModule): string
proc printParamBindings(ctx: var PrintCtx, kernels: seq[CirProc]): string

proc emitGlslScalar(t: CirType): string =
  ## GLSL name of a non-array leaf type, "" when the caller builds the type name.
  getLeafTypeName(TargetVocabs[bkGlsl].scalarNames, glslVectorRule, t)

proc glslTypeName(t: CirType): string =
  ## GLSL name of a type with no identifier, the printer appends the array
  ## dimensions to the type name (`float[4]`).
  if t.kind == ctArray:
    glslTypeName(t.aElem) & "[" & $t.aN & "]"
  else:
    emitGlslScalar(t)

proc emitTypeName(ctx: var PrintCtx, t: CirType): string =
  ## GLSL name of `t` with no identifier, matching the cast and constructor form.
  glslTypeName(t)

proc refTypeName*(t: CirType): string =
  ## GLSL name of the buffer-reference block binding `t`,
  ## the `Ref_` prefix. The ref-block analysis pass dedups blocks by this name.
  "Ref_" & glslTypeName(t).multiReplace(
    (" ", "_"), ("[", ""), ("]", ""), (",", "_"), ("(", ""), (")", ""))

proc emitGlslRefBlockMember(ctx: var PrintCtx, pointee: CirType): string =
  ## Member declaration of one buffer-reference block, the runtime array `x[]`:
  ## a fixed-array element nests after the runtime dimension (`uint8_t x[][32]`),
  ## the unsized dimension is only legal in the outermost position.
  case pointee.kind
  of ctArray:
    var base = pointee
    var dims = ""
    while base.kind == ctArray:
      dims &= "[" & $base.aN & "]"
      base = base.aElem
    result = emitGlslScalar(base) & " x[]" & dims & ";"
  of ctVoid, ctProc:
    ctx.fail("buffer reference to " & $pointee.kind)
    result = ""
  else:
    result = emitGlslScalar(pointee) & " x[];"

proc emitGlslPtrDecl(ctx: var PrintCtx, t: CirType, ident: string): string =
  ## GLSL declaration of a buffer pointer: the reference block handle,
  ## `Ref_T ident`.
  if t.space != caDevice:
    ctx.fail("buffer pointer in " & $t.space & " space")
    return ""
  result = refTypeName(t.pElem) & ' ' & ident

proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  ## GLSL declaration text of one struct member: a buffer pointer member is
  ## the reference block handle, the printer stacks the dimension behind the type name for an array.
  if f.typ.kind == ctArray:
    return emitGlslScalar(f.typ.aElem) & ' ' & f.name & "[" & $f.typ.aN & "]"
  if f.typ.kind in {ctAptr, ctPtr}:
    return ctx.emitGlslPtrDecl(f.typ, f.name)
  return ctx.emitTypeName(f.typ) & ' ' & f.name

# ═══════════════════════════════════════════════════════════════════════
#  Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

proc getGlslCoordName(k: CirCoordBuiltinKind): string =
  ## GLSL name of a canonical coordinate builtin: all six are native
  ## GLSL builtins, a direct rename.
  case k
  of cbkThreadPositionInGrid: "gl_GlobalInvocationID"
  of cbkThreadgroupPositionInGrid: "gl_WorkGroupID"
  of cbkThreadPositionInThreadgroup: "gl_LocalInvocationID"
  of cbkThreadsPerThreadgroup: "gl_WorkGroupSize"
  of cbkThreadgroupsPerGrid: "gl_NumWorkGroups"
  of cbkThreadIndexInThreadgroup: "gl_LocalInvocationIndex"
  else: ""


proc coordRef(ctx: var PrintCtx, e: CirExpr): string =
  if e.coordBuiltin == cbkThreadIndexInThreadgroup:
    return "int(gl_LocalInvocationIndex)"
  let g = getGlslCoordName(e.coordBuiltin)
  "ivec3(int(" & g & ".x), int(" & g & ".y), int(" & g & ".z))"

proc coordDot(ctx: var PrintCtx, e: CirExpr): string =
  ## One component of an unsigned GLSL builtin, cast to the canonical int.
  "int(" & getGlslCoordName(e.dotBase.coordBuiltin) & "." & e.fieldName & ")"

# ═══════════════════════════════════════════════════════════════════════
#  Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## A reference type constructs from its element type.
  let t = e.castType
  if t.kind in {ctAptr, ctPtr}:
    return refTypeName(t.pElem) & "(" & ctx.printExpr(e.castOperand) & ")"
  return ctx.emitTypeName(t) & "(" & ctx.printExpr(e.castOperand) & ")"

proc printGlslAtomicFetchAdd(ctx: var PrintCtx, args: seq[CirExpr]): string =
  ## GLSL text of the atomic fetch-and-add call, shared by the value
  ## and statement paths: int is core, fp needs GL_EXT_shader_atomic_float,
  ## the address operand is the reference's element 0.
  let addrTxt = if getExprType(args[0]) != nil and getExprType(
      args[0]).kind in {ctAptr, ctPtr}:
    ctx.printExpr(args[0]) & ".x[0]"
  else:
    ctx.printExpr(args[0])
  return "atomicAdd(" & addrTxt & ", " & ctx.printExpr(args[1]) & ")"

proc getAtomicCallText(ctx: var PrintCtx, callee: CirExpr, argExprs: seq[CirExpr], args: seq[string]): string =
  ## atomicFetchAdd prints its own arguments, a pre-print into the shared
  ## arg list would print each argument twice.
  ctx.printGlslAtomicFetchAdd(argExprs)

proc printSizeof(ctx: var PrintCtx, e: CirExpr): string =
  ctx.fail(getUnhandledExprRejection(e.kind))
  ""

proc printExtraExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## GLSL routes the addressing kinds itself: subscript, dereference and address-of over buffer references.
  case e.kind
  of eAt, ePat:
    let base = if e.kind == eAt: e.atBase else: e.patBase
    let (baseTxt, idxTxt) = ctx.printSubscriptParts(e)
    let bt = getExprType(base)
    if bt != nil and bt.kind == ctAptr:
      return baseTxt & ".x[" & idxTxt & "]"
    if bt != nil and bt.kind == ctArray:
      return baseTxt & "[" & idxTxt & "]"
    if bt != nil and bt.kind == ctPtr:
      ctx.fail("index through a pointer is outside the GLSL subset")
      return ""
    ctx.fail("index of a " & (if bt == nil: "untyped" else: $bt.kind) & " base is outside the GLSL subset")
    return ""
  of eDeref:
    let op = e.derefOperand
    let ot = getExprType(op)
    if ot != nil and ot.kind == ctAptr:
      return ctx.printExpr(op) & ".x[0]"
    if ot != nil and ot.kind == ctPtr:
      ctx.fail("deref of a pointer is outside the GLSL subset")
      return ""
    ctx.fail("deref of a " & (if ot == nil: "untyped" else: $ot.kind) & " operand is outside the GLSL subset")
    return ""
  of eAddr:
    # a buffer element's address is its indexed reference access
    if e.addrOperand.kind == ePat:
      return ctx.printExpr(e.addrOperand)
    ctx.fail("address of a local is outside the GLSL subset")
    return ""
  else:
    ctx.fail(getUnhandledExprRejection(e.kind))
    ""

proc getStmtBuiltinText(ctx: var PrintCtx, s: CirStmt, pad: string, args: seq[string]): string =
  case s.callCallee.synchroBuiltin
  of cbkThreadgroupBarrier:
    pad & "barrier();"
  of cbkThreadgroupBarrierDevice:
    # GLSL's barrier() orders shared memory only, buffer writes are
    # ordered before it by a leading memoryBarrierBuffer() call
    pad & "memoryBarrierBuffer();\n" & pad & "barrier();"
  else:
    if s.callCallee.atomicsBuiltin == cbkAtomicFetchAdd:
      # a GLSL call statement discards the fetched old value
      return pad & ctx.printGlslAtomicFetchAdd(s.callArgs) & ";"
    ""

# ═══════════════════════════════════════════════════════════════════════
#  Statements: the GLSL space policy and declaration forms
# ═══════════════════════════════════════════════════════════════════════

proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool =
  ## GLSL has no function-storage addressing, the GLSL lowering pass rewrites
  ## pointer locals to their pointees. The storage spaces relocate to program
  ## scope through the GLSL lowering pass.
  if s.declType.kind == ctPtr:
    ctx.fail("pointer local " & s.name & ", the GLSL lowering pass rewrites pointer locals to their pointees")
    return false
  if s.space == caThreadgroup:
    ctx.fail("local " & s.name & " in threadgroup space, the GLSL lowering pass hoists threadgroup locals to program scope")
    return false
  if s.space == caConstant:
    ctx.fail("local " & s.name & " in constant space, the GLSL lowering pass hoists const_mem locals to program scope")
    return false
  if s.space == caDevice:
    ctx.fail("local " & s.name & " in device space, GLSL locals are private")
    return false
  true

proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string =
  if s.declType.kind == ctAptr:
    # a buffer pointer local is the reference block handle
    return ctx.emitGlslPtrDecl(s.declType, s.name)
  if s.declType.kind == ctArray:
    return emitGlslScalar(s.declType.aElem) & ' ' & s.name &
      "[" & $s.declType.aN & "]"
  ctx.emitTypeName(s.declType) & ' ' & s.name

proc checkConstDecl(ctx: var PrintCtx, s: CirStmt): bool =
  true

proc constDecl(ctx: var PrintCtx, s: CirStmt): string =
  "const " & ctx.emitTypeName(s.declType) & ' ' & s.name

# ═══════════════════════════════════════════════════════════════════════
#  Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string] =
  ## GLSL parameter texts of one proc or kernel, in declaration order.
  ##
  ## - a kernel takes zero parameters: every kernel parameter packs into the module's shared push-constant block, deduped across kernels
  ## - a device function's buffer pointer prints as the reference block
  ##   handle of either pointer kind, a `var` binding gains `inout`
  if fn.isKernel:
    return
  for p in fn.params:
    case p.typ.kind
    of ctAptr, ctPtr:
      result.add ctx.emitGlslPtrDecl(p.typ, p.name)
    else:
      # a `var` binding gains an inout qualifier, the printer appends the dimension after the variable name for an array
      let binding = if p.varParam: "inout " else: ""
      if p.typ.kind == ctArray:
        result.add binding & emitGlslScalar(p.typ.aElem) & ' ' & p.name & "[" & $p.typ.aN & "]"
      else:
        result.add binding & ctx.emitTypeName(p.typ) & ' ' & p.name

proc emitConstDecl(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope `const` declaration of one hoisted const_mem local.
  ## The lowering pass refuses an initializer-less const_mem local before relocation, this check fails on a malformed node.
  if s.init == nil:
    ctx.refuseUninitializedConstMem(s.name)
    return ""
  if s.declType.kind == ctArray:
    result = "const " & emitGlslScalar(s.declType.aElem) & ' ' & s.name &
      "[" & $s.declType.aN & "]"
  else:
    result = "const " & ctx.emitTypeName(s.declType) & ' ' & s.name
  result &= " = " & ctx.printExpr(s.init) & ";"

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope declaration text of one relocated storage local.
  if s.space == caThreadgroup:
    # program-scope `shared` declaration of a threadgroup local
    if s.declType.kind == ctArray:
      return "shared " & emitGlslScalar(s.declType.aElem) & ' ' & s.name &
        "[" & $s.declType.aN & "];"
    return "shared " & ctx.emitTypeName(s.declType) & ' ' & s.name & ";"
  if s.init == nil:
    ctx.refuseUninitializedConstMem(s.name)
    return ""
  ctx.emitConstDecl(s)

proc printProcHead(ctx: var PrintCtx, fn: CirProc): string =
  ## Head of one GLSL proc: the kernel's `local_size` layout qualifier precedes
  ## the shared C signature. The engine launches each entry point by name.
  if fn.isKernel:
    let wg = getWorkgroup(fn)
    result = "layout(local_size_x = " & $wg[0] & ", local_size_y = " &
      $wg[1] & ", local_size_z = " & $wg[2] & ") in;\n"
    result &= printSig(ctx, fn) & " {\n"
  else:
    result = printSig(ctx, fn)

proc printPrelude(ctx: var PrintCtx, m: CirModule): string =
  ## GLSL extension block over the stored module feature scan `CirModule.features`.
  ## `#version 450` prints first, then one line per feature in use.
  result = "#version 450\n"
  result.add "#extension GL_EXT_buffer_reference : enable\n"
  result.add "#extension GL_EXT_buffer_reference2 : enable\n"
  result.add declareExtensions(m.features, [
    (m.features.controlFlow, "#extension GL_EXT_control_flow_attributes : require\n"),
    (m.features.fp64, "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable\n"),
    (m.features.fp16, "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\n"),
    (m.features.fp16, "#extension GL_EXT_shader_16bit_storage : enable\n"),
    (m.features.int64, "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable\n"),
    (m.features.int16, "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable\n"),
    (m.features.int16, "#extension GL_EXT_shader_16bit_storage : enable\n"),
    (m.features.int8, "#extension GL_EXT_shader_explicit_int8 : enable\n"),
    (m.features.int8, "#extension GL_EXT_shader_8bit_storage : enable\n"),
    (m.features.subgroup, "#extension GL_KHR_shader_subgroup_basic : enable\n"),
    (m.features.subgroup, "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"),
    (m.features.subgroupShuffle, "#extension GL_KHR_shader_subgroup_basic : enable\n"),
    (m.features.subgroupShuffle, "#extension GL_KHR_shader_subgroup_shuffle : enable\n"),
    (m.features.atomicFloat, "#extension GL_EXT_shader_atomic_float : enable\n")])

proc printRefBlocks(ctx: var PrintCtx, m: CirModule): string =
  ## Buffer-reference block declarations, one per `CirModule.refBlocks` entry in registration order.
  ##
  ## Filled by the ref-block analysis pass after its last transform, never registered by the printer.
  for name, pointee in m.refBlocks:
    let member = ctx.emitGlslRefBlockMember(pointee)
    result &= "layout(buffer_reference, std430) buffer " & name &
      " { " & member & " };\n"
    if ctx.failing():
      return ""
  if m.refBlocks.len > 0:
    result &= "\n"

proc printParamBindings(ctx: var PrintCtx, kernels: seq[CirProc]): string =
  ## Push-constant block: one module-wide block over the union of every
  ## kernel's parameters, same-name dedup by declaration text.
  ##
  ## - a pointer of either kind packs as the buffer-reference handle
  ## - each launch serializes its own kernel's arguments from offset 0
  ## - a by-value scalar cannot share the block across kernels
  ## - multi-kernel sources with scalar block members refuse at ingest
  var pushDecls: seq[string]
  for kernel in kernels:
    for p in kernel.params:
      var decl: string
      case p.typ.kind
      of ctAptr, ctPtr:
        decl = ctx.emitGlslPtrDecl(p.typ, p.name)
      of ctArray:
        ctx.fail("kernel parameter " & p.name & " of array type, arrays do not pack into the push block")
        decl = ""
      else:
        decl = ctx.emitTypeName(p.typ) & ' ' & p.name
      if decl notin pushDecls:
        pushDecls.add decl
  if ctx.failing():
    return ""
  if pushDecls.len > 0:
    result &= "layout(push_constant, std430) uniform KernelParams {\n"
    for decl in pushDecls:
      result &= "  " & decl & ";\n"
    result &= "};\n\n"

proc printGlslModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full GLSL translation unit through the shared module skeleton, returning "" as the error when clean, else the first refusal.
  var ctx = initPrintCtx(glslWalkPolicy, glslVectorRule, glslEmit, bkGlsl)
  let kernels = getModuleKernels(m)
  if kernels.len == 0:
    ctx.fail("module declares no kernel")
    return ("", ctx.err)
  printModule(ctx, m, kernels, glslModuleShape)

proc codegen*(m: CirModule): string =
  codegenModule(printGlslModule(m))
