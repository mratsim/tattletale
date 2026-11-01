# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## OpenCL C printer: a structured CIRv2 module rendered as OpenCL C source,
## table-driven over the closed set of type kinds.
##
## Emission contract:
## - every kernel of the module prints, in encounter order: one OpenCL program per module,
##   the device functions print alongside, forward-declared
## - the kernel signature is `__kernel void <name>(...)`: one parameter per kernel parameter, the host `store` packing order
## - device pointers render as `__global T* restrict` parameters, element access `p[index]`
## - `eSizeof` prints C `sizeof`, OpenCL C is C99
##
## Addressing and memory classes:
## - a `ptr` local is a plain C pointer, `(&x)` addresses a local, `(*p)` dereferences
## - `{.smem.}` locals declare `__local` in the kernel body
## - a loop prints `while (cond)`, `{.unroll.}` adds `#pragma unroll`
##
## Storage relocation:
## - `{.const_mem.}` locals relocate into `CirModule.globals` through the OpenCL lowering pass, program-scope `__constant` output
## - the printer refuses a `{.const_mem.}` local surviving outside the relocation, OpenCL C has no function-scope constant storage

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, sequtils, sets, tables]
import ../ir/ast_datatypes
import ./print_common
import ./print_compute
import ./targets_registry

type
  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    policy: KindRow
    ## Kind row of the target. The shared walk reads its policy fields.
    lit: LitRow
    ## Literal token forms of the target. The shared literal head reads them.
    row: ScalarRow
    ## Scalar row of the target. The narrow integer cast names its type from it.
    inKernel: bool
    ## False inside a device function: `__local` and `__constant` locals are
    ## kernel-only storage, the pass refuses a device function declaring one.

const printerTarget = "opencl"

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

# OpenCL C bool storage is 1 byte, the launch ABI delivers 4-byte ints
proc emitOpenclScalar(t: CirType): string =
  ## OpenCL C name of a non-array leaf type, "" when the caller composes the name.
  leafName(openclScalarRow, openclVectorRule, t)

proc emitTypeBare(t: CirType): string =
  ## OpenCL C name of `t` with no identifier, pointer stars trailing the element name, the form of casts and compound literals.
  bareName(openclScalarRow, openclVectorRule, t)

proc emitOpenclArrayDecl(t: CirType, ident: string): string =
  ## OpenCL C declaration of an array local or field, the lengths move behind the identifier
  ## (`float ident[4]`, nested arrays stack every dimension `float ident[4][8]`).
  var dims = ""
  var leaf = t
  while leaf.kind == ctArray:
    dims &= "[" & $leaf.aN & "]"
    leaf = leaf.aElem
  emitOpenclScalar(leaf) & ' ' & ident & dims

proc emitOpenclPtrField(t: CirType, ident: string): string =
  ## OpenCL C declaration of a pointer field or local, the address space qualifier leads.
  ## `restrict` is a function-parameter qualifier, struct fields and locals drop it.
  let qualifier = case t.space
    of caThread: ""
    of caThreadgroup: "__local"
    of caConstant: "__constant"
    of caDevice: "__global"
  let sep = if qualifier.len > 0: " " else: ""
  result = qualifier & sep & emitTypeBare(t) & ' ' & ident

proc emitComputeStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  ## OpenCL C declaration text of one struct member.
  ##
  ## - an array stacks every dimension behind the name (`float a[4][8]`)
  ## - a pointer member prints its address-space qualifier, and only device
  ##   pointers nest in structs
  if f.typ.kind == ctArray:
    return emitOpenclArrayDecl(f.typ, f.name)
  if f.typ.kind in {ctAptr, ctPtr}:
    if f.typ.kind == ctAptr and f.typ.space != caDevice:
      ctx.fail("struct field " & t.name & "." & f.name &
        " in " & $f.typ.space & " space, only device pointers nest in structs")
      return ""
    return emitOpenclPtrField(f.typ, f.name)
  return emitTypeBare(f.typ) & ' ' & f.name

# ═══════════════════════════════════════════════════════════════════════
# Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

proc openclCoordComponent(kind: CoordBuiltinKind, field: string): string =
  ## OpenCL C call of one canonical coordinate builtin component, the dimension as a literal.
  ## "" when the field is not a component.
  let d = case field
    of "x": "0"
    of "y": "1"
    of "z": "2"
    else: ""
  if d.len == 0:
    return ""
  case kind
  of cbkThreadPositionInGrid: "(int)get_global_id(" & d & ")"
  of cbkThreadgroupPositionInGrid: "(int)get_group_id(" & d & ")"
  of cbkThreadPositionInThreadgroup: "(int)get_local_id(" & d & ")"
  of cbkThreadsPerThreadgroup: "(int)get_local_size(" & d & ")"
  of cbkThreadgroupsPerGrid: "(int)get_num_groups(" & d & ")"
  of cbkNone, cbkThreadIndexInThreadgroup: ""

## x-major flat thread index, parenthesized so a trailing `* k` cannot
## mis-associate into `+ get_local_id(0) * k`. OpenCL index builtins are
## size_t, the canonical Nim type is int, the printer casts.
const OpenclFlatThreadIndex =
  "(int)(get_local_id(2)*get_local_size(0)*get_local_size(1) + " &
  "get_local_id(1)*get_local_size(0) + get_local_id(0))"

proc atomicOperandPtr(e: CirExpr): CirType =
  ## Pointer type of an atomicFetchAdd address operand.
  ## The base of an element access, else the operand itself when it is
  ## already a pointer. Nil when the operand has no pointer.
  if e.kind == ePat:
    let t = getExprType(e.patBase)
    if t != nil and t.kind in {ctPtr, ctAptr}:
      return t
  let t = getExprType(e)
  if t != nil and t.kind in {ctPtr, ctAptr}:
    return t

proc emitExtensions(features: ModuleFeatures): string =
  ## OpenCL C extension pragma block from the stored module feature scan
  ## (`CirModule.features`, filled by the feature-scan pass), one line per
  ## feature in use.
  emitFeatureLines(features, [
    (features.fp64, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"),
    (features.fp16, "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"),
    (features.atomicFloat, "#pragma OPENCL EXTENSION cl_ext_float_atomics : enable\n"),
    (features.subgroup, "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"),
  ])

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## One conversion, the C-style cast form.
  ##
  ## - a pointer cast includes the address space, the C compiler rejects a silent change of the pointee's space
  ## - other types take the bare name
  let target = if e.castType.kind in {ctPtr, ctAptr}:
      emitOpenclPtrField(e.castType, "").strip()
    else:
      emitTypeBare(e.castType)
  return "(" & target & ")" & ctx.printExpr(e.castOperand)


proc printBuiltinCall(ctx: var PrintCtx, call: CirExpr, args: seq[string]): string =
  ## Returns the OpenCL C text of an expression-position builtin call,
  ## "" when `call.callee` is not a builtin.
  if call.callee.kind == eSym:
    # builtin vocabulary dispatches on the typed kinds, never on names
    case call.callee.reductionBuiltin
    of rbkReduceAdd:
      return "sub_group_reduce_add(" & args.join(", ") & ")"
    of rbkReduceMax:
      return "sub_group_reduce_max(" & args.join(", ") & ")"
    of rbkNone: discard
  if call.callee.kind == eSym:
    case call.callee.shuffleBuiltin
    of shkSimdShuffle:
      # `sub_group_shuffle` takes the subgroup-relative index as `uint`
      return "sub_group_shuffle(" & args[0] & ", (uint)(" & args[1] & "))"
    of shkSimdShuffleXor:
      # `sub_group_shuffle_xor` takes the xor mask as `uint`, not the Nim `int`
      return "sub_group_shuffle_xor(" & args[0] & ", (uint)(" & args[1] & "))"
    of shkNone: discard
  if call.callee.kind == eSym and call.callee.builtin and
      call.callee.name in ["min", "max", "abs"] and call.args.len > 0:
    # float operands take the f-prefixed OpenCL names, integer operands keep the plain names
    let at = getExprType(call.args[0])
    if at != nil and at.kind == ctFloat:
      let mapped = case call.callee.name
        of "max": "fmax"
        of "min": "fmin"
        else: "fabs"
      return mapped & "(" & args.join(", ") & ")"
  if call.callee.kind == eSym and call.callee.atomicsBuiltin == abkAtomicFetchAdd:
    let pt = atomicOperandPtr(call.args[0])
    if pt == nil or pt.pElem == nil:
      ctx.fail("atomicFetchAdd on an operand with no pointee type")
      return ""
    let qualifier = case pt.space
      of caDevice: "__global"
      of caThreadgroup: "__local"
      of caConstant: "__constant"
      of caThread:
        ctx.fail("atomicFetchAdd on thread-private storage")
        return ""
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
      ctx.fail("atomicFetchAdd on " & emitOpenclScalar(pt.pElem) &
        ", only 32-bit int/uint/float pointees")
      return ""
    # atomic_fetch_add acts on `atomic_*` pointees: cast the raw
    # qualified pointer at the call site
    return "atomic_fetch_add((volatile " & qualifier & " " & atomicTy & "*)" &
      ctx.printExpr(call.args[0]) & ", " & ctx.printExpr(call.args[1]) & ")"
  if call.callee.kind == eSym and call.callee.synchroBuiltin != sbkNone:
    ctx.fail("barrier is a statement, a value-position barrier is outside the OpenCL subset")
    return ""
  if call.callee.kind == eSym and call.callee.builtin and
      openclConvBuiltinNames.hasKey(call.callee.name):
    return openclConvBuiltinNames.getOrDefault(call.callee.name) % args.join(", ")
  return ""
proc emitLocalDecl(s: CirStmt): string =
  ## Declaration text of one local, the array form appends the length to the name, the init remains with the caller.
  ## Pointer types include the address-space qualifier in their own text, the caller adds no storage prefix (OpenCL rejects a doubled qualifier).
  if s.declType.kind == ctArray:
    result = emitOpenclArrayDecl(s.declType, s.name)
  elif s.declType.kind in {ctPtr, ctAptr}:
    result = emitOpenclPtrField(s.declType, s.name)
  else:
    result = emitTypeBare(s.declType) & ' ' & s.name

# ═══════════════════════════════════════════════════════════════════════
# Statements: per-target policy arms, the shared dispatch lives in print_common
# ═══════════════════════════════════════════════════════════════════════

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## OpenCL C text of the kinds in `targetExprKinds`.
  case e.kind
  of eSym:
    # a whole-vector coordinate read meets the printer's refusal, the flat thread index
    # takes its canonical token, a name passes through
    if e.coordBuiltin != cbkNone:
      if e.coordBuiltin == cbkThreadIndexInThreadgroup:
        return OpenclFlatThreadIndex
      ctx.fail("whole-vector coordinate read " & e.name &
        ", OpenCL C has no vector form, read the components")
      return ""
    return e.name
  of eDot:
    # a coordinate component takes its canonical token, a vector component
    # takes its letter, names pass through
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      let mapped = openclCoordComponent(e.dotBase.coordBuiltin, e.fieldName)
      if mapped.len == 0:
        ctx.fail("no coordinate component " & e.fieldName)
        return ""
      return mapped
    let bt = getExprType(e.dotBase)
    if bt != nil and bt.kind == ctVector:
      let comp = emitVectorComponent(e.fieldName)
      if comp.len == 0:
        ctx.fail("no vector component " & e.fieldName)
        return ""
      return ctx.printExpr(e.dotBase) & "." & comp
    return ctx.printExpr(e.dotBase) & "." & e.fieldName
  of eCast:
    return ctx.printCast(e, openclCastRules, emitOpenclScalar(e.castType))
  of eConv:
    return ctx.printConv(e)
  of eSizeof:
    # an array type names through the declaration form, the bare-name composer `emitTypeBare` prints "" for an array leaf
    if e.ofType.kind == ctArray:
      return "sizeof(" & emitOpenclArrayDecl(e.ofType, "") & ")"
    return "sizeof(" & emitTypeBare(e.ofType) & ")"
  of eOconstr:
    # C99 compound literal, a bare braced init list cannot be member-accessed
    if e.kvFields.len == 0:
      # a zero-field struct, the padding member zero-initializes
      return "(" & emitTypeBare(e.ocType) & "){0}"
    result = "(" & emitTypeBare(e.ocType) & "){"
    for i, kvv in e.kvFields:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(kvv.value)
    result.add "}"
  of eAconstr:
    # plain braces, OpenCL C has no array compound-literal cast,
    # object constructions cast `(Type){...}`
    let bare = emitTypeBare(e.acType)
    result = if bare.len > 0: "(" & bare & "){" else: "{"
    for i, el in e.elems:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(el)
    result.add "}"
  of eCall:
    let args = printCallArgs(ctx, e)
    return printCallText(ctx, e, args, printBuiltinCall(ctx, e, args))
  else:
    # dispatcher sends only the kinds above, anything else is a routing bug
    ctx.fail("expression kind " & $e.kind & " has no OpenCL C emission")


proc printVar(ctx: var PrintCtx, s: CirStmt): string =
  ## One local declaration: storage-class refusals, the address-space qualifier, then the declaration text.
  let pad = spaces(ctx.indent * 2)
  if s.space == caThreadgroup:
    if not ctx.inKernel:
      ctx.fail("smem local " & s.name & " in a device function, __local is kernel storage")
      return ""
    # no initializer survives here, the rewrite ran before printing
    var decl = (if s.declType.kind in {ctPtr, ctAptr}: "" else: "__local ") &
      emitLocalDecl(s)
    result = pad & decl & ";"
  elif s.space == caConstant:
    # OpenCL lowering pass relocates const_mem locals at every scope,
    # a survivor here was never relocated
    ctx.fail("local " & s.name & " in constant space, the OpenCL lowering" &
      " pass hoists const_mem locals to program scope")
    return ""
  elif s.space == caDevice:
    ctx.fail("local " & s.name & " in device space, OpenCL locals are private or local")
    return ""
  else:
    var decl = (if s.declType.kind in {ctPtr, ctAptr}: "" else: "__private ") &
      emitLocalDecl(s)
    result = pad & decl
    if s.init != nil:
      if s.declType.kind == ctArray and s.init.kind == eAconstr:
        var elems: seq[string]
        for el in s.init.elems:
          elems.add ctx.printExpr(el)
        result &= " = {" & elems.join(", ") & "}"
      else:
        result &= " = " & ctx.printExpr(s.init)
    result &= ";"

proc printConst(ctx: var PrintCtx, s: CirStmt): string =
  ## One local constant declaration, the `const` form.
  let pad = spaces(ctx.indent * 2)
  result = pad & "const " & emitTypeBare(s.declType) & ' ' &
    s.name & " = " & ctx.printExpr(s.init) & ";"

proc printCall(ctx: var PrintCtx, s: CirStmt): string =
  ## One call statement: the barrier vocabulary, then the plain call form.
  let pad = spaces(ctx.indent * 2)
  var parts: seq[string]
  parts.add ctx.printExpr(s.callCallee)
  for a in s.callArgs:
    parts.add ctx.printExpr(a)
  if s.callCallee.kind == eSym and
      s.callCallee.synchroBuiltin == sbkThreadgroupBarrier:
    result = pad & "barrier(CLK_LOCAL_MEM_FENCE);"
  elif s.callCallee.kind == eSym and
      s.callCallee.synchroBuiltin == sbkThreadgroupBarrierDevice:
    # global fence bit orders device-address-space accesses
    result = pad & "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);"
  else:
    result = pad & parts[0] & "(" & parts[1 ..^ 1].join(", ") & ");"

proc printAsm(ctx: var PrintCtx, s: CirStmt): string =
  ## One inline assembly statement: refused, the text names the target.
  ctx.fail("inline assembly has no OpenCL C form")
  result = ""

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc emitParam(ctx: var PrintCtx, p: CirParam, isKernel: bool): string =
  ## Parameter declaration of one kernel or device-function parameter.
  ## A raw `ptr` parameter is legal C between device functions, the printer refuses it on a kernel.
  ## The engine binds device buffers and values, never raw pointers.
  case p.typ.kind
  of ctAptr:
    let inner = emitOpenclScalar(p.typ.pElem)
    case p.typ.space
    of caDevice:
      result = "__global " & inner & "* restrict " & p.name
    of caThreadgroup:
      result = "__local " & inner & "* " & p.name
    of caConstant:
      result = "__constant " & inner & "* " & p.name
    of caThread:
      result = inner & "* " & p.name
  of ctPtr:
    if isKernel:
      ctx.fail("kernel parameter " & p.name &
        " of pointer type, the engine binds device buffers and values")
      return ""
    result = emitTypeBare(p.typ) & ' ' & p.name
  of ctArray:
    ctx.fail("array parameter " & p.name &
      ", OpenCL C array parameters decay to pointers, the engine packs the value")
    return ""
  else:
    result = emitTypeBare(p.typ) & ' ' & p.name

proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string] =
  ## OpenCL C parameter texts of one proc or kernel, in declaration order,
  ## through the per-parameter `emitParam` policy.
  for p in fn.params:
    result.add ctx.emitParam(p, fn.isKernel)

proc emitComputePrefix(ctx: var PrintCtx, fn: CirProc): string =
  ## OpenCL C linkage prefix of one proc or kernel.
  ##
  ## - a kernel is `__kernel`
  ## - a device function is `static inline`: C99 linkage that keeps the name
  ##   file-local to the program
  if fn.isKernel:
    return "__kernel "
  return "static inline "

proc emitConstDecl(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope `__constant` declaration of a const_mem local.
  if s.declType.kind == ctArray:
    result = "__constant " & emitOpenclArrayDecl(s.declType, s.name)
    if s.init != nil and s.init.kind == eAconstr:
      var elems: seq[string]
      for el in s.init.elems:
        elems.add ctx.printExpr(el)
      result &= " = {" & elems.join(", ") & "};"
      return
  else:
    result = "__constant " & emitTypeBare(s.declType) & ' ' & s.name
  result &= " = " & ctx.printExpr(s.init) & ";"

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope declaration text of one relocated const_mem local.
  ## OpenCL C forbids declaring `__constant` storage without an initializer:
  ## the printer refuses a const_mem local that lacks one.
  if s.init == nil:
    ctx.fail("const_mem local " & s.name & " without an initializer")
    return ""
  ctx.emitConstDecl(s)

proc printOpenclModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full OpenCL C translation unit: extension pragmas, struct typedefs,
  ## program-scope `__constant` declarations, the device functions,
  ## and the kernels.
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>", policy: openclKindRow,
    lit: openclLitRow, row: openclScalarRow)
  let kernels = moduleKernels(m)
  if kernels.len == 0:
    ctx.fail("no kernel in the module")
    return ("", ctx.err)

  result.src = emitExtensions(m.features)

  result.src &= printComputeStructs(ctx, m, true, "char _padding_")
  if ctx.failing():
    return ("", ctx.err)

  # program-scope `__constant` declarations from `m.globals`, relocated there by the OpenCL lowering pass in encounter order
  result.src &= printComputeGlobals(ctx, m, {caConstant})
  if ctx.failing():
    return ("", ctx.err)

  # forward declarations, then device function definitions, then the kernel
  let fns = moduleDeviceFns(m)
  for fn in fns:
    result.src &= printComputeSig(ctx, fn) & ";\n"
    if ctx.failing():
      return ("", ctx.err)
  if fns.len > 0:
    result.src &= "\n"
  for fn in fns:
    # device functions declare no kernel storage, printVar refuses it there
    ctx.inKernel = false
    result.src &= printComputeProc(ctx, fn) & "\n\n"
  if ctx.failing():
    return ("", ctx.err)

  # every kernel of the module, in encounter order: one OpenCL program per module, the engine launches each kernel by name
  for kernel in kernels:
    ctx.inKernel = true
    result.src &= printComputeProc(ctx, kernel) & "\n"
    if ctx.failing():
      return ("", ctx.err)
  return (result.src, "")


proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as OpenCL C source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printOpenclModule(ctx.module))
