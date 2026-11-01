# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GLSL 450 (Vulkan compute) printer: a structured CIRv2 module
## rendered as GLSL source, table-driven over the closed set of type kinds.
##
## Emission contract:
## - one `{.global.}` kernel per module, the device functions it calls emitted alongside
## - kernel signature `void <name>()`: one push-constant block member per kernel parameter, in host `store` packing order
## - buffer pointers render as buffer-reference handles (`GL_EXT_buffer_reference`), element access `<handle>.x[index]`
## - `eSizeof` reports the packed byte size from the stored module layout records, GLSL has no `sizeof`
##
## Lowering-pass contract (`passes_lowering_glsl.nim` runs before printing):
## - `ptr` locals rewrite to their pointees (GLSL has no function-storage addressing),
##   the pass refuses a surviving pointer local, the refusal message
##   identifies the pass
## - a `var T` device-function parameter binds as `inout`, the qualify pass retyped it to its pointee
## - `{.smem.}` and `{.const_mem.}` kernel locals relocate into `CirModule.globals`, emitted as program-scope `shared`/`const` declarations
## - the ref-block pass registers every buffer-reference block into
##   `CirModule.refBlocks`, the printer declares one per entry in order
##
## Remaining checks:
## - the printer refuses a threadgroup/constant local surviving in a nested scope or device function, and a `caDevice`-space local too
## - a loop prints `while (cond)`, `{.unroll.}` adds `[[unroll]]`

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, tables]
import ../ir/ast_datatypes
import ./print_common
import ./print_compute
import ./targets_registry
import ../passes/layout_std430

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
    layouts: Table[string, StructLayout]
    ## Stored struct layouts of the module, the layout pass filled them.

const printerTarget = "glsl"

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitGlslScalar(t: CirType): string =
  ## GLSL name of a non-array leaf type, "" when the caller composes the name.
  getLeafName(glslScalarRow, glslVectorRule, t)

proc emitTypeBare(t: CirType): string =
  ## GLSL name of `t` with no identifier, array dimensions composed into
  ## the name (`float[4]`), the form of casts and constructors.
  case t.kind
  of ctArray:
    result = emitTypeBare(t.aElem) & "[" & $t.aN & "]"
  else:
    result = emitGlslScalar(t)

proc refTypeName*(t: CirType): string =
  ## Returns the GLSL name of the buffer-reference block binding `t`.
  ## A pointer renders as its reference block type. The GLSL ref-block
  ## analysis pass dedups blocks by this name.
  result = "Ref_" & emitTypeBare(t).multiReplace(
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

proc emitGlslParam(t: CirType, name: string, varParam: bool): string =
  ## GLSL parameter of a device function. A `var` binding gains an `inout` qualifier.
  ## An array composes the dimension after the name (`float a[8]`).
  let binding = if varParam: "inout " else: ""
  if t.kind == ctArray:
    result = binding & emitGlslScalar(t.aElem) & ' ' & name & "[" & $t.aN & "]"
  else:
    result = binding & emitTypeBare(t) & ' ' & name

proc emitComputeStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  ## GLSL declaration text of one struct member: a buffer pointer member is
  ## the reference block handle, an array stacks the dimension behind the name.
  if f.typ.kind == ctArray:
    return emitGlslScalar(f.typ.aElem) & ' ' & f.name & "[" & $f.typ.aN & "]"
  if f.typ.kind in {ctAptr, ctPtr}:
    return ctx.emitGlslPtrDecl(f.typ, f.name)
  return emitTypeBare(f.typ) & ' ' & f.name

# ═══════════════════════════════════════════════════════════════════════
# Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

proc getGlslCoordName(kind: CoordBuiltinKind): string =
  ## GLSL name of a canonical coordinate builtin kind: all six are native
  ## GLSL builtins, a plain rename.
  case kind
  of cbkThreadPositionInGrid: "gl_GlobalInvocationID"
  of cbkThreadgroupPositionInGrid: "gl_WorkGroupID"
  of cbkThreadPositionInThreadgroup: "gl_LocalInvocationID"
  of cbkThreadsPerThreadgroup: "gl_WorkGroupSize"
  of cbkThreadgroupsPerGrid: "gl_NumWorkGroups"
  of cbkThreadIndexInThreadgroup: "gl_LocalInvocationIndex"
  of cbkNone: ""

proc emitExtensions(features: ModuleFeatures): string =
  ## GLSL extension block from the stored module feature scan (`CirModule.features`):
  ## `#version 450` prints first, then one line per feature in use.
  result = "#version 450\n"
  result.add "#extension GL_EXT_buffer_reference : enable\n"
  result.add "#extension GL_EXT_buffer_reference2 : enable\n"
  result.add emitFeatureLines(features, [
    (features.controlFlow, "#extension GL_EXT_control_flow_attributes : require\n"),
    (features.fp64, "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable\n"),
    (features.fp16, "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\n"),
    (features.fp16, "#extension GL_EXT_shader_16bit_storage : enable\n"),
    (features.int64, "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable\n"),
    (features.int16, "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable\n"),
    (features.int16, "#extension GL_EXT_shader_16bit_storage : enable\n"),
    (features.int8, "#extension GL_EXT_shader_explicit_int8 : enable\n"),
    (features.int8, "#extension GL_EXT_shader_8bit_storage : enable\n"),
    (features.subgroup, "#extension GL_KHR_shader_subgroup_basic : enable\n"),
    (features.subgroup, "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"),
    (features.subgroupShuffle, "#extension GL_KHR_shader_subgroup_basic : enable\n"),
    (features.subgroupShuffle, "#extension GL_KHR_shader_subgroup_shuffle : enable\n"),
    (features.atomicFloat, "#extension GL_EXT_shader_atomic_float : enable\n")])

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## One conversion, the constructor-call form: a reference type constructs
  ## from its element type.
  let t = e.castType
  if t.kind in {ctAptr, ctPtr}:
    return refTypeName(t.pElem) & "(" & ctx.printExpr(e.castOperand) & ")"
  return emitTypeBare(t) & "(" & ctx.printExpr(e.castOperand) & ")"


proc printGlslAtomicFetchAdd(ctx: var PrintCtx, args: seq[CirExpr]): string =
  ## GLSL vocabulary for the atomic fetch-and-add builtin, shared by the value and statement call paths:
  ##
  ## - int atomicAdd is core
  ## - fp needs GL_EXT_shader_atomic_float, enabled at module scope
  ##   for an fp atomicFetchAdd in the module
  ## - atomicAdd increments through one indirection, the address operand
  ##   is the reference's element 0
  let addrTxt = if getExprType(args[0]) != nil and getExprType(
      args[0]).kind in {ctAptr, ctPtr}:
    ctx.printExpr(args[0]) & ".x[0]"
  else:
    ctx.printExpr(args[0])
  return "atomicAdd(" & addrTxt & ", " & ctx.printExpr(args[1]) & ")"


proc printBuiltinCall(ctx: var PrintCtx, call: CirExpr, args: seq[string]): string =
  ## Returns the GLSL text of an expression-position builtin call,
  ## "" when `call.callee` is not a builtin.
  if call.callee.kind == eSym and call.callee.atomicsBuiltin == abkAtomicFetchAdd:
    # atomicFetchAdd prints its own arguments, a pre-print into the shared arg list would print each argument twice
    return ctx.printGlslAtomicFetchAdd(call.args)
  if call.callee.kind == eSym:
    # builtin vocabulary dispatches on the typed kinds, never on names
    case call.callee.reductionBuiltin
    of rbkReduceAdd:
      return "subgroupAdd(" & args.join(", ") & ")"
    of rbkReduceMax:
      return "subgroupMax(" & args.join(", ") & ")"
    of rbkNone: discard
  if call.callee.kind == eSym:
    case call.callee.shuffleBuiltin
    of shkSimdShuffle:
      # subgroupShuffle takes the subgroup-relative index as `uint`. A plain cast
      # compiles without GL_NV_explicit_typecast
      return "subgroupShuffle(" & args[0] & ", uint(" & args[1] & "))"
    of shkSimdShuffleXor:
      return "subgroupShuffleXor(" & args[0] & ", uint(" & args[1] & "))"
    of shkNone: discard
    if call.callee.synchroBuiltin != sbkNone:
      ctx.fail("barrier is a statement, a value-position barrier is outside the GLSL subset")
      return ""
  if call.callee.kind == eSym and call.callee.builtin and
      glslConvBuiltinNames.hasKey(call.callee.name):
    return glslConvBuiltinNames.getOrDefault(call.callee.name) % args.join(", ")
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════


proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## GLSL text of the kinds in `targetExprKinds`.
  case e.kind
  of eNil:
    ctx.fail("nil literal is outside the GLSL subset")
    return ""
  of eSym:
    # a pointer-value refusal, coordinate builtins take component casts,
    # names pass through
    if e.coordBuiltin != cbkNone:
      if e.coordBuiltin == cbkThreadIndexInThreadgroup:
        return "int(gl_LocalInvocationIndex)"
      let g = getGlslCoordName(e.coordBuiltin)
      return "ivec3(int(" & g & ".x), int(" & g & ".y), int(" & g & ".z))"
    if e.typ != nil and e.typ.kind == ctPtr and e.typ.space != caDevice:
      ctx.fail("pointer local " & e.name & " used as a value, GLSL has no" &
        " pointer values, the GLSL lowering pass rewrites pointer locals to their pointees")
      return ""
    return e.name
  of eAt, ePat:
    # reference blocks index their storage element, arrays index plainly,
    # pointers and untyped bases refuse
    let base = if e.kind == eAt: e.atBase else: e.patBase
    let idx = if e.kind == eAt: e.atIndex else: e.patIndex
    let idxTxt = ctx.printExpr(idx)
    let bt = getExprType(base)
    var baseTxt = ctx.printExpr(base)
    if base.kind in {eCast, eConv}:
      # a cast binds looser than the subscript: parenthesize the base
      baseTxt = "(" & baseTxt & ")"
    if bt != nil and bt.kind == ctAptr:
      return baseTxt & ".x[" & idxTxt & "]"
    if bt != nil and bt.kind == ctArray:
      return baseTxt & "[" & idxTxt & "]"
    if bt != nil and bt.kind == ctPtr:
      ctx.fail("index through a pointer, the GLSL lowering pass rewrites" &
        " pointer locals to their pointees")
      return ""
    ctx.fail("index of a " & (if bt == nil: "untyped" else: $bt.kind) &
      " base is outside the GLSL subset")
    return ""
  of eDot:
    # a coordinate component casts to the canonical int, a vector component
    # takes its letter, names pass through
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      # a component of an unsigned GLSL builtin, cast to the canonical int
      return "int(" & getGlslCoordName(e.dotBase.coordBuiltin) & "." & e.fieldName & ")"
    let bt = getExprType(e.dotBase)
    if bt != nil and bt.kind == ctVector:
      let comp = emitVectorComponent(e.fieldName)
      if comp.len == 0:
        ctx.fail("no vector component " & e.fieldName)
        return ""
      return ctx.printExpr(e.dotBase) & "." & comp
    return ctx.printExpr(e.dotBase) & "." & e.fieldName
  of eDeref:
    # a reference block reads its storage element, pointers and untyped operands refuse
    let op = e.derefOperand
    let ot = getExprType(op)
    if ot != nil and ot.kind == ctAptr:
      return ctx.printExpr(op) & ".x[0]"
    if ot != nil and ot.kind == ctPtr:
      ctx.fail("deref of a pointer, GLSL has no function-storage addressing," &
        " the GLSL lowering pass rewrites pointer locals to their pointees")
      return ""
    ctx.fail("deref of a " & (if ot == nil: "untyped" else: $ot.kind) &
      " operand is outside the GLSL subset")
    return ""
  of eAddr:
    # a buffer element's address is the indexed reference access, the printer refuses anything else
    if e.addrOperand.kind == ePat:
      return ctx.printExpr(e.addrOperand)
    ctx.fail("cannot take the address of a local, GLSL has no " &
      "function-storage addressing; only the address of a buffer element")
    return ""
  of eCast:
    return ctx.printCast(e, glslCastRules, emitTypeBare(e.castType))
  of eConv:
    return ctx.printConv(e)
  of eSizeof:
    return $typeByteSize(ctx.layouts, e.ofType, std430BoolWidth)
  of eOconstr:
    # constructor-call form, a zero-field struct's padding member zero-initializes
    if e.kvFields.len == 0:
      return emitTypeBare(e.ocType) & "(0)"
    result = emitTypeBare(e.ocType) & "("
    for i, kvv in e.kvFields:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(kvv.value)
    result.add ")"
  of eAconstr:
    # constructor-call form
    result = emitTypeBare(e.acType) & "("
    for i, el in e.elems:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(el)
    result.add ")"
  of eCall:
    let args = printCallArgs(ctx, e)
    return printCallText(ctx, e, args, printBuiltinCall(ctx, e, args))
  else:
    # dispatcher sends only the kinds above, anything else is a routing bug
    ctx.fail("expression kind " & $e.kind & " has no GLSL emission")


# ═══════════════════════════════════════════════════════════════════════
# Statements: per-target policy arms, the shared dispatch lives in print_common
# ═══════════════════════════════════════════════════════════════════════
proc printVar(ctx: var PrintCtx, s: CirStmt): string =
  ## One local declaration: pointer and storage-class refusals, then the declaration text.
  let pad = spaces(ctx.indent * 2)
  if s.declType.kind == ctPtr:
    ctx.fail("pointer local " & s.name & ", GLSL has no function-storage" &
      " addressing, the GLSL lowering pass rewrites pointer locals to their pointees")
    return ""
  if s.space == caThreadgroup:
    ctx.fail("local " & s.name & " in threadgroup space, the GLSL lowering" &
      " pass hoists threadgroup locals to program scope")
    return ""
  if s.space == caConstant:
    ctx.fail("local " & s.name & " in constant space, the GLSL lowering" &
      " pass hoists const_mem locals to program scope")
    return ""
  if s.space == caDevice:
    ctx.fail("local " & s.name & " in device space, GLSL locals are private")
    return ""
  var decl: string
  if s.declType.kind == ctAptr:
    # a buffer pointer local is the reference block handle
    decl = ctx.emitGlslPtrDecl(s.declType, s.name)
  elif s.declType.kind == ctArray:
    decl = emitGlslScalar(s.declType.aElem) & ' ' & s.name &
      "[" & $s.declType.aN & "]"
  else:
    decl = emitTypeBare(s.declType) & ' ' & s.name
  result = pad & decl
  if s.init != nil:
    result &= " = " & ctx.printExpr(s.init)
  result &= ";"

proc printConst(ctx: var PrintCtx, s: CirStmt): string =
  ## One local constant declaration, the `const` form.
  let pad = spaces(ctx.indent * 2)
  result = pad & "const " & emitTypeBare(s.declType) & ' ' &
    s.name & " = " & ctx.printExpr(s.init) & ";"

proc printAsgn(ctx: var PrintCtx, s: CirStmt): string =
  ## One assignment: the pointer-rebind refusal, then the assignment text.
  let pad = spaces(ctx.indent * 2)
  if s.lhs.kind == eSym and s.lhs.typ != nil and s.lhs.typ.kind == ctPtr:
    ctx.fail("pointer local " & s.lhs.name & " rebound, GLSL has no" &
      " pointer values, the GLSL lowering pass rewrites pointer locals to their pointees")
    return ""
  result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"

proc printCall(ctx: var PrintCtx, s: CirStmt): string =
  ## One call statement: the barrier and atomic vocabulary, then the plain call form.
  let pad = spaces(ctx.indent * 2)
  if s.callCallee.kind == eSym and
      s.callCallee.synchroBuiltin == sbkThreadgroupBarrier:
    result = pad & "barrier();"
  elif s.callCallee.kind == eSym and
      s.callCallee.synchroBuiltin == sbkThreadgroupBarrierDevice:
    # GLSL's barrier() orders shared memory only, buffer writes are
    # ordered before it by a leading memoryBarrierBuffer() call
    result = pad & "memoryBarrierBuffer();\n" & pad & "barrier();"
  elif s.callCallee.kind == eSym and
      s.callCallee.atomicsBuiltin == abkAtomicFetchAdd:
    # a GLSL call statement discards the fetched old value
    result = pad & ctx.printGlslAtomicFetchAdd(s.callArgs) & ";"
  else:
    var parts: seq[string]
    parts.add ctx.printExpr(s.callCallee)
    for a in s.callArgs:
      parts.add ctx.printExpr(a)
    result = pad & parts[0] & "(" & parts[1 ..^ 1].join(", ") & ");"

proc printAsm(ctx: var PrintCtx, s: CirStmt): string =
  ## One inline assembly statement: refused, the text names the target.
  ctx.fail("inline assembly has no GLSL form")
  result = ""

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string] =
  ## GLSL parameter texts of one proc or kernel, in declaration order.
  ##
  ## - a kernel takes zero parameters: every kernel parameter packs into the module's shared push-constant block, deduped across kernels
  ## - a device function's buffer pointer is the reference block handle,
  ##   a `var` binding gains the `inout` qualifier
  if fn.isKernel:
    return
  for p in fn.params:
    case p.typ.kind
    of ctAptr:
      result.add ctx.emitGlslPtrDecl(p.typ, p.name)
    of ctPtr:
      ctx.fail("pointer parameter " & p.name &
        ", GLSL device functions take buffer references or values")
    else:
      result.add emitGlslParam(p.typ, p.name, p.varParam)

proc emitComputePrefix(ctx: var PrintCtx, fn: CirProc): string =
  ## GLSL linkage prefix: none, GLSL knows no linkage qualifiers.
  ""

proc emitSmemDecl(t: CirType, name: string): string =
  ## Program-scope `shared` declaration of a threadgroup local.
  if t.kind == ctArray:
    result = "shared " & emitGlslScalar(t.aElem) & ' ' & name &
      "[" & $t.aN & "];"
  else:
    result = "shared " & emitTypeBare(t) & ' ' & name & ";"

proc emitConstDecl(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope `const` declaration of one hoisted const_mem local.
  ## Shape guard: the lowering pass refuses an initializer-less const_mem
  ## local before relocation, this check fails loudly on a malformed node.
  if s.init == nil:
    ctx.fail("const_mem local " & s.name & " without an initializer")
    return ""
  if s.declType.kind == ctArray:
    result = "const " & emitGlslScalar(s.declType.aElem) & ' ' & s.name &
      "[" & $s.declType.aN & "]"
  else:
    result = "const " & emitTypeBare(s.declType) & ' ' & s.name
  result &= " = " & ctx.printExpr(s.init) & ";"

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope declaration text of one relocated storage local.
  ##
  ## - a threadgroup local declares bare `shared` storage
  ## - a constant local declares `const`, the printer refuses an initializer-less one
  if s.space == caThreadgroup:
    return emitSmemDecl(s.declType, s.name)
  if s.init == nil:
    ctx.fail("const_mem local " & s.name & " without an initializer")
    return ""
  ctx.emitConstDecl(s)

# ═══════════════════════════════════════════════════════════════════════
# Shared push-constant block contract (emission site)
# ═══════════════════════════════════════════════════════════════════════

proc pushParamText(p: CirParam): string =
  ## Compact text of one kernel parameter for the refusal message,
  ## `(i +32)`-style Leng type vocabulary.
  const spaceName: array[AddressSpace, string] =
    ["thread", "threadgroup", "constant", "device"]
  case p.typ.kind
  of ctBool: result = "bool"
  of ctInt: result = "i" & $p.typ.bits
  of ctUint: result = "u" & $p.typ.bits
  of ctFloat: result = "f" & $p.typ.bits
  of ctVector: result = "vec" & $p.typ.vN
  of ctPtr: result = "ptr " & spaceName[p.typ.space]
  of ctAptr: result = "aptr " & spaceName[p.typ.space]
  of ctStruct: result = p.typ.name
  of ctArray: result = "array"
  of ctSpan: result = "span " & spaceName[p.typ.sSpace]
  else: result = $p.typ.kind
  result = p.name & ": " & result

proc paramsText(params: seq[CirParam]): string =
  ## One kernel's ordered parameter list in the refusal-message vocabulary.
  for i, p in params:
    if i > 0:
      result.add ", "
    result.add pushParamText(p)

proc pushConstLayoutRejection(kernels: seq[CirProc]): string =
  ## Refusal when the kernels of one module declare different ordered params,
  ## "" when every kernel's params agree elementwise.
  ##
  ## Layout mechanics:
  ##
  ## - the GLSL printer unions every kernel's parameters into one module-wide
  ##   push-constant block, same-name dedup by declaration text
  ## - the engine serializes each launched kernel's arguments from offset 0
  ##   in that kernel's own param order
  ## - a member reads the same offset for every kernel only under identical
  ##   ordered layouts
  ## - scalars and pointers alike pack through the block, a pointer packs
  ##   as its buffer-reference address
  if kernels.len < 2:
    return ""
  let reference = kernels[0]
  let referenceParams = reference.params.paramsText
  for k in kernels[1 ..^ 1]:
    block mismatch:
      if k.params.len != reference.params.len:
        break mismatch
      for i in 0 ..< k.params.len:
        if k.params[i].name != reference.params[i].name or
            not sameType(k.params[i].typ, reference.params[i].typ):
          break mismatch
      continue
    return "multi-kernel GLSL sources share one push-constant block, all kernels " &
      "must declare the same ordered params, got kernel `" & k.name & "` (" &
      k.params.paramsText & ") vs kernel `" & reference.name & "` (" &
      referenceParams & "): a launch serializes its args at offsets another " &
      "kernel reads as different members"
  return ""


proc printGlslModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full GLSL translation unit: version and extensions,
  ## buffer-reference blocks, struct declarations, the push-constant block,
  ## program-scope `shared` and `const` declarations, the device functions,
  ## and the kernel with its `local_size` layout qualifier.
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>", policy: glslKindRow,
    lit: glslLitRow, row: glslScalarRow)
  ctx.layouts = m.layouts
  let kernels = moduleKernels(m)
  if kernels.len == 0:
    ctx.fail("no kernel in the module")
    return ("", ctx.err)

  # buffer-reference blocks, one declaration per entry of the registration
  # record in `CirModule.refBlocks`, which the ref-block analysis pass
  # filled after the last transform. The printer never registers a block.
  result.src = emitExtensions(m.features)

  for name, pointee in m.refBlocks:
    let member = ctx.emitGlslRefBlockMember(pointee)
    result.src &= "layout(buffer_reference, std430) buffer " & name &
      " { " & member & " };\n"
    if ctx.failing():
      return ("", ctx.err)
  if m.refBlocks.len > 0:
    result.src &= "\n"

  result.src &= printComputeStructs(ctx, m, false, "uint _padding")
  if ctx.failing():
    return ("", ctx.err)

  # one push block for the whole module: the union of every kernel's parameters,
  # same-name dedup by declaration text. The engine serializes each launched
  # kernel's arguments from offset 0, so the block is sound only under
  # identical ordered layouts and the emission site refuses a divergence
  let pushRejection = pushConstLayoutRejection(kernels)
  if pushRejection.len > 0:
    ctx.fail(pushRejection)
    return ("", ctx.err)
  var pushDecls: seq[string]
  for kernel in kernels:
    for p in kernel.params:
      var decl: string
      case p.typ.kind
      of ctAptr:
        # a device pointer packs as the buffer-reference handle: the engine passes the device address and the block dereferences it
        decl = ctx.emitGlslPtrDecl(p.typ, p.name)
      of ctPtr:
        ctx.fail("kernel parameter " & p.name &
          " of pointer type, kernels take values and device buffers" &
          " (`ptr UncheckedArray[T]`), pass an object's fields separately")
        decl = ""
      of ctArray:
        ctx.fail("kernel parameter " & p.name &
          " of array type, arrays do not pack into the push block")
        decl = ""
      else:
        decl = emitTypeBare(p.typ) & ' ' & p.name
      if decl notin pushDecls:
        pushDecls.add decl
  if ctx.failing():
    return ("", ctx.err)
  if pushDecls.len > 0:
    result.src &= "layout(push_constant, std430) uniform KernelParams {\n"
    for decl in pushDecls:
      result.src &= "  " & decl & ";\n"
    result.src &= "};\n\n"

  # program-scope `shared` and `const` declarations relocated into
  # `m.globals` by the GLSL lowering pass, in encounter order
  result.src &= printComputeGlobals(ctx, m, {caThreadgroup, caConstant})
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
    result.src &= printComputeProc(ctx, fn) & "\n\n"
  if ctx.failing():
    return ("", ctx.err)

  # every kernel of the module, in encounter order, one local_size layout
  # qualifier per kernel, the engine launches each entry point by name
  for kernel in kernels:
    let wg = (if kernel.workgroup == [0, 0, 0]: [256, 1, 1] else: kernel.workgroup)
    result.src &= "layout(local_size_x = " & $wg[0] & ", local_size_y = " &
      $wg[1] & ", local_size_z = " & $wg[2] & ") in;\n"
    result.src &= printComputeProc(ctx, kernel) & "\n\n"
    if ctx.failing():
      return ("", ctx.err)
  return (result.src, "")


proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as GLSL source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printGlslModule(ctx.module))
