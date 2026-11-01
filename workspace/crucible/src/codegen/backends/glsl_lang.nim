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
## - `ptr` locals dissolve into their pointees (GLSL has no function-storage addressing), a surviving pointer local refuses, naming the pass
## - a `var T` device-function parameter binds as `inout`, the qualify pass retyped it to its pointee
## - `{.smem.}` and `{.const_mem.}` kernel locals relocate into `CirModule.globals`, emitted as program-scope `shared`/`const` declarations
##
## Remaining checks:
## - a threadgroup/constant local surviving in a nested scope or device function refuses, a `caDevice`-space local refuses too
## - a loop prints `while (cond)`, a conditionless loop prints `for (;;)`, `{.unroll.}` adds `[[unroll]]`

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, tables]
import ../ir/ast_datatypes
import ./emit_common
import ../passes/layout_std430

type
  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    ## Literal token table of the target, the shared literal head reads it.
    refBlocks: OrderedTable[string, string]
    ## Buffer-reference block declarations to emit, block name -> member text.
    layouts: Table[string, StructLayout]
    ## Stored struct layouts of the module, the layout pass filled them.

const printerTarget = "glsl"

## Optional expression kinds this printer routes through
## its own `printTargetExpr`, the shared dispatcher emits the rest.
const targetExprKinds = {eNil, eAt, ePat, eDeref, eAddr}

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

const glslScalarTokens = ScalarTokens(
  void: "void", bool: "bool",
  i8: "int8_t", i16: "int16_t", i32: "int", i64: "int64_t",
  u8: "uint8_t", u16: "uint16_t", u32: "uint", u64: "uint64_t",
  f16: "float16_t", f32: "float", f64: "double")

proc emitGlslScalar(t: CirType): string =
  ## GLSL name of a non-array leaf type, "" when the caller composes the name.
  case t.kind
  of ctVoid, ctBool, ctInt, ctUint, ctFloat:
    result = emitScalarToken(glslScalarTokens, t)
  of ctVector:
    let elem = emitGlslScalar(t.vElem)
    let prefix = case elem
      of "float": "vec"
      of "double": "dvec"
      of "float16_t": "f16vec"
      of "int": "ivec"
      of "uint": "uvec"
      of "bool": "bvec"
      else: ""
    if prefix.len > 0:
      result = prefix & $t.vN
  of ctStruct:
    result = t.name
  of ctPtr, ctAptr, ctArray, ctProc, ctSpan:
    result = ""

proc emitGlslTypeBare(t: CirType): string =
  ## GLSL name of `t` with no identifier, array dimensions composed into
  ## the name (`float[4]`), the form of casts and constructors.
  case t.kind
  of ctArray:
    result = emitGlslTypeBare(t.aElem) & "[" & $t.aN & "]"
  else:
    result = emitGlslScalar(t)

proc refTypeName(t: CirType): string =
  ## GLSL name of the buffer-reference block binding `t`:
  ## a pointer renders as its reference block type.
  result = "Ref_" & emitGlslTypeBare(t).multiReplace(
    (" ", "_"), ("[", ""), ("]", ""), (",", "_"), ("(", ""), (")", ""))

proc requireRefBlock(ctx: var PrintCtx, pointee: CirType) =
  ## Registers the buffer-reference block of one runtime-array pointee,
  ## deduped by block name. The block's member is the runtime array `x[]`:
  ## a fixed-array element nests after the runtime dimension (`uint8_t x[][32]`),
  ## the unsized dimension is only legal in the outermost position.
  let name = refTypeName(pointee)
  if ctx.refBlocks.hasKey(name):
    return
  let member =
    case pointee.kind
    of ctArray:
      var base = pointee
      var dims = ""
      while base.kind == ctArray:
        dims &= "[" & $base.aN & "]"
        base = base.aElem
      emitGlslScalar(base) & " x[]" & dims & ";"
    of ctVoid, ctProc:
      ctx.fail("buffer reference to " & $pointee.kind)
      ""
    else:
      emitGlslScalar(pointee) & " x[];"
  if member.len > 0:
    ctx.refBlocks[name] = member

proc emitGlslPtrDecl(ctx: var PrintCtx, t: CirType, ident: string): string =
  ## GLSL declaration of a buffer pointer: the reference block handle,
  ## `Ref_T ident`.
  if t.space != caDevice:
    ctx.fail("buffer pointer in " & $t.space & " space")
    return ""
  ctx.requireRefBlock(t.pElem)
  result = refTypeName(t.pElem) & ' ' & ident

proc emitGlslParam(t: CirType, name: string, varParam: bool): string =
  ## GLSL parameter of a device function. A `var` binding gains an `inout` qualifier.
  ## An array composes the dimension after the name (`float a[8]`).
  let binding = if varParam: "inout " else: ""
  if t.kind == ctArray:
    result = binding & emitGlslScalar(t.aElem) & ' ' & name & "[" & $t.aN & "]"
  else:
    result = binding & emitGlslTypeBare(t) & ' ' & name

# ═══════════════════════════════════════════════════════════════════════
# Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

proc glslCoordName(kind: CoordBuiltinKind): string =
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
  ## GLSL extension block from the stored module feature scan
  ## (`CirModule.features`, filled by the feature-scan pass), `#version 450` first.
  result = "#version 450\n"
  result.add "#extension GL_EXT_buffer_reference : enable\n"
  result.add "#extension GL_EXT_buffer_reference2 : enable\n"
  if features.controlFlow:
    result.add "#extension GL_EXT_control_flow_attributes : require\n"
  if features.fp64:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable\n"
  if features.fp16:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\n"
    result.add "#extension GL_EXT_shader_16bit_storage : enable\n"
  if features.int64:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable\n"
  if features.int16:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable\n"
    result.add "#extension GL_EXT_shader_16bit_storage : enable\n"
  if features.int8:
    result.add "#extension GL_EXT_shader_explicit_int8 : enable\n"
    result.add "#extension GL_EXT_shader_8bit_storage : enable\n"
  if features.subgroup:
    result.add "#extension GL_KHR_shader_subgroup_basic : enable\n"
    result.add "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"
  if features.subgroupShuffle:
    result.add "#extension GL_KHR_shader_subgroup_basic : enable\n"
    result.add "#extension GL_KHR_shader_subgroup_shuffle : enable\n"
  if features.atomicFloat:
    result.add "#extension GL_EXT_shader_atomic_float : enable\n"
  result.add "\n"

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

# GLSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
# `uintBitsToFloat` (32 bit) and the width conversion at the use site.
# Wide integer literals print as constructor calls (`int64_t(5)`), glslang rejects the `LL` suffix.
const targetLitTokens = LitTokens(
  scalar: glslScalarTokens,
  int: IntLitTokens(
    u64Prefix: "uint64_t(", u64Suffix: ")",
    u32Suffix: "U",
    i64Prefix: "int64_t(", i64Suffix: ")",
    narrowCast: true),
  f16Prefix: "float16_t(", f16Suffix: ")",
  f32Suffix: "f",
  f64Prefix: "double(", f64Suffix: ")",
  nonFinite: FloatLitTokens(
    inf16: "float16_t(uintBitsToFloat(0x7f800000u))",
    negInf16: "float16_t(uintBitsToFloat(0xff800000u))",
    nan16: "float16_t(uintBitsToFloat(0x7fc00000u))",
    inf32: "uintBitsToFloat(0x7f800000u)",
    negInf32: "uintBitsToFloat(0xff800000u)",
    nan32: "uintBitsToFloat(0x7fc00000u)",
    inf64: "double(uintBitsToFloat(0x7f800000u))",
    negInf64: "double(uintBitsToFloat(0xff800000u))",
    nan64: "double(uintBitsToFloat(0x7fc00000u))"))

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## One conversion, the constructor-call form: a reference type constructs
  ## from its element type.
  let t = e.castType
  if t.kind in {ctAptr, ctPtr}:
    ctx.requireRefBlock(t.pElem)
    return refTypeName(t.pElem) & "(" & ctx.printExpr(e.castOperand) & ")"
  return emitGlslTypeBare(t) & "(" & ctx.printExpr(e.castOperand) & ")"

proc printCast(ctx: var PrintCtx, e: CirExpr): string =
  ## One bit reinterpret, the GLSL floatBits builtins.
  ##
  ## - `uintBitsToFloat(u)` reads the pattern, a `T()` constructor call
  ##   numerically converts
  ## - core GLSL floatBits builtins cover the 32-bit float pairs only
  ## - a same-width integer pair keeps the constructor, the modular
  ##   conversion preserves every bit
  ## - a wider float pair refuses, a numeric conversion would lose the bits
  let t = e.castType
  let src = getExprType(e.castOperand)
  let scalar = t.kind in {ctInt, ctUint, ctFloat} and src != nil and
    src.kind in {ctInt, ctUint, ctFloat} and src.bits == t.bits
  if scalar and src.kind == ctFloat and t.kind != ctFloat:
    if src.bits == 32:
      return (if t.kind == ctUint: "floatBitsToUint" else: "floatBitsToInt") &
        "(" & ctx.printExpr(e.castOperand) & ")"
    ctx.fail("bit cast of a 64-bit float has no GLSL emission, only the" &
      " 32-bit floatBits builtins")
    return ""
  if scalar and t.kind == ctFloat and src.kind != ctFloat:
    if t.bits == 32:
      return (if src.kind == ctUint: "uintBitsToFloat" else: "intBitsToFloat") &
        "(" & ctx.printExpr(e.castOperand) & ")"
    ctx.fail("bit cast to a 64-bit float has no GLSL emission, only the" &
      " 32-bit floatBits builtins")
    return ""
  # pointers and same-width integer pairs keep the constructor call
  return ctx.printConv(e)

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
      let g = glslCoordName(e.coordBuiltin)
      return "ivec3(int(" & g & ".x), int(" & g & ".y), int(" & g & ".z))"
    if e.typ != nil and e.typ.kind == ctPtr and e.typ.space != caDevice:
      ctx.fail("pointer local " & e.name & " used as a value, GLSL has no" &
        " pointer values, the GLSL lowering pass dissolves pointer locals")
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
      ctx.requireRefBlock(bt.pElem)
      return baseTxt & ".x[" & idxTxt & "]"
    if bt != nil and bt.kind == ctArray:
      return baseTxt & "[" & idxTxt & "]"
    if bt != nil and bt.kind == ctPtr:
      ctx.fail("index through a pointer, the GLSL lowering pass dissolves" &
        " pointer locals")
      return ""
    ctx.fail("index of a " & (if bt == nil: "untyped" else: $bt.kind) &
      " base is outside the GLSL subset")
    return ""
  of eDot:
    # a coordinate component casts to the canonical int, a vector component
    # takes its letter, names pass through
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      # a component of an unsigned GLSL builtin, cast to the canonical int
      return "int(" & glslCoordName(e.dotBase.coordBuiltin) & "." & e.fieldName & ")"
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
      ctx.requireRefBlock(ot.pElem)
      return ctx.printExpr(op) & ".x[0]"
    if ot != nil and ot.kind == ctPtr:
      ctx.fail("deref of a pointer, GLSL has no function-storage addressing," &
        " the GLSL lowering pass dissolves pointer locals")
      return ""
    ctx.fail("deref of a " & (if ot == nil: "untyped" else: $ot.kind) &
      " operand is outside the GLSL subset")
    return ""
  of eAddr:
    # a buffer element's address is the indexed reference access, anything else refuses
    if e.addrOperand.kind == ePat:
      return ctx.printExpr(e.addrOperand)
    ctx.fail("cannot take the address of a local, GLSL has no " &
      "function-storage addressing; only the address of a buffer element")
    return ""
  of eCast:
    return ctx.printCast(e)
  of eConv:
    return ctx.printConv(e)
  of eSizeof:
    return $typeByteSize(ctx.layouts, e.ofType, std430BoolWidth)
  of eOconstr:
    # constructor-call form, a zero-field struct's padding member zero-initializes
    if e.kvFields.len == 0:
      return emitGlslTypeBare(e.ocType) & "(0)"
    result = emitGlslTypeBare(e.ocType) & "("
    for i, kvv in e.kvFields:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(kvv.value)
    result.add ")"
  of eAconstr:
    # constructor-call form
    result = emitGlslTypeBare(e.acType) & "("
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


const glslLoopStyle = LoopStyle(
  unrollDirective: "[[unroll]]",
  infiniteHead: "for (;;) {\n",
  braceSameLine: true)

# ═══════════════════════════════════════════════════════════════════════
# Statements: per-target policy arms, the shared dispatch lives in emit_common
# ═══════════════════════════════════════════════════════════════════════
proc printVar(ctx: var PrintCtx, s: CirStmt): string =
  ## One local declaration: pointer and storage-class refusals, then the declaration text.
  let pad = spaces(ctx.indent * 2)
  if s.declType.kind == ctPtr:
    ctx.fail("pointer local " & s.name & ", GLSL has no function-storage" &
      " addressing, the GLSL lowering pass dissolves pointer locals")
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
    decl = emitGlslTypeBare(s.declType) & ' ' & s.name
  result = pad & decl
  if s.init != nil:
    result &= " = " & ctx.printExpr(s.init)
  result &= ";"

proc printConst(ctx: var PrintCtx, s: CirStmt): string =
  ## One local constant declaration, the `const` form.
  let pad = spaces(ctx.indent * 2)
  result = pad & "const " & emitGlslTypeBare(s.declType) & ' ' &
    s.name & " = " & ctx.printExpr(s.init) & ";"

proc printAsgn(ctx: var PrintCtx, s: CirStmt): string =
  ## One assignment: the pointer-rebind refusal, then the assignment text.
  let pad = spaces(ctx.indent * 2)
  if s.lhs.kind == eSym and s.lhs.typ != nil and s.lhs.typ.kind == ctPtr:
    ctx.fail("pointer local " & s.lhs.name & " rebound, GLSL has no" &
      " pointer values, the GLSL lowering pass dissolves pointer locals")
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

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt]): string =
  ## Prints one statement list, the shared walk with the GLSL loop style.
  printStmtList(ctx, stmts, glslLoopStyle)

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc printFn(ctx: var PrintCtx, fn: CirProc): string =
  ## One device function: signature, body, closing brace.
  ctx.fnName = fn.name
  ctx.indent = 0
  let bodyTxt = ctx.printStmts(fn.body)
  if ctx.failing():
    return ""
  var params: seq[string]
  for p in fn.params:
    case p.typ.kind
    of ctAptr:
      params.add ctx.emitGlslPtrDecl(p.typ, p.name)
    of ctPtr:
      ctx.fail("pointer parameter " & p.name &
        ", GLSL device functions take buffer references or values")
      return ""
    else:
      params.add emitGlslParam(p.typ, p.name, p.varParam)
  let ret = emitGlslTypeBare(fn.ret)
  let sig = ret & ' ' & fn.name & '(' & params.join(", ") & ")"
  return sig & " {\n" & bodyTxt & '\n' & "}"

proc emitSmemDecl(t: CirType, name: string): string =
  ## Program-scope `shared` declaration of a threadgroup local.
  if t.kind == ctArray:
    result = "shared " & emitGlslScalar(t.aElem) & ' ' & name &
      "[" & $t.aN & "];"
  else:
    result = "shared " & emitGlslTypeBare(t) & ' ' & name & ";"

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
    result = "const " & emitGlslTypeBare(s.declType) & ' ' & s.name
  result &= " = " & ctx.printExpr(s.init) & ";"

proc registerTypeRefBlocks(ctx: var PrintCtx, t: CirType) =
  ## Registers the buffer-reference blocks of every device pointer
  ## reachable from the type tree, through pointer targets and array
  ## elements and object fields, dedup by reference-block name.
  if t.isNil:
    return
  case t.kind
  of ctAptr:
    ctx.requireRefBlock(t.pElem)
    ctx.registerTypeRefBlocks(t.pElem)
  of ctPtr:
    ctx.registerTypeRefBlocks(t.pElem)
  of ctArray:
    ctx.registerTypeRefBlocks(t.aElem)
  of ctVector:
    ctx.registerTypeRefBlocks(t.vElem)
  of ctStruct:
    for f in t.fields:
      ctx.registerTypeRefBlocks(f.typ)
  else:
    discard

proc registerExprRefBlocks(ctx: var PrintCtx, e: CirExpr) =
  ## Registers the buffer-reference blocks reachable from one expression:
  ## every buffer pointer subexpression's pointee needs its block declared
  ## before the module body prints, a cast target included.
  if e.isNil:
    return
  let t = getExprType(e)
  if t != nil and t.kind == ctAptr:
    ctx.requireRefBlock(t.pElem)
  case e.kind
  of eCall:
    ctx.registerExprRefBlocks(e.callee)
    for a in e.args:
      ctx.registerExprRefBlocks(a)
  of eAt:
    ctx.registerExprRefBlocks(e.atBase)
    ctx.registerExprRefBlocks(e.atIndex)
  of ePat:
    ctx.registerExprRefBlocks(e.patBase)
    ctx.registerExprRefBlocks(e.patIndex)
  of eDot:
    ctx.registerExprRefBlocks(e.dotBase)
  of eDeref:
    ctx.registerExprRefBlocks(e.derefOperand)
  of eAddr:
    ctx.registerExprRefBlocks(e.addrOperand)
  of eCast, eConv:
    ctx.registerExprRefBlocks(e.castOperand)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    ctx.registerExprRefBlocks(e.left)
    ctx.registerExprRefBlocks(e.right)
  of eAnd, eOr:
    ctx.registerExprRefBlocks(e.andLeft)
    ctx.registerExprRefBlocks(e.andRight)
  of eTernary:
    ctx.registerExprRefBlocks(e.cond)
    ctx.registerExprRefBlocks(e.thenE)
    ctx.registerExprRefBlocks(e.elseE)
  of eEq, eNeq, eLt, eLe:
    ctx.registerExprRefBlocks(e.a)
    ctx.registerExprRefBlocks(e.b)
  of eBitNot, eNeg:
    ctx.registerExprRefBlocks(e.unOperand)
  of eNot:
    ctx.registerExprRefBlocks(e.notOperand)
  of eOconstr:
    for kvf in e.kvFields:
      ctx.registerExprRefBlocks(kvf.value)
  of eAconstr:
    for el in e.elems:
      ctx.registerExprRefBlocks(el)
  of eNil, eIntLit, eFloatLit, eTrue, eFalse, eSizeof, eSym:
    discard

proc registerStmtRefBlocks(ctx: var PrintCtx, ss: seq[CirStmt]) =
  ## Registers the buffer-reference blocks reachable from one statement list.
  for s in ss:
    case s.kind
    of sVar, sConst:
      if s.declType.kind == ctAptr:
        ctx.requireRefBlock(s.declType.pElem)
      ctx.registerExprRefBlocks(s.init)
    of sAsgn:
      ctx.registerExprRefBlocks(s.lhs)
      ctx.registerExprRefBlocks(s.rhs)
    of sIte:
      ctx.registerExprRefBlocks(s.cond)
      ctx.registerStmtRefBlocks(s.thenBody)
      ctx.registerStmtRefBlocks(s.elseBody)
    of sLoop:
      ctx.registerExprRefBlocks(s.loopCond)
      ctx.registerStmtRefBlocks(s.loopBody)
    of sCase:
      ctx.registerExprRefBlocks(s.selector)
      for br in s.branches:
        ctx.registerStmtRefBlocks(br.body)
      ctx.registerStmtRefBlocks(s.caseElse)
    of sScope:
      ctx.registerStmtRefBlocks(s.scopeBody)
    of sAssume:
      ctx.registerExprRefBlocks(s.assumeCond)
    of sRet:
      ctx.registerExprRefBlocks(s.retVal)
    of sDiscard:
      ctx.registerExprRefBlocks(s.discardVal)
    of sCall:
      ctx.registerExprRefBlocks(s.callCallee)
      for a in s.callArgs:
        ctx.registerExprRefBlocks(a)
    of sAsm, sEmit:
      discard

proc printGlslModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full GLSL translation unit: version and extensions,
  ## buffer-reference blocks, struct declarations, the push-constant block,
  ## program-scope `shared` and `const` declarations, the device functions,
  ## and the kernel with its `local_size` layout qualifier.
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>")
  ctx.layouts = m.layouts
  var kernels: seq[CirProc]
  for fn in m.procs:
    if fn.isKernel:
      kernels.add fn
  if kernels.len == 0:
    ctx.fail("no kernel in the module")
    return ("", ctx.err)

  # ref-block pre-pass: registers the blocks the declarations below use
  for t in m.types:
    ctx.registerTypeRefBlocks(t)
  for fn in m.procs:
    for p in fn.params:
      if p.typ.kind in {ctAptr, ctPtr}:
        ctx.requireRefBlock(p.typ.pElem)
    ctx.registerStmtRefBlocks(fn.body)

  result.src = emitExtensions(m.features)

  for name, member in ctx.refBlocks:
    result.src &= "layout(buffer_reference, std430) buffer " & name &
      " { " & member & " };\n"
  if ctx.refBlocks.len > 0:
    result.src &= "\n"

  for t in m.types:
    if t.kind != ctStruct:
      ctx.fail("module type " & t.name & " is not a struct")
      break
    result.src &= "struct " & t.name & " {\n"
    if t.fields.len == 0:
      result.src &= "  uint _padding;\n"
    for f in t.fields:
      var field = emitGlslTypeBare(f.typ) & ' ' & f.name
      if f.typ.kind == ctArray:
        field = emitGlslScalar(f.typ.aElem) & ' ' & f.name &
          "[" & $f.typ.aN & "]"
      elif f.typ.kind in {ctAptr, ctPtr}:
        field = ctx.emitGlslPtrDecl(f.typ, f.name)
      result.src &= "  " & field & ";\n"
    result.src &= "};\n\n"

  # one push block for the whole module: the union of every kernel's parameters, same-name parameters dedup by their declaration text
  var pushDecls: seq[string]
  for kernel in kernels:
    for p in kernel.params:
      var decl: string
      case p.typ.kind
      of ctAptr, ctPtr:
        # a device pointer packs as the buffer-reference handle: the engine passes the device address, the block dereferences it
        decl = ctx.emitGlslPtrDecl(p.typ, p.name)
      of ctArray:
        ctx.fail("kernel parameter " & p.name &
          " of array type, arrays do not pack into the push block")
        decl = ""
      else:
        decl = emitGlslTypeBare(p.typ) & ' ' & p.name
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
  var hoisted: seq[string]
  for s in m.globals:
    if s.kind == sVar and s.space == caThreadgroup:
      hoisted.add emitSmemDecl(s.declType, s.name)
    elif s.kind == sVar and s.space == caConstant:
      if s.init == nil:
        ctx.fail("const_mem local " & s.name & " without an initializer")
      else:
        hoisted.add ctx.emitConstDecl(s)
  for d in hoisted:
    result.src &= d & "\n"
  if hoisted.len > 0:
    result.src &= "\n"
  if ctx.failing():
    return ("", ctx.err)

  # forward declarations, then device function definitions, then the kernel
  var fns: seq[CirProc]
  for fn in m.procs:
    if not fn.isKernel:
      fns.add fn
  for fn in fns:
    ctx.fnName = fn.name
    var params: seq[string]
    for p in fn.params:
      case p.typ.kind
      of ctAptr:
        params.add ctx.emitGlslPtrDecl(p.typ, p.name)
      of ctPtr:
        ctx.fail("pointer parameter " & p.name &
          ", GLSL device functions take buffer references or values")
      else:
        params.add emitGlslParam(p.typ, p.name, p.varParam)
    if ctx.failing():
      return ("", ctx.err)
    result.src &= emitGlslTypeBare(fn.ret) & ' ' & fn.name & '(' &
      params.join(", ") & ");\n"
  if fns.len > 0:
    result.src &= "\n"
  for fn in fns:
    result.src &= ctx.printFn(fn) & "\n\n"
  if ctx.failing():
    return ("", ctx.err)

  # every kernel of the module, in encounter order, one local_size layout
  # qualifier per kernel, the engine launches each entry point by name
  for kernel in kernels:
    let wg = (if kernel.workgroup == [0, 0, 0]: [256, 1, 1] else: kernel.workgroup)
    result.src &= "layout(local_size_x = " & $wg[0] & ", local_size_y = " &
      $wg[1] & ", local_size_z = " & $wg[2] & ") in;\n"
    ctx.fnName = kernel.name
    ctx.indent = 0
    let bodyTxt = ctx.printStmts(kernel.body)
    if ctx.failing():
      return ("", ctx.err)
    # exportc renames a kernel entry for the host runtime lookup.
    # A device function keeps the name its call sites reference.
    let entryName = if kernel.isKernel and kernel.exportName.len > 0: kernel.exportName else: kernel.name
    result.src &= "void " & entryName & "() {\n" & bodyTxt & '\n' & "}\n\n"
  return (result.src, "")


proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as GLSL source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printGlslModule(ctx.module))
