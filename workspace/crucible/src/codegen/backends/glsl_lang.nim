# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GLSL 450 (Vulkan compute) printer: a structured CIRv2 module
## rendered as GLSL source, table-driven over the closed type lattice.
##
## Emission contract:
## - one `{.global.}` kernel per module, the device functions it calls emitted alongside
## - kernel signature `void <name>()`: one push-constant block member per kernel parameter, in host `store` packing order
## - buffer pointers render as buffer-reference handles (`GL_EXT_buffer_reference`), element access `<handle>.x[index]`
## - `eSizeof` computes the std430 byte size, GLSL has no `sizeof`
##
## Lowering-pass contract (`passes_lowering_glsl.nim` runs before printing):
## - `ptr` locals dissolve into their pointees (GLSL has no function-storage addressing), a surviving pointer local refuses, naming the pass
## - `{.smem.}` and `{.const_mem.}` kernel locals relocate into `CirModule.globals`, emitted as program-scope `shared`/`const` declarations
## - a threadgroup/constant local surviving in a nested scope or device function refuses, a `caDevice`-space local refuses too
## - jmp to the loop exit label is `break`, to the continue label `continue`, any other lab/jmp refuses

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, tables]
import ../ir/ast_datatypes
import ./emit_common

type
  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    refBlocks: OrderedTable[string, string]
    ## Buffer-reference block declarations to emit, block name -> member text.

const printerTarget = "glsl"

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc spellGlslScalar(t: CirType): string =
  ## GLSL name of a non-array leaf type, "" when the caller composes the name.
  case t.kind
  of ctVoid:
    result = "void"
  of ctBool:
    result = "bool"
  of ctInt:
    case t.bits
    of 8:
      result = "int8_t"
    of 16:
      result = "int16_t"
    of 32:
      result = "int"
    of 64:
      result = "int64_t"
    else:
      result = ""
  of ctUint:
    case t.bits
    of 8:
      result = "uint8_t"
    of 16:
      result = "uint16_t"
    of 32:
      result = "uint"
    of 64:
      result = "uint64_t"
    else:
      result = ""
  of ctFloat:
    case t.bits
    of 16:
      result = "float16_t"
    of 32:
      result = "float"
    of 64:
      result = "double"
    else:
      result = ""
  of ctVector:
    let elem = spellGlslScalar(t.vElem)
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
  of ctPtr, ctAptr, ctArray, ctProc:
    result = ""

proc spellGlslTypeBare(t: CirType): string =
  ## GLSL name of `t` with no identifier, array dimensions composed into
  ## the name (`float[4]`), the form of casts and constructors.
  case t.kind
  of ctArray:
    result = spellGlslTypeBare(t.aElem) & "[" & $t.aN & "]"
  else:
    result = spellGlslScalar(t)

proc refTypeName(t: CirType): string =
  ## GLSL name of the buffer-reference block binding `t`:
  ## a pointer renders as its reference block type.
  result = "Ref_" & spellGlslTypeBare(t).multiReplace(
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
      spellGlslScalar(base) & " x[]" & dims & ";"
    of ctVoid, ctProc:
      ctx.fail("buffer reference to " & $pointee.kind)
      ""
    else:
      spellGlslScalar(pointee) & " x[];"
  if member.len > 0:
    ctx.refBlocks[name] = member

proc spellGlslPtrDecl(ctx: var PrintCtx, t: CirType, ident: string): string =
  ## GLSL declaration of a buffer pointer: the reference block handle,
  ## `Ref_T ident`.
  if t.space != caDevice:
    ctx.fail("buffer pointer in " & $t.space & " space")
    return ""
  ctx.requireRefBlock(t.pElem)
  result = refTypeName(t.pElem) & ' ' & ident

# ═══════════════════════════════════════════════════════════════════════
# std430 size (GLSL has no sizeof)
# ═══════════════════════════════════════════════════════════════════════

proc std430Align(t: CirType): int
  ## Forward declaration, struct alignment recurses into members.

proc std430Size(t: CirType): int =
  ## std430 byte size of a type, the `eSizeof` value.
  case t.kind
  of ctVoid, ctBool, ctInt, ctUint, ctFloat:
    scalarByteSize(t)
  of ctVector:
    scalarByteSize(t.vElem) * t.vN
  of ctArray:
    std430Size(t.aElem) * t.aN
  of ctStruct:
    var off = 0
    var align = 1
    for f in t.fields:
      let fa = std430Align(f.typ)
      off = roundUpMultiple(off, fa) + std430Size(f.typ)
      align = max(align, fa)
    roundUpMultiple(off, align)
  of ctPtr, ctAptr: 8
  of ctProc: 0

proc std430Align(t: CirType): int =
  ## std430 byte alignment of a type.
  case t.kind
  of ctStruct:
    var align = 1
    for f in t.fields:
      align = max(align, std430Align(f.typ))
    align
  of ctArray: std430Align(t.aElem)
  of ctVector: max(scalarByteSize(t.vElem), 1)
  of ctPtr, ctAptr: 8
  else: max(scalarByteSize(t), 1)

# ═══════════════════════════════════════════════════════════════════════
# Module feature scan (extensions)
# ═══════════════════════════════════════════════════════════════════════

type
  ModuleFeatures = object
    fp64, fp16, int8, int16, int64: bool
    subgroup: bool
    subgroupShuffle: bool
    atomicFloat: bool
    controlFlow: bool
    ## an `{.unroll.}` loop, `[[unroll]]` needs GL_EXT_control_flow_attributes


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

proc scanType(t: CirType, f: var ModuleFeatures) =
  if t.isNil:
    return
  case t.kind
  of ctFloat:
    if t.bits == 64:
      f.fp64 = true
    if t.bits == 16:
      f.fp16 = true
  of ctInt, ctUint:
    if t.bits == 8:
      f.int8 = true
    if t.bits == 16:
      f.int16 = true
    if t.bits == 64:
      f.int64 = true
  of ctVector: scanType(t.vElem, f)
  of ctArray: scanType(t.aElem, f)
  of ctPtr, ctAptr: scanType(t.pElem, f)
  of ctStruct:
    for fl in t.fields:
      scanType(fl.typ, f)
  of ctVoid, ctBool, ctProc:
    discard

proc scanExpr(e: CirExpr, f: var ModuleFeatures) =
  if e.isNil:
    return
  case e.kind
  of eCall:
    if e.callee.kind == eSym:
      if e.callee.reductionBuiltin in {rbkReduceAdd, rbkReduceMax}:
        f.subgroup = true
      if e.callee.builtin and e.callee.name in ["simd_shuffle", "simd_shuffle_xor"]:
        f.subgroupShuffle = true
      if e.callee.atomicsBuiltin == abkAtomicFetchAdd:
        let pt = getExprType(e.args[0])
        if pt != nil and pt.kind in {ctPtr, ctAptr} and
            pt.pElem != nil and pt.pElem.kind == ctFloat and pt.pElem.bits == 32:
          f.atomicFloat = true
  else:
    discard
  for c in getExprChildren(e):
    scanExpr(c, f)
  scanType(getExprType(e), f)

proc scanStmts(ss: seq[CirStmt], f: var ModuleFeatures) =
  for s in ss:
    case s.kind
    of sVar, sConst:
      scanType(s.declType, f)
      if s.init != nil:
        scanExpr(s.init, f)
    of sAsgn:
      scanExpr(s.lhs, f)
      scanExpr(s.rhs, f)
    of sIte:
      scanExpr(s.cond, f)
      scanStmts(s.thenBody, f)
      scanStmts(s.elseBody, f)
    of sLoop:
      if s.unroll:
        f.controlFlow = true
      scanStmts(s.loopBody, f)
    of sScope:
      scanStmts(s.scopeBody, f)
    of sRet:
      if s.retVal != nil:
        scanExpr(s.retVal, f)
    of sCall:
      scanExpr(s.callCallee, f)
      for a in s.callArgs:
        scanExpr(a, f)
    of sAssume:
      scanExpr(s.assumeCond, f)
    of sJmp, sLab, sDiscard:
      discard
    of sCase, sAsm, sEmit:
      discard

proc scanModule(m: CirModule, f: var ModuleFeatures) =
  for t in m.types:
    scanType(t, f)
  # feature scan: the globals a lowering pass relocated out of the kernel
  # body keep their declarations and initializers
  scanStmts(m.globals, f)
  for fn in m.procs:
    scanType(fn.ret, f)
    for p in fn.params:
      scanType(p.typ, f)
    scanStmts(fn.body, f)

proc spellExtensions(f: ModuleFeatures): string =
  ## GLSL extension block, `#version 450` first.
  result = "#version 450\n"
  result.add "#extension GL_EXT_buffer_reference : enable\n"
  result.add "#extension GL_EXT_buffer_reference2 : enable\n"
  if f.controlFlow:
    result.add "#extension GL_EXT_control_flow_attributes : require\n"
  if f.fp64:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable\n"
  if f.fp16:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\n"
    result.add "#extension GL_EXT_shader_16bit_storage : enable\n"
  if f.int64:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable\n"
  if f.int16:
    result.add "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable\n"
    result.add "#extension GL_EXT_shader_16bit_storage : enable\n"
  if f.int8:
    result.add "#extension GL_EXT_shader_explicit_int8 : enable\n"
    result.add "#extension GL_EXT_shader_8bit_storage : enable\n"
  if f.subgroup:
    result.add "#extension GL_KHR_shader_subgroup_basic : enable\n"
    result.add "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"
  if f.subgroupShuffle:
    result.add "#extension GL_KHR_shader_subgroup_basic : enable\n"
    result.add "#extension GL_KHR_shader_subgroup_shuffle : enable\n"
  if f.atomicFloat:
    result.add "#extension GL_EXT_shader_atomic_float : enable\n"
  result.add "\n"

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printExpr(ctx: var PrintCtx, e: CirExpr): string
  ## Forward declaration, expressions recurse into themselves.

proc spellGlslLit(e: CirExpr): string =
  ## GLSL token of a literal, typed by its literal type.
  ## Wide integer literals print as constructor calls (`int64_t(5)`),
  ## the `LL` suffix is rejected by glslang.
  case e.kind
  of eIntLit:
    case e.litType.kind
    of ctUint:
      case e.litType.bits
      of 64: result = "uint64_t(" & $e.value & ")"
      of 32: result = $e.value & "U"
      else: result = "(" & spellGlslScalar(e.litType) & ")" & $e.value
    of ctInt:
      case e.litType.bits
      of 64: result = "int64_t(" & $e.value & ")"
      of 32: result = $e.value
      else: result = "(" & spellGlslScalar(e.litType) & ")" & $e.value
    else:
      result = $e.value
  of eFloatLit:
    case e.fLitType.bits
    of 16: result = "float16_t(" & $e.fValue & ")"
    of 32: result = $e.fValue & "f"
    of 64: result = "double(" & $e.fValue & ")"
    else: result = $e.fValue
  of eTrue: result = "true"
  of eFalse: result = "false"
  else: result = "<glsl: not a literal>"

proc spellVectorComponent(name: string): string =
  ## GLSL component letter of a CIRv2 vector component name (`F0`..`F3`).
  case name
  of "F0": "x"
  of "F1": "y"
  of "F2": "z"
  of "F3": "w"
  else: ""

proc printExpr(ctx: var PrintCtx, e: CirExpr): string =
  if ctx.failing():
    return ""
  case e.kind
  of eNil:
    ctx.fail("nil literal is outside the GLSL subset")
    return ""
  of eTrue, eFalse, eIntLit, eFloatLit:
    return spellGlslLit(e)
  of eSym:
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
    let base = (if e.kind == eAt: e.atBase else: e.patBase)
    let idx = ctx.printExpr(if e.kind == eAt: e.atIndex else: e.patIndex)
    let bt = getExprType(base)
    var baseTxt = ctx.printExpr(base)
    if base.kind in {eCast, eConv}:
      # a cast binds looser than the subscript: parenthesize the base
      baseTxt = "(" & baseTxt & ")"
    if bt != nil and bt.kind == ctAptr:
      ctx.requireRefBlock(bt.pElem)
      return baseTxt & ".x[" & idx & "]"
    if bt != nil and bt.kind == ctArray:
      return baseTxt & "[" & idx & "]"
    if bt != nil and bt.kind == ctPtr:
      ctx.fail("index through a pointer, the GLSL lowering pass dissolves" &
        " pointer locals")
      return ""
    ctx.fail("index of a " & (if bt == nil: "untyped" else: $bt.kind) &
      " base is outside the GLSL subset")
    return ""
  of eDeref:
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
  of eDot:
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      # a component of an unsigned GLSL builtin, cast to the canonical int
      return "int(" & glslCoordName(e.dotBase.coordBuiltin) & "." & e.fieldName & ")"
    let bt = getExprType(e.dotBase)
    if bt != nil and bt.kind == ctVector:
      let comp = spellVectorComponent(e.fieldName)
      if comp.len == 0:
        ctx.fail("no vector component " & e.fieldName)
        return ""
      return ctx.printExpr(e.dotBase) & "." & comp
    return ctx.printExpr(e.dotBase) & "." & e.fieldName
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    let op = case e.kind
      of eAdd: "+"
      of eSub: "-"
      of eMul: "*"
      of eDiv: "/"
      of eMod: "%"
      of eShl: "<<"
      of eShr: ">>"
      of eBitAnd: "&"
      of eBitOr: "|"
      else: "^"
    return "(" & ctx.printExpr(e.left) & ' ' & op & ' ' & ctx.printExpr(e.right) & ")"
  of eEq, eNeq, eLt, eLe:
    let op = case e.kind
      of eEq: "=="
      of eNeq: "!="
      of eLt: "<"
      else: "<="
    return "(" & ctx.printExpr(e.a) & ' ' & op & ' ' & ctx.printExpr(e.b) & ")"
  of eNot:
    return "(!" & ctx.printExpr(e.notOperand) & ")"
  of eAnd, eOr:
    let op = if e.kind == eAnd: "&&" else: "||"
    return "(" & ctx.printExpr(e.andLeft) & ' ' & op & ' ' & ctx.printExpr(e.andRight) & ")"
  of eBitNot:
    return "(~" & ctx.printExpr(e.unOperand) & ")"
  of eNeg:
    return "(-" & ctx.printExpr(e.unOperand) & ")"
  of eCast, eConv:
    let t = e.castType
    if t.kind in {ctAptr, ctPtr}:
      ctx.requireRefBlock(t.pElem)
      return refTypeName(t.pElem) & "(" & ctx.printExpr(e.castOperand) & ")"
    return spellGlslTypeBare(t) & "(" & ctx.printExpr(e.castOperand) & ")"
  of eSizeof:
    return $std430Size(e.ofType)
  of eAddr:
    # a buffer element's address is the indexed reference access
    if e.addrOperand.kind == ePat:
      return ctx.printExpr(e.addrOperand)
    ctx.fail("cannot take the address of a local, GLSL has no " &
      "function-storage addressing; only the address of a buffer element")
    return ""
  of eOconstr:
    if e.kvFields.len == 0:
      # a zero-field struct: the emitted padding member zero-initializes
      return spellGlslTypeBare(e.ocType) & "(0)"
    result = spellGlslTypeBare(e.ocType) & "("
    for i, kvv in e.kvFields:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(kvv.value)
    result.add ")"
  of eAconstr:
    result = spellGlslTypeBare(e.acType) & "("
    for i, el in e.elems:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(el)
    result.add ")"
  of eCall:
    let calleeName =
      if e.callee.kind == eSym: e.callee.name
      else: ctx.printExpr(e.callee)
    var args: seq[string]
    for a in e.args:
      args.add ctx.printExpr(a)
    if e.callee.kind == eSym:
      # builtin vocabulary dispatches on the typed kinds, never on names
      case e.callee.reductionBuiltin
      of rbkReduceAdd:
        return "subgroupAdd(" & args.join(", ") & ")"
      of rbkReduceMax:
        return "subgroupMax(" & args.join(", ") & ")"
      of rbkNone: discard
    if e.callee.kind == eSym and e.callee.builtin:
      case e.callee.name
      of "simd_shuffle":
        # subgroupShuffle takes the subgroup-relative index as `uint`. A plain cast
        # compiles without GL_NV_explicit_typecast
        return "subgroupShuffle(" & args[0] & ", uint(" & args[1] & "))"
      of "simd_shuffle_xor":
        return "subgroupShuffleXor(" & args[0] & ", uint(" & args[1] & "))"
      else: discard
      if e.callee.atomicsBuiltin == abkAtomicFetchAdd:
        # int atomicAdd is core. fp needs GL_EXT_shader_atomic_float,
        # emitted at module scope for an fp atomicFetchAdd in the module.
        # atomicAdd increments through one indirection: the address operand
        # is the reference's element 0
        let addrTxt = if getExprType(e.args[0]) != nil and getExprType(
            e.args[0]).kind in {ctAptr, ctPtr}:
          args[0] & ".x[0]"
        else:
          args[0]
        return "atomicAdd(" & addrTxt & ", " & args[1] & ")"
      if e.callee.synchroBuiltin != sbkNone:
        ctx.fail("barrier is a statement, a value-position barrier is outside the GLSL subset")
        return ""
    return calleeName & "(" & args.join(", ") & ")"

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt], exitL, contL: string): string
  ## Forward declaration, the loop context (break/continue labels) is a parameter.

proc printStmt(ctx: var PrintCtx, s: CirStmt, exitL, contL: string): string =
  ## One structured statement, `exitL`/`contL` the enclosing loop's labels.
  if ctx.failing():
    return ""
  let pad = spaces(ctx.indent * 2)
  case s.kind
  of sVar:
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
      decl = ctx.spellGlslPtrDecl(s.declType, s.name)
    elif s.declType.kind == ctArray:
      decl = spellGlslScalar(s.declType.aElem) & ' ' & s.name &
        "[" & $s.declType.aN & "]"
    else:
      decl = spellGlslTypeBare(s.declType) & ' ' & s.name
    result = pad & decl
    if s.init != nil:
      result &= " = " & ctx.printExpr(s.init)
    result &= ";"
  of sConst:
    result = pad & "const " & spellGlslTypeBare(s.declType) & ' ' &
      s.name & " = " & ctx.printExpr(s.init) & ";"
  of sAsgn:
    if s.lhs.kind == eSym and s.lhs.typ != nil and s.lhs.typ.kind == ctPtr:
      ctx.fail("pointer local " & s.lhs.name & " rebound, GLSL has no" &
        " pointer values, the GLSL lowering pass dissolves pointer locals")
      return ""
    result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"
  of sIte:
    let cond = ctx.printExpr(s.cond)
    ctx.indent.inc
    let thenTxt = ctx.printStmts(s.thenBody, exitL, contL)
    ctx.indent.dec
    result = pad & "if (" & cond & ") {\n" & thenTxt & '\n' & pad & '}'
    if s.elseBody.len > 0:
      ctx.indent.inc
      let elseTxt = ctx.printStmts(s.elseBody, exitL, contL)
      ctx.indent.dec
      result &= " else {\n" & elseTxt & '\n' & pad & '}'
  of sLoop:
    # loops print through printStmts, which derives the loop's own labels
    result = ""
  of sLab:
    # an exit label after a loop is the break target marker, dropped
    result = ""
  of sJmp:
    if s.jmpTarget == exitL and exitL.len > 0:
      result = pad & "break;"
    elif s.jmpTarget == contL and contL.len > 0:
      result = pad & "continue;"
    else:
      ctx.fail("unstructured jump to " & s.jmpTarget)
      result = ""
  of sScope:
    result = pad & "{\n"
    ctx.indent.inc
    result &= ctx.printStmts(s.scopeBody, exitL, contL)
    ctx.indent.dec
    result &= '\n' & pad & '}'
  of sAssume:
    # a loop-bound hint, GLSL has no assume construct
    result = ""
  of sRet:
    if s.retVal != nil:
      result = pad & "return " & ctx.printExpr(s.retVal) & ";"
    else:
      result = pad & "return;"
  of sDiscard:
    result = ""
  of sCall:
    var parts: seq[string]
    parts.add ctx.printExpr(s.callCallee)
    for a in s.callArgs:
      parts.add ctx.printExpr(a)
    if s.callCallee.kind == eSym and
        s.callCallee.synchroBuiltin == sbkThreadgroupBarrier:
      result = pad & "barrier();"
    elif s.callCallee.kind == eSym and
        s.callCallee.synchroBuiltin == sbkThreadgroupBarrierDevice:
      # GLSL's barrier() orders shared memory only, buffer writes are
      # ordered before it by a leading memoryBarrierBuffer() call
      result = pad & "memoryBarrierBuffer();\n" & pad & "barrier();"
    else:
      result = pad & parts[0] & "(" & parts[1 ..^ 1].join(", ") & ");"
  of sAsm:
    ctx.fail("inline assembly has no GLSL form")
    result = ""
  of sEmit:
    # `{.emit.}` raw source: verbatim text spliced with the CIRv2 names,
    # no terminator appended, the text supplies its own terminators
    var text = ""
    for i, part in s.emitParts:
      text.add(part)
      if i < s.emitOperands.len:
        text.add(s.emitOperands[i].name)
    result = pad & text
  of sCase:
    ctx.fail("case statements are outside the CIRv2 closure, the frontend refuses them")
    result = ""

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt], exitL, contL: string): string =
  ## Prints one statement list. Plain statements take the caller's loop context,
  ## a `loop` derives its own labels: the exit label right after it and the continue label inside.
  var lines: seq[string]
  var i = 0
  while i < stmts.len:
    let s = stmts[i]
    var txt: string
    if s.kind == sLoop:
      let lExit = if i + 1 < stmts.len and stmts[i + 1].kind == sLab:
        stmts[i + 1].labName
      else:
        ""
      var lCont = ""
      for t in s.loopBody:
        if t.kind == sLab and t.labName.startsWith(ContinueLabelPrefix):
          lCont = t.labName
      ctx.indent.inc
      let bodyTxt = ctx.printStmts(s.loopBody, lExit, lCont)
      ctx.indent.dec
      let directive = if s.unroll:
        spaces(ctx.indent * 2) & "[[unroll]]\n"
      else:
        ""
      txt = directive & spaces(ctx.indent * 2) & "for (;;) {\n" & bodyTxt & "\n" &
        spaces(ctx.indent * 2) & '}'
      if lExit.len > 0:
        i.inc  # exit label, consumed
    else:
      txt = ctx.printStmt(s, exitL, contL)
    if txt.len > 0:
      lines.add txt
    i.inc
  return lines.join("\n")

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc printFn(ctx: var PrintCtx, fn: CirProc): string =
  ## One device function: signature, body, closing brace.
  ctx.fnName = fn.name
  ctx.indent = 0
  let bodyTxt = ctx.printStmts(fn.body, "", "")
  if ctx.failing():
    return ""
  var params: seq[string]
  for p in fn.params:
    case p.typ.kind
    of ctAptr:
      params.add ctx.spellGlslPtrDecl(p.typ, p.name)
    of ctPtr:
      ctx.fail("pointer parameter " & p.name &
        ", GLSL device functions take buffer references or values")
      return ""
    else:
      params.add spellGlslTypeBare(p.typ) & ' ' & p.name
  let ret = spellGlslTypeBare(fn.ret)
  let sig = ret & ' ' & fn.name & '(' & params.join(", ") & ")"
  return sig & " {\n" & bodyTxt & '\n' & "}"

proc spellSmemDecl(t: CirType, name: string): string =
  ## Program-scope `shared` declaration of a threadgroup local.
  if t.kind == ctArray:
    result = "shared " & spellGlslScalar(t.aElem) & ' ' & name &
      "[" & $t.aN & "];"
  else:
    result = "shared " & spellGlslTypeBare(t) & ' ' & name & ";"

proc spellConstDecl(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope `const` declaration of one hoisted const_mem local.
  ## Shape guard: the lowering pass refuses an initializer-less const_mem
  ## local before relocation, this check fails loudly on a malformed node.
  if s.init == nil:
    ctx.fail("const_mem local " & s.name & " without an initializer")
    return ""
  if s.declType.kind == ctArray:
    result = "const " & spellGlslScalar(s.declType.aElem) & ' ' & s.name &
      "[" & $s.declType.aN & "]"
  else:
    result = "const " & spellGlslTypeBare(s.declType) & ' ' & s.name
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
    of sLab, sJmp, sAsm, sEmit:
      discard

proc printGlslModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full GLSL translation unit: version and extensions,
  ## buffer-reference blocks, struct declarations, the push-constant block,
  ## program-scope `shared` and `const` declarations, the device functions,
  ## and the kernel with its `local_size` layout qualifier.
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>")
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

  var features = ModuleFeatures()
  scanModule(m, features)
  result.src = spellExtensions(features)

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
      var field = spellGlslTypeBare(f.typ) & ' ' & f.name
      if f.typ.kind == ctArray:
        field = spellGlslScalar(f.typ.aElem) & ' ' & f.name &
          "[" & $f.typ.aN & "]"
      elif f.typ.kind in {ctAptr, ctPtr}:
        field = ctx.spellGlslPtrDecl(f.typ, f.name)
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
        decl = ctx.spellGlslPtrDecl(p.typ, p.name)
      of ctArray:
        ctx.fail("kernel parameter " & p.name &
          " of array type, arrays do not pack into the push block")
        decl = ""
      else:
        decl = spellGlslTypeBare(p.typ) & ' ' & p.name
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
      hoisted.add spellSmemDecl(s.declType, s.name)
    elif s.kind == sVar and s.space == caConstant:
      if s.init == nil:
        ctx.fail("const_mem local " & s.name & " without an initializer")
      else:
        hoisted.add ctx.spellConstDecl(s)
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
        params.add ctx.spellGlslPtrDecl(p.typ, p.name)
      of ctPtr:
        ctx.fail("pointer parameter " & p.name &
          ", GLSL device functions take buffer references or values")
      else:
        params.add spellGlslTypeBare(p.typ) & ' ' & p.name
    if ctx.failing():
      return ("", ctx.err)
    result.src &= spellGlslTypeBare(fn.ret) & ' ' & fn.name & '(' &
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
    let bodyTxt = ctx.printStmts(kernel.body, "", "")
    if ctx.failing():
      return ("", ctx.err)
    result.src &= "void " & kernel.name & "() {\n" & bodyTxt & '\n' & "}\n\n"
  return (result.src, "")



proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as GLSL source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printGlslModule(ctx.module))
