# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## WebGPU WGSL printer: a structured CIRv2 module rendered as WGSL source,
## table-driven over the closed set of type kinds.
##
## Kernel parameter bindings, one `@group(0) @binding(i)` per parameter in host `store` packing order, output first:
## - device buffers declare `var<storage, read_write> name: array<T>;`, the engine binds them writable
## - scalars and value structs declare `var<storage, read> name: T;`, packed into the 256-byte-aligned scratch buffer
## - a bool scalar declares `u32` (bool is not host-shareable), the WGSL lowering pass rewrites the body reads to `name != 0u`
## - the kernel signature takes only `@builtin(...)` coordinate parameters, first-use order
##
## Addressing, memory classes, jumps:
## - a pointer local or pointer-holding struct local refuses, the WGSL lowering pass dissolves the shapes into pointees and leaf symbols
## - a smem/const_mem kernel local relocates into `CirModule.globals`, the WGSL lowering pass owns the relocation
## - struct declarations emit for the referenced value structs only, a referenced pointer-holding struct refuses
## - jmp to the loop exit label is `break`, to the continue label `continue`, any other lab/jmp refuses

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import ../passes/pass_helpers
import std/[macros, sets, strutils, tables]
import ../ir/ast_datatypes
import ./emit_common
import ../passes/layout_std430

type
  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    inKernel: bool
    ## False inside a device function, a device function reading a coordinate builtin refuses.
    paramNames: HashSet[string]
    ## Parameter names of the proc in flight, a pointer-typed parameter prints
    ## as-is (WGSL device functions take real pointer parameters).
    ## Only pointer locals refuse as values.
    builtins: OrderedSet[CoordBuiltinKind]
    ## Canonical coordinate builtin names the kernel body references, first-use order.
    neededStructs: HashSet[string]
    ## Struct type names referenced by the emitted declarations and expressions.
    layouts: Table[string, StructLayout]
    ## Stored struct layouts of the module, the layout pass filled them.

const printerTarget = "wgsl"

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitWgslScalar(ctx: var PrintCtx, t: CirType): string =
  ## WGSL name of a non-array leaf type, "" when the caller composes the name.
  ## A struct name registers the struct for reference-driven emission.
  case t.kind
  of ctVoid:
    result = "void"
  of ctBool:
    result = "bool"
  of ctInt:
    case t.bits
    of 32:
      result = "i32"
    of 64:
      result = "i64"
    else:
      ctx.fail($t.bits & "-bit integers have no WGSL type, only 32 and 64")
      result = ""
  of ctUint:
    case t.bits
    of 32:
      result = "u32"
    of 64:
      result = "u64"
    else:
      ctx.fail($t.bits & "-bit integers have no WGSL type, only 32 and 64")
      result = ""
  of ctFloat:
    case t.bits
    of 16:
      result = "f16"
    of 32:
      result = "f32"
    of 64:
      result = "f64"
    else:
      result = ""
  of ctVector:
    let elem = ctx.emitWgslScalar(t.vElem)
    if elem.len > 0:
      result = "vec" & $t.vN & "<" & elem & ">"
  of ctStruct:
    ctx.neededStructs.incl(t.name)
    result = t.name
  of ctPtr, ctAptr, ctArray, ctProc:
    result = ""

proc emitWgslTypeBare(ctx: var PrintCtx, t: CirType): string =
  ## WGSL name of `t` with no identifier, the form of casts and constructors.
  case t.kind
  of ctArray:
    let elem = ctx.emitWgslTypeBare(t.aElem)
    if t.aN == 0:
      result = "array<" & elem & ">"
    else:
      result = "array<" & elem & ", " & $t.aN & ">"
  else:
    result = ctx.emitWgslScalar(t)

proc emitWgslPtrType(ctx: var PrintCtx, t: CirType): string =
  ## WGSL pointer type of a `ptr` local or device-function parameter,
  ## the address space leads, storage pointers declare the access mode.
  let elem = ctx.emitWgslTypeBare(t.pElem)
  case t.space
  of caThread:
    result = "ptr<function, " & elem & ">"
  of caThreadgroup:
    result = "ptr<workgroup, " & elem & ">"
  of caDevice:
    if t.kind == ctAptr:
      ctx.fail("device array pointer type has no WGSL form, the binding is the array itself")
      return ""
    result = "ptr<storage, " & elem & ", read_write>"
  of caConstant:
    ctx.fail("pointer to constant space, WGSL consts have no address")
    return ""

# ═══════════════════════════════════════════════════════════════════════
# Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

type WgslBuiltinParam = tuple[attribute, param, wgslType: string]
  ## Coordinate builtin kind -> WGSL `@builtin(...)` kernel parameter.
  ## - `attribute`: the WGSL builtin attribute.
  ## - `param`: the identifier the kernel body reads.
  ## - `wgslType`: the WGSL type (scalar for the flat thread index).

proc wgslBuiltinParam(kind: CoordBuiltinKind): WgslBuiltinParam =
  ## Returns the WGSL parameter of a canonical coordinate builtin kind, an empty tuple when none exists.
  ## `threads_per_threadgroup` is absent, WGSL exposes the threadgroup size
  ## only as the `@workgroup_size` entry-point attribute, a read refuses.
  case kind
  of cbkThreadPositionInGrid:
    ("global_invocation_id", "global_id", "vec3<u32>")
  of cbkThreadgroupPositionInGrid:
    ("workgroup_id", "workgroup_id", "vec3<u32>")
  of cbkThreadPositionInThreadgroup:
    ("local_invocation_id", "local_invocation_id", "vec3<u32>")
  of cbkThreadgroupsPerGrid:
    ("num_workgroups", "num_workgroups", "vec3<u32>")
  of cbkThreadIndexInThreadgroup:
    ("local_invocation_index", "local_invocation_index", "u32")
  of cbkNone, cbkThreadsPerThreadgroup:
    ("", "", "")

proc wgslCoordComponent(kind: CoordBuiltinKind, field: string): string =
  ## WGSL read of one canonical coordinate builtin component, "" for a non-component field.
  case field
  of "x", "y", "z":
    let p = wgslBuiltinParam(kind)
    if p.attribute.len > 0:
      result = "i32(" & p.param & "." & field & ")"
  else:
    result = ""

proc wgslWholeVector(kind: CoordBuiltinKind): string =
  ## WGSL read of a canonical coordinate builtin whole, the i32 vector
  ## composed from the u32 parameter components.
  let p = wgslBuiltinParam(kind)
  result = "vec3<i32>(i32(" & p.param & ".x), i32(" & p.param & ".y), i32(" &
    p.param & ".z))"

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printExpr(ctx: var PrintCtx, e: CirExpr): string
  ## Forward declaration, expressions recurse into themselves.

# WGSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
# `bitcast` from the unsigned integer of the same width.
const wgslFloatTokens = FloatLitTokens(
  inf16: "bitcast<f16>(0x7c00u)", negInf16: "bitcast<f16>(0xfc00u)", nan16: "bitcast<f16>(0x7e00u)",
  inf32: "bitcast<f32>(0x7f800000u)",
  negInf32: "bitcast<f32>(0xff800000u)",
  nan32: "bitcast<f32>(0x7fc00000u)",
  inf64: "bitcast<f64>(0x7ff0000000000000u)",
  negInf64: "bitcast<f64>(0xfff0000000000000u)",
  nan64: "bitcast<f64>(0x7ff8000000000000u)")

proc emitWgslLit(e: CirExpr): string =
  ## WGSL token of a literal, typed by its literal type.
  ## Bare literals are abstract in WGSL and work in any typed context,
  ## u32 gains the `u` suffix, f16 the `h` suffix.
  case e.kind
  of eIntLit:
    case e.litType.kind
    of ctUint:
      if e.litType.bits == 32:
        result = $e.value & "u"
      else:
        result = $e.value
    else:
      result = $e.value
  of eFloatLit:
    result = nonFiniteFloatLit(e, wgslFloatTokens)
    if result.len == 0:
      case e.fLitType.bits
      of 16: result = $e.fValue & "h"
      else: result = $e.fValue
  of eTrue: result = "true"
  of eFalse: result = "false"
  else: result = "<wgsl: not a literal>"

proc emitVectorComponent(name: string): string =
  ## WGSL component letter of a CIRv2 vector component name (`F0`..`F3`).
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
    ctx.fail("nil literal is outside the WGSL subset")
    return ""
  of eTrue, eFalse, eIntLit, eFloatLit:
    return emitWgslLit(e)
  of eSym:
    if e.typ != nil and e.typ.kind == ctPtr and e.name notin ctx.paramNames:
      ctx.fail("pointer local " & e.name & " used as a value, WGSL has no" &
        " pointer values, the WGSL lowering pass dissolves pointer locals")
      return ""
    if e.typ != nil and e.typ.kind == ctStruct and e.typ.containsPtrDeep and
        e.name notin ctx.paramNames:
      ctx.fail("struct local " & e.name &
        " used whole, WGSL has no pointer struct members, the WGSL lowering" &
        " pass dissolves it into its fields")
      return ""
    if e.coordBuiltin != cbkNone:
      if not ctx.inKernel:
        ctx.fail("coordinate builtin " & e.name &
          " in a device function, WGSL builtins are entry-point parameters")
        return ""
      if e.coordBuiltin == cbkThreadsPerThreadgroup:
        ctx.fail("threads_per_threadgroup has no WGSL builtin: WGSL has only" &
          " the @workgroup_size attribute, the threadgroup size is not" &
          " addressable from WGSL")
        return ""
      ctx.builtins.incl(e.coordBuiltin)
      if e.coordBuiltin == cbkThreadIndexInThreadgroup:
        return "i32(" & wgslBuiltinParam(e.coordBuiltin).param & ")"
      return wgslWholeVector(e.coordBuiltin)
    return e.name
  of eAt, ePat:
    let base = (if e.kind == eAt: e.atBase else: e.patBase)
    let idx = ctx.printExpr(if e.kind == eAt: e.atIndex else: e.patIndex)
    let bt = getExprType(base)
    var baseTxt = ctx.printExpr(base)
    if base.kind in {eCast, eConv}:
      # a cast binds looser than the subscript: parenthesize the base
      baseTxt = "(" & baseTxt & ")"
    if bt != nil and bt.kind in {ctPtr, ctAptr} and
        not (bt.kind == ctAptr and bt.space == caDevice):
      # a pointer local indexes through an explicit dereference
      return "(*" & baseTxt & ")[" & idx & "]"
    return baseTxt & "[" & idx & "]"
  of eDeref:
    return "(*" & ctx.printExpr(e.derefOperand) & ")"
  of eDot:
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      if not ctx.inKernel:
        ctx.fail("coordinate builtin " & e.dotBase.name &
          " in a device function, WGSL builtins are entry-point parameters")
        return ""
      let mapped = wgslCoordComponent(e.dotBase.coordBuiltin, e.fieldName)
      if mapped.len == 0:
        if e.dotBase.coordBuiltin == cbkThreadsPerThreadgroup:
          ctx.fail("threads_per_threadgroup has no WGSL builtin: WGSL has only" &
            " the @workgroup_size attribute, the threadgroup size is not" &
            " addressable from WGSL")
        else:
          ctx.fail("no coordinate component " & e.fieldName)
        return ""
      ctx.builtins.incl(e.dotBase.coordBuiltin)
      return mapped
    let bt = getExprType(e.dotBase)
    if bt != nil and bt.kind == ctVector:
      let comp = emitVectorComponent(e.fieldName)
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
  of eCast:
    return "bitcast<" & ctx.emitWgslTypeBare(e.castType) & ">(" &
      ctx.printExpr(e.castOperand) & ")"
  of eConv:
    if e.castType.kind == ctBool:
      ctx.fail("conversion to bool, WGSL has no bool constructor, compare" &
        " against zero")
      return ""
    let bt = getExprType(e.castOperand)
    if bt != nil and bt.kind == ctBool:
      ctx.fail("conversion from bool, WGSL has no bool-to-numeric conversion")
      return ""
    return ctx.emitWgslTypeBare(e.castType) & "(" & ctx.printExpr(e.castOperand) & ")"
  of eSizeof:
    return $typeByteSize(ctx.layouts, e.ofType, launchAbiBoolWidth)
  of eAddr:
    return "(&" & ctx.printExpr(e.addrOperand) & ")"
  of eOconstr:
    if e.kvFields.len == 0:
      # a zero-field struct, the padding member zero-initializes
      return ctx.emitWgslTypeBare(e.ocType) & "(0u)"
    result = ctx.emitWgslTypeBare(e.ocType) & "("
    for i, kvv in e.kvFields:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(kvv.value)
    result.add ")"
  of eAconstr:
    result = ctx.emitWgslTypeBare(e.acType) & "("
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
      # builtin vocabulary dispatches on the typed kinds, never on names,
      # `rsqrt` stays a name case, a math intrinsic outside the four categories
      case e.callee.reductionBuiltin
      of rbkReduceAdd:
        return "subgroupAdd(" & args.join(", ") & ")"
      of rbkReduceMax:
        return "subgroupMax(" & args.join(", ") & ")"
      of rbkNone: discard
      if e.callee.atomicsBuiltin == abkAtomicFetchAdd:
        # WGSL atomics cover i32/u32 on atomic<T> storage types only,
        # f32 atomics do not exist, and the atomic<T> retype of the binding
        # is a declaration-site change the printer does not make
        ctx.fail("atomicFetchAdd has no WGSL form, WGSL atomics cover i32/u32" &
          " pointees through atomic<T> storage types only")
        return ""
      if e.callee.synchroBuiltin != sbkNone:
        ctx.fail("barrier is a statement, a value-position barrier is outside the WGSL subset")
        return ""
    # a math intrinsic reroutes only when the callee IS the builtin
    # a user proc that shares the name stays an ordinary call
    if e.callee.kind == eSym and e.callee.builtin:
      case calleeName
      of "rsqrt":
        return "inverseSqrt(" & args.join(", ") & ")"
      else: discard
    return calleeName & "(" & args.join(", ") & ")"

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt], exitL, contL: string): string
  ## Forward declaration, the loop context (break/continue labels) is a parameter.

proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string =
  ## Declaration text of one local, the init stays with the caller.
  ## A pointer-typed local refuses earlier, the WGSL lowering pass dissolves it.
  case s.declType.kind
  of ctArray:
    result = s.name & ": " & ctx.emitWgslTypeBare(s.declType)
  else:
    result = s.name & ": " & ctx.emitWgslTypeBare(s.declType)

proc printStmt(ctx: var PrintCtx, s: CirStmt, exitL, contL: string): string =
  ## One structured statement, `exitL`/`contL` the enclosing loop's labels.
  if ctx.failing():
    return ""
  let pad = spaces(ctx.indent * 2)
  case s.kind
  of sVar:
    if s.space == caThreadgroup:
      # WGSL lowering pass hoists threadgroup locals to `CirModule.globals`,
      # a survivor here was never relocated
      ctx.fail("local " & s.name & " in threadgroup space, the WGSL lowering" &
        " pass hoists threadgroup locals to program scope")
      return ""
    elif s.space == caConstant:
      ctx.fail("local " & s.name & " in constant space, the WGSL lowering" &
        " pass hoists const_mem locals to program scope")
      return ""
    elif s.space == caDevice:
      ctx.fail("local " & s.name & " in device space, WGSL locals are" &
        " function or workgroup")
      return ""
    elif s.declType.kind in {ctPtr, ctAptr}:
      ctx.fail("pointer local " & s.name & ", WGSL has no pointer values," &
        " the WGSL lowering pass dissolves pointer locals")
      return ""
    elif s.declType.kind == ctStruct and s.declType.containsPtrDeep:
      ctx.fail("struct local " & s.name & ", WGSL has no pointer struct" &
        " members, the WGSL lowering pass dissolves pointer locals")
      return ""
    else:
      result = pad & "var " & ctx.emitLocalDecl(s)
      if s.init != nil:
        result &= " = " & ctx.printExpr(s.init)
      result &= ";"
  of sConst:
    result = pad & "let " & s.name & ": " &
      ctx.emitWgslTypeBare(s.declType) & " = " & ctx.printExpr(s.init) & ";"
  of sAsgn:
    if s.lhs.kind == eSym and s.lhs.typ != nil and s.lhs.typ.kind == ctPtr and
        s.lhs.name notin ctx.paramNames:
      ctx.fail("pointer local " & s.lhs.name & " rebound, WGSL has no" &
        " pointer values, the WGSL lowering pass dissolves pointer locals")
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
    # a loop-bound hint, WGSL has no assume construct
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
      result = pad & "workgroupBarrier();"
    elif s.callCallee.kind == eSym and
        s.callCallee.synchroBuiltin == sbkThreadgroupBarrierDevice:
      # WGSL has no combined barrier: the workgroup fence, then the storage fence
      result = pad & "workgroupBarrier();\n" & pad & "storageBarrier();"
    else:
      result = pad & parts[0] & "(" & parts[1 ..^ 1].join(", ") & ");"
  of sAsm:
    ctx.fail("inline assembly has no WGSL form")
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
        # WGSL has no unroll directive, surface the drop in the artifact
        spaces(ctx.indent * 2) & "// {.unroll.} requested but not honored on WebGPU\n"
      else:
        ""
      txt = directive & spaces(ctx.indent * 2) & "loop {\n" & bodyTxt & "\n" &
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
  ctx.inKernel = false
  ctx.indent = 0
  ctx.paramNames = HashSet[string]()
  for p in fn.params:
    ctx.paramNames.incl p.name
  let bodyTxt = ctx.printStmts(fn.body, "", "")
  if ctx.failing():
    return ""
  var params: seq[string]
  for p in fn.params:
    case p.typ.kind
    of ctAptr:
      ctx.fail("device pointer parameter " & p.name & " in device function " &
        fn.name & ", WGSL device functions take no buffer pointers, the" &
        " call sites have no pointer form for a storage binding")
      return ""
    of ctPtr:
      params.add p.name & ": " & ctx.emitWgslPtrType(p.typ)
    else:
      params.add p.name & ": " & ctx.emitWgslTypeBare(p.typ)
  let ret = ctx.emitWgslTypeBare(fn.ret)
  let sig = "fn " & fn.name & "(" & params.join(", ") & ")"
  let arrow = if ret.len > 0: " -> " & ret else: ""
  return sig & arrow & " {\n" & bodyTxt & '\n' & "}"

proc registerTypeStructs(ctx: var PrintCtx, t: CirType) =
  ## Registers the struct names reachable from `t`, the reference-driven struct
  ## emission works from the parameter and declaration types.
  if t.isNil:
    return
  case t.kind
  of ctStruct:
    ctx.neededStructs.incl(t.name)
    for f in t.fields:
      ctx.registerTypeStructs(f.typ)
  of ctArray:
    ctx.registerTypeStructs(t.aElem)
  of ctVector:
    ctx.registerTypeStructs(t.vElem)
  of ctPtr, ctAptr:
    ctx.registerTypeStructs(t.pElem)
  else:
    discard

proc emitBinding(ctx: var PrintCtx, p: CirParam, i: int): string =
  ## `@group(0) @binding(i)` declaration of one kernel parameter in host `store` packing order:
  ## - device buffers bind writable storage, `var<storage, read_write>`
  ## - scalars and value structs bind read-only storage, packed in the engine scratch buffer
  let attr = "@group(0) @binding(" & $i & ") "
  case p.typ.kind
  of ctAptr:
    if p.typ.space != caDevice:
      ctx.fail("kernel parameter " & p.name & " in " & $p.typ.space &
        " space, the WebGPU engine binds device buffers only")
      return ""
    result = attr & "var<storage, read_write> " & p.name & ": array<" &
      ctx.emitWgslTypeBare(p.typ.pElem) & ">;"
  of ctPtr:
    ctx.fail("kernel parameter " & p.name &
      " of pointer type, the engine binds device buffers and values")
    return ""
  of ctArray:
    ctx.fail("kernel parameter " & p.name &
      " of array type, the engine packs the value, the parameter is a struct or a scalar")
    return ""
  of ctStruct:
    for f in p.typ.fields:
      let ft = (if f.typ.kind == ctVector: f.typ.vElem else: f.typ)
      if f.typ.kind == ctVector:
        ctx.fail("kernel parameter " & p.name & " field " & f.name &
          " is a vector, the WGSL storage layout of vectors (16-byte alignment" &
          " for 3-component vectors) does not match the host packing")
        return ""
      if ft.kind == ctBool:
        ctx.fail("kernel parameter " & p.name & " field " & f.name &
          " is a bool, bool is not host-shareable in WGSL storage")
        return ""
    if p.typ.containsPtrDeep:
      ctx.fail("kernel parameter " & p.name &
        " holds a pointer, the parameter explosion removes those before printing")
      return ""
    result = attr & "var<storage, read> " & p.name & ": " &
      ctx.emitWgslTypeBare(p.typ) & ";"
  of ctBool:
    result = attr & "var<storage, read> " & p.name & ": u32;"
  else:
    result = attr & "var<storage, read> " & p.name & ": " &
      ctx.emitWgslTypeBare(p.typ) & ";"

proc emitWgslStruct(ctx: var PrintCtx, t: CirType): string =
  ## One struct declaration, WGSL requires at least one member.
  if t.containsPtrDeep:
    ctx.fail("struct " & t.name &
      " holds a pointer, WGSL has no pointer struct members")
    return ""
  result = "struct " & t.name & " {\n"
  if t.fields.len == 0:
    result &= "  _padding: u32,\n"
  else:
    for f in t.fields:
      var field = f.name & ": " & ctx.emitWgslTypeBare(f.typ)
      if f.typ.kind == ctArray:
        field = f.name & ": " & ctx.emitWgslTypeBare(f.typ)
      result &= "  " & field & ",\n"
  result &= '}'

proc printWgslModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full WGSL module, one `@group(0) @binding(i)` per kernel parameter:
  ## - `enable f16` when any type in the module uses f16
  ## - the referenced struct declarations
  ## - the hoisted module-scope `var<workgroup>` and `const` declarations
  ## - the device functions, then the `@compute` kernel with its builtin parameters
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>")
  ctx.layouts = m.layouts
  var kernels: seq[CirProc]
  for fn in m.procs:
    if fn.isKernel:
      kernels.add fn
  if kernels.len != 1:
    ctx.fail("exactly one kernel per module, found " & $kernels.len)
    return ("", ctx.err)
  let kernel = kernels[0]
  # a pointer-typed kernel parameter would refuse at the binding, and until that refusal the body's symbol reads print as-is
  for p in kernel.params:
    ctx.paramNames.incl p.name

  for p in kernel.params:
    ctx.registerTypeStructs(p.typ)
  # relocated const_mem declarations, struct-typed ones included
  for s in m.globals:
    if s.kind in {sVar, sConst}:
      ctx.registerTypeStructs(s.declType)

  # `enable f16` from the stored module feature scan
  # (`CirModule.features`, filled by the feature-scan pass)
  if m.features.fp16:
    result.src = "enable f16;\n\n"

  # device functions first, their bodies register the structs they reference
  var fns: seq[CirProc]
  for fn in m.procs:
    if not fn.isKernel:
      fns.add fn
  var fnTxts: seq[string]
  for fn in fns:
    let txt = ctx.printFn(fn)
    if ctx.failing():
      return ("", ctx.err)
    fnTxts.add txt

  # kernel body: the signature consumes the builtin set it references
  ctx.fnName = kernel.name
  ctx.inKernel = true
  ctx.indent = 0
  let bodyTxt = ctx.printStmts(kernel.body, "", "")
  if ctx.failing():
    return ("", ctx.err)
  var builtinArgs: seq[string]
  for kind in ctx.builtins:
    let p = wgslBuiltinParam(kind)
    builtinArgs.add "@builtin(" & p.attribute & ") " & p.param & ": " & p.wgslType
  let wg = (if kernel.workgroup == [0, 0, 0]: [64, 1, 1] else: kernel.workgroup)
  # exportc renames a kernel entry for the host runtime lookup.
  # A device function keeps the name its call sites reference.
  let entryName = if kernel.isKernel and kernel.exportName.len > 0: kernel.exportName else: kernel.name
  let kernelSig = "@compute @workgroup_size(" & $wg[0] & ", " & $wg[1] & ", " &
    $wg[2] & ")\nfn " & entryName & "(" & builtinArgs.join(", ") & ")"

  # module-scope workgroup and const declarations relocated into
  # `m.globals` by the WGSL lowering pass, in encounter order
  var hoisted: seq[string]
  for s in m.globals:
    if s.kind == sVar and s.space == caThreadgroup:
      if s.init != nil:
        ctx.fail("smem local " & s.name &
          " with an initializer, WGSL workgroup variables are zero-initialized")
      else:
        hoisted.add "var<workgroup> " & s.name & ": " &
          ctx.emitWgslTypeBare(s.declType) & ";"
    elif s.kind == sVar and s.space == caConstant:
      if s.init == nil:
        ctx.fail("const_mem local " & s.name & " without an initializer")
      else:
        hoisted.add "const " & s.name & ": " &
          ctx.emitWgslTypeBare(s.declType) & " = " & ctx.printExpr(s.init) & ";"
  if ctx.failing():
    return ("", ctx.err)

  # struct declarations for the referenced value structs, the closure over
  # their struct-typed members included, pointer-holding structs refuse
  var toEmit = ctx.neededStructs
  var changed = true
  while changed:
    changed = false
    for t in m.types:
      if t.kind == ctStruct and t.name in toEmit:
        for f in t.fields:
          var ft = f.typ
          while ft != nil and ft.kind in {ctArray}:
            ft = ft.aElem
          if ft != nil and ft.kind == ctStruct and ft.name notin toEmit:
            toEmit.incl(ft.name)
            changed = true
  for t in m.types:
    if t.kind != ctStruct:
      continue
    if t.name notin toEmit:
      continue
    result.src &= ctx.emitWgslStruct(t) & "\n\n"
  if ctx.failing():
    return ("", ctx.err)

  # kernel parameter bindings, one per parameter in order
  var bindings: seq[string]
  for i, p in kernel.params:
    let b = ctx.emitBinding(p, i)
    if ctx.failing():
      return ("", ctx.err)
    bindings.add b
  for b in bindings:
    result.src &= b & "\n"
  if bindings.len > 0:
    result.src &= "\n"

  for d in hoisted:
    result.src &= d & "\n"
  if hoisted.len > 0:
    result.src &= "\n"

  for txt in fnTxts:
    result.src &= txt & "\n\n"
  result.src &= kernelSig & " {\n" & bodyTxt & '\n' & "}\n"
  return (result.src, "")



proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as WGSL source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printWgslModule(ctx.module))
