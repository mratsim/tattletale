# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## OpenCL C printer: a structured CIRv2 module rendered as OpenCL C source,
## table-driven over the closed type lattice.
##
## Emission contract:
## - one `{.global.}` kernel per module, the device functions it calls emitted alongside
## - the kernel signature is `__kernel void <name>(...)`: one parameter per kernel parameter, the host `store` packing order
## - device pointers render as `__global T* restrict` parameters, element access `p[index]`
## - `eSizeof` prints C `sizeof`, OpenCL C is C99
##
## Addressing, memory classes, jumps:
## - a `ptr` local is a plain C pointer, `(&x)` addresses a local, `(*p)` dereferences
## - `{.smem.}` locals declare `__local` in the kernel body
## - jmp to the loop exit label is `break`, to the continue label `continue`, any other lab/jmp refuses
##
## Storage relocation:
## - `{.const_mem.}` locals relocate into `CirModule.globals` through the OpenCL lowering pass, program-scope `__constant` output
## - a `{.const_mem.}` local surviving outside the relocation refuses, OpenCL C has no function-scope constant storage

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, sequtils, sets, tables]
import ../ir/ast_datatypes

type
  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    inKernel: bool
    ## False inside a device function: `__local` and `__constant` locals are
    ## kernel-only storage, a device function declaring one refuses.

proc failing(ctx: PrintCtx): bool =
  ctx.err.len > 0

proc fail(ctx: var PrintCtx, detail: string) =
  ## Records the first refusal, naming the proc.
  if ctx.err.len == 0:
    ctx.err = "cirv2 printer opencl: at " & ctx.fnName & ": " & detail

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc spellOpenclScalar(t: CirType): string =
  ## OpenCL C name of a non-array leaf type, "" when the caller composes the name.
  case t.kind
  of ctVoid:
    result = "void"
  of ctBool:
    # OpenCL C bool storage is 1 byte, the launch ABI delivers 4-byte ints
    result = "int"
  of ctInt:
    case t.bits
    of 8:
      result = "char"
    of 16:
      result = "short"
    of 32:
      result = "int"
    of 64:
      result = "long"
    else:
      result = ""
  of ctUint:
    case t.bits
    of 8:
      result = "uchar"
    of 16:
      result = "ushort"
    of 32:
      result = "uint"
    of 64:
      result = "ulong"
    else:
      result = ""
  of ctFloat:
    case t.bits
    of 16:
      result = "half"
    of 32:
      result = "float"
    of 64:
      result = "double"
    else:
      result = ""
  of ctVector:
    let elem = spellOpenclScalar(t.vElem)
    if elem.len > 0:
      result = elem & $t.vN
  of ctStruct:
    result = t.name
  of ctPtr, ctAptr, ctArray, ctProc:
    result = ""

proc spellOpenclTypeBare(t: CirType): string =
  ## OpenCL C name of `t` with no identifier, pointer stars trailing the element name, the form of casts and compound literals.
  case t.kind
  of ctPtr, ctAptr:
    result = spellOpenclTypeBare(t.pElem) & "*"
  else:
    result = spellOpenclScalar(t)

proc spellOpenclArrayDecl(t: CirType, ident: string): string =
  ## OpenCL C declaration of an array local or field, the length moves
  ## behind the identifier (`float ident[4]`).
  spellOpenclScalar(t.aElem) & ' ' & ident & "[" & $t.aN & "]"

proc spellOpenclPtrField(t: CirType, ident: string): string =
  ## OpenCL C declaration of a pointer field or local, the address space qualifier leads.
  ## `restrict` is a function-parameter qualifier, struct fields and locals drop it.
  let qualifier = case t.space
    of caDevice: "__global "
    of caThreadgroup: "__local "
    of caConstant: "__constant "
    of caThread: ""
  result = qualifier & spellOpenclTypeBare(t) & ' ' & ident

# ═══════════════════════════════════════════════════════════════════════
# Module feature scan (extensions)
# ═══════════════════════════════════════════════════════════════════════

type
  ModuleFeatures = object
    fp64, fp16: bool
    subgroup: bool
    atomicFloat: bool


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
  ## already a pointer. Nil when the operand names no pointer.
  if e.kind == ePat:
    let t = getExprType(e.patBase)
    if t != nil and t.kind in {ctPtr, ctAptr}:
      return t
  let t = getExprType(e)
  if t != nil and t.kind in {ctPtr, ctAptr}:
    return t

proc scanType(t: CirType, f: var ModuleFeatures) =
  if t.isNil:
    return
  case t.kind
  of ctFloat:
    if t.bits == 64:
      f.fp64 = true
    if t.bits == 16:
      f.fp16 = true
  of ctVector: scanType(t.vElem, f)
  of ctArray: scanType(t.aElem, f)
  of ctPtr, ctAptr: scanType(t.pElem, f)
  of ctStruct:
    for fl in t.fields:
      scanType(fl.typ, f)
  of ctVoid, ctBool, ctInt, ctUint, ctProc:
    discard

proc scanExpr(e: CirExpr, f: var ModuleFeatures) =
  if e.isNil:
    return
  case e.kind
  of eCall:
    if e.callee.kind == eSym:
      if e.callee.reductionBuiltin in {rbkReduceAdd, rbkReduceMax}:
        f.subgroup = true
      if e.callee.atomicsBuiltin == abkAtomicFetchAdd:
        let pt = atomicOperandPtr(e.args[0])
        if pt != nil and pt.pElem != nil and pt.pElem.kind == ctFloat and
            pt.pElem.bits == 32:
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
    of sCase, sAsm:
      discard

proc scanModule(m: CirModule, f: var ModuleFeatures) =
  for t in m.types:
    scanType(t, f)
  scanStmts(m.globals, f)
  for fn in m.procs:
    scanType(fn.ret, f)
    for p in fn.params:
      scanType(p.typ, f)
    scanStmts(fn.body, f)

proc spellExtensions(f: ModuleFeatures): string =
  ## OpenCL C extension pragma block, one line per feature in use.
  if f.fp64:
    result.add "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
  if f.fp16:
    result.add "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
  if f.atomicFloat:
    result.add "#pragma OPENCL EXTENSION cl_ext_float_atomics : enable\n"
  if f.subgroup:
    result.add "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
  result.add "\n"

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printExpr(ctx: var PrintCtx, e: CirExpr): string
  ## Forward declaration, expressions recurse into themselves.

proc spellOpenclLit(e: CirExpr): string =
  ## OpenCL C token of a literal, typed by its literal type.
  ## C suffixes mark the 32-bit and 64-bit widths, a cast forms the narrow widths.
  case e.kind
  of eIntLit:
    case e.litType.kind
    of ctUint:
      case e.litType.bits
      of 64: result = $e.value & "ULL"
      of 32: result = $e.value & "U"
      else: result = "(" & spellOpenclScalar(e.litType) & ")" & $e.value
    of ctInt:
      case e.litType.bits
      of 64: result = $e.value & "LL"
      of 32: result = $e.value
      else: result = "(" & spellOpenclScalar(e.litType) & ")" & $e.value
    else:
      result = $e.value
  of eFloatLit:
    case e.fLitType.bits
    of 16: result = "(half)" & $e.fValue
    of 32: result = $e.fValue & "f"
    else: result = $e.fValue
  of eTrue: result = "true"
  of eFalse: result = "false"
  else: result = "<opencl: not a literal>"

proc spellVectorComponent(name: string): string =
  ## OpenCL C component letter of a CIRv2 vector component name (`F0`..`F3`).
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
    ctx.fail("nil literal is outside the OpenCL subset")
    return ""
  of eTrue, eFalse, eIntLit, eFloatLit:
    return spellOpenclLit(e)
  of eSym:
    if e.coordBuiltin != cbkNone:
      if e.coordBuiltin == cbkThreadIndexInThreadgroup:
        return OpenclFlatThreadIndex
      ctx.fail("whole-vector coordinate read " & e.name &
        ", OpenCL C has no vector form, read the components")
      return ""
    return e.name
  of eAt, ePat:
    let base = (if e.kind == eAt: e.atBase else: e.patBase)
    let idx = ctx.printExpr(if e.kind == eAt: e.atIndex else: e.patIndex)
    var baseTxt = ctx.printExpr(base)
    if base.kind in {eCast, eConv}:
      # a cast binds looser than the subscript: parenthesize the base
      baseTxt = "(" & baseTxt & ")"
    return baseTxt & "[" & idx & "]"
  of eDeref:
    return "(*" & ctx.printExpr(e.derefOperand) & ")"
  of eDot:
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      let mapped = openclCoordComponent(e.dotBase.coordBuiltin, e.fieldName)
      if mapped.len == 0:
        ctx.fail("no coordinate component " & e.fieldName)
        return ""
      return mapped
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
    return "(" & spellOpenclTypeBare(e.castType) & ")" &
      ctx.printExpr(e.castOperand)
  of eSizeof:
    return "sizeof(" & spellOpenclTypeBare(e.ofType) & ")"
  of eAddr:
    return "(&" & ctx.printExpr(e.addrOperand) & ")"
  of eOconstr:
    # C99 compound literal, a bare braced init list cannot be member-accessed
    if e.kvFields.len == 0:
      # a zero-field struct, the padding member zero-initializes
      return "(" & spellOpenclTypeBare(e.ocType) & "){0}"
    result = "(" & spellOpenclTypeBare(e.ocType) & "){"
    for i, kvv in e.kvFields:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(kvv.value)
    result.add "}"
  of eAconstr:
    result = "(" & spellOpenclTypeBare(e.acType) & "){"
    for i, el in e.elems:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(el)
    result.add "}"
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
        return "sub_group_reduce_add(" & args.join(", ") & ")"
      of rbkReduceMax:
        return "sub_group_reduce_max(" & args.join(", ") & ")"
      of rbkNone: discard
    if e.callee.kind == eSym and e.callee.atomicsBuiltin == abkAtomicFetchAdd:
      let pt = atomicOperandPtr(e.args[0])
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
        ctx.fail("atomicFetchAdd on " & spellOpenclScalar(pt.pElem) &
          ", only 32-bit int/uint/float pointees")
        return ""
      # atomic_fetch_add acts on `atomic_*` pointees: cast the raw
      # qualified pointer at the call site
      return "atomic_fetch_add((volatile " & qualifier & " " & atomicTy & "*)" &
        ctx.printExpr(e.args[0]) & ", " & ctx.printExpr(e.args[1]) & ")"
    if e.callee.kind == eSym and e.callee.synchroBuiltin != sbkNone:
      ctx.fail("barrier is a statement, a value-position barrier is outside the OpenCL subset")
      return ""
    return calleeName & "(" & args.join(", ") & ")"

# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt], exitL, contL: string): string
  ## Forward declaration, the loop context (break/continue labels) is a parameter.

proc spellLocalDecl(s: CirStmt): string =
  ## Declaration text of one local, the array form appends the length to the name, the init stays with the caller.
  if s.declType.kind == ctArray:
    result = spellOpenclArrayDecl(s.declType, s.name)
  elif s.declType.kind in {ctPtr, ctAptr}:
    result = spellOpenclPtrField(s.declType, s.name)
  else:
    result = spellOpenclTypeBare(s.declType) & ' ' & s.name

proc printStmt(ctx: var PrintCtx, s: CirStmt, exitL, contL: string): string =
  ## One structured statement, `exitL`/`contL` the enclosing loop's labels.
  if ctx.failing():
    return ""
  let pad = spaces(ctx.indent * 2)
  case s.kind
  of sVar:
    if s.space == caThreadgroup:
      if not ctx.inKernel:
        ctx.fail("smem local " & s.name & " in a device function, __local is kernel storage")
        return ""
      var decl = "__local " & spellLocalDecl(s)
      result = pad & decl
      if s.init != nil:
        result &= " = " & ctx.printExpr(s.init)
      result &= ";"
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
      var decl = "__private " & spellLocalDecl(s)
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
  of sConst:
    result = pad & "const " & spellOpenclTypeBare(s.declType) & ' ' &
      s.name & " = " & ctx.printExpr(s.init) & ";"
  of sAsgn:
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
    # a loop-bound hint, OpenCL C has no assume construct
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
      result = pad & "barrier(CLK_LOCAL_MEM_FENCE);"
    elif s.callCallee.kind == eSym and
        s.callCallee.synchroBuiltin == sbkThreadgroupBarrierDevice:
      # global fence bit orders device-address-space accesses
      result = pad & "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);"
    else:
      result = pad & parts[0] & "(" & parts[1 ..^ 1].join(", ") & ");"
  of sAsm:
    ctx.fail("inline assembly has no OpenCL C form")
    result = ""
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
        spaces(ctx.indent * 2) & "#pragma unroll\n"
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

proc spellParam(ctx: var PrintCtx, p: CirParam, isKernel: bool): string =
  ## Parameter declaration of one kernel or device-function parameter.
  ## A raw `ptr` parameter is legal C between device functions, a kernel refuses it.
  ## The engine binds device buffers and values, never raw pointers.
  case p.typ.kind
  of ctAptr:
    let inner = spellOpenclScalar(p.typ.pElem)
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
    result = spellOpenclTypeBare(p.typ) & ' ' & p.name
  of ctArray:
    ctx.fail("array parameter " & p.name &
      ", OpenCL C array parameters decay to pointers, the engine packs the value")
    return ""
  else:
    result = spellOpenclTypeBare(p.typ) & ' ' & p.name

proc printFn(ctx: var PrintCtx, fn: CirProc): string =
  ## One device function: signature, body, closing brace.
  ctx.fnName = fn.name
  ctx.inKernel = false
  ctx.indent = 0
  var params: seq[string]
  for p in fn.params:
    params.add ctx.spellParam(p, false)
  if ctx.failing():
    return ""
  let bodyTxt = ctx.printStmts(fn.body, "", "")
  if ctx.failing():
    return ""
  let sig = "static inline " & spellOpenclTypeBare(fn.ret) & ' ' &
    fn.name & '(' & params.join(", ") & ")"
  return sig & " {\n" & bodyTxt & '\n' & "}"

proc spellConstDecl(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope `__constant` declaration of a const_mem local.
  if s.declType.kind == ctArray:
    result = "__constant " & spellOpenclArrayDecl(s.declType, s.name)
    if s.init != nil and s.init.kind == eAconstr:
      var elems: seq[string]
      for el in s.init.elems:
        elems.add ctx.printExpr(el)
      result &= " = {" & elems.join(", ") & "}"
      return
  else:
    result = "__constant " & spellOpenclTypeBare(s.declType) & ' ' & s.name
  result &= " = " & ctx.printExpr(s.init) & ";"

proc printOpenclModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full OpenCL C translation unit: extension pragmas, struct typedefs,
  ## program-scope `__constant` declarations, the device functions,
  ## and the kernel.
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>")
  var kernels: seq[CirProc]
  for fn in m.procs:
    if fn.isKernel:
      kernels.add fn
  if kernels.len != 1:
    ctx.fail("exactly one kernel per module, found " & $kernels.len)
    return ("", ctx.err)
  let kernel = kernels[0]

  var features = ModuleFeatures()
  scanModule(m, features)
  result.src = spellExtensions(features)

  for t in m.types:
    if t.kind != ctStruct:
      ctx.fail("module type " & t.name & " is not a struct")
      break
    result.src &= "typedef struct {\n"
    if t.fields.len == 0:
      # OpenCL C requires at least one field in a struct
      result.src &= "  char _padding_;\n"
    for f in t.fields:
      var field: string
      if f.typ.kind == ctArray:
        field = spellOpenclArrayDecl(f.typ, f.name)
      elif f.typ.kind in {ctAptr, ctPtr}:
        if f.typ.kind == ctAptr and f.typ.space != caDevice:
          ctx.fail("struct field " & t.name & "." & f.name &
            " in " & $f.typ.space & " space, only device pointers nest in structs")
          field = ""
        else:
          field = spellOpenclPtrField(f.typ, f.name)
      else:
        field = spellOpenclTypeBare(f.typ) & ' ' & f.name
      result.src &= "  " & field & ";\n"
    result.src &= "} " & t.name & ";\n\n"
  if ctx.failing():
    return ("", ctx.err)

  # program-scope `__constant` declarations from `m.globals`, relocated there by the OpenCL lowering pass in encounter order
  var hoisted: seq[string]
  for s in m.globals:
    if s.kind == sVar and s.space == caConstant:
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
      params.add ctx.spellParam(p, false)
    if ctx.failing():
      return ("", ctx.err)
    result.src &= "static inline " & spellOpenclTypeBare(fn.ret) & ' ' &
      fn.name & '(' & params.join(", ") & ");\n"
  if fns.len > 0:
    result.src &= "\n"
  for fn in fns:
    result.src &= ctx.printFn(fn) & "\n\n"
  if ctx.failing():
    return ("", ctx.err)

  ctx.fnName = kernel.name
  ctx.inKernel = true
  ctx.indent = 0
  var params: seq[string]
  for p in kernel.params:
    params.add ctx.spellParam(p, true)
  if ctx.failing():
    return ("", ctx.err)
  let bodyTxt = ctx.printStmts(kernel.body, "", "")
  if ctx.failing():
    return ("", ctx.err)
  result.src &= "__kernel void " & kernel.name & '(' &
    params.join(", ") & ") {\n" & bodyTxt & '\n' & "}\n"
  return (result.src, "")



proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as OpenCL C source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  let (src, err) = printOpenclModule(ctx.module)
  if err.len > 0:
    error(err)
  result = src
