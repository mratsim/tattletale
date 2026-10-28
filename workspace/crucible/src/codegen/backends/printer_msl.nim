# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal Shading Language (MSL) printer: a structured CIRv2 module to MSL source,
## table-driven over the closed type lattice.
##
## Emission contract, one buffer slot per kernel parameter in order:
## - pointers: `device T* name [[buffer(n)]]`
## - scalars: unattributed `constant T& name`, Metal auto-assigns the free slot
## - coordinate builtins: attributed `uint3`/`uint` params, no slot
##
## Other emission rules:
## - `{.smem.}` locals: a kernel's top-level local stays in-body as kernel-scope
##   `threadgroup` storage (Metal allocates it per threadgroup),
##   any initializer is discarded, a local outside a kernel body drops,
##   a device function still owning a smem local refuses in the pass
## - `{.const_mem.}` locals relocate into `CirModule.globals` through the MSL lowering pass,
##   program-scope `constant` output, a const_mem survivor refuses at the printer
## - `jmp` to the loop exit label is `break`, to the continue label `continue`, any other lab/jmp refuses
## - the typed builtin kinds dispatch to the MSL vocabulary, `{.builtin.}` names pass
##   through except the numeric conversions, which resolve to the MSL native call text

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, sequtils, sets, tables, math]

type
  PrintError* = object
    ## Refusal of the printer, names the proc and the cause.
    procName*: string
    msg*: string

  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    ## Per-kernel state.
    builtins: OrderedSet[string]
    ## Canonical builtin names the kernel body references, first-use order.

proc failing(ctx: PrintCtx): bool =
  ctx.err.len > 0

proc fail(ctx: var PrintCtx, detail: string) =
  ## Records the first refusal, naming the proc.
  if ctx.err.len == 0:
    ctx.err = "cirv2 printer msl: at " & ctx.fnName & ": " & detail

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc spellMslSpace(space: CirAddrSpace): string =
  ## MSL address-space qualifier of a memory class.
  case space
  of caThread: "thread"
  of caThreadgroup: "threadgroup"
  of caConstant: "constant"
  of caDevice: "device"

proc spellMslType(t: CirType, ident: string, allowEmptyIdent: bool): string
  ## Forward declaration, arrays recurse into themselves.

## `{.builtin.}` numeric conversions resolve to the MSL native call text:
## each entry is a format string whose `$1` stands for the call's argument list.
## The CIRv2 lattice has no fp16 or bf16 kind: `float16` and `bfloat16` are `distinct uint16` types holding raw bits in `ushort` storage.
## Value-producing conversions move the bit pattern, not the numeric value:
## - `toFp16(x)` prints `as_type<ushort>(half(x))`
## - `toBf16(x)` prints `as_type<ushort>((bfloat)(x))`
## - `toFp32`/`toBf32` reinterpret the stored bits back through
##   `as_type<half>`/`as_type<bfloat>` before the `float` widening
## - `asFp16`/`asU16` are identities on that storage
## fp8 conversions have no MSL native: the `{.builtin.}` name passes through and refuses at Metal compile time.
const MslConversionBuiltinNames: Table[string, string] = {
  "toFp16": "as_type<ushort>(half($1))",
  "toFp32": "float(as_type<half>($1))",
  "toBf16": "as_type<ushort>((bfloat)($1))",
  "toBf32": "float(as_type<bfloat>($1))",
  "asFp16": "$1",
  "asU16": "$1",
}.toTable()

proc spellMslScalar(t: CirType): string =
  ## MSL name of a non-array leaf type, "" when the caller composes the name.
  case t.kind
  of ctVoid:
    result = "void"
  of ctBool:
    result = "bool"
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
    else:
      result = ""
  of ctVector:
    result = spellMslScalar(t.vElem)
    if result.len > 0:
      result.add $t.vN
  of ctPtr, ctAptr:
    result = spellMslScalar(t.pElem) & "*"
  of ctStruct:
    result = t.name
  of ctArray, ctProc:
    result = ""


proc spellMslType(t: CirType, ident: string, allowEmptyIdent: bool): string =
  ## MSL name of `t` with `ident` appended. Arrays take the identifier into
  ## the declarator (`T ident[N]`, pointer-to-array `T (*ident)[N]`).
  ## `allowEmptyIdent` permits no identifier (return types, casts), an array without one otherwise refuses.
  case t.kind
  of ctArray:
    if ident.len == 0 and not allowEmptyIdent:
      return "<msl: array needs an identifier>"
    if t.aElem.kind == ctArray:
      result = spellMslType(t.aElem, "(" & ident & ")" & "[" & $t.aN & "]",
        allowEmptyIdent)
    else:
      result = spellMslType(t.aElem, "", true) & ' ' & ident & '[' & $t.aN & ']'
  else:
    result = spellMslScalar(t)
    if ident.len > 0:
      result.add ' ' & ident

proc spellMslPtr(t: CirType, ident: string): string =
  ## MSL name of a pointer or array pointer with its address space:
  ## `device float* p`, pointer-to-array `device float (*p)[N]`.
  let inner = t.pElem
  if inner.kind == ctArray:
    result = spellMslSpace(t.space) & ' ' & spellMslType(inner, "(*" & ident & ")", true)
  else:
    result = spellMslSpace(t.space) & ' ' & spellMslScalar(inner) & "* " & ident

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printExpr(ctx: var PrintCtx, e: CirExpr): string
  ## Forward declaration, expressions recurse into themselves.

proc spellMslLit(e: CirExpr): string =
  ## MSL token of a literal, typed by its literal type.
  case e.kind
  of eIntLit:
    case e.litType.kind
    of ctUint:
      case e.litType.bits
      of 64: result = $e.value & "ULL"
      of 32: result = $e.value & "U"
      else: result = "(" & spellMslScalar(e.litType) & ")" & $e.value
    of ctInt:
      case e.litType.bits
      of 64: result = $e.value & "LL"
      of 32: result = $e.value
      else: result = "(" & spellMslScalar(e.litType) & ")" & $e.value
    else:
      result = $e.value
  of eFloatLit:
    # MSL has no `inf`/`nan` tokens and Nim's `$` renders `inf`/`nan`.
    # Emission text is the <math.h> macros the MSL compiler accepts.
    # Literal width converts at the use site.
    case e.fValue.classify
    of fcInf:
      result = "INFINITY"
    of fcNegInf:
      result = "-INFINITY"
    of fcNan:
      result = "NAN"
    else:
      case e.fLitType.bits
      of 16: result = $e.fValue & "h"
      of 32: result = $e.fValue & "f"
      else: result = $e.fValue
  of eNil: result = "nullptr"
  of eTrue: result = "true"
  of eFalse: result = "false"
  else: result = "<msl: not a literal>"


proc printBuiltinRef(ctx: var PrintCtx, name: string, whole: bool): string =
  ## MSL token of a builtin reference, the attributed parameter name.
  ##
  ## - whole-value use of a vector builtin: an `int3` constructor,
  ##   the attribute params are `uint3`, the canonical Nim type is int
  ## - component use: a cast to `int`
  ## - the flat thread index: a scalar `uint`, whole value is the bare name
  ##   (uint converts to int implicitly)
  ctx.builtins.incl(name)
  if whole and name != "thread_index_in_threadgroup":
    return "int3((int)" & name & ".x, (int)" & name & ".y, (int)" & name & ".z)"
  return name

proc spellMslBuiltinCall(callee: CirExpr, args: seq[string]): string =
  ## Returns the MSL text of a builtin call, "" when `callee` is not a builtin.
  ##
  ## Shared by expression- and statement-position calls: the statement path
  ## bypasses `printExpr`, so a barrier printed as a bare name would lose
  ## its memory flags, text MSL refuses to compile.
  ##
  ## `args` contains the already-printed argument texts.
  if callee.kind != eSym:
    return ""
  # builtin vocabulary dispatches on the typed kinds, never on names
  case callee.synchroBuiltin
  of sbkThreadgroupBarrier:
    return "threadgroup_barrier(mem_flags::mem_threadgroup)"
  of sbkThreadgroupBarrierDevice:
    return "threadgroup_barrier(mem_flags::mem_device)"
  of sbkNone: discard
  case callee.reductionBuiltin
  of rbkReduceAdd:
    return "simd_sum(" & args.join(", ") & ")"
  of rbkReduceMax:
    return "simd_max(" & args.join(", ") & ")"
  of rbkNone: discard
  if callee.atomicsBuiltin == abkAtomicFetchAdd:
    # device float atomics: the pointer casts to the atomic type
    # device address space implies the scope, relaxed order
    return "atomic_fetch_add_explicit((volatile device atomic_float*)" &
      "(" & args[0] & "), " & args[1] & ", memory_order_relaxed)"
  if callee.builtin and MslConversionBuiltinNames.hasKey(callee.name):
    return MslConversionBuiltinNames.getOrDefault(callee.name) % args.join(", ")
  return ""


proc printExpr(ctx: var PrintCtx, e: CirExpr): string =
  if ctx.failing():
    return ""
  case e.kind
  of eNil, eTrue, eFalse, eIntLit, eFloatLit:
    return spellMslLit(e)
  of eSym:
    if e.coordBuiltin != cbkNone:
      return ctx.printBuiltinRef(e.name, true)
    return e.name
  of eAt:
    let base = ctx.printExpr(e.atBase)
    return base & '[' & ctx.printExpr(e.atIndex) & ']'
  of ePat:
    let base = ctx.printExpr(e.patBase)
    return base & '[' & ctx.printExpr(e.patIndex) & ']'
  of eDot:
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      # a component of an unsigned MSL builtin, cast to the canonical int
      ctx.builtins.incl(e.dotBase.name)
      return "(int)(" & e.dotBase.name & '.' & e.fieldName & ")"
    let base = ctx.printExpr(e.dotBase)
    if e.dotBase.kind == eDeref:
      return "(" & base & ")." & e.fieldName
    return base & '.' & e.fieldName
  of eDeref:
    return "(*" & ctx.printExpr(e.derefOperand) & ")"
  of eAddr:
    return "(&" & ctx.printExpr(e.addrOperand) & ")"
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
    # pure bool operators over side-effect-free operands: the frontend materializes
    # value-position short-circuits into temps and the remainder prints directly
    let op = if e.kind == eAnd: "&&" else: "||"
    return "(" & ctx.printExpr(e.andLeft) & ' ' & op & ' ' & ctx.printExpr(e.andRight) & ")"
  of eBitNot:
    return "(~" & ctx.printExpr(e.unOperand) & ")"
  of eNeg:
    return "(-" & ctx.printExpr(e.unOperand) & ")"
  of eConv:
    # a numeric conversion prints as the C-style cast
    let t = e.castType
    var castTyp = spellMslType(t, "", true)
    if t.kind in {ctPtr, ctAptr}:
      castTyp = spellMslSpace(t.space) & ' ' & castTyp
    return "(" & castTyp & ")" & ctx.printExpr(e.castOperand)
  of eCast:
    # A same-size scalar `cast` reinterprets the bits: the SPIR-V printer
    # emits OpBitcast and the MSL text is `as_type<T>`. A C-style cast
    # converts numerically instead: 0xFF7FFFFF read as float32 gives
    # +4.29e9 and the same bits reinterpreted give -FLT_MAX.
    #
    # Pointer and machine-word casts keep the C-style text.
    # as_type accepts neither pointers nor width changes.
    let t = e.castType
    let src = getExprType(e.castOperand)
    if t.kind in {ctInt, ctUint, ctFloat} and src != nil and
        src.kind in {ctInt, ctUint, ctFloat} and
        t.bits in {8, 16, 32, 64} and src.bits == t.bits:
      return "as_type<" & spellMslType(t, "", true) & ">(" & ctx.printExpr(e.castOperand) & ")"
    var castTyp = spellMslType(t, "", true)
    if t.kind in {ctPtr, ctAptr}:
      castTyp = spellMslSpace(t.space) & ' ' & castTyp
    return "(" & castTyp & ")" & ctx.printExpr(e.castOperand)
  of eSizeof:
    return "sizeof(" & spellMslType(e.ofType, "", true) & ")"
  of eOconstr:
    result = spellMslType(e.ocType, "", true) & "{"
    for i, kv in e.kvFields:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(kv.value)
    result.add "}"
  of eAconstr:
    result = "{"
    for i, el in e.elems:
      if i > 0:
        result.add ", "
      result.add "(" & spellMslType(e.acType.aElem, "", true) & ")" &
        ctx.printExpr(el)
    result.add "}"
  of eCall:
    let calleeName =
      if e.callee.kind == eSym: e.callee.name
      else: ctx.printExpr(e.callee)
    var args: seq[string]
    for a in e.args:
      args.add ctx.printExpr(a)
    let builtinTxt = spellMslBuiltinCall(e.callee, args)
    if builtinTxt.len > 0:
      return builtinTxt
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
    if s.space == caThreadgroup:
      # kernel-scope threadgroup storage: the MSL lowering pass keeps only
      # a kernel's top-level smem locals here, an inline device function's
      # smem local merges into the calling kernel, any initializer is discarded
      result = pad & "threadgroup " & spellMslType(s.declType, s.name, false) & ";"
      return
    if s.space == caConstant:
      # MSL lowering pass relocates const_mem locals at every scope,
      # a survivor here was never relocated
      ctx.fail("local " & s.name & " in constant space, the MSL lowering" &
        " pass hoists const_mem locals to program scope")
      return ""
    var decl = spellMslType(s.declType, s.name, false)
    if s.declType.kind in {ctPtr, ctAptr}:
      decl = spellMslPtr(s.declType, s.name)
    result = pad & decl
    if s.init != nil:
      result &= " = " & ctx.printExpr(s.init)
    result &= ";"
  of sConst:
    if s.declType.kind == ctArray:
      result = pad & "constexpr " &
        spellMslType(s.declType, s.name, false) & " = " & ctx.printExpr(s.init) & ";"
    else:
      result = pad & "constexpr " & spellMslScalar(s.declType) & ' ' & s.name &
        " = " & ctx.printExpr(s.init) & ";"
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
    # a loop-bound hint, MSL has no assume construct
    result = ""
  of sRet:
    if s.retVal != nil:
      result = pad & "return " & ctx.printExpr(s.retVal) & ";"
    else:
      result = pad & "return;"
  of sDiscard:
    result = ""
  of sCall:
    let calleeName =
      if s.callCallee.kind == eSym: s.callCallee.name
      else: ctx.printExpr(s.callCallee)
    var args: seq[string]
    for a in s.callArgs:
      args.add ctx.printExpr(a)
    let builtinTxt = spellMslBuiltinCall(s.callCallee, args)
    if builtinTxt.len > 0:
      result = pad & builtinTxt & ";"
    else:
      result = pad & calleeName & "(" & args.join(", ") & ");"
  of sAsm:
    # inline assembly: the verbatim template spliced with the CIRv2 names
    var text = ""
    for i, part in s.asmParts:
      text.add(part)
      if i < s.asmOperands.len:
        text.add(s.asmOperands[i].name)
    result = pad & text
  of sCase:
    ctx.fail("case statements are outside the CIRv2 closure, the frontend refuses them")

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
        spaces(ctx.indent * 2) & "#pragma clang loop unroll(full)\n"
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

proc spellMslSmemParam(p: CirParam): string =
  ## MSL parameter of a relocated smem local: threadgroup storage by pointer
  ## for an array local (`threadgroup T* name`), by reference otherwise
  ## (`threadgroup T& name`).
  ##
  ## No binding, no buffer slot, the appended parameters trail the regular
  ## parameters of the kernel signature.
  let inner = p.typ.pElem
  if inner.kind == ctArray:
    return "threadgroup " & spellMslType(inner.aElem, "", true) & "* " & p.name
  return "threadgroup " & spellMslScalar(inner) & "& " & p.name

proc spellBuiltinParam(name: string): string =
  ## Attributed MSL parameter of a canonical builtin reference:
  ## `uint3 name [[name]]` for the coordinate vectors, a scalar `uint` for the flat thread index.
  ## No binding, no buffer slot.
  if name == "thread_index_in_threadgroup":
    return "uint " & name & " [[" & name & "]]"
  return "uint3 " & name & " [[" & name & "]]"

proc printProc(ctx: var PrintCtx, fn: CirProc): string =
  ## One proc or kernel: signature, body, closing brace. The body prints first,
  ## the kernel signature consumes the builtin set the body references.
  ctx.fnName = fn.name
  ctx.builtins = initOrderedSet[string]()
  ctx.indent = 0
  let bodyTxt = ctx.printStmts(fn.body, "", "")
  if ctx.failing():
    return ""
  let name = if fn.cudaName.len > 0: fn.cudaName else: fn.name
  var params: seq[string]
  var slot = 0
  if fn.isKernel:
    for p in fn.params:
      if p.typ.kind == ctPtr and p.typ.space == caThreadgroup:
        # a relocated smem local, no buffer slot
        params.add spellMslSmemParam(p)
        continue
      case p.typ.kind
      of ctAptr, ctPtr:
        let inner = p.typ.pElem
        params.add "device " & spellMslType(inner, "", true) & "* " & p.name &
          " [[buffer(" & $slot & ")]]"
      else:
        var elem = spellMslScalar(p.typ)
        if p.typ.kind == ctBool:
          elem = "int"
        params.add "constant " & elem & "& " & p.name
      inc slot
    for b in ctx.builtins:
      # coordinate builtins are attributed params, no binding, no slot
      params.add spellBuiltinParam(b)
  else:
    for p in fn.params:
      case p.typ.kind
      of ctAptr, ctPtr:
        params.add spellMslPtr(p.typ, p.name)
      else:
        params.add spellMslType(p.typ, p.name, true)
  let ret = spellMslType(fn.ret, "", true)
  let prefix = if fn.isKernel: "kernel "
    elif fn.forceInline: "inline "
    else: ""
  let sig = prefix & ret & ' ' & name & '(' & params.join(", ") & ")"
  return sig & " {\n" & bodyTxt & '\n' & "}"

proc printModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full MSL translation unit: metal headers, struct declarations, program-scope constants,
  ## device-function forward declarations, then every function body.
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>")
  var src = "#include <metal_stdlib>\n"
  src &= "#include <metal_atomic>\n"
  src &= "using namespace metal;\n\n"
  for t in m.types:
    if t.kind != ctStruct:
      ctx.fail("module type " & t.name & " is not a struct")
      break
    src &= "struct " & t.name & " {\n"
    if t.fields.len == 0:
      src &= "  char _;\n"
    for f in t.fields:
      var field = spellMslType(f.typ, f.name, false)
      if f.typ.kind in {ctPtr, ctAptr}:
        field = spellMslPtr(f.typ, f.name)
      src &= "  " & field & ";\n"
    src &= "};\n\n"
  # program-scope `constant` declarations relocated into `m.globals` by the MSL lowering pass,
  # in encounter order
  var hoisted: seq[string]
  for s in m.globals:
    if s.kind == sVar and s.space == caConstant:
      if s.init == nil:
        ctx.fail("const_mem local " & s.name & " without an initializer")
      else:
        var decl = spellMslType(s.declType, s.name, false)
        if s.declType.kind in {ctPtr, ctAptr}:
          decl = spellMslPtr(s.declType, s.name)
        hoisted.add "constant " & decl & " = " & ctx.printExpr(s.init) & ";"
  for d in hoisted:
    src &= d & "\n"
  if hoisted.len > 0:
    src &= "\n"
  if ctx.failing():
    return ("", ctx.err)
  for fn in m.procs:
    let txt = ctx.printProc(fn)
    if ctx.err.len > 0:
      break
    src &= txt & "\n\n"
  result = (src, ctx.err)
