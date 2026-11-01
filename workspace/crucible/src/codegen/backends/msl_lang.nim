# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal Shading Language (MSL) printer: a structured CIRv2 module to MSL source,
## table-driven over the closed set of type kinds.
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
## - a loop prints `while (cond)`, a conditionless loop prints `for (;;)`, `{.unroll.}` adds the Metal unroll pragma
## - the typed builtin kinds dispatch to the MSL vocabulary, `{.builtin.}` names pass
##   through except the numeric conversions, which resolve to the MSL native call text

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, sequtils, sets, tables, math]
import ../ir/ast_datatypes
import ./emit_common

type
  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    ## Literal token table of the target, the shared literal head reads it.
    ## Per-kernel state.
    builtins: OrderedSet[string]
    ## Canonical builtin names that the kernel body references, first-use order.

const printerTarget = "msl"

## Optional expression kinds this printer routes through
## its own `printTargetExpr`, the shared dispatcher emits the rest.
const targetExprKinds: set[CirExprKind] = {}

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitMslSpace(space: CirAddrSpace): string =
  ## MSL address-space qualifier of a memory class.
  case space
  of caThread: "thread"
  of caThreadgroup: "threadgroup"
  of caConstant: "constant"
  of caDevice: "device"

proc emitMslType(t: CirType, ident: string, allowEmptyIdent: bool): string
  ## Forward declaration, arrays recurse into themselves.

## `{.builtin.}` numeric conversions resolve to the MSL native call text:
## each entry is a format string whose `$1` stands for the call's argument list.
## CIRv2 has no fp16 or bf16 type kind: `float16` and `bfloat16` are `distinct uint16` types holding raw bits in `ushort` storage.
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

const mslScalarTokens = ScalarTokens(
  void: "void", bool: "bool",
  i8: "char", i16: "short", i32: "int", i64: "long",
  u8: "uchar", u16: "ushort", u32: "uint", u64: "ulong",
  f16: "half", f32: "float")

proc emitMslScalar(t: CirType): string =
  ## MSL name of a non-array leaf type, "" when the caller composes the name.
  case t.kind
  of ctVoid, ctBool, ctInt, ctUint, ctFloat:
    result = emitScalarToken(mslScalarTokens, t)
  of ctPtr, ctAptr:
    result = emitMslScalar(t.pElem) & "*"
  of ctStruct:
    result = t.name
  of ctVector:
    # coordinate builtins carry `ivec3`, so the builtin-threading
    # parameters and whole-value reads print this MSL vector name
    case t.vElem.kind
    of ctInt, ctUint:
      if t.vElem.bits == 32:
        result = (if t.vElem.kind == ctInt: "int" else: "uint") & $t.vN
      else:
        result = ""
    else:
      result = ""
  of ctArray, ctProc, ctSpan:
    result = ""


proc emitMslType(t: CirType, ident: string, allowEmptyIdent: bool): string =
  ## MSL name of `t` with `ident` appended. Arrays take the identifier into
  ## the declarator (`T ident[N]`, pointer-to-array `T (*ident)[N]`).
  ## `allowEmptyIdent` permits no identifier (return types, casts), an array without one otherwise refuses.
  case t.kind
  of ctArray:
    if ident.len == 0 and not allowEmptyIdent:
      return "<msl: array needs an identifier>"
    if t.aElem.kind == ctArray:
      result = emitMslType(t.aElem, "(" & ident & ")" & "[" & $t.aN & "]",
        allowEmptyIdent)
    else:
      result = emitMslType(t.aElem, "", true) & ' ' & ident & '[' & $t.aN & ']'
  else:
    result = emitMslScalar(t)
    if ident.len > 0:
      result.add ' ' & ident

proc emitMslPtr(t: CirType, ident: string): string =
  ## MSL name of a pointer or array pointer with its address space:
  ## `device float* p`, pointer-to-array `device float (*p)[N]`.
  let inner = t.pElem
  if inner.kind == ctArray:
    result = emitMslSpace(t.space) & ' ' & emitMslType(inner, "(*" & ident & ")", true)
  else:
    result = emitMslSpace(t.space) & ' ' & emitMslScalar(inner) & "* " & ident

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

# `INFINITY`, `-INFINITY` and `NAN` convert at the use site, one form covers every width.
const targetLitTokens = LitTokens(
  scalar: mslScalarTokens,
  int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
  f16Suffix: "h", f32Suffix: "f",
  nilToken: "nullptr",
  nonFinite: FloatLitTokens(
    inf16: "INFINITY", negInf16: "-INFINITY", nan16: "NAN",
    inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
    inf64: "INFINITY", negInf64: "-INFINITY", nan64: "NAN"))

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

proc emitMslBuiltinCall(callee: CirExpr, args: seq[string]): string =
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
    # pointer casts to the atomic pointee type, relaxed order only,
    # fp32 needs Metal 3+, device address space implies the scope
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
  if callee.builtin and MslConversionBuiltinNames.hasKey(callee.name):
    return MslConversionBuiltinNames.getOrDefault(callee.name) % args.join(", ")
  return ""


proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## One numeric conversion, the C-style cast form, pointer casts name
  ## their address space.
  let t = e.castType
  var castTyp = emitMslType(t, "", true)
  if t.kind in {ctPtr, ctAptr}:
    castTyp = emitMslSpace(t.space) & ' ' & castTyp
  return "(" & castTyp & ")" & ctx.printExpr(e.castOperand)

proc printCast(ctx: var PrintCtx, e: CirExpr): string =
  ## One bit reinterpret, `as_type<T>`.
  ##
  ## - a C-style cast converts numerically, 0xFF7FFFFF read as float32
  ##   gives +4.29e9 and the same bits reinterpreted give -FLT_MAX
  ## - pointer and machine-word casts keep the C-style text
  ## - as_type accepts neither pointers nor width changes
  let t = e.castType
  let src = getExprType(e.castOperand)
  if t.kind in {ctInt, ctUint, ctFloat} and src != nil and
      src.kind in {ctInt, ctUint, ctFloat} and
      t.bits in {8, 16, 32, 64} and src.bits == t.bits:
    return "as_type<" & emitMslType(t, "", true) & ">(" & ctx.printExpr(e.castOperand) & ")"
  return ctx.printConv(e)

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## MSL text of the kinds in `targetExprKinds`.
  case e.kind
  of eSym:
    # a coordinate builtin takes its attributed parameter name, a name passes through
    if e.coordBuiltin != cbkNone:
      return ctx.printBuiltinRef(e.name, true)
    return e.name
  of eDot:
    # a coordinate component casts to the canonical int, a dereferenced base parenthesizes
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      ctx.builtins.incl(e.dotBase.name)
      return "(int)(" & e.dotBase.name & '.' & e.fieldName & ")"
    let base = ctx.printExpr(e.dotBase)
    if e.dotBase.kind == eDeref:
      return "(" & base & ")." & e.fieldName
    return base & '.' & e.fieldName
  of eCast:
    return ctx.printCast(e)
  of eConv:
    return ctx.printConv(e)
  of eSizeof:
    return "sizeof(" & emitMslType(e.ofType, "", true) & ")"
  of eOconstr:
    # one struct construction in braces
    result = emitMslType(e.ocType, "", true) & "{"
    for i, kv in e.kvFields:
      if i > 0:
        result.add ", "
      result.add ctx.printExpr(kv.value)
    result.add "}"
  of eAconstr:
    # every element casts to the element type
    result = "{"
    for i, el in e.elems:
      if i > 0:
        result.add ", "
      result.add "(" & emitMslType(e.acType.aElem, "", true) & ")" &
        ctx.printExpr(el)
    result.add "}"
  of eCall:
    let args = printCallArgs(ctx, e)
    return printCallText(ctx, e, args, emitMslBuiltinCall(e.callee, args))
  else:
    # dispatcher sends only the kinds above, anything else is a routing bug
    ctx.fail("expression kind " & $e.kind & " has no MSL emission")


# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

const mslLoopStyle = LoopStyle(
  unrollDirective: "#pragma clang loop unroll(full)",
  infiniteHead: "for (;;) {\n",
  braceSameLine: true)

# ═══════════════════════════════════════════════════════════════════════
# Statements: per-target policy arms, the shared dispatch lives in emit_common
# ═══════════════════════════════════════════════════════════════════════
proc printVar(ctx: var PrintCtx, s: CirStmt): string =
  ## One local declaration: kernel-scope threadgroup storage and storage-class refusals, then the declaration text.
  let pad = spaces(ctx.indent * 2)
  if s.space == caThreadgroup:
    # kernel-scope threadgroup storage: the MSL lowering pass keeps only
    # a kernel's top-level smem locals here, a device function's smem local
    # refuses in the pass, any initializer is discarded
    result = pad & "threadgroup " & emitMslType(s.declType, s.name, false) & ";"
    return
  if s.space == caConstant:
    # MSL lowering pass relocates const_mem locals at every scope,
    # a survivor here was never relocated
    ctx.fail("local " & s.name & " in constant space, the MSL lowering" &
      " pass hoists const_mem locals to program scope")
    return ""
  var decl = emitMslType(s.declType, s.name, false)
  if s.declType.kind in {ctPtr, ctAptr}:
    decl = emitMslPtr(s.declType, s.name)
  result = pad & decl
  if s.init != nil:
    result &= " = " & ctx.printExpr(s.init)
  result &= ";"

proc printConst(ctx: var PrintCtx, s: CirStmt): string =
  ## One local constant declaration, the `constexpr` form.
  let pad = spaces(ctx.indent * 2)
  if s.declType.kind == ctArray:
    result = pad & "constexpr " &
      emitMslType(s.declType, s.name, false) & " = " & ctx.printExpr(s.init) & ";"
  else:
    result = pad & "constexpr " & emitMslScalar(s.declType) & ' ' & s.name &
      " = " & ctx.printExpr(s.init) & ";"

proc printAsgn(ctx: var PrintCtx, s: CirStmt): string =
  ## One assignment.
  let pad = spaces(ctx.indent * 2)
  result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"

proc printCall(ctx: var PrintCtx, s: CirStmt): string =
  ## One call statement: the builtin call vocabulary, then the plain call form.
  let pad = spaces(ctx.indent * 2)
  let calleeName =
    if s.callCallee.kind == eSym: s.callCallee.name
    else: ctx.printExpr(s.callCallee)
  var args: seq[string]
  for a in s.callArgs:
    args.add ctx.printExpr(a)
  let builtinTxt = emitMslBuiltinCall(s.callCallee, args)
  if builtinTxt.len > 0:
    result = pad & builtinTxt & ";"
  else:
    result = pad & calleeName & "(" & args.join(", ") & ");"

proc printAsm(ctx: var PrintCtx, s: CirStmt): string =
  ## One inline assembly statement: the spliced text, MSL has no asm form.
  let pad = spaces(ctx.indent * 2)
  var text = ""
  for i, part in s.asmParts:
    text.add(part)
    if i < s.asmOperands.len:
      text.add(s.asmOperands[i].name)
  result = pad & text

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt]): string =
  ## Prints one statement list, the shared walk with the MSL loop style.
  printStmtList(ctx, stmts, mslLoopStyle)

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc emitMslSmemParam(p: CirParam): string =
  ## MSL parameter of a relocated smem local: threadgroup storage by pointer
  ## for an array local (`threadgroup T* name`), by reference otherwise
  ## (`threadgroup T& name`).
  ##
  ## No binding, no buffer slot, the appended parameters trail the regular
  ## parameters of the kernel signature.
  let inner = p.typ.pElem
  if inner.kind == ctArray:
    return "threadgroup " & emitMslType(inner.aElem, "", true) & "* " & p.name
  return "threadgroup " & emitMslScalar(inner) & "& " & p.name

proc emitBuiltinParam(name: string): string =
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
  let bodyTxt = ctx.printStmts(fn.body)
  if ctx.failing():
    return ""
  # exportc renames a kernel entry for the host runtime lookup.
  # A device function keeps the name its call sites reference.
  let name = if fn.isKernel and fn.exportName.len > 0: fn.exportName else: fn.name
  var params: seq[string]
  var slot = 0
  if fn.isKernel:
    for p in fn.params:
      if p.typ.kind == ctPtr and p.typ.space == caThreadgroup:
        # a relocated smem local, no buffer slot
        params.add emitMslSmemParam(p)
        continue
      case p.typ.kind
      of ctAptr, ctPtr:
        let inner = p.typ.pElem
        params.add "device " & emitMslType(inner, "", true) & "* " & p.name &
          " [[buffer(" & $slot & ")]]"
      else:
        var elem = emitMslScalar(p.typ)
        if p.typ.kind == ctBool:
          elem = "int"
        params.add "constant " & elem & "& " & p.name
      inc slot
    for b in ctx.builtins:
      # coordinate builtins are attributed params, no binding, no slot
      params.add emitBuiltinParam(b)
  else:
    for p in fn.params:
      case p.typ.kind
      of ctAptr, ctPtr:
        # a threadgroup-space pointer parameter is a relocated smem arena,
        # caller-owned storage (Metal rejects threadgroup allocation inside device functions)
        # - an `aptr` over the element type prints a pointer:
        #   `threadgroup T* name`
        # - a `ptr` over the object type prints a reference:
        #   `threadgroup T& name`
        if p.typ.space == caThreadgroup and p.typ.kind == ctPtr:
          params.add emitMslSmemParam(p)
        else:
          params.add emitMslPtr(p.typ, p.name)
      else:
        params.add emitMslType(p.typ, p.name, true)
  let ret = emitMslType(fn.ret, "", true)
  let prefix = if fn.isKernel: "kernel "
    elif fn.inline: "inline "
    else: ""
  let sig = prefix & ret & ' ' & name & '(' & params.join(", ") & ")"
  return sig & " {\n" & bodyTxt & '\n' & "}"

proc printMslModule*(m: CirModule): tuple[src: string, err: string] =
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
      var field = emitMslType(f.typ, f.name, false)
      if f.typ.kind in {ctPtr, ctAptr}:
        field = emitMslPtr(f.typ, f.name)
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
        var decl = emitMslType(s.declType, s.name, false)
        if s.declType.kind in {ctPtr, ctAptr}:
          decl = emitMslPtr(s.declType, s.name)
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


proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as Metal Shading Language (MSL) source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printMslModule(ctx.module))
