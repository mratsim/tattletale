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
## - `{.smem.}` locals: the pass keeps a kernel's top-level local in-body
##   as kernel-scope `threadgroup` storage (Metal allocates it per threadgroup),
##   any initializer is discarded, a local outside a kernel body drops,
##   the pass refuses a device function still owning a smem local
## - `{.const_mem.}` locals relocate into `CirModule.globals` through the MSL lowering pass,
##   program-scope `constant` output, the printer refuses a const_mem survivor
## - a loop prints `while (cond)`, `{.unroll.}` adds the Metal unroll pragma
## - the typed builtin kinds dispatch to the MSL vocabulary, `{.builtin.}` names pass
##   through except the numeric conversions, which resolve to the MSL native call text

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, sequtils, sets, tables, math]
import ../ir/ast_datatypes
import ./print_common
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
    ## Literal token forms of the target. The shared literal head reads them, per kernel state.
    row: ScalarRow
    ## Scalar row of the target. The narrow integer cast names its type from it., per kernel state.
    builtins: OrderedSet[string]
    ## Canonical builtin names that the kernel body references, first-use order.

const printerTarget = "msl"

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitMslType(t: CirType, ident: string, allowEmptyIdent: bool): string
  ## Forward declaration, arrays recurse into themselves.

proc emitMslScalar(t: CirType): string =
  ## MSL bare name of a non-array leaf type, "" when the caller composes the name.
  getBareName(mslScalarRow, mslVectorRule, t)

proc emitMslType(t: CirType, ident: string, allowEmptyIdent: bool): string =
  ## MSL declaration text of `t` with `ident` appended, the shared C declarator
  ## over the MSL rows. Arrays take the identifier into the declarator
  ## (`T ident[N]`, pointer-to-array `T (*ident)[N]`), the printer refuses an array without an identifier.
  cDecl(mslScalarRow, mslVectorRule, printerTarget, t, ident, allowEmptyIdent)

proc ptrField(row: ScalarRow, rule: VectorRule, target: string, qualifier: string, t: CirType, ident: string): string =
  ## Declaration text of a pointer or array pointer with its address-space qualifier:
  ## `device float* p`, pointer-to-array `device float (*p)[N]`.
  ##
  ## - `qualifier`: the MSL pointer space qualifier, "" never occurs on MSL
  ##   (all four address spaces carry a qualifier)
  ## - an array pointee takes the identifier into the parenthesized declarator
  let sep = if qualifier.len > 0: " " else: ""
  let inner = t.pElem
  if inner.kind == ctArray:
    return qualifier & sep & cDecl(row, rule, target, inner, "(*" & ident & ")", true)
  return qualifier & sep & getBareName(row, rule, inner) & "* " & ident

proc getMslPtrQualifier(space: AddressSpace): string =
  ## MSL qualifier before a pointer declaration or cast, one word per space.
  case space
  of caThread: "thread"
  of caThreadgroup: "threadgroup"
  of caConstant: "constant"
  of caDevice: "device"

proc emitMslPtr(t: CirType, ident: string): string =
  ## MSL declaration text of a pointer or array pointer with its address space:
  ## `device float* p`, pointer-to-array `device float (*p)[N]`.
  ptrField(mslScalarRow, mslVectorRule, printerTarget,
    getMslPtrQualifier(t.space), t, ident)

proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  ## MSL declaration text of one struct member: a pointer member names
  ## the address space of its pointee (`device float* p`), the rest takes
  ## the shared C declarator form.
  if f.typ.kind in {ctPtr, ctAptr}:
    return emitMslPtr(f.typ, f.name)
  return emitMslType(f.typ, f.name, false)

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════


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

proc emitBuiltinCall(callee: CirExpr, args: seq[string]): string =
  ## Returns the MSL text of a builtin call, "" when `callee` is not a builtin.
  ##
  ## Shared by expression- and statement-position calls: the statement path
  ## bypasses `printExpr`, so a barrier printed as a bare name would lose
  ## its memory flags, the Metal compiler refuses the text.
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
  if callee.builtin and mslConvBuiltinNames.hasKey(callee.name):
    return mslConvBuiltinNames.getOrDefault(callee.name) % args.join(", ")
  return ""


proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## One numeric conversion, the C-style cast form, pointer casts name
  ## their address space.
  let t = e.castType
  var castTyp = emitMslType(t, "", true)
  if t.kind in {ctPtr, ctAptr}:
    castTyp = getMslPtrQualifier(t.space) & ' ' & castTyp
  return "(" & castTyp & ")" & ctx.printExpr(e.castOperand)


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
    return ctx.printCast(e, mslCastRules, emitMslType(e.castType, "", true))
  of eConv:
    return ctx.printConv(e)
  of eSizeof:
    return "sizeof(" & emitMslType(e.ofType, "", true) & ")"
  of eOconstr:
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
    return printCallText(ctx, e, args, emitBuiltinCall(e.callee, args))
  else:
    ctx.fail("expression kind " & $e.kind & " has no MSL emission")


# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Statements: per-target policy arms, the shared dispatch lives in print_common
# ═══════════════════════════════════════════════════════════════════════
proc printVar(ctx: var PrintCtx, s: CirStmt): string =
  ## One local declaration: kernel-scope threadgroup storage and storage-class refusals, then the declaration text.
  let pad = spaces(ctx.indent * 2)
  if s.space == caThreadgroup:
    # kernel-scope threadgroup storage: the MSL lowering pass keeps only
    # a kernel's initializer-less smem locals here, a device function's
    # smem local and any initializer refuse in the pass
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

proc printAsm(ctx: var PrintCtx, s: CirStmt): string =
  ## One inline assembly statement: the spliced text, MSL has no asm form.
  let pad = spaces(ctx.indent * 2)
  var text = ""
  for i, part in s.asmParts:
    text.add(part)
    if i < s.asmOperands.len:
      text.add(s.asmOperands[i].name)
  result = pad & text

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

proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string] =
  ## MSL parameter texts of one proc or kernel, in declaration order.
  ##
  ## Kernel parameters:
  ## - a relocated smem local binds as threadgroup storage, no buffer slot
  ## - a device pointer takes the next `[[buffer(n)]]` slot
  ## - a value parameter binds as `constant T&`, bool retypes to `int`
  ##   (Metal has no bool constant storage)
  ## - the coordinate builtins that the body references append as attributed parameters, no slot
  ##
  ## Device-function parameters:
  ## - a threadgroup `ptr` parameter is a relocated smem arena, caller-owned
  ##   storage (Metal rejects threadgroup allocation inside device functions)
  ## - other pointers keep their address space, values print plainly
  var slot = 0
  if fn.isKernel:
    for p in fn.params:
      if p.typ.kind == ctPtr and p.typ.space == caThreadgroup:
        result.add emitMslSmemParam(p)
        continue
      case p.typ.kind
      of ctAptr, ctPtr:
        let inner = p.typ.pElem
        result.add "device " & emitMslType(inner, "", true) & "* " & p.name &
          " [[buffer(" & $slot & ")]]"
      else:
        var elem = emitMslScalar(p.typ)
        if p.typ.kind == ctBool:
          elem = "int"
        result.add "constant " & elem & "& " & p.name
      inc slot
    for b in ctx.builtins:
      # coordinate builtins are attributed params, no binding, no slot
      result.add emitBuiltinParam(b)
  else:
    for p in fn.params:
      case p.typ.kind
      of ctAptr, ctPtr:
        if p.typ.space == caThreadgroup and p.typ.kind == ctPtr:
          result.add emitMslSmemParam(p)
        else:
          result.add emitMslPtr(p.typ, p.name)
      else:
        result.add emitMslType(p.typ, p.name, true)

proc emitComputePrefix(ctx: var PrintCtx, fn: CirProc): string =
  ## MSL linkage prefix: a kernel is `kernel`, an `{.inline.}` device function
  ## is `inline`, every other device function prints none.
  if fn.isKernel:
    return "kernel "
  if fn.inline:
    return "inline "
  return ""

proc emitTypeBare(t: CirType): string =
  ## MSL name of `t` with no identifier, as casts and sizeof print it.
  emitMslType(t, "", true)

proc prepProc(ctx: var PrintCtx, fn: CirProc) =
  ## Per-proc state of the MSL walk: the builtin set resets before the walk,
  ## the body walk fills it, and the signature reads it after the body prints.
  ctx.builtins = initOrderedSet[string]()

const mslModuleShape = ModuleShape(
  spaces: {caConstant},
  structTypedef: false,
  structPadding: "char _",
  structFieldTerm: ';',
  structClose: "};",
  fwdDecls: false,
  moduleOrderProcs: true,
  kernelSep: "\n\n",
  customProcHead: false,
  printsRefBlocks: false,
  printsParamBindings: false,
  printsReduceTemplates: false)

proc printPrelude(ctx: var PrintCtx, m: CirModule): string =
  ## Translation-unit head of every MSL module: the metal headers.
  result = "#include <metal_stdlib>\n"
  result &= "#include <metal_atomic>\n"
  result &= "using namespace metal;\n\n"

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope `constant` declaration of one relocated const_mem local.
  ## MSL forbids declaring constant storage without an initializer.
  if s.init == nil:
    ctx.fail("const_mem local " & s.name & " without an initializer")
    return ""
  var decl = emitMslType(s.declType, s.name, false)
  if s.declType.kind in {ctPtr, ctAptr}:
    decl = emitMslPtr(s.declType, s.name)
  return "constant " & decl & " = " & ctx.printExpr(s.init) & ";"

proc printMslModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full MSL translation unit through the shared module skeleton, returning "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>", policy: mslKindRow,
    lit: mslLitRow, row: mslScalarRow)
  printModule(ctx, m, moduleKernels(m), mslModuleShape)

proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as Metal Shading Language (MSL) source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printMslModule(ctx.module))
