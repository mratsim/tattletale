# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CUDA C++ printer: a structured CIRv2 module to NVRTC-compilable CUDA C++ source,
## table-driven over the closed set of type kinds, structured like the MSL printer.
##
## Signature and storage contract:
##
## - kernels: `extern "C" __global__ void name(args)`, the host runtime
##   looks up the kernel by unmangled name
## - device functions: `__device__` with no extern "C", each device function keeps its C++ mangled name.
## - parameters print plainly, CUDA has one generic address space,
##   `{.smem.}` locals print `__shared__` in the body, and program-scope variables in `CirModule.globals` print `__constant__`
##
## Statement and builtin contract:
##
## - coordinate builtins map to the CUDA vocabulary: blockIdx, blockDim,
##   gridDim, threadIdx. The canonical Nim type is int, the CUDA builtins are unsigned, reads cast to int
## - `{.unroll.}` prints `#pragma unroll` in front of the loop, a loop prints `while (cond)` or `for (;;)`
## - the typed builtin kinds dispatch to the CUDA vocabulary, `{.builtin.}` names pass through, the warp reductions are the exception
## - a warp-reduction call makes its butterfly template emit at module scope on first use

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, sequtils, tables, math]
import ../ir/ast_datatypes
import ./emit_common

type
  PrintCtx = object
    fnName: string
    ## First refusal of the walk, "" while clean.
    err: string
    indent: int
    ## Literal token table of the target, the shared literal head reads it.

const printerTarget = "cuda"

## Optional expression kinds this printer routes through
## its own `printTargetExpr`, the shared dispatcher emits the rest.
const targetExprKinds: set[CirExprKind] = {}

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

const cudaScalarTokens = ScalarTokens(
  void: "void", bool: "bool",
  i8: "signed char", i16: "short", i32: "int", i64: "long long",
  u8: "unsigned char", u16: "unsigned short", u32: "unsigned int",
  u64: "unsigned long long",
  f32: "float", f64: "double")

proc emitCudaScalar(t: CirType): string =
  ## CUDA name of a non-array leaf type, "" when the caller composes the name.
  case t.kind
  of ctVoid, ctBool, ctInt, ctUint, ctFloat:
    result = emitScalarToken(cudaScalarTokens, t)
  of ctPtr, ctAptr:
    result = emitCudaScalar(t.pElem) & "*"
  of ctStruct:
    result = t.name
  of ctVector, ctArray, ctProc, ctSpan:
    result = ""

proc emitCudaType(t: CirType, ident: string, allowEmptyIdent: bool): string =
  ## CUDA name of `t` with `ident` appended. Arrays take the identifier into
  ## the declarator (`T ident[N]`, pointer-to-array `T (*ident)[N]`).
  ## `allowEmptyIdent` permits no identifier, return types and casts use it.
  ## An array without an identifier otherwise refuses.
  case t.kind
  of ctArray:
    if ident.len == 0 and not allowEmptyIdent:
      return "<cuda: array needs an identifier>"
    if t.aElem.kind == ctArray:
      result = emitCudaType(t.aElem, "(" & ident & ")" & "[" & $t.aN & "]",
        allowEmptyIdent)
    else:
      result = emitCudaType(t.aElem, "", true) & ' ' & ident & '[' & $t.aN & ']'
  else:
    result = emitCudaScalar(t)
    if ident.len > 0:
      result.add ' ' & ident

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════

# `INFINITY`, `-INFINITY` and `NAN` convert at the use site, one form covers every width.
const targetLitTokens = LitTokens(
  scalar: cudaScalarTokens,
  int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
  f32Suffix: "f",
  nilToken: "nullptr",
  nonFinite: FloatLitTokens(
    inf16: "INFINITY", negInf16: "-INFINITY", nan16: "NAN",
    inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
    inf64: "INFINITY", negInf64: "-INFINITY", nan64: "NAN"))

proc cudaCoordToken(kind: CoordBuiltinKind, name: string, whole: bool, field: string, coordTy: CirType): string =
  ## CUDA text of a canonical coordinate reference.
  ## - whole-value use of a vector builtin: a braced init of the coordinate
  ##   struct the module registered. The CUDA builtins are unsigned and
  ##   the canonical Nim type is int, the components cast
  ## - component use: a cast to int, the flat thread index linearizes
  ##   x-major and parenthesizes so a trailing `* k` cannot mis-associate
  let vecTy = emitCudaScalar(coordTy)
  case kind
  of cbkThreadgroupPositionInGrid:
    if whole:
      result = vecTy & "{(int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z}"
    else:
      result = "(int)(blockIdx." & field & ")"
  of cbkThreadPositionInThreadgroup:
    if whole:
      result = vecTy & "{(int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z}"
    else:
      result = "(int)(threadIdx." & field & ")"
  of cbkThreadsPerThreadgroup:
    if whole:
      result = vecTy & "{(int)blockDim.x, (int)blockDim.y, (int)blockDim.z}"
    else:
      result = "(int)(blockDim." & field & ")"
  of cbkThreadgroupsPerGrid:
    if whole:
      result = vecTy & "{(int)gridDim.x, (int)gridDim.y, (int)gridDim.z}"
    else:
      result = "(int)(gridDim." & field & ")"
  of cbkThreadIndexInThreadgroup:
    if whole:
      result = "(int)(threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x)"
    else:
      result = "(int)(threadIdx." & field & ")"
  of cbkThreadPositionInGrid:
    # whole-value use has no CUDA text, the canonical name passes through
    if whole:
      result = name
    else:
      result = "(int)(blockIdx." & field & "*blockDim." & field &
        "+threadIdx." & field & ")"
  of cbkNone:
    result = name

proc emitCudaBuiltinCall(callee: CirExpr, args: seq[string]): string =
  ## Returns the CUDA text of a builtin call, "" when `callee` is not a builtin.
  ##
  ## Shared by expression- and statement-position calls: the statement path
  ## bypasses `printExpr`, so a barrier printed as a bare name would lose
  ## its memory ordering, the CUDA text is `__syncthreads()`.
  ##
  ## `args` contains the already-printed argument texts.
  if callee.kind != eSym:
    return ""
  case callee.synchroBuiltin
  of sbkThreadgroupBarrier, sbkThreadgroupBarrierDevice:
    # __syncthreads() already orders global memory, the device-scope barrier needs no separate instruction
    return "__syncthreads()"
  of sbkNone: discard
  case callee.reductionBuiltin
  of rbkReduceAdd:
    return "warpReduceAdd(" & args.join(", ") & ")"
  of rbkReduceMax:
    return "warpReduceMax(" & args.join(", ") & ")"
  of rbkNone: discard
  if callee.atomicsBuiltin == abkAtomicFetchAdd:
    # atomicAdd overloads on the pointee type, no cast needed
    return "atomicAdd(" & args[0] & ", " & args[1] & ")"
  case callee.shuffleBuiltin
  of shkSimdShuffle, shkSimdShuffleXor:
    # full mask: every warp lane participates and receives the result,
    # xor exchanges lane pairs bitwise
    let shfl = if callee.shuffleBuiltin == shkSimdShuffle: "__shfl_sync"
               else: "__shfl_xor_sync"
    return shfl & "(0xffffffffu, " & args[0] & ", " & args[1] & ")"
  of shkNone: discard
  return ""

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## One numeric conversion, the C-style cast form.
  let t = e.castType
  return "(" & emitCudaType(t, "", true) & ")" & ctx.printExpr(e.castOperand)

proc printCast(ctx: var PrintCtx, e: CirExpr): string =
  ## One bit reinterpret, the CUDA float-as-integer intrinsics.
  ##
  ## - `__float_as_uint(1.5f)` reads 0x3FC00000, a C-style cast numerically
  ##   converts and gives 1u
  ## - float-as-integer intrinsics cover the 32- and 64-bit float pairs
  ## - a same-width integer pair keeps the C-style cast, the modular
  ##   conversion preserves every bit
  ## - a same-width float pair without an intrinsic refuses, a numeric
  ##   conversion would lose the bits
  let t = e.castType
  let src = getExprType(e.castOperand)
  let scalar = t.kind in {ctInt, ctUint, ctFloat} and src != nil and
    src.kind in {ctInt, ctUint, ctFloat} and src.bits == t.bits
  if scalar and src.kind == ctFloat and t.kind != ctFloat:
    if src.bits == 32:
      return (if t.kind == ctUint: "__float_as_uint" else: "__float_as_int") &
        "(" & ctx.printExpr(e.castOperand) & ")"
    if src.bits == 64:
      return "__double_as_longlong(" & ctx.printExpr(e.castOperand) & ")"
    ctx.fail("bit cast of a 16-bit float has no CUDA emission")
    return ""
  if scalar and t.kind == ctFloat and src.kind != ctFloat:
    if t.bits == 32:
      return (if src.kind == ctUint: "__uint_as_float" else: "__int_as_float") &
        "(" & ctx.printExpr(e.castOperand) & ")"
    if t.bits == 64:
      return "__longlong_as_double(" & ctx.printExpr(e.castOperand) & ")"
    ctx.fail("bit cast to a 16-bit float has no CUDA emission")
    return ""
  # pointers, same-width integer pairs and float-to-float keep the C-style cast
  return ctx.printConv(e)

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## CUDA text of the kinds in `targetExprKinds`.
  case e.kind
  of eSym:
    # a coordinate builtin takes the CUDA vocabulary, a name passes through
    if e.coordBuiltin != cbkNone:
      return cudaCoordToken(e.coordBuiltin, e.name, true, "", e.typ)
    return e.name
  of eDot:
    # a coordinate component takes its CUDA token, a dereferenced base parenthesizes
    if e.dotBase.kind == eSym and e.dotBase.coordBuiltin != cbkNone:
      return cudaCoordToken(e.dotBase.coordBuiltin, e.dotBase.name, false,
        e.fieldName, e.dotBase.typ)
    let base = ctx.printExpr(e.dotBase)
    if e.dotBase.kind == eDeref:
      return "(" & base & ")." & e.fieldName
    return base & '.' & e.fieldName
  of eCast:
    return ctx.printCast(e)
  of eConv:
    return ctx.printConv(e)
  of eSizeof:
    return "sizeof(" & emitCudaType(e.ofType, "", true) & ")"
  of eOconstr:
    # braced init list `TypeName{val, ...}`, the functional-style cast form
    # keeps the result a valid C++ expression, bare braced-init-lists are
    # not expressions and cannot be used with member access
    result = emitCudaType(e.ocType, "", true) & "{"
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
      result.add "(" & emitCudaType(e.acType.aElem, "", true) & ")" &
        ctx.printExpr(el)
    result.add "}"
  of eCall:
    let args = printCallArgs(ctx, e)
    return printCallText(ctx, e, args, emitCudaBuiltinCall(e.callee, args))
  else:
    # dispatcher sends only the kinds above, anything else is a routing bug
    ctx.fail("expression kind " & $e.kind & " has no CUDA emission")


# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

const cudaLoopStyle = LoopStyle(
  unrollDirective: "#pragma unroll",
  infiniteHead: "for (;;) {\n",
  braceSameLine: true)

# ═══════════════════════════════════════════════════════════════════════
# Statements: per-target policy arms, the shared dispatch lives in emit_common
# ═══════════════════════════════════════════════════════════════════════
proc printVar(ctx: var PrintCtx, s: CirStmt): string =
  ## One local declaration: kernel-scope shared storage and storage-class refusals, then the declaration text.
  let pad = spaces(ctx.indent * 2)
  if s.space == caThreadgroup:
    # kernel-scope shared storage in `__shared__` form,
    # NVRTC allocates it per threadgroup, any initializer is discarded
    result = pad & "__shared__ " & emitCudaType(s.declType, s.name, false) & ";"
    return
  if s.space == caConstant:
    # program-scope storage belongs to CirModule.globals, a survivor here
    # was never relocated
    ctx.fail("local " & s.name & " in constant space, CUDA constant storage" &
      " is program scope only")
    return ""
  var decl = emitCudaType(s.declType, s.name, false)
  result = pad & decl
  if s.init != nil:
    result &= " = " & ctx.printExpr(s.init)
  result &= ";"

proc printConst(ctx: var PrintCtx, s: CirStmt): string =
  ## One local constant declaration, the `constexpr` form.
  let pad = spaces(ctx.indent * 2)
  if s.declType.kind == ctArray:
    result = pad & "constexpr " &
      emitCudaType(s.declType, s.name, false) & " = " & ctx.printExpr(s.init) & ";"
  else:
    result = pad & "constexpr " & emitCudaScalar(s.declType) & ' ' & s.name &
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
  let builtinTxt = emitCudaBuiltinCall(s.callCallee, args)
  if builtinTxt.len > 0:
    result = pad & builtinTxt & ";"
  else:
    result = pad & calleeName & "(" & args.join(", ") & ");"

proc printAsm(ctx: var PrintCtx, s: CirStmt): string =
  ## One inline assembly statement: the spliced text in an `asm(...)` wrapper.
  let pad = spaces(ctx.indent * 2)
  var text = ""
  for i, part in s.asmParts:
    text.add(part)
    if i < s.asmOperands.len:
      text.add(s.asmOperands[i].name)
  result = pad & "asm(" & text.strip() & ");"

proc printStmts(ctx: var PrintCtx, stmts: seq[CirStmt]): string =
  ## Prints one statement list, the shared walk with the CUDA loop style.
  printStmtList(ctx, stmts, cudaLoopStyle)

proc emitWarpReduceTemplate(kind: ReductionBuiltinKind): string =
  ## Butterfly warp-reduction template for one reduction kind: CUDA has no
  ## float warp-reduce intrinsic, the tree walks the xor mask 16/8/4/2/1.
  let op = case kind
    of rbkReduceAdd: "v += "
    else: "v = max(v, "
  let close = case kind
    of rbkReduceAdd: ""
    else: ")"
  result = "template <typename T> __forceinline__ __device__ T warpReduce" &
    (if kind == rbkReduceAdd: "Add" else: "Max") & "(T v) {\n"
  result &= "  #pragma unroll\n"
  result &= "  for (int delta = 16; delta > 0; delta >>= 1) {\n"
  result &= "    " & op & "__shfl_xor_sync(0xffffffff, v, delta, 32)" & close & ";\n"
  result &= "  }\n"
  result &= "  return v;\n"
  result &= "}\n\n"

# ═══════════════════════════════════════════════════════════════════════
# Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc printProc(ctx: var PrintCtx, fn: CirProc): string =
  ## One proc or kernel: signature, body, closing brace.
  ctx.fnName = fn.name
  ctx.indent = 0
  let bodyTxt = ctx.printStmts(fn.body)
  if ctx.failing():
    return ""
  # exportc renames a kernel entry for the host runtime lookup.
  # A device function keeps the name its call sites reference.
  let name = if fn.isKernel and fn.exportName.len > 0: fn.exportName else: fn.name
  var params: seq[string]
  if fn.isKernel:
    for p in fn.params:
      case p.typ.kind
      of ctAptr, ctPtr:
        params.add emitCudaScalar(p.typ.pElem) & "* " & p.name
      of ctArray:
        # by-value array params decay to pointers, they are read-only,
        # const-qualified so the read-only contract reaches the callee
        params.add "const " & emitCudaType(p.typ, p.name, false)
      else:
        params.add emitCudaType(p.typ, p.name, true)
  else:
    for p in fn.params:
      case p.typ.kind
      of ctAptr, ctPtr:
        params.add emitCudaScalar(p.typ.pElem) & "* " & p.name
      else:
        params.add emitCudaType(p.typ, p.name, true)
  let ret = emitCudaType(fn.ret, "", true)
  var prefix = ""
  if fn.isKernel:
    # extern "C" keeps the unmangled name that the host runtime looks up
    prefix = "extern \"C\" __global__ "
  else:
    if fn.inline:
      prefix = "__forceinline__ "
    prefix &= "__device__ "
  let sig = prefix & ret & ' ' & name & '(' & params.join(", ") & ")"
  return sig & " {\n" & bodyTxt & '\n' & "}"

proc printCudaModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full CUDA translation unit: struct declarations, program-scope constants,
  ## the warp-reduction templates in use, then every function body.
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>")
  var src = ""
  for t in m.types:
    if t.kind != ctStruct:
      ctx.fail("module type " & t.name & " is not a struct")
      break
    src &= "struct " & t.name & " {\n"
    if t.fields.len == 0:
      # CUDA requires at least one field in a struct
      src &= "  char _;\n"
    for f in t.fields:
      src &= "  " & emitCudaType(f.typ, f.name, false) & ";\n"
    src &= "};\n\n"
  for s in m.globals:
    if s.kind != sVar:
      ctx.fail("module global " & s.name & " is not a variable")
      break
    var decl = emitCudaType(s.declType, s.name, false)
    if s.init != nil:
      decl &= " = " & ctx.printExpr(s.init)
    src &= "__constant__ " & decl & ";\n"
  if m.globals.len > 0:
    src &= "\n"
  # warp-reduction template set from the stored module feature scan
  # (`CirModule.features`, filled by the feature-scan pass)
  let reduceKinds = m.features.reductions
  if rbkReduceAdd in reduceKinds:
    src &= emitWarpReduceTemplate(rbkReduceAdd)
  if rbkReduceMax in reduceKinds:
    src &= emitWarpReduceTemplate(rbkReduceMax)
  if ctx.failing():
    return ("", ctx.err)
  for fn in m.procs:
    let txt = ctx.printProc(fn)
    if ctx.err.len > 0:
      break
    src &= txt & "\n\n"
  result = (src, ctx.err)

proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as CUDA C++ source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printCudaModule(ctx.module))
