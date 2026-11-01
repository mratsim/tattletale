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
    ## False inside a device function. Its smem local owns no storage and refuses an initializer there.

const printerTarget = "cuda"

proc printTargetExpr(ctx: var PrintCtx, e: CirExpr): string

# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

proc emitCudaScalar(t: CirType): string =
  ## CUDA bare name of a non-array leaf type, "" when the caller composes the name.
  bareName(cudaScalarRow, cudaVectorRule, t)

proc emitCudaType(t: CirType, ident: string, allowEmptyIdent: bool): string =
  ## CUDA declaration text of `t` with `ident` appended, the shared C declarator
  ## over the CUDA rows. Arrays take the identifier into the declarator
  ## (`T ident[N]`, pointer-to-array `T (*ident)[N]`), an array without an identifier refuses.
  cDecl(cudaScalarRow, cudaVectorRule, printerTarget, t, ident, allowEmptyIdent)

proc emitComputeStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  ## CUDA declaration text of one struct member via the shared C declarator.
  emitCudaType(f.typ, f.name, false)

# ═══════════════════════════════════════════════════════════════════════
# Expressions
# ═══════════════════════════════════════════════════════════════════════


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

proc emitBuiltinCall(callee: CirExpr, args: seq[string]): string =
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

proc cudaMathCallText(call: CirExpr, args: seq[string]): string =
  ## CUDA text of a `{.builtin.}` scalar-math call whose CUDA name differs
  ## from the canonical name (`cudaMathBuiltinNames`), "" otherwise.
  ##
  ## - only a single float operand selects the mapped name, an integer
  ##   operand or a width outside the table keeps the canonical name
  ## - a user proc that shares a builtin's name stays an ordinary call
  if call.callee.kind != eSym or not call.callee.builtin or call.args.len != 1:
    return ""
  let at = getExprType(call.args[0])
  if at == nil or at.kind != ctFloat:
    return ""
  let name = cudaMathBuiltinNames.getOrDefault((call.callee.name, at.bits), "")
  if name.len == 0:
    return ""
  return name & "(" & args.join(", ") & ")"

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## One numeric conversion, the C-style cast form.
  let t = e.castType
  return "(" & emitCudaType(t, "", true) & ")" & ctx.printExpr(e.castOperand)


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
    return ctx.printCast(e, cudaCastRules, emitCudaType(e.castType, "", true))
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
    var builtinTxt = emitBuiltinCall(e.callee, args)
    if builtinTxt.len == 0:
      builtinTxt = cudaMathCallText(e, args)
    return printCallText(ctx, e, args, builtinTxt)
  else:
    # dispatcher sends only the kinds above, anything else is a routing bug
    ctx.fail("expression kind " & $e.kind & " has no CUDA emission")


# ═══════════════════════════════════════════════════════════════════════
# Statements
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Statements: per-target policy arms, the shared dispatch lives in print_common
# ═══════════════════════════════════════════════════════════════════════
proc printVar(ctx: var PrintCtx, s: CirStmt): string =
  ## One local declaration: kernel-scope shared storage and storage-class refusals, then the declaration text.
  let pad = spaces(ctx.indent * 2)
  if s.space == caThreadgroup:
    # kernel-scope shared storage in `__shared__` form,
    # NVRTC allocates it per threadgroup
    # a device function's smem local owns no storage, an initializer on it refuses
    # kernel-body initializers rewrote to stores before printing
    if s.init != nil and not ctx.inKernel:
      ctx.fail("smem local " & s.name &
        " with an initializer, CUDA C++ forbids initializing __shared__ variables")
      return ""
    result = pad & "__shared__" & ' ' &
      emitCudaType(s.declType, s.name, false) & ";"
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

proc printAsm(ctx: var PrintCtx, s: CirStmt): string =
  ## One inline assembly statement: the spliced text in an `asm(...)` wrapper.
  let pad = spaces(ctx.indent * 2)
  var text = ""
  for i, part in s.asmParts:
    text.add(part)
    if i < s.asmOperands.len:
      text.add(s.asmOperands[i].name)
  result = pad & "asm(" & text.strip() & ");"

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

proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string] =
  ## CUDA parameter texts of one proc or kernel, in declaration order.
  ##
  ## - pointers print `T* name` (the engine binds device buffers and values)
  ## - a kernel's by-value array parameter is const-qualified: the array
  ##   decays to a pointer and is read-only
  ## - every other parameter prints the plain declarator form
  for p in fn.params:
    case p.typ.kind
    of ctAptr, ctPtr:
      result.add emitCudaScalar(p.typ.pElem) & "* " & p.name
    of ctArray:
      if fn.isKernel:
        result.add "const " & emitCudaType(p.typ, p.name, false)
      else:
        result.add emitCudaType(p.typ, p.name, true)
    else:
      result.add emitCudaType(p.typ, p.name, true)

proc emitComputePrefix(ctx: var PrintCtx, fn: CirProc): string =
  ## CUDA linkage prefix of one proc or kernel.
  ##
  ## - a kernel is `extern "C" __global__`: extern "C" keeps the unmangled
  ##   name that the host runtime looks up
  ## - a device function is `__device__`, plus the force-inline qualifier
  ##   under `{.inline.}`
  if fn.isKernel:
    return "extern \"C\" __global__ "
  if fn.inline:
    return "__forceinline__ __device__ "
  return "__device__ "

proc emitTypeBare(t: CirType): string =
  ## CUDA C++ name of `t` with no identifier, as casts and sizeof print it.
  emitCudaType(t, "", true)

proc printCudaModule*(m: CirModule): tuple[src: string, err: string] =
  ## Emits the full CUDA translation unit: struct declarations, program-scope constants,
  ## the warp-reduction templates in use, then every function body.
  ##
  ## Returns "" as the error when clean, else the first refusal.
  var ctx = PrintCtx(fnName: "<module>", policy: cudaKindRow,
    lit: cudaLitRow, row: cudaScalarRow)
  var src = printComputeStructs(ctx, m, false, "char _")
  if ctx.failing():
    return ("", ctx.err)
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
    ctx.inKernel = fn.isKernel
    let txt = printComputeProc(ctx, fn)
    if ctx.err.len > 0:
      break
    src &= txt & "\n\n"
  result = (src, ctx.err)

proc codegen*(ctx: var GpuContext): string =
  ## Prints the module in `ctx` as CUDA C++ source, the `backends_api` entry point.
  ## A printer refusal is a compile error naming the construct.
  codegenModule(printCudaModule(ctx.module))
