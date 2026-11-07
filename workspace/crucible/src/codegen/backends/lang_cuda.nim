# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import std/[macros, strutils, sequtils, sets, tables, math]
import ./print_common
import ./targets_registry
import ../../builtins/builtins_gpu_types
import ../../crucible_codegen
import ../../builtins/builtins_decl_helpers

# ############################################################
#
#            CUDA C++ printer
#
# ############################################################
{.experimental: "codeReordering".}

const printerTarget = "cuda"

type PrintCtx = object of WalkCtx
  ## Per-target walk context. The shared walk procs instantiate against
  ## the derived type and the overlay procs below resolve per target.



# ═══════════════════════════════════════════════════════════════════════
#  Target tables
# ═══════════════════════════════════════════════════════════════════════

const cudaVocab = BuiltinVocab(
  synchroText: makeTexts([
    # __syncthreads() already orders global memory, the device-scope barrier
    # emits no separate instruction
    (cbkThreadgroupBarrier, "__syncthreads()"),
    (cbkThreadgroupBarrierDevice, "__syncthreads()")]),
  reductionText: makeTexts([
    (cbkReduceAdd, "warpReduceAdd($1)"),
    (cbkReduceMax, "warpReduceMax($1)")]),
  atomicsText: makeTexts([
    # atomicAdd overloads on the pointee type, no cast needed
    (cbkAtomicFetchAdd, "atomicAdd($1, $2)")]),
  shuffleText: makeTexts([
    # full mask: every warp lane participates and receives the result,
    # xor exchanges lane pairs bitwise
    (cbkSimdShuffle, "__shfl_sync(0xffffffffu, $1, $2)"),
    (cbkSimdShuffleXor, "__shfl_xor_sync(0xffffffffu, $1, $2)")]))

const cudaEmit = TargetEmit(
  vocab: cudaVocab,
  functionalCtor: false,
  ctorOpen: "{",
  ctorClose: "}",
  ctorCast: false,
  ctorZero: "",
  aconstrCasts: true,
  dotDerefParens: true,
  vectorDots: false,
  noPointerValues: (spaces: {}, text: ""),
  noPointerRebind: "",
  prefixGlobalFn: "extern \"C\" __global__ ",
  prefixInline: "__forceinline__ __device__ ",
  prefixDevice: "__device__ ")

const cudaModuleShape = ModuleShape(
  spaces: {caThread, caThreadgroup, caConstant, caDevice},
  structTypedef: false,
  structPadding: "char _",
  structFieldTerm: ';',
  structClose: "};",
  fwdDecls: false,
  moduleOrderProcs: true,
  globalFnSep: "\n\n",
  customProcHead: false,
  printsRefBlocks: false,
  printsParamBindings: false,
  printsReduceTemplates: true)

# ═══════════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════════

# Shared-walk mixin targets, declared so the overlay procs are module-scope
# symbols when the walk's generics resolve (mixin ignores codeReordering).
proc emitTypeName(ctx: var PrintCtx, t: CirType): string
proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string
proc coordRef(ctx: var PrintCtx, e: CirExpr): string
proc coordDot(ctx: var PrintCtx, e: CirExpr): string
proc printConv(ctx: var PrintCtx, e: CirExpr): string
proc getStmtBuiltinText(ctx: var PrintCtx, s: CirStmt, pad: string, args: seq[string]): string
proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool
proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string
proc printAsm(ctx: var PrintCtx, s: CirStmt): string
proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string]
proc printPrelude(ctx: var PrintCtx, m: CirModule): string
proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string
proc printReduceTemplates(ctx: var PrintCtx, m: CirModule): string

proc emitCudaScalar(t: CirType): string =
  getFullTypeName(TargetVocabs[cgCuda].scalarNames, cudaVectorRule, t)

proc emitCudaType(t: CirType, ident: string, allowEmptyIdent: bool): string =
  printCDecl(TargetVocabs[cgCuda].scalarNames, cudaVectorRule, printerTarget, t, ident, allowEmptyIdent)

proc emitTypeName(ctx: var PrintCtx, t: CirType): string =
  emitCudaType(t, "", true)

proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  emitCudaType(f.typ, f.name, false)

# ═══════════════════════════════════════════════════════════════════════
#  Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

proc getCudaCoordToken(k: CirCoordBuiltinKind, whole: bool, field: string, coordTy: CirType): string =
  ## CUDA text of a canonical coordinate reference.
  ## - whole-value use of a vector builtin: a braced init of the coordinate
  ##   struct the module registered. The CUDA builtins are unsigned and
  ##   the canonical Nim type is int, the components cast
  ## - component use: a cast to int, the flat thread index linearizes
  ##   x-major and parenthesizes so a trailing `* k` cannot mis-associate
  let vecTy = emitCudaScalar(coordTy)
  case k
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
    if whole:
      result = $k
    else:
      result = "(int)(blockIdx." & field & "*blockDim." & field &
        "+threadIdx." & field & ")"
  else: doAssert false


proc coordRef(ctx: var PrintCtx, e: CirExpr): string =
  getCudaCoordToken(e.coordBuiltin, true, "", e.typ)

proc coordDot(ctx: var PrintCtx, e: CirExpr): string =
  getCudaCoordToken(e.dotBase.coordBuiltin, false, e.fieldName, e.dotBase.typ)

# ═══════════════════════════════════════════════════════════════════════
#  Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  "(" & emitCudaType(e.castType, "", true) & ")" & ctx.printExpr(e.castOperand)

proc getStmtBuiltinText(ctx: var PrintCtx, s: CirStmt, pad: string, args: seq[string]): string =
  let txt = ctx.emitBuiltinCall(s.callCallee, args)
  if txt.len == 0:
    return ""
  pad & txt & ";"

# ═══════════════════════════════════════════════════════════════════════
#  Statement declarations
# ═══════════════════════════════════════════════════════════════════════

proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool =
  ## CUDA C++ forbids initializing shared variables.
  if s.addressSpace == caThreadgroup:
    if s.init != nil and not ctx.inGlobalFn:
      raiseAssert("smem local " & s.name & " with an initializer, CUDA C++ forbids initializing __shared__ variables")
    return true
  if s.addressSpace == caConstant:
    raiseAssert("local " & s.name & " in constant addressSpace, CUDA constant storage is program scope only")
  true

proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string =
  if s.addressSpace == caThreadgroup:
    return "__shared__ " & emitCudaType(s.declType, s.name, false)
  emitCudaType(s.declType, s.name, false)

proc printAsm(ctx: var PrintCtx, s: CirStmt): string =
  let pad = spaces(ctx.indent * 2)
  pad & "asm(" & s.asmSpliceText().strip() & ");"

# ═══════════════════════════════════════════════════════════════════════
#  Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc emitWarpReduceTemplate(name: string): string =
  ## Butterfly warp-reduction template for one reduction: CUDA has no
  ## float warp-reduce intrinsic, the tree walks the xor mask 16/8/4/2/1.
  let add = name == "reduceAdd"
  let op = if add: "v += " else: "v = max(v, "
  let close = if add: "" else: ")"
  result = "template <typename T> __forceinline__ __device__ T warpReduce" &
    (if add: "Add" else: "Max") & "(T v) {\n"
  result &= "  #pragma unroll\n"
  result &= "  for (int delta = 16; delta > 0; delta >>= 1) {\n"
  result &= "    " & op & "__shfl_xor_sync(0xffffffff, v, delta, 32)" & close & ";\n"
  result &= "  }\n"
  result &= "  return v;\n"
  result &= "}\n\n"

proc emitComputeParams(ctx: var PrintCtx, fn: CirProc): seq[string] =
  ## CUDA parameter texts of one proc or entrypoint, in declaration order.
  ##
  ## An entrypoint's by-value array parameter is const-qualified, the printer lowers it to a read-only pointer.
  for p in fn.params:
    if p.passByRef and not fn.isGlobalFn:
      # const Type& name — C++ reference, no body changes needed
      result.add "const " & emitTypeName(ctx, p.typ) & "& " & p.name
      continue
    case p.typ.kind
    of ctAptr, ctPtr:
      result.add emitCudaType(p.typ, p.name, true)
    of ctArray:
      if fn.isGlobalFn:
        result.add "const " & emitCudaType(p.typ, p.name, false)
      else:
        result.add emitCudaType(p.typ, p.name, true)
    else:
      result.add emitCudaType(p.typ, p.name, true)

proc printPrelude(ctx: var PrintCtx, m: CirModule): string =
  ## NVRTC's default include path provides the fp16 and bf16 headers.
  "#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n"

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  var decl = emitCudaType(s.declType, s.name, false)
  if s.init != nil:
    decl &= " = " & ctx.printExpr(s.init)
  return "__constant__ " & decl & ";"

proc printReduceTemplates(ctx: var PrintCtx, m: CirModule): string =
  let reduceKinds = m.features.reductions
  if "reduceAdd" in reduceKinds:
    result &= emitWarpReduceTemplate("reduceAdd")
  if "reduceMax" in reduceKinds:
    result &= emitWarpReduceTemplate("reduceMax")

proc printCudaModule*(m: CirModule): string =
  var ctx = initPrintCtx(cudaWalkPolicy, cudaVectorRule, cudaEmit, cgCuda)
  ctx.mathNames = cudaMathBuiltinNames
  printModule(ctx, m, getModuleGlobalFns(m), cudaModuleShape)

proc codegen*(m: CirModule): string =
  codegenModule(printCudaModule(m))
