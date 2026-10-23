# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            HLSL validation passes
#
# ############################################################
#
# Rejections over what only the HLSL pipeline can check:
# - constructs the glslang HLSL frontend cannot parse or bind
# - constructs outside the HLSL printer subset
# Every message names the rejected construct.

import std / [macros, sets, tables]
import ../../builtins/builtins_gpu_types
import ../ir/ast_datatypes
import ./pass_datatypes
import ./pass_registry

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerHlslValidationPasses*(reg: var PassRegistry) =
  reg.register("validateHlslKernelShape", pkValidation, phaseMain,
    "Rejects multi-kernel sources, non-void kernels, and kernel parameters the glslang HLSL frontend cannot bind",
    proc(ctx: var GpuContext): void =
      ctx.validateHlslKernelShape()
  )
  reg.register("validateHlslBody", pkValidation, phaseMain,
    "Rejects constructs outside the HLSL printer subset: address-of, bare derefs, whole-vector coordinate builtins, pointer and smem locals, expression ifs",
    proc(ctx: var GpuContext): void =
      ctx.validateHlslBody()
  )

# ═══════════════════════════════════════════════════════════════════════
# Kernel shape
# ═══════════════════════════════════════════════════════════════════════

proc validateHlslParam(p: GpuParam, kernelName: string) =
  ## Parameter shapes the HLSL resource model cannot bind
  let name = p.ident.ident()
  if p.flavor == kVar:
    error "HLSL: `var` (pass-by-reference) parameter " & name & " in kernel `" & kernelName &
      "` has no binding, buffers pass as `ptr UncheckedArray[T]`"
  if p.passByRef:
    error "HLSL: by-reference parameter " & name & " in kernel `" & kernelName & "` has no binding"
  if p.addressSpace == asSMEM:
    error "HLSL: {.smem.} parameter " & name & " in kernel `" & kernelName & "` has no binding"
  case p.typ.kind
  of gtVoidPtr:
    error "HLSL: void pointer parameter " & name & " in kernel `" & kernelName & "` has no resource binding"
  of gtPtr:
    if p.typ.to.kind != gtUncheckedArray:
      error "HLSL: buffer parameter " & name & " in kernel `" & kernelName &
        "` must be `ptr UncheckedArray[T]`, got ptr " & $p.typ.to.kind
  of gtObject, gtGenericInst:
    # gtObject carries the plain type name, gtGenericInst the generic's name
    let tname = if p.typ.kind == gtGenericInst: p.typ.gName else: p.typ.name
    if tname in ["TensorView", "TensorOwned"]:
      error "HLSL: " & tname & " parameter " & name & " in kernel `" & kernelName &
        "` is outside the HLSL subset (HLSL structs cannot hold resources)"
    error "HLSL: struct parameter " & name & " in kernel `" & kernelName & "` has no binding"
  of gtInt64, gtUint64:
    error "HLSL: 64-bit integer parameter " & name & " in kernel `" & kernelName &
      "` is rejected: int64_t is a parse error in the glslang HLSL frontend and uint64_t pulls the SPIR-V Int64 capability"
  of gtArray, gtUncheckedArray:
    error "HLSL: " & $p.typ.kind & " parameter " & name & " in kernel `" & kernelName &
      "` has no binding, buffers pass as `ptr UncheckedArray[T]`"
  of gtFloat16, gtBf16, gtF8E4M3, gtF8E5M2, gtUint8, gtUint16, gtInt16, gtSpan, gtString:
    error "HLSL: parameter " & name & " of type " & $p.typ.kind & " in kernel `" & kernelName &
      "` has no glslang HLSL type name"
  else:
    discard

proc validateHlslKernelShape(ctx: var GpuContext) =
  ## One global kernel per source and shape checks on its parameters.
  ## glslang compiles one entry point per invocation, the kernel emits as `main`.
  var kernel: GpuAst = nil
  var kernelName = ""
  for fn in ctx.fnTab.values:
    if fn.kind != gpuProc:
      continue
    if not fn.isGlobal():
      # the inline pass spliced every callee reachable from a kernel,
      # the dead fnTab entries dropped
      continue
    doAssert kernel.isNil,
      "HLSL compiles one kernel per source, found `" & kernelName & "` and `" & fn.pName.ident() & "`"
    kernel = fn
    kernelName = fn.pName.ident()
  doAssert kernel.isNil.not, "HLSL found no global kernel"
  if kernel.pRetType.kind != gtVoid:
    error "HLSL: kernel `" & kernelName & "` must return void, got " & $kernel.pRetType.kind
  for p in kernel.pParams:
    p.validateHlslParam(kernelName)

# ═══════════════════════════════════════════════════════════════════════
# Body subset
# ═══════════════════════════════════════════════════════════════════════

type HlslParamShape = object
  bufferISyms: HashSet[string] ## iSyms of the kernel's buffer (ptr) parameters
  pushISyms: HashSet[string]   ## iSyms of the kernel's scalar (push-constant) parameters

proc validateHlslExpr(ctx: GpuContext, n: GpuAst, shape: HlslParamShape, fnName: string)
proc validateHlslStmt(ctx: GpuContext, n: GpuAst, shape: HlslParamShape, fnName: string)

proc validateHlslCall(ctx: GpuContext, n: GpuAst, shape: HlslParamShape, fnName: string) =
  let sym = n.cName.symbol
  if sym.reductionBuiltin != gbkNone:
    error "HLSL: subgroup reductions have no emission, wave intrinsics are outside this backend"
  if sym.atomicsBuiltin != gbkNone:
    error "HLSL: atomic builtins have no emission in this backend"
  # fp16 conversions have no HLSL mapping: float16 types are outside the subset,
  # a raw conversion call would surface as a glslang "unknown function" error
  if isFp16ConversionBuiltin(sym.name):
    error "HLSL: fp16 conversion builtin `" & sym.name & "` has no emission, float16 types are outside the HLSL subset"
  # synchro builtins (barriers) and intrinsics (max/min/abs, exp2, rsqrt) emit,
  # device function calls are spliced away by the inline pass before validation
  for arg in n.cArgs:
    ctx.validateHlslExpr(arg, shape, fnName)

proc validateHlslIndexBase(ctx: GpuContext, base: GpuAst, fnName: string) =
  ## An index base is a plain array value (a buffer parameter, a fixed array)
  ## or a gpuDeref the printer reads through, anything else is rejected
  case base.kind
  of gpuIdent:
    let t = base.symbol.typ
    if t.isNil or t.kind notin {gtPtr, gtArray, gtUncheckedArray}:
      error "HLSL: indexing something other than a buffer parameter or a fixed array: " & base.symbol.name
  of gpuDeref:
    ctx.validateHlslIndexBase(base.dOf, fnName)
  else:
    error "HLSL: indexed base is not a plain identifier, got " & $base.kind

proc validateHlslExpr(ctx: GpuContext, n: GpuAst, shape: HlslParamShape, fnName: string) =
  ## Expression-position rejections: address-of, bare derefs, bare buffer pointer values, whole-vector coordinate builtins
  case n.kind
  of gpuIndex:
    # the base may be a deref-wrapped buffer parameter, the printer reads through it
    ctx.validateHlslIndexBase(n.iArr, fnName)
    ctx.validateHlslExpr(n.iIndex, shape, fnName)
  of gpuDeref:
    error "HLSL: bare dereference in expression position has no emission, buffers are index-only resources"
  of gpuAddr:
    error "HLSL: address-of has no emission, buffers are index-only resources"
  of gpuIdent:
    let sym = n.symbol
    if sym.iSym in shape.bufferISyms:
      error "HLSL: bare pointer value use of buffer parameter " & sym.name & " in `" & fnName & "`"
    if sym.coordBuiltin != gbkNone and sym.coordBuiltin != gbkThreadIndexInThreadgroup:
      error "HLSL: whole-vector use of coordinate builtin " & sym.name & " has no emission, index a component"
  of gpuDot:
    if n.dParent.kind == gpuIdent and n.dParent.symbol != nil and
        n.dParent.symbol.coordBuiltin != gbkNone:
      if n.dParent.symbol.coordBuiltin notin {gbkThreadPositionInGrid, gbkThreadgroupPositionInGrid}:
        error "HLSL: component access on " & n.dParent.ident() &
          " has no emission (scalar or unsupported coordinate builtin)"
      # component read of a supported vector builtin: the parent was type-checked above, recursing would re-flag it as a whole-vector use
      ctx.validateHlslExpr(n.dField, shape, fnName)
    else:
      ctx.validateHlslExpr(n.dParent, shape, fnName)
      ctx.validateHlslExpr(n.dField, shape, fnName)
  of gpuCall:
    ctx.validateHlslCall(n, shape, fnName)
  else:
    for ch in n:
      ctx.validateHlslExpr(ch, shape, fnName)

proc validateHlslStmt(ctx: GpuContext, n: GpuAst, shape: HlslParamShape, fnName: string) =
  case n.kind
  of gpuBlock:
    for s in n.statements:
      ctx.validateHlslStmt(s, shape, fnName)
  of gpuVar:
    let name = n.vName.ident()
    if n.vType.kind == gtPtr:
      error "HLSL: pointer-valued local " & name & " in `" & fnName & "` has no emission"
    if n.addressSpace == asSMEM:
      error "HLSL: {.smem.} local " & name & " in `" & fnName & "` has no emission"
    if n.vType.kind == gtGenericInst and n.vType.gName in ["TensorView", "TensorOwned"]:
      error "HLSL: " & n.vType.gName & " local " & name & " in `" & fnName &
        "` is outside the HLSL subset (HLSL structs cannot hold resources)"
    if n.vType.kind in [gtInt64, gtUint64]:
      error "HLSL: 64-bit integer local " & name & " in `" & fnName &
        "` is rejected: int64_t is a parse error in the glslang HLSL frontend and uint64_t pulls the SPIR-V Int64 capability"
    if n.vType.kind == gtSpan:
      error "HLSL: openArray local " & name & " in `" & fnName & "` has no binding"
    ctx.validateHlslExpr(n.vInit, shape, fnName)
  of gpuAssign:
    if n.aLeft.kind == gpuIdent and n.aLeft.symbol.iSym in shape.pushISyms:
      error "HLSL: assignment to push-constant parameter " & n.aLeft.ident() &
        " in `" & fnName & "`, push constants are read-only inputs"
    ctx.validateHlslExpr(n.aLeft, shape, fnName)
    ctx.validateHlslExpr(n.aRight, shape, fnName)
  of gpuIf:
    if n.ifIsExpr:
      error "HLSL: expression if in `" & fnName & "` has no emission, use a statement if or a ternary"
    ctx.validateHlslExpr(n.ifCond, shape, fnName)
    ctx.validateHlslStmt(n.ifThen, shape, fnName)
    for el in n.ifElifs:
      ctx.validateHlslExpr(el.cond, shape, fnName)
      ctx.validateHlslStmt(el.body, shape, fnName)
    if n.ifElse.kind != gpuDiscard:
      ctx.validateHlslStmt(n.ifElse, shape, fnName)
  of gpuFor:
    ctx.validateHlslExpr(n.fStart, shape, fnName)
    ctx.validateHlslExpr(n.fEnd, shape, fnName)
    ctx.validateHlslExpr(n.fStep, shape, fnName)
    ctx.validateHlslStmt(n.fBody, shape, fnName)
  of gpuWhile:
    ctx.validateHlslExpr(n.wCond, shape, fnName)
    ctx.validateHlslStmt(n.wBody, shape, fnName)
  of gpuReturn:
    if n.rValue.kind != gpuDiscard:
      error "HLSL: non-void return in kernel `" & fnName & "`, kernels are void"
  of gpuInlineAsm:
    error "HLSL: inline assembly has no emission"
  else:
    ctx.validateHlslExpr(n, shape, fnName)

proc validateHlslBody(ctx: GpuContext) =
  ## Construct-level rejections over every kernel body
  for fn in ctx.fnTab.values:
    if fn.kind != gpuProc or not fn.isGlobal():
      continue
    var shape = HlslParamShape()
    for p in fn.pParams:
      if p.typ.kind == gtPtr:
        shape.bufferISyms.incl p.ident.symbol.iSym
      elif p.ident.symbol.coordBuiltin == gbkNone:
        shape.pushISyms.incl p.ident.symbol.iSym
    ctx.validateHlslStmt(fn.pBody, shape, fn.pName.ident())
