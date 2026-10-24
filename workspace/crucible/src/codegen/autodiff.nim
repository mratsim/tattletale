# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Compile-time kernel-pair artifact of `bwd_diff`.
## The artifact carries the printed sources of the pair plus the signature metadata a runtime backend needs, nothing else.
## No Nim AST is stored, the compile-time stage is a pure function of the kernel symbol.
##
## Backward signature rule, the only signature authority of the pair ABI:
##
## | Block          | Content                                                                    |
## | -------------- | -------------------------------------------------------------------------- |
## | `bwdSignature` | `<d_-block>, <forward signature verbatim>`                                 |
## | d_-block       | one gradient slot per forward parameter with a cotangent, forward order    |
## | buffer grads   | same shape as the primal buffer (`d_<name>`)                               |
## | scalar grads   | 1-element device buffers (`kind = buffer(1)`), matched by name, never type |
## | block 2        | the forward signature verbatim, forward outputs at their forward position  |
## | `save:` order  | the forward output order                                                   |
## | workspaces     | reduce-partials live only in the backward kernel's IR signature            |
##
## A declared backward (`bwd_diff(fwd, bwd, save = [...])`) skips the transposer.
## Validation against this mirror runs at expansion: gradient slots in forward order,
## block 2 the forward signature verbatim, the kernel named `<fwd>_bwd`.
##
## | Section    | Symbols                                                          |
## | ---------- | ---------------------------------------------------------------- |
## | Pair       | `BwdDiffKernels`                                                 |
## | Metadata   | `BwdParamMeta`, `BwdParamKind`, `BwdGradKind`, `BwdExtentSource` |
## | Derivation | `deriveBwdParams`, `mirrorSignature`                             |
## | Pipeline   | `codegenBwdDiffKernels`, `codegenBwdDiffKernelsDeclared`         |

import std/[macros, sets, strutils, tables]

import ir/ast_datatypes

import ./frontend/nim_to_crucible
import ./gpu_compiler # registerCommonPasses
import ./passes/pass_datatypes
import ./passes/pass_registry
import ./passes/passes_autodiff
import ./passes/passes_lowering_glsl
import ./passes/passes_lowering_msl
import ./passes/passes_validations
import ./backends/backends_api

export ast_datatypes.BackendKind

# ═══════════════════════════════════════════════════════════════════════
# ▸ Types
# ═══════════════════════════════════════════════════════════════════════

type ReduceSource* = object
  ## One generated axis-reduce kernel of the pair: its name and standalone source.
  name*: string
  source*: string

type ScalarWorkspace* = object
  ## Reduce-partials scratch of one scalar cotangent: the backward kernel
  ## accumulates per-index partials into it, the reduce kernel folds it into
  ## the scalar's 1-element gradient buffer.
  gradName*: string        ## `d_<name>`, the 1-element gradient buffer
  workspaceName*: string   ## `d_<name>_partials`, the backward kernel's scratch
  tripCount*: int          ## Static index-loop trip count, -1 when runtime-derived
  tripCountParam*: string  ## Name of the kernel parameter carrying the trip count, runtime-derived
  tripCount64*: bool       ## Trip count is a 64-bit value, else 32-bit

type BwdParamKind* = enum
  pkBuffer    ## Read-only input buffer, gradient is a same-shape buffer
  pkScalar    ## Scalar parameter (integer or float), integer scalars carry no gradient
  pkResidual  ## Forward-output buffer the forward kernel writes, available to the backward kernel at its block-2 position

type BwdGradKind* = enum
  gkNone    ## No gradient slot in the backward signature
  gkBuffer  ## Gradient buffer, same shape as the primal argument
  gkBuffer1 ## 1-element gradient device buffer, the scalar cotangent written by the generated reduce kernel

type BwdExtentSource* = enum
  xeWrapperField ## Buffer length lives on the caller's host wrapper, `DeviceMem` carries no sizes
  xeStaticShape  ## Pointer to a static array, the length is compile-time
  xeLenParam     ## A separate integer kernel parameter carries the length

type BwdParamMeta* = object
  ## Signature record of one forward parameter.
  name*: string            ## Forward parameter name
  kind*: BwdParamKind
  gradName*: string        ## `d_<name>`, empty when `gradKind == gkNone`
  gradKind*: BwdGradKind
  extent*: BwdExtentSource ## Buffer-length source, buffers only, unused for scalars
  extentParam*: string     ## `xeLenParam`: the integer parameter's name
  staticLen*: int          ## `xeStaticShape`: the compile-time element count

type BwdDiffKernels* = object
  ## Const artifact of one `bwd_diff` expansion:
  ##
  ## - two printed sources
  ## - the reduce kernels of the scalar cotangents
  ## - the parameter metadata and the derived backward signature
  kernelName*: string                          ## Forward kernel's name
  backend*: BackendKind                        ## Backend the sources target
  fwdSource*: string                           ## Forward kernel source, standalone translation unit
  bwdSource*: string                           ## Backward kernel source, `<kernel>_bwd`
  reduceSources*: seq[ReduceSource]                ## One axis-reduce kernel per scalar cotangent
  params*: seq[BwdParamMeta]                   ## Forward parameters in declaration order
  scalarWorkspaces*: seq[ScalarWorkspace]      ## Reduce-partials scratch per scalar cotangent, forward order
  bwdSignature*: seq[string]                   ## Backward signature, see the module header
  bwdKernelParams*: seq[string]                ## Printed backward kernel's parameter names, post-lowering, in binding order
  saveNames*: seq[string]                      ## `save = [...]` forward outputs kept alive between forward and backward, declared pairs only

# ═══════════════════════════════════════════════════════════════════════
# ▸ Derivation
# ═══════════════════════════════════════════════════════════════════════
#
# The metadata derives mechanically from the post-pass IR procs the pipeline produced, never from hand-written tables:
# - a buffer parameter carries a gradient when the backward kernel's signature has its `d_<name>` shadow adjoint
# - a float scalar carries a gradient when the transposer generated its `<kernel>_bwd_d_<name>_reduce` kernel
# - a buffer is a forward output when the forward body assigns through it

const PairFloatKinds = {gtFloat32, gtFloat64}
  ## Float widths that carry cotangents in the pair ABI, matching the transposer's differentiable float widths.
  ## Grow both together when the subset grows.

proc writtenParamISyms(n: GpuAst, paramISyms: seq[string], found: var HashSet[string]) =
  ## Collects the iSyms of forward parameters the forward body writes through (assignment targets rooted at a parameter index chain).
  if n.isNil:
    return
  if n.kind == gpuAssign and n.aLeft.kind == gpuIndex:
    var root = n.aLeft.iArr
    while root.kind == gpuDeref:
      root = root.dOf
    if root.kind == gpuIdent and root.symbol != nil and
        root.symbol.iSym in paramISyms:
      found.incl root.symbol.iSym
  for ch in n.items:
    ch.writtenParamISyms(paramISyms, found)

proc firstLoopBound(n: GpuAst): GpuAst =
  ## Trip-count expression of the first loop in statement order, nil when the kernel has no loop.
  ## The differentiable subset refuses nested loops, so there is at most one.
  if n.isNil:
    return nil
  if n.kind == gpuFor:
    return n.fEnd
  for ch in n.items:
    let bound = ch.firstLoopBound()
    if not bound.isNil:
      return bound
  return nil

proc bufferExtent(p: GpuParam, fwd: GpuAst, paramISyms: seq[string]): BwdParamMeta =
  ## Length source of one buffer parameter, in the order of authority: the parameter's own type, then the index-loop bound, then the wrapper.
  result = BwdParamMeta(name: p.ident.ident(), extent: xeWrapperField)
  if p.typ.kind == gtPtr and p.typ.to.kind == gtArray:
    result.extent = xeStaticShape
    result.staticLen = p.typ.to.aLen
    return
  let bound = fwd.pBody.firstLoopBound()
  if not bound.isNil and bound.kind == gpuIdent and bound.symbol != nil and
      bound.symbol.iSym in paramISyms:
    result.extent = xeLenParam
    result.extentParam = bound.symbol.name

proc deriveBwdParams*(fwd, bwd: GpuAst, reduceNames: seq[string]): seq[BwdParamMeta] =
  ## Derives the backward-signature metadata from the post-pass procs of one kernel.
  ##
  ## Expected input:
  ## - `fwd`: the forward kernel as the pipeline produced it
  ## - `bwd`: its backward kernel, `<fwd>_bwd`, transposed or declared
  ## - `reduceNames`: names of the generated `<fwd>_bwd_d_<scalar>_reduce` kernels,
  ##   empty for a declared backward
  ##
  ## A float scalar carries `gkBuffer1` when the backward's parameter list has a `d_<name>` slot:
  ##
  ## - declared backward: the hand-written kernel writes the 1-element buffer itself
  ## - transposed backward: a generated reduce kernel fills the buffer
  ##
  ## Output: one `BwdParamMeta` per forward parameter, declaration order.
  doAssert fwd.kind == gpuProc and bwd.kind == gpuProc
  var paramISyms: seq[string]
  for p in fwd.pParams:
    paramISyms.add p.ident.symbol.iSym
  # The backward's gradient slots match by parameter name, never by symbol id:
  # symbol ids are namespaced per proc.
  # A declared backward is a proc of its own, so mirror parameters carry different ids across the two procs even with equal names.
  # The mirror validation already checked the names positionally.
  var bwdParamNames: HashSet[string]
  for p in bwd.pParams:
    bwdParamNames.incl p.ident.ident()
  var written: HashSet[string]
  fwd.pBody.writtenParamISyms(paramISyms, written)

  for p in fwd.pParams:
    let name = p.ident.ident()
    let iSym = p.ident.symbol.iSym
    if p.typ.kind == gtPtr:
      var meta = p.bufferExtent(fwd, paramISyms)
      if iSym in written:
        meta.kind = pkResidual
      else:
        meta.kind = pkBuffer
      if "d_" & name in bwdParamNames:
        meta.gradKind = gkBuffer
        meta.gradName = "d_" & name
      result.add meta
    elif p.typ.kind in PairFloatKinds:
      var meta = BwdParamMeta(name: name, kind: pkScalar)
      if "d_" & name in bwdParamNames:
        # declared backward: the hand-written kernel carries the scalar's
        # 1-element gradient slot itself, no reduce kernel exists
        meta.gradKind = gkBuffer1
        meta.gradName = "d_" & name
      else:
        for rn in reduceNames:
          if rn.endsWith("_bwd_d_" & name & "_reduce"):
            meta.gradKind = gkBuffer1
            meta.gradName = "d_" & name
      result.add meta
    else:
      result.add BwdParamMeta(name: name, kind: pkScalar)

proc mirrorSignature*(params: seq[BwdParamMeta]): seq[string] =
  ## Returns the backward signature of a pair.
  ## Block 1 holds the gradient slots in forward order, block 2 the forward signature verbatim, see the module header.
  for m in params:
    if m.gradKind != gkNone:
      result.add m.gradName
  for m in params:
    result.add m.name

# ═══════════════════════════════════════════════════════════════════════
# ▸ Declared backward
# ═══════════════════════════════════════════════════════════════════════
#
# The declared path of `bwd_diff(fwd, bwd, save = [...])` skips the transposer:
# the hand-written backward kernel is the backward. The mirror rule still
# governs its signature, derived from the forward symbol alone, and a mismatch
# refuses at expansion.

proc referencedParamISyms(n: GpuAst, paramISyms: seq[string], found: var HashSet[string]) =
  ## Collects the iSyms of forward parameters referenced anywhere in the body,
  ## write and read positions alike: index chains and bare identifiers.
  if n.isNil:
    return
  if n.kind == gpuIdent and n.symbol != nil and n.symbol.iSym in paramISyms:
    found.incl n.symbol.iSym
  for ch in n.items:
    ch.referencedParamISyms(paramISyms, found)

proc mirrorCotangentNames*(fwd: GpuAst): seq[string] =
  ## d_-block of the mirror signature, derived from the forward symbol alone.
  ##
  ## Expected input:
  ## - `fwd`: the forward kernel as the pipeline produced it
  ##
  ## Output: one `d_<name>` slot per forward parameter the forward body references,
  ## buffers and float scalars alike, forward order.
  ##
  ## The transposer seeds its own d_-block from the same buffer walk,
  ## plus the scalar cotangents it actually demands, so the two agree for kernels in the differentiable subset.
  ##
  ## One narrow case may differ: a float scalar referenced only in a zero-cotangent position,
  ## say a comparison, carries no transposed slot but does carry a declared slot.
  var paramISyms: seq[string]
  for p in fwd.pParams:
    paramISyms.add p.ident.symbol.iSym
  var referenced: HashSet[string]
  fwd.pBody.referencedParamISyms(paramISyms, referenced)
  for p in fwd.pParams:
    let iSym = p.ident.symbol.iSym
    if (p.typ.kind == gtPtr or p.typ.kind in PairFloatKinds) and
        iSym in referenced:
      result.add "d_" & p.ident.ident()

proc declaredMirrorSignature*(fwd: GpuAst): seq[string] =
  ## Full mirror signature a declared backward must reproduce exactly:
  ## the d_-block of `mirrorCotangentNames`, then the forward signature verbatim.
  result = fwd.mirrorCotangentNames()
  for p in fwd.pParams:
    result.add p.ident.ident()

proc validateDeclaredBwd*(fwd, bwd: GpuAst) =
  ## Refuses a declared backward that does not mirror its forward.
  ## The contract, checked in order:
  ##
  ## - the backward kernel is named `<fwd>_bwd`, the name the pair's backward
  ##   launches under
  ## - its parameters equal `declaredMirrorSignature(fwd)` positionally,
  ##   the refusal names the first differing position and the rule it breaks
  let fwdName = fwd.pName.ident()
  let bwdName = bwd.pName.ident()
  if bwdName != fwdName & "_bwd":
    error "bwd_diff: the declared backward '" & bwdName & "' must be named '" &
      fwdName & "_bwd', the name the pair's backward launches under"
  let mirror = fwd.declaredMirrorSignature()
  var declared: seq[string]
  for p in bwd.pParams:
    declared.add p.ident.ident()
  if declared == mirror:
    return
  let dBlock = fwd.mirrorCotangentNames().len
  var i = 0
  while i < declared.len and i < mirror.len and declared[i] == mirror[i]:
    i.inc
  let rule = if i < dBlock:
    "gradients sit in forward-parameter order"
  else:
    "block 2 is the forward signature verbatim"
  if i < declared.len and i < mirror.len:
    error "bwd_diff: the declared backward '" & bwdName & "' does not mirror '" &
      fwdName & "': parameter " & $(i + 1) & " is '" & declared[i] &
      "', the mirror rule derives '" & mirror[i] & "' (" & rule & ")"
  elif i < declared.len:
    error "bwd_diff: the declared backward '" & bwdName & "' does not mirror '" &
      fwdName & "': parameter " & $(i + 1) & " is '" & declared[i] &
      "', beyond the mirror (" & rule & ")"
  else:
    error "bwd_diff: the declared backward '" & bwdName & "' does not mirror '" &
      fwdName & "': it stops at parameter " & $declared.len &
      ", the mirror rule derives '" & mirror[i] & "' (" & rule & ")"

proc validateSaveNames*(fwdName: string, metas: seq[BwdParamMeta], saveNames: seq[string]) =
  ## Refuses `save = [...]` names that are not forward outputs of the pair's
  ## forward kernel: a save names a buffer the forward kernel writes.
  for s in saveNames:
    var known = false
    for m in metas:
      if m.name == s:
        known = true
        if m.kind != pkResidual:
          error "bwd_diff: save name '" & s & "' is not a forward output of '" &
            fwdName & "', the forward kernel does not write it"
    if not known:
      error "bwd_diff: save name '" & s & "' is not a parameter of '" &
        fwdName & "'"

const PairWideIntKinds = {gtInt64, gtUint64, gtSize_t}
  ## Integer widths the reduce-kernel trip count may carry, grow with the IR's int kinds.

proc deriveScalarWorkspaces*(fwd: GpuAst, metas: seq[BwdParamMeta]): seq[ScalarWorkspace] =
  ## Derives the reduce-partials scratch records of a pair's scalar cotangents.
  ##
  ## The trip count is the forward kernel's index-loop bound:
  ##
  ## - a literal bound folds to a static trip count
  ## - an identifier bound names the parameter carrying the trip count
  ## - a loop-free kernel has no scalar cotangent: its accumulation
  ##   degenerates to a single partial, so the pair derives zero workspaces
  ##   and zero reduce kernels
  ##
  ## Expected input:
  ## - `fwd`: the forward kernel as the pipeline produced it
  ## - `metas`: `deriveBwdParams`' output
  ##
  ## Output: one `ScalarWorkspace` per `gkBuffer1` parameter, forward order.
  var scalarGrads: seq[BwdParamMeta]
  for m in metas:
    if m.gradKind == gkBuffer1:
      scalarGrads.add m
  if scalarGrads.len == 0:
    return
  let bound = fwd.pBody.firstLoopBound()
  doAssert not bound.isNil,
    "scalar cotangent without an index-loop trip count"
  var tripCount = -1
  var tripCountParam = ""
  var tripCount64 = false
  if bound.kind == gpuLit:
    tripCount = bound.lValue.parseInt()
    tripCount64 = bound.lType.kind in PairWideIntKinds
  else:
    doAssert bound.kind == gpuIdent and bound.symbol != nil,
      "the reduce-kernel trip count is neither a literal nor a kernel parameter"
    tripCountParam = bound.symbol.name
    tripCount64 = bound.symbol.typ.kind in PairWideIntKinds
  for m in scalarGrads:
    result.add ScalarWorkspace(
      gradName: m.gradName,
      workspaceName: "d_" & m.name & "_partials",
      tripCount: tripCount,
      tripCountParam: tripCountParam,
      tripCount64: tripCount64)
proc printSingleProc(ctx: var GpuContext, bk: BackendKind, fn: GpuAst): string =
  ## Prints one proc as a standalone translation unit of backend `bk`.
  ##
  ## The printer sees only this kernel in the function table.
  ## Each kernel prints with its own type declarations and push-constant blocks.
  let saved = ctx.fnTab
  ctx.fnTab = {fn.pName: fn}.toOrderedTable
  case bk
  of bkMetal:
    result = ctx.codegenMetal()
  of bkGlsl:
    result = ctx.codegenGLSL()
  else:
    discard
  ctx.fnTab = saved

proc lowerAndPrintKernels(ctx: var GpuContext, bk: BackendKind, fwd, bwd: GpuAst, reduces: seq[GpuAst], donePasses: HashSet[string]): tuple[fwdSource: string, bwdSource: string, reduceSources: seq[ReduceSource], bwdKernelParams: seq[string]] =
  ## Stage 2 of the pair pipelines: backend lowering over the pair's kernels and reduce kernels, then per-kernel standalone printing.
  ##
  ## The GLSL shared-push-constant-layout validation does not apply.
  ## It guards multi-kernel sources sharing one block.
  ## Each kernel of the pair prints and ingests standalone, carrying its own push-constant block.
  var regLow = PassRegistry.new()
  # Lowering passes carry `dependsOn` edges into the common passes
  # (e.g. GLSL's insertPointerDeref needs the legalized rewriteIndexDeref).
  # The dependency set continues across the stages.
  regLow.donePasses = donePasses
  case bk
  of bkMetal:
    regLow.registerMetalLoweringPasses()
  of bkGlsl:
    regLow.registerGlslLoweringPasses()
  else:
    discard
  regLow.registerValidationPostPasses()
  runPasses(ctx, regLow)

  for rk in reduces:
    result.reduceSources.add ReduceSource(name: rk.pName.ident(),
                                          source: ctx.printSingleProc(bk, rk))
  for p in bwd.pParams:
    result.bwdKernelParams.add p.ident.ident()
  result.fwdSource = ctx.printSingleProc(bk, fwd)
  result.bwdSource = ctx.printSingleProc(bk, bwd)

proc codegenBwdDiffKernels*(bk: BackendKind, body: NimNode): BwdDiffKernels =
  ## Pipeline of one `bwd_diff` expansion for backend `bk`:
  ##
  ##   Nim body → frontend → common passes → transposer → metadata
  ##           → backend lowering → per-kernel print
  ##
  ## - the typed kernel body goes through the frontend, the common passes
  ##   and the autodiff transposer
  ## - the signature metadata derives from the transposer's output
  ## - each kernel prints standalone after the backend lowering
  ##
  ## Compile-time only, called from the `bwd_diff` macro.
  ##
  ## A `static` backend-kind macro parameter cannot host the pipeline:
  ##
  ## - a static parameter makes the macro generic
  ## - instantiation re-sem-checks the typed kernel body
  ## - statement pragmas like `{.emit: ... .}` refuse there
  ## - the differentiable subset must diagnose those pragmas itself
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toCrucibleAst(typeReg, body)
  ctx.types = typeReg.types
  ctx.ingestTopLevel(gpuAst, typesFirst = true)

  # Stage 1, differentiation: metadata derives from the transposer's output,
  # before backend lowering renames identifiers.
  # The ABI keeps Nim-facing names (GLSL renames `out` to `out_nim`).
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  reg.registerAutodiffPasses()
  runPasses(ctx, reg)

  var fwd: GpuAst = nil
  var bwd: GpuAst = nil
  var reduces: seq[GpuAst] = @[]
  for fn in ctx.allFnTab.values():
    if fn.kind != gpuProc or attGlobal notin fn.pAttributes:
      continue
    if fwd.isNil:
      fwd = fn
    elif fn.pName.ident() == fwd.pName.ident() & "_bwd":
      bwd = fn
    elif fn.pName.ident().startsWith(fwd.pName.ident() & "_bwd_d_") and
        fn.pName.ident().endsWith("_reduce"):
      reduces.add fn
    else:
      error "bwd_diff: unexpected global proc '" & fn.pName.ident() &
        "' in the pair pipeline"
  if fwd.isNil:
    error "bwd_diff: no {.global.} kernel to differentiate"
  if bwd.isNil:
    error "bwd_diff: no backward kernel was generated for '" &
      fwd.pName.ident() & "'"

  var reduceNames: seq[string] = @[]
  for rk in reduces:
    reduceNames.add rk.pName.ident()
  let metas = deriveBwdParams(fwd, bwd, reduceNames)

  let printed = ctx.lowerAndPrintKernels(bk, fwd, bwd, reduces, reg.donePasses)
  result = BwdDiffKernels(
    kernelName: fwd.pName.ident(),
    backend: bk,
    fwdSource: printed.fwdSource,
    bwdSource: printed.bwdSource,
    reduceSources: printed.reduceSources,
    params: metas,
    scalarWorkspaces: fwd.deriveScalarWorkspaces(metas),
    bwdSignature: metas.mirrorSignature(),
    bwdKernelParams: printed.bwdKernelParams,
    saveNames: @[])

proc codegenBwdDiffKernelsDeclared*(bk: BackendKind, fwdBody, bwdBody: NimNode, saveNames: seq[string]): BwdDiffKernels =
  ## Pipeline of one declared-backward `bwd_diff(fwd, bwd, save = [...])`
  ## expansion for backend `bk`:
  ##
  ##   Nim bodies → frontend → common passes → mirror validation
  ##             → backend lowering → per-kernel print
  ##
  ## Contract:
  ##
  ## - No autodiff pass runs: the hand-written kernel is the pair's backward.
  ## - The pair's metadata still derives from the forward symbol.
  ## - A declared backward whose signature deviates from the mirror refuses at expansion with the `bwd_diff:` prefix (`validateDeclaredBwd`).
  ## - Every save name must be a forward output of the forward kernel (`validateSaveNames`),
  ##   saved outputs stay alive between forward and backward.
  ##
  ## Compile-time only, called from the `bwd_diff` macro.
  var ctx = GpuContext()
  var typeReg = TypeRegistry(types: ctx.types)
  let gpuAst = ctx.toCrucibleAst(typeReg, nnkStmtList.newTree(fwdBody, bwdBody))
  ctx.types = typeReg.types
  ctx.ingestTopLevel(gpuAst, typesFirst = true)

  # Stage 1: common passes only, the declared backward is hand-written, the transposer never runs for the pair.
  var reg = PassRegistry.new()
  reg.registerCommonPasses()
  runPasses(ctx, reg)

  # The pair ingests in declaration order, the forward kernel first.
  var globals: seq[GpuAst]
  for fn in ctx.allFnTab.values():
    if fn.kind == gpuProc and attGlobal in fn.pAttributes:
      globals.add fn
  if globals.len != 2:
    error "bwd_diff: the declared pair takes one {.global.} forward kernel " &
      "and one '<forward>_bwd' backward, got " & $globals.len & " kernels"
  let fwd = globals[0]
  let bwd = globals[1]

  # The mirror rule is the only signature authority: validate before deriving.
  validateDeclaredBwd(fwd, bwd)
  let metas = deriveBwdParams(fwd, bwd, @[])
  validateSaveNames(fwd.pName.ident(), metas, saveNames)

  let printed = ctx.lowerAndPrintKernels(bk, fwd, bwd, @[], reg.donePasses)
  result = BwdDiffKernels(
    kernelName: fwd.pName.ident(),
    backend: bk,
    fwdSource: printed.fwdSource,
    bwdSource: printed.bwdSource,
    reduceSources: printed.reduceSources,
    params: metas,
    scalarWorkspaces: @[],
    bwdSignature: metas.mirrorSignature(),
    bwdKernelParams: printed.bwdKernelParams,
    saveNames: saveNames)

