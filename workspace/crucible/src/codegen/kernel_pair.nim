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
## | Section    | Symbols                                                          |
## | ---------- | ---------------------------------------------------------------- |
## | Pair       | `KernelPair`                                                     |
## | Metadata   | `BwdParamMeta`, `BwdParamKind`, `BwdGradKind`, `BwdExtentSource` |
## | Derivation | `deriveBwdParams`, `mirrorSignature`                             |

import std/[sets, strutils]

import ir/ast_datatypes

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

type KernelPair* = object
  ## Const artifact of one `bwd_diff` expansion:
  ## two printed sources, the reduce kernels of the scalar cotangents,
  ## the parameter metadata and the derived backward signature.
  kernelName*: string                          ## Forward kernel's name
  backend*: BackendKind                        ## Backend the sources target
  fwdSource*: string                           ## Forward kernel source, standalone translation unit
  bwdSource*: string                           ## Backward kernel source, `<kernel>_bwd`
  reduceSources*: seq[ReduceSource]                ## One axis-reduce kernel per scalar cotangent
  params*: seq[BwdParamMeta]                   ## Forward parameters in declaration order
  scalarWorkspaces*: seq[ScalarWorkspace]      ## Reduce-partials scratch per scalar cotangent, forward order
  bwdSignature*: seq[string]                   ## Backward signature, see the module header
  bwdKernelParams*: seq[string]                ## Printed backward kernel's parameter names, post-lowering, in binding order

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
  ## - `bwd`: its transposed backward kernel, `<fwd>_bwd`
  ## - `reduceNames`: names of the generated `<fwd>_bwd_d_<scalar>_reduce` kernels
  ##
  ## Output: one `BwdParamMeta` per forward parameter, declaration order.
  doAssert fwd.kind == gpuProc and bwd.kind == gpuProc
  var paramISyms: seq[string]
  for p in fwd.pParams:
    paramISyms.add p.ident.symbol.iSym
  var bwdParamISyms: HashSet[string]
  for p in bwd.pParams:
    bwdParamISyms.incl p.ident.symbol.iSym
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
      if "d_" & iSym in bwdParamISyms:
        meta.gradKind = gkBuffer
        meta.gradName = "d_" & name
      result.add meta
    elif p.typ.kind in PairFloatKinds:
      var meta = BwdParamMeta(name: name, kind: pkScalar)
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

const PairWideIntKinds = {gtInt64, gtUint64, gtSize_t}
  ## Integer widths the reduce-kernel trip count may carry, grow with the IR's int kinds.

proc deriveScalarWorkspaces*(fwd: GpuAst, metas: seq[BwdParamMeta]): seq[ScalarWorkspace] =
  ## Derives the reduce-partials scratch records of a pair's scalar cotangents.
  ##
  ## The trip count is the forward kernel's index-loop bound.
  ## A literal bound folds to a static trip count.
  ## An identifier bound names the parameter carrying the trip count.
  ##
  ## Expected input:
  ## - `fwd`: the forward kernel as the pipeline produced it
  ## - `metas`: `deriveBwdParams`' output
  ##
  ## Output: one `ScalarWorkspace` per `gkBuffer1` parameter, forward order.
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
  for m in metas:
    if m.gradKind == gkBuffer1:
      result.add ScalarWorkspace(
        gradName: m.gradName,
        workspaceName: "d_" & m.name & "_partials",
        tripCount: tripCount,
        tripCountParam: tripCountParam,
        tripCount64: tripCount64)
