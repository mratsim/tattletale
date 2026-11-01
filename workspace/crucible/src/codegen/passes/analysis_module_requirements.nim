# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Module requirements analysis: one `pkAnalysis` pass per text target, filling
## `CirModule.features` with the constructs the kernel source contains.
##
##   last target transform → glslScanModuleFeatures | openclScanModuleFeatures
##                         | wgslScanModuleFeatures | cudaScanModuleFeatures → validation
##
## Consumers of the stored record:
## - the GLSL extension block and the OpenCL pragma block (`declareExtensions`)
## - the WGSL `enable f16` module directive
## - the CUDA warp-reduction template list
##
## Contract:
## - each target registers its pass in `gpu_compiler.nim` after its last transform, before check-only validation
## - analysis results are valid only if no transform runs after them, which runPasses enforces
## - the printer reads the stored record, it never rescans

import ../ir/cirv2_lang
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Shared walker, one per pass, profile-driven
# ═══════════════════════════════════════════════════════════════════════

type ScanProfile = object
  ## Detector knobs of one target's consumers.
  ##
  ## - a knob set enables its detector for that target's pass
  ## - the base detectors run for every profile: 64- and 16-bit floats set
  ##   `fp64`/`fp16` for any target whose source contains them
  ## - a reduction builtin sets `subgroup` unless the `reductions` knob
  ##   collects the kind into the `reductions` set
  intWidths: bool
  ## GLSL maps 8/16/64-bit integers, so their presence selects extension lines.
  shuffleNames: bool
  ## GLSL maps `simd_shuffle`/`simd_shuffle_xor` to subgroup shuffles.
  unrollAttr: bool
  ## GLSL emits `{.unroll.}` loops through an extension-gated attribute.
  atomicFloatViaPattern: bool
  ## An `atomicFetchAdd` address operand that indexes its pointee
  ## (`buf[i]`) names a float pointer for OpenCL.
  reductions: bool
  ## CUDA's knob, read by the warp-reduction template emission.

const glslProfile = ScanProfile(
  intWidths: true, shuffleNames: true, unrollAttr: true,
  atomicFloatViaPattern: false)

const openclProfile = ScanProfile(
  intWidths: false, shuffleNames: false, unrollAttr: false,
  atomicFloatViaPattern: true)

const wgslProfile = ScanProfile(
  intWidths: false, shuffleNames: false, unrollAttr: false,
  atomicFloatViaPattern: false)

const cudaProfile = ScanProfile(
  reductions: true)

proc isFloat32Pointee(t: CirType): bool =
  t != nil and t.kind in {ctPtr, ctAptr} and t.pElem != nil and
    t.pElem.kind == ctFloat and t.pElem.bits == 32

proc scanType(t: CirType, f: var ModuleFeatures, profile: ScanProfile) =
  if t.isNil:
    return
  case t.kind
  of ctFloat:
    if t.bits == 64:
      f.fp64 = true
    if t.bits == 16:
      f.fp16 = true
  of ctInt, ctUint:
    if profile.intWidths:
      if t.bits == 8:
        f.int8 = true
      if t.bits == 16:
        f.int16 = true
      if t.bits == 64:
        f.int64 = true
  of ctVector: scanType(t.vElem, f, profile)
  of ctArray: scanType(t.aElem, f, profile)
  of ctPtr, ctAptr: scanType(t.pElem, f, profile)
  of ctStruct:
    for fl in t.fields:
      scanType(fl.typ, f, profile)
  of ctBf16:
    # no target needs a module feature for bfloat16
    discard
  of ctVoid, ctBool, ctProc, ctSpan:
    discard

proc scanCallee(callee: CirExpr, args: seq[CirExpr], f: var ModuleFeatures, profile: ScanProfile) =
  ## Both call shapes share this helper. The caller passes the callee and the arguments of either shape.
  ##
  ## Requires: an `abkAtomicFetchAdd` callee takes one pointer argument.
  if callee.kind != eSym:
    return
  case callee.reductionBuiltin
  of rbkReduceAdd, rbkReduceMax:
    if profile.reductions:
      f.reductions.incl callee.reductionBuiltin
    else:
      f.subgroup = true
  of rbkNone:
    discard
  if profile.shuffleNames and callee.shuffleBuiltin != shkNone:
    f.subgroupShuffle = true
  if callee.atomicsBuiltin == abkAtomicFetchAdd:
    var pt = getExprType(args[0])
    if profile.atomicFloatViaPattern and not pt.isFloat32Pointee() and
        args[0].kind == ePat:
      let bt = getExprType(args[0].patBase)
      if bt.isFloat32Pointee():
        pt = bt
    if pt.isFloat32Pointee():
      f.atomicFloat = true

proc scanExprNode(e: CirExpr, f: var ModuleFeatures, profile: ScanProfile) =
  ## One expression node of the walk: the callee detectors of a call node,
  ## then the node's type. The shared walker visits the children.
  if e.kind == eCall and e.callee.kind == eSym:
    scanCallee(e.callee, e.args, f, profile)
  scanType(getExprType(e), f, profile)

proc scanStmts(ss: seq[CirStmt], f: var ModuleFeatures, profile: ScanProfile) =
  var features = f
  let onExpr = proc(e: CirExpr) =
    scanExprNode(e, features, profile)
  let onStmt = proc(s: CirStmt) =
    case s.kind
    of sVar, sConst:
      scanType(s.declType, features, profile)
    of sWhile:
      if profile.unrollAttr and s.whileUnroll:
        features.controlFlow = true
    of sCall:
      scanCallee(s.callCallee, s.callArgs, features, profile)
    else:
      discard
  ss.walkStmts(onStmt, onExpr)
  f = features

proc scanModuleRequirements(m: CirModule, f: var ModuleFeatures, profile: ScanProfile) =
  for t in m.types:
    scanType(t, f, profile)
  # scan coverage includes the program-scope declarations a lowering pass
  # relocated out of a kernel body, initializers included
  scanStmts(m.globals, f, profile)
  for fn in m.procs:
    scanType(fn.ret, f, profile)
    for p in fn.params:
      scanType(p.typ, f, profile)
    scanStmts(fn.body, f, profile)

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

proc registerGlslFeatureScan*(reg: var PassRegistry) =
  ## Fills `CirModule.features` with the flags the GLSL extension block maps.
  reg.register("glslScanModuleFeatures", pkAnalysis,
    "Scans the module once for the features the GLSL extension block maps:" &
    " float and integer widths, subgroups, shuffles, atomic float, unroll loops",
    run = proc(m: CirModule): string {.nimcall.} =
      scanModuleRequirements(m, m.features, glslProfile))

proc registerOpenclFeatureScan*(reg: var PassRegistry) =
  ## Fills `CirModule.features` with the flags the OpenCL pragma block maps.
  reg.register("openclScanModuleFeatures", pkAnalysis,
    "Scans the module once for the features the OpenCL pragma block maps:" &
    " float widths, subgroups, atomic float",
    run = proc(m: CirModule): string {.nimcall.} =
      scanModuleRequirements(m, m.features, openclProfile))

proc registerWgslFeatureScan*(reg: var PassRegistry) =
  ## Fills `CirModule.features` with the `enable f16` trigger of the WGSL module directive.
  reg.register("wgslScanModuleFeatures", pkAnalysis,
    "Scans the module once for 16-bit float values, the WGSL `enable f16` trigger",
    run = proc(m: CirModule): string {.nimcall.} =
      scanModuleRequirements(m, m.features, wgslProfile))

proc registerCudaFeatureScan*(reg: var PassRegistry) =
  ## Fills `CirModule.features` with the warp-reduction kinds the CUDA templates emit.
  reg.register("cudaScanModuleFeatures", pkAnalysis,
    "Scans the proc bodies once for the reduction kinds the warp-reduction" &
    " templates emit",
    run = proc(m: CirModule): string {.nimcall.} =
      scanModuleRequirements(m, m.features, cudaProfile))
