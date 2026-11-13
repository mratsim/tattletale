# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/sets
import ../ir/cirv2_lang
import ./pass_datatypes
import ./pass_helpers

# ############################################################
#
#            Module requirements analysis
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Shared walker
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
  fp16AsStorage: bool
  ## OpenCL stores fp16/bf16 values as 16-bit bit patterns in a `ushort`.
  ## That storage works without the `cl_khr_fp16` extension, the opencl profile
  ## skips the `fp16` detector.
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
  atomicFloatViaPattern: true, fp16AsStorage: true)

const wgslProfile = ScanProfile(
  intWidths: false, shuffleNames: false, unrollAttr: false,
  atomicFloatViaPattern: false)

const cudaProfile = ScanProfile(
  reductions: true)

proc isFloat32Pointee(t: CirType): bool =
  t != nil and t.kind in {ctPtr, ctAptr} and t.pElem != nil and
    t.pElem.kind == ctFloat and t.pElem.bits == 32

proc scanType(f: var ModuleFeatures, t: CirType, profile: ScanProfile) =
  if t.isNil:
    return
  case t.kind
  of ctFloat:
    if t.bits == 64:
      f.fp64 = true
    if t.bits == 16 and not profile.fp16AsStorage:
      f.fp16 = true
  of ctInt, ctUint:
    if profile.intWidths:
      if t.bits == 8:
        f.int8 = true
      if t.bits == 16:
        f.int16 = true
      if t.bits == 64:
        f.int64 = true
  of ctVector: scanType(f, t.vElem, profile)
  of ctArray: scanType(f, t.aElem, profile)
  of ctPtr, ctAptr: scanType(f, t.pElem, profile)
  of ctStruct:
    for fl in t.fields:
      scanType(f, fl.typ, profile)
  of ctBf16:
    # no target needs a module feature for bfloat16
    discard
  of ctVoid, ctBool, ctProc, ctSpan, ctString:
    discard

proc scanCallee(f: var ModuleFeatures, callee: CirExpr, args: seq[CirExpr], profile: ScanProfile) =
  ## Both call shapes share this helper. The caller passes the callee and the arguments of either shape.
  ##
  ## Requires: an `atomicFetchAdd` callee supplies one pointer argument.
  if callee.kind != eSym or not callee.builtin:
    return
  if callee.name in ["reduceAdd", "reduceMax"]:
    if profile.reductions:
      f.reductions.incl callee.name
    else:
      f.subgroup = true
  if profile.shuffleNames and callee.name in ["simd_shuffle", "simd_shuffle_xor"]:
    f.subgroupShuffle = true
  if callee.nilBuiltin == cbkIsNil:
    # GLSL prints the check as a `uint64_t` comparison, which needs the int64
    # extension even when the module declares no 64-bit type.
    f.int64 = true
  if callee.name == "atomicFetchAdd":
    var pt = getExprType(args[0])
    if profile.atomicFloatViaPattern and not pt.isFloat32Pointee() and
        args[0].kind == ePat:
      let bt = getExprType(args[0].patBase)
      if bt.isFloat32Pointee():
        pt = bt
    if pt.isFloat32Pointee():
      f.atomicFloat = true

proc scanExprNode(f: var ModuleFeatures, e: CirExpr, profile: ScanProfile) =
  if e.kind == eCall and e.callee.kind == eSym:
    scanCallee(f, e.callee, e.args, profile)
  scanType(f, getExprType(e), profile)

proc scanStmts(f: var ModuleFeatures, ss: seq[CirStmt], profile: ScanProfile) =
  var features = f
  let onExpr = proc(e: CirExpr) =
    scanExprNode(features, e, profile)
  let onStmt = proc(s: CirStmt) =
    case s.kind
    of sVar:
      scanType(features, s.declType, profile)
    of sWhile:
      if profile.unrollAttr and s.whileUnroll:
        features.controlFlow = true
    of sCall:
      scanCallee(features, s.callCallee, s.callArgs, profile)
    else:
      discard
  ss.walkStmts(onStmt, onExpr)
  f = features

proc scanModuleRequirements(f: var ModuleFeatures, m: CirModule, profile: ScanProfile) =
  for t in m.types:
    scanType(f, t, profile)
  # scan coverage includes the program-scope declarations a lowering pass
  # relocated out of a kernel body, initializers included
  scanStmts(f, m.globals, profile)
  for fn in m.procs:
    scanType(f, fn.ret, profile)
    for p in fn.params:
      scanType(f, p.typ, profile)
    scanStmts(f, fn.body, profile)

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

proc registerGlslFeatureScan*(reg: var PassRegistry) =
  reg.register("glslScanModuleFeatures", pkAnalysis,
    "Scans the module once for the features the GLSL extension block maps:" &
    " float and integer widths, subgroups, shuffles, atomic float, unroll loops",
    run = proc(m: CirModule) {.nimcall.} =
      scanModuleRequirements(m.features, m, glslProfile))

proc registerOpenclFeatureScan*(reg: var PassRegistry) =
  reg.register("openclScanModuleFeatures", pkAnalysis,
    "Scans the module once for the features the OpenCL pragma block maps:" &
    " float widths, subgroups, atomic float",
    run = proc(m: CirModule) {.nimcall.} =
      scanModuleRequirements(m.features, m, openclProfile))

proc registerWgslFeatureScan*(reg: var PassRegistry) =
  reg.register("wgslScanModuleFeatures", pkAnalysis,
    "Scans the module once for 16-bit float values, the WGSL `enable f16` trigger",
    run = proc(m: CirModule) {.nimcall.} =
      scanModuleRequirements(m.features, m, wgslProfile))

proc registerCudaFeatureScan*(reg: var PassRegistry) =
  reg.register("cudaScanModuleFeatures", pkAnalysis,
    "Scans the proc bodies once for the reduction kinds the warp-reduction" &
    " templates emit",
    run = proc(m: CirModule) {.nimcall.} =
      scanModuleRequirements(m.features, m, cudaProfile))
