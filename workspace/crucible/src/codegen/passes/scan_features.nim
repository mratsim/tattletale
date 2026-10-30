# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Module feature scan: one `pkAnalysis` pass per text target, filling
## `CirModule.features` with the module's feature metadata.
##
## Contract:
## - each target registers its scan in `gpu_compiler.nim` after its last transform,
##   before check-only validation
## - analysis results are valid only if no transform runs after them:
##   the registry runs in registration order and never invalidates
## - the printer reads the stored record, it never rescans
##
## Detector set per target:
## - GLSL and OpenCL share one walker whose profile selects the flags each
##   target's extension block consumes
## - the profiles differ on one detector: an `atomicFetchAdd` address operand
##   indexing its pointee (`buf[i]`) names a float pointer for OpenCL,
##   not for GLSL
## - WGSL: type-only, the `enable f16` trigger
## - CUDA: warp-reduction template kinds

import ../ir/cirv2_lang
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Shared GLSL/OpenCL scan
# ═══════════════════════════════════════════════════════════════════════

type ScanProfile = object
  ## Detector knobs of one target's extension block.
  intWidths: bool
  ## GLSL maps 8/16/64-bit integers, so their presence selects extension lines.
  shuffleNames: bool
  ## GLSL maps `simd_shuffle`/`simd_shuffle_xor` to subgroup shuffles.
  unrollAttr: bool
  ## GLSL emits `{.unroll.}` loops through an extension-gated attribute.
  atomicFloatViaPattern: bool
  ## An `atomicFetchAdd` address operand that indexes its pointee
  ## (`buf[i]`) names a float pointer for OpenCL.

const glslProfile = ScanProfile(
  intWidths: true, shuffleNames: true, unrollAttr: true,
  atomicFloatViaPattern: false)

const openclProfile = ScanProfile(
  intWidths: false, shuffleNames: false, unrollAttr: false,
  atomicFloatViaPattern: true)

proc isFloat32Pointee(t: CirType): bool =
  ## True when `t` is a pointer whose pointee is a 32-bit float.
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
  of ctVoid, ctBool, ctProc:
    discard

proc scanExpr(e: CirExpr, f: var ModuleFeatures, profile: ScanProfile) =
  if e.isNil:
    return
  if e.kind == eCall and e.callee.kind == eSym:
    if e.callee.reductionBuiltin in {rbkReduceAdd, rbkReduceMax}:
      f.subgroup = true
    if profile.shuffleNames and e.callee.builtin and
        e.callee.name in ["simd_shuffle", "simd_shuffle_xor"]:
      f.subgroupShuffle = true
    if e.callee.atomicsBuiltin == abkAtomicFetchAdd:
      var pt = getExprType(e.args[0])
      if profile.atomicFloatViaPattern and not pt.isFloat32Pointee() and
          e.args[0].kind == ePat:
        let bt = getExprType(e.args[0].patBase)
        if bt.isFloat32Pointee():
          pt = bt
      if pt.isFloat32Pointee():
        f.atomicFloat = true
  for c in getExprChildren(e):
    scanExpr(c, f, profile)
  scanType(getExprType(e), f, profile)

proc scanStmts(ss: seq[CirStmt], f: var ModuleFeatures, profile: ScanProfile) =
  for s in ss:
    case s.kind
    of sVar, sConst:
      scanType(s.declType, f, profile)
      if s.init != nil:
        scanExpr(s.init, f, profile)
    of sAsgn:
      scanExpr(s.lhs, f, profile)
      scanExpr(s.rhs, f, profile)
    of sIte:
      scanExpr(s.cond, f, profile)
      scanStmts(s.thenBody, f, profile)
      scanStmts(s.elseBody, f, profile)
    of sLoop:
      if profile.unrollAttr and s.unroll:
        f.controlFlow = true
      scanStmts(s.loopBody, f, profile)
    of sScope:
      scanStmts(s.scopeBody, f, profile)
    of sRet:
      if s.retVal != nil:
        scanExpr(s.retVal, f, profile)
    of sCall:
      scanExpr(s.callCallee, f, profile)
      for a in s.callArgs:
        scanExpr(a, f, profile)
    of sAssume:
      scanExpr(s.assumeCond, f, profile)
    of sJmp, sLab, sDiscard, sCase, sAsm, sEmit:
      discard

proc scanModuleFeatures(m: CirModule, f: var ModuleFeatures, profile: ScanProfile) =
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
# WGSL scan: type-only
# ═══════════════════════════════════════════════════════════════════════

proc wgslScanType(t: CirType, f: var ModuleFeatures) =
  if t.isNil:
    return
  case t.kind
  of ctFloat:
    if t.bits == 16:
      f.fp16 = true
  of ctVector: wgslScanType(t.vElem, f)
  of ctArray: wgslScanType(t.aElem, f)
  of ctPtr, ctAptr: wgslScanType(t.pElem, f)
  of ctStruct:
    for fl in t.fields:
      wgslScanType(fl.typ, f)
  of ctVoid, ctBool, ctInt, ctUint, ctProc:
    discard

proc wgslScanStmts(ss: seq[CirStmt], f: var ModuleFeatures) =
  for s in ss:
    case s.kind
    of sVar, sConst:
      wgslScanType(s.declType, f)
      if s.init != nil:
        wgslScanType(getExprType(s.init), f)
    of sAsgn:
      wgslScanType(getExprType(s.lhs), f)
      wgslScanType(getExprType(s.rhs), f)
    of sIte:
      wgslScanStmts(s.thenBody, f)
      wgslScanStmts(s.elseBody, f)
    of sLoop:
      wgslScanStmts(s.loopBody, f)
    of sScope:
      wgslScanStmts(s.scopeBody, f)
    of sRet:
      if s.retVal != nil:
        wgslScanType(getExprType(s.retVal), f)
    of sCall, sJmp, sLab, sDiscard, sAssume, sCase, sAsm, sEmit:
      discard

proc wgslScanFeatures(m: CirModule): string {.nimcall.} =
  var f = m.features
  for t in m.types:
    wgslScanType(t, f)
  for fn in m.procs:
    wgslScanType(fn.ret, f)
    for p in fn.params:
      wgslScanType(p.typ, f)
    wgslScanStmts(fn.body, f)
  wgslScanStmts(m.globals, f)
  m.features = f

# ═══════════════════════════════════════════════════════════════════════
# CUDA scan: warp-reduction kinds
# ═══════════════════════════════════════════════════════════════════════

proc cudaScanStmts(ss: seq[CirStmt], f: var ModuleFeatures)

proc cudaScanExpr(e: CirExpr, f: var ModuleFeatures) =
  ## Collects the reduction kinds one expression tree needs.
  ## Warp-reduction templates emit at module scope when a proc body
  ## references one kind.
  if e.isNil:
    return
  if e.kind == eCall and e.callee.kind == eSym:
    case e.callee.reductionBuiltin
    of rbkReduceAdd:
      f.reductions.incl rbkReduceAdd
    of rbkReduceMax:
      f.reductions.incl rbkReduceMax
    of rbkNone:
      discard
  for ch in getExprChildren(e):
    cudaScanExpr(ch, f)

proc cudaScanStmts(ss: seq[CirStmt], f: var ModuleFeatures) =
  for s in ss:
    case s.kind
    of sVar, sConst:
      cudaScanExpr(s.init, f)
    of sAsgn:
      cudaScanExpr(s.lhs, f)
      cudaScanExpr(s.rhs, f)
    of sIte:
      cudaScanExpr(s.cond, f)
      cudaScanStmts(s.thenBody, f)
      cudaScanStmts(s.elseBody, f)
    of sLoop:
      cudaScanStmts(s.loopBody, f)
    of sScope:
      cudaScanStmts(s.scopeBody, f)
    of sAssume:
      cudaScanExpr(s.assumeCond, f)
    of sRet:
      cudaScanExpr(s.retVal, f)
    of sDiscard:
      cudaScanExpr(s.discardVal, f)
    of sCall:
      for a in s.callArgs:
        cudaScanExpr(a, f)
    of sJmp, sLab, sCase, sAsm, sEmit:
      discard

proc cudaScanFeatures(m: CirModule): string {.nimcall.} =
  var f = m.features
  for fn in m.procs:
    cudaScanStmts(fn.body, f)
  m.features = f

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

proc glslRun(m: CirModule): string {.nimcall.} =
  scanModuleFeatures(m, m.features, glslProfile)

proc openclRun(m: CirModule): string {.nimcall.} =
  scanModuleFeatures(m, m.features, openclProfile)

proc registerGlslFeatureScan*(reg: var PassRegistry) =
  ## Registers the GLSL feature scan over CIRv2 modules.
  ##
  ## Contract:
  ## - fills `CirModule.features` with the flags the GLSL extension block maps
  ## - registers after the target's last transform, before the check-only
  ##   validation stage: analysis results are valid only if no transform
  ##   runs after them
  reg.register("glslScanModuleFeatures", pkAnalysis, phaseMain,
    "Scans the module once for the features the GLSL extension block maps:" &
    " float and integer widths, subgroups, shuffles, atomic float, unroll loops",
    run = glslRun)

proc registerOpenclFeatureScan*(reg: var PassRegistry) =
  ## Registers the OpenCL feature scan over CIRv2 modules.
  ##
  ## Contract:
  ## - fills `CirModule.features` with the flags the OpenCL extension pragma
  ##   block maps
  ## - registers after the target's last transform, before the check-only
  ##   validation stage: analysis results are valid only if no transform
  ##   runs after them
  reg.register("openclScanModuleFeatures", pkAnalysis, phaseMain,
    "Scans the module once for the features the OpenCL pragma block maps:" &
    " float widths, subgroups, atomic float",
    run = openclRun)

proc registerWgslFeatureScan*(reg: var PassRegistry) =
  ## Registers the WGSL feature scan over CIRv2 modules.
  ##
  ## Contract:
  ## - fills `CirModule.features` with the `enable f16` trigger of the WGSL
  ##   module directive
  ## - registers after the target's last transform, before the check-only
  ##   validation stage: analysis results are valid only if no transform
  ##   runs after them
  reg.register("wgslScanModuleFeatures", pkAnalysis, phaseMain,
    "Scans the module once for 16-bit float values, the WGSL `enable f16` trigger",
    run = wgslScanFeatures)

proc registerCudaFeatureScan*(reg: var PassRegistry) =
  ## Registers the CUDA feature scan over CIRv2 modules.
  ##
  ## Contract:
  ## - fills `CirModule.features` with the warp-reduction kinds the CUDA
  ##   templates emit
  ## - registers after the target's last transform, before the check-only
  ##   validation stage: analysis results are valid only if no transform
  ##   runs after them
  reg.register("cudaScanModuleFeatures", pkAnalysis, phaseMain,
    "Scans the proc bodies once for the reduction kinds the warp-reduction" &
    " templates emit",
    run = cudaScanFeatures)
