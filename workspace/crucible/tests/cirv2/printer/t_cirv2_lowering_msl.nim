# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## MSL lowering fixtures: hand-built CIRv2 modules run through the registered
## Metal pipeline (the registry `codegenMetal` builds), the per-target lowering
## transform relocates `{.smem.}` kernel locals into trailing `threadgroup` parameters.
##
## Like the printer fixtures, the modules build and run at compile time:
## the pipeline and the printer are the same VM code the `metal:` macro runs.
##
## Without the MSL lowering pass the emission fixtures fail, the bare pipeline tests check the raw output the pass replaces.
##
## No refusal lives in the pass, today's smem handling discards an initializer
## on a relocated local and drops a device-function smem local without any parameter.
## Device-function output has a dedicated same-text test.
##
## - Run: `nim c -r --outdir:build/tests tests/cirv2/printer/t_cirv2_lowering_msl.nim`

import std/[strutils, unittest]
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/ir/cirv2_verify
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_legalizations
import workspace/crucible/src/codegen/passes/passes_normalizations
import workspace/crucible/src/codegen/passes/passes_optimizations
import workspace/crucible/src/codegen/passes/passes_lowering
import workspace/crucible/src/codegen/passes/passes_low_reserved
import workspace/crucible/src/codegen/passes/passes_lowering_msl
import workspace/crucible/src/codegen/passes/passes_validations
import workspace/crucible/src/codegen/backends/printer_msl

proc mslPipeline(m: CirModule): tuple[src: string, err: string] =
  ## Metal pipeline from the registry `codegenMetal`, shared stages first:
  ##
  ## - per-target rename
  ## - the MSL lowering pass
  ## - validation
  ## - the printer
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("metalReservedIdentifiers", MetalReserved)
  reg.registerMslLoweringPasses()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    return ("", perr)
  let (src, serr) = printModule(m)
  return (src, serr)

proc barePipeline(m: CirModule): tuple[src: string, err: string] =
  ## Pipeline with the MSL lowering pass skipped, what the printer alone
  ## does with the smem shapes the pass relocates.
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("metalReservedIdentifiers", MetalReserved)
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    return ("", perr)
  let (src, serr) = printModule(m)
  return (src, serr)

# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

proc buildSmemModule(): CirModule =
  ## kernel: a smem scratch array and a smem scalar, read and written by the kernel body,
  ## both destined for trailing `threadgroup` parameters
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let scratchTy = tArray(8, f32)
  let kern = procD("kern_lower_smem", @[
    param("dst", aptrD)], tVoid(), @[
    varD("scratch", scratchTy, nil, caThreadgroup),
    varD("scale", f32, nil, caThreadgroup),
    asgnS(atE(sym("scratch", scratchTy), intLit(0, i32)),
      mulE(f32, sym("scale", f32), fltLit(3.0, f32))),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      atE(sym("scratch", scratchTy), intLit(0, i32))),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_smem", @[], @[kern])

proc buildSmemBuiltinModule(): CirModule =
  ## kernel: a smem array plus a coordinate builtin read, the signature order
  ## is regular parameters with buffer slots, the threadgroup parameters,
  ## then the builtin attributed parameters
  let f32 = tFloat(32)
  let u32 = tUint(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let scratchTy = tArray(4, u32)
  let coordTy = tStruct("uvec3", fld("x", u32), fld("y", u32), fld("z", u32))
  let kern = procD("kern_lower_smembuiltin", @[
    param("dst", aptrD)], tVoid(), @[
    varD("scratch", scratchTy, nil, caThreadgroup),
    varD("gx", u32, dotE(sym("thread_position_in_grid", coordTy,
      builtin = true, coordBuiltin = cbkThreadPositionInGrid), "x")),
    asgnS(atE(sym("scratch", scratchTy), intLit(0, i32)), sym("gx", u32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_smembuiltin", @[coordTy], @[kern])

proc buildDeviceSmemModule(): CirModule =
  ## device function: a top-level smem local, both pipelines drop it without
  ## appending a parameter
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let scratchTy = tArray(4, f32)
  let helper = procD("helper_scratch", @[param("dst", aptrD)], tVoid(), @[
    varD("scratch", scratchTy, nil, caThreadgroup),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      atE(sym("scratch", scratchTy), intLit(0, i32))),
    retS(),
  ])
  moduleD("lower_devsmem", @[], @[helper])

proc emitPipe(build: proc(): CirModule): tuple[src: string, err: string] =
  ## VM pipeline run for one fixture module, the entry point for the consts.
  let m = build()
  result = mslPipeline(m)

proc emitBare(build: proc(): CirModule): string =
  ## VM run of the pass-less pipeline for one fixture module.
  let m = build()
  result = barePipeline(m).src

proc emitVerified(build: proc(): CirModule): string =
  ## Verification message of one fixture module after the full pipeline.
  let m = build()
  discard mslPipeline(m)
  result = verifyMessage(m)

const
  smemPipe = emitPipe(buildSmemModule)
  smemBuiltinPipe = emitPipe(buildSmemBuiltinModule)
  deviceSmemPipe = emitPipe(buildDeviceSmemModule)
  deviceSmemBare = emitBare(buildDeviceSmemModule)
  bareSmemSrc = emitBare(buildSmemModule)
  smemVerified = emitVerified(buildSmemModule)

# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════

suite "MSL lowering fixtures":

  test "smem locals relocate into trailing threadgroup parameters":
    check smemPipe.err == ""
    # array local takes the pointer form, the scalar local the reference form,
    # both trail the regular parameter and take no buffer slot
    check "kernel void kern_lower_smem(device float* dst [[buffer(0)]]," &
      " threadgroup float* scratch, threadgroup float& scale)" in smemPipe.src
    # declarations left the kernel body, the reads stay
    check "threadgroup float" notin smemPipe.src.replace(
      "threadgroup float* scratch, threadgroup float& scale", "")
    check "scratch[0] = (scale * 3.0f);" in smemPipe.src
    check "dst[0] = scratch[0];" in smemPipe.src
    check smemVerified == ""

  test "threadgroup parameters sit before the builtin attributed parameters":
    check smemBuiltinPipe.err == ""
    check "device float* dst [[buffer(0)]], threadgroup uint* scratch," &
      " uint3 thread_position_in_grid [[thread_position_in_grid]])" in
      smemBuiltinPipe.src

  test "device function smem handling matches the bare printer":
    ## Pass output on a device function: the smem declaration drops and no
    ## parameter is appended, the same text the bare printer produces.
    check deviceSmemPipe.err == ""
    check deviceSmemPipe.src == deviceSmemBare
    check "threadgroup" notin deviceSmemPipe.src

  test "bare pipeline leaves no threadgroup parameter behind":
    ## Control for the transform: without the pass the smem declarations drop
    ## at the printer and the kernel signature keeps only the regular parameter.
    check "threadgroup" notin bareSmemSrc
    check "kernel void kern_lower_smem(device float* dst [[buffer(0)]])" in
      bareSmemSrc

# ═══════════════════════════════════════════════════════════════════════
#  Const_mem survivor fixtures
# ═══════════════════════════════════════════════════════════════════════

proc buildConstModule(): CirModule =
  ## kernel + device function, three const_mem declaration sites:
  ##
  ## - kernel top level: `cmBias`, read in both `if` branches
  ## - nested scope: `cmMult`, declared inside the taken branch
  ## - device function: `cmK`, declared in `helper_scale`
  ##
  ## Every site relocates to a program-scope `constant` declaration.
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let sumTy = tProcT(f32, f32)
  let helper = procD("helper_scale", @[param("v", f32)], f32, @[
    varD("cmK", f32, fltLit(3.0, f32), caConstant),
    varD("result", f32, nil),
    asgnS(sym("result", f32), mulE(f32, sym("v", f32), sym("cmK", f32))),
    retS(sym("result", f32)),
  ])
  let kern = procD("kern_lower_const", @[
    param("dst", aptrD)], tVoid(), @[
    varD("cmBias", f32, fltLit(1.5, f32), caConstant),
    iteS(ltE(fltLit(0.0, f32), patE(sym("dst", aptrD), intLit(0, i32))), @[
      varD("cmMult", f32, fltLit(2.0, f32), caConstant),
      asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
        addE(f32, mulE(f32, patE(sym("dst", aptrD), intLit(0, i32)),
          sym("cmMult", f32)), sym("cmBias", f32))),
    ], @[
      asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("cmBias", f32)),
    ]),
    asgnS(patE(sym("dst", aptrD), intLit(1, i32)),
      callE(sym("helper_scale", sumTy), @[patE(sym("dst", aptrD), intLit(1, i32))])),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_const", @[], @[helper, kern])

proc buildConstNoInitModule(): CirModule =
  ## kernel: a const_mem local without an initializer, kept alive by a body
  ## read so the unused-local passes leave it for the hoist pass to refuse
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_lower_constnoinit", @[
    param("dst", aptrD)], tVoid(), @[
    varD("cmBias", f32, nil, caConstant),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("cmBias", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_constnoinit", @[], @[kern])

proc emitBareErr(build: proc(): CirModule): string =
  ## Refusal of the pass-less pipeline for one fixture module.
  let m = build()
  result = barePipeline(m).err

const
  constPipe = emitPipe(buildConstModule)
  constNoInitErr = mslPipeline(buildConstNoInitModule()).err
  constBareErr = emitBareErr(buildConstModule)
  constVerified = emitVerified(buildConstModule)

suite "MSL lowering fixtures: const_mem survivors":

  test "const_mem locals relocate to program-scope constant declarations":
    check constPipe.err == ""
    # every relocation site emits at program scope, before any function body
    check "constant float cmBias = 1.5f;" in constPipe.src
    check "constant float cmMult = 2.0f;" in constPipe.src
    check "constant float cmK = 3.0f;" in constPipe.src
    check constPipe.src.find("constant float cmK") <
      constPipe.src.find("helper_scale")
    # declarations left their bodies, the reads stay
    check constPipe.src.count("float cmBias") == 1
    check constPipe.src.count("float cmMult") == 1
    check constPipe.src.count("float cmK") == 1
    check "dst[0] = cmBias;" in constPipe.src
    check constVerified == ""

  test "bare pipeline refuses a surviving const_mem local":
    ## Control for the transform: without the pass the const_mem declarations
    ## survive to the printer and the refusal names the owning pass.
    check constBareErr.contains("in constant space, the MSL lowering" &
      " pass hoists const_mem locals to program scope")

  test "loud refusal: const_mem local without an initializer":
    check constNoInitErr.contains("const_mem local cmBias without an initializer")
