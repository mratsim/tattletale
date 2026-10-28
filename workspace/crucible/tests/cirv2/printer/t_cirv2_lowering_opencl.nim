# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## OpenCL lowering fixtures: hand-built CIRv2 modules run through the registered
## OpenCL pipeline (the registry `codegenOpenCL` builds), the per-target lowering
## transform relocates `{.const_mem.}` kernel locals into `CirModule.globals`
## and the printer emits them as program-scope `__constant` declarations.
##
## The modules build and run at compile time, the pipeline and the printer are
## the same VM code the `opencl:` macro runs. Omitting the lowering pass fails
## the relocation fixtures, a surviving const_mem local drops at the printer.
##
## - Run: `nim c -r --outdir:build/tests tests/cirv2/printer/t_cirv2_lowering_opencl.nim`

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
import workspace/crucible/src/codegen/passes/passes_lowering_opencl
import workspace/crucible/src/codegen/passes/passes_validations
import workspace/crucible/src/codegen/backends/printer_opencl

proc openclPipeline(m: CirModule): tuple[src: string, err: string] =
  ## OpenCL pipeline from the registry `codegenOpenCL`, shared stages first:
  ##
  ## - per-target rename
  ## - the OpenCL lowering pass
  ## - validation
  ## - the printer
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("openclReservedIdentifiers", OpenclReserved)
  reg.registerOpenclLoweringPasses()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    return ("", perr)
  let (src, serr) = printOpenclModule(m)
  return (src, serr)

proc barePipeline(m: CirModule): tuple[src: string, err: string] =
  ## Pipeline with the OpenCL lowering pass skipped, what the printer alone
  ## does with the const_mem shapes the pass relocates.
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("openclReservedIdentifiers", OpenclReserved)
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    return ("", perr)
  let (src, serr) = printOpenclModule(m)
  return (src, serr)

# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

proc buildConstModule(): CirModule =
  ## kernel: a const_mem local with an initializer, read in the body,
  ## destined for a program-scope `__constant` declaration
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_lower_const", @[
    param("dst", aptrD)], tVoid(), @[
    varD("c", f32, fltLit(1.5, f32), caConstant),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("c", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_const", @[], @[kern])

proc buildPlainModule(): CirModule =
  ## kernel: no const_mem local, the pass leaves the body untouched
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_lower_plain", @[
    param("dst", aptrD)], tVoid(), @[
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), fltLit(2.0, f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_plain", @[], @[kern])

proc buildConstNoInitModule(): CirModule =
  ## kernel: a const_mem local without an initializer, kept alive by a body
  ## read so the unused-local passes leave it for the hoist pass to refuse
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_lower_constnoinit", @[
    param("dst", aptrD)], tVoid(), @[
    varD("c", f32, nil, caConstant),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("c", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_constnoinit", @[], @[kern])

proc emitPipe(build: proc(): CirModule): tuple[src: string, err: string] =
  ## VM pipeline run for one fixture module, the entry point for the consts.
  let m = build()
  result = openclPipeline(m)

proc emitBareErr(build: proc(): CirModule): string =
  ## Refusal of the pass-less pipeline for one fixture module.
  let m = build()
  result = barePipeline(m).err

proc emitVerified(build: proc(): CirModule): string =
  ## Verification message of one fixture module after the full pipeline.
  let m = build()
  discard openclPipeline(m)
  result = verifyMessage(m)

const
  constPipe = emitPipe(buildConstModule)
  plainPipe = emitPipe(buildPlainModule)
  constNoInitErr = openclPipeline(buildConstNoInitModule()).err
  bareConstErr = emitBareErr(buildConstModule)
  constVerified = emitVerified(buildConstModule)

# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════

suite "OpenCL lowering fixtures":

  test "const_mem local relocates to a program-scope __constant declaration":
    check constPipe.err == ""
    check "__constant float c = 1.5f;" in constPipe.src
    # relocated declaration emits after the typedefs, before the kernel
    check constPipe.src.find("__constant float c = 1.5f;") <
      constPipe.src.find("__kernel void")
    # declaration left the kernel body, the body read stays
    check constPipe.src.count("float c") == 1
    check "dst[0] = c;" in constPipe.src
    check constVerified == ""

  test "plain module passes through untouched":
    check plainPipe.err == ""
    check "__constant" notin plainPipe.src
    check "dst[0] = 2.0f;" in plainPipe.src

  test "loud refusal: const_mem local without an initializer":
    check constNoInitErr.contains("const_mem local c without an initializer")

  test "bare pipeline refuses a surviving const_mem local":
    ## Control for the transform: without the pass the const_mem declaration
    ## survives to the printer and the refusal names the owning pass.
    check bareConstErr.contains("in constant space, the OpenCL lowering" &
      " pass hoists const_mem locals to program scope")

# ═══════════════════════════════════════════════════════════════════════
#  Const_mem survivor fixtures
# ═══════════════════════════════════════════════════════════════════════

proc buildSurvivorModule(): CirModule =
  ## kernel + device function, three const_mem declaration sites:
  ##
  ## - kernel top level: `cmBias`, read in both `if` branches
  ## - nested scope: `cmMult`, declared inside the taken branch
  ## - device function: `cmK`, declared in `helper_scale`
  ##
  ## Every site relocates to a program-scope `__constant` declaration.
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
  let kern = procD("kern_lower_survivor", @[
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
  moduleD("lower_survivor", @[], @[helper, kern])

const
  survivorPipe = emitPipe(buildSurvivorModule)
  survivorBareErr = emitBareErr(buildSurvivorModule)
  survivorVerified = emitVerified(buildSurvivorModule)

suite "OpenCL lowering fixtures: const_mem survivors":

  test "nested-scope and device-fn const_mem relocate to program scope":
    check survivorPipe.err == ""
    # every relocation site emits at program scope, before any function
    check "__constant float cmBias = 1.5f;" in survivorPipe.src
    check "__constant float cmMult = 2.0f;" in survivorPipe.src
    check "__constant float cmK = 3.0f;" in survivorPipe.src
    check survivorPipe.src.find("__constant float cmK") <
      survivorPipe.src.find("helper_scale")
    # declarations left their bodies, the reads stay
    check survivorPipe.src.count("float cmBias") == 1
    check survivorPipe.src.count("float cmMult") == 1
    check survivorPipe.src.count("float cmK") == 1
    check "dst[0] = cmBias;" in survivorPipe.src
    check survivorVerified == ""

  test "bare pipeline refuses a surviving const_mem local":
    ## Control for the transform: without the pass the const_mem declarations
    ## drop at the printer and the body reads reference undeclared names.
    check survivorBareErr.contains("in constant space, the OpenCL lowering" &
      " pass hoists const_mem locals to program scope")
