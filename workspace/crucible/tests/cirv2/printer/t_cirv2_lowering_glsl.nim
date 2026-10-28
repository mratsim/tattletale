# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GLSL lowering fixtures: hand-built CIRv2 modules run through the registered
## GLSL pipeline (the registry `codegenGLSL` builds), the per-target lowering
## transforms dissolve pointer locals and hoist program-scope declarations.
##
## Like the printer fixtures, the modules build and run at compile time:
## the pipeline and the printer are the same VM code the `glsl:` macro runs.
##
## Without the GLSL lowering passes the emission fixtures fail, the bare printer
## refuses the surviving shapes naming the pass, the "bare pipeline" tests
## check that refusal.
##
## - Run: `nim c -r --outdir:build/tests tests/cirv2/printer/t_cirv2_lowering_glsl.nim`

import std/[os, osproc, strutils, unittest]
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/ir/cirv2_pretty
import workspace/crucible/src/codegen/ir/cirv2_verify
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_legalizations
import workspace/crucible/src/codegen/passes/passes_normalizations
import workspace/crucible/src/codegen/passes/passes_optimizations
import workspace/crucible/src/codegen/passes/passes_lowering
import workspace/crucible/src/codegen/passes/passes_low_reserved
import workspace/crucible/src/codegen/passes/passes_lowering_glsl
import workspace/crucible/src/codegen/passes/passes_validations
import workspace/crucible/src/codegen/backends/printer_glsl

proc glslPipeline(m: CirModule): tuple[src: string, err: string] =
  ## GLSL pipeline from the registry `codegenGLSL`, shared stages first:
  ## per-target rename and lowering passes, validation, then the printer.
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
  reg.registerGlslLoweringPasses()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    return ("", perr)
  let (src, serr) = printGlslModule(m)
  return (src, serr)

proc barePipeline(m: CirModule): tuple[src: string, err: string] =
  ## Pipeline with the GLSL lowering passes skipped, what the printer alone
  ## does with the shapes the passes dissolve and hoist.
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    return ("", perr)
  let (src, serr) = printGlslModule(m)
  return (src, serr)

proc validate(name: string, entry: string, src: string): bool =
  ## Runs glslangValidator over the printed source, true when clean.
  ## The flags match the Vulkan engine's ingest: entry point is the kernel name,
  ## source entry point is `main`, subgroup sources pass --target-env vulkan1.1.
  let srcPath = getTempDir() / name & ".comp"
  writeFile(srcPath, src)
  var args = "-V -e " & entry & " --source-entrypoint main"
  if "GL_KHR_shader_subgroup" in src:
    args.add " --target-env vulkan1.1"
  let cmd = "glslangValidator " & args & " " & srcPath
  let (outp, code) = execCmdEx(cmd)
  if code != 0:
    echo "glslangValidator failed for " & name & ":\n" & outp & "\n" & src
  result = code == 0

# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

proc buildAliasModule(): CirModule =
  ## kernel: a ptr local bound to the address of a local struct and dereferenced
  ## on both sides of an assignment through the pointee's array member
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let arrTy = tArray(4, f32)
  let bufTy = tStruct("Buf", fld("data", arrTy))
  let ptrTy = tPtr(caThread, bufTy)
  let kern = procD("kern_lower_alias", @[
    param("dst", aptrD)], tVoid(), @[
    varD("tmp", bufTy, oconstrE(bufTy, kv("data",
      aconstrE(arrTy, fltLit(1.0, f32), fltLit(2.0, f32),
        fltLit(3.0, f32), fltLit(4.0, f32))))),
    varD("p", ptrTy),
    asgnS(sym("p", ptrTy), addrE(ptrTy, sym("tmp", bufTy))),
    asgnS(atE(dotE(derefE(sym("p", ptrTy)), "data"), intLit(2, i32)),
      fltLit(9.0, f32)),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      atE(dotE(derefE(sym("p", ptrTy)), "data"), intLit(2, i32))),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_alias", @[bufTy], @[kern])

proc buildHoistModule(): CirModule =
  ## kernel: a smem scratch array and a const_mem scale, written and read in the body,
  ## both destined for program scope
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let scratchTy = tArray(8, f32)
  let kern = procD("kern_lower_hoist", @[
    param("dst", aptrD)], tVoid(), @[
    varD("scratch", scratchTy, nil, caThreadgroup),
    varD("scale", f32, fltLit(2.0, f32), caConstant),
    asgnS(atE(sym("scratch", scratchTy), intLit(0, i32)),
      mulE(f32, sym("scale", f32), fltLit(3.0, f32))),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      atE(sym("scratch", scratchTy), intLit(0, i32))),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_hoist", @[], @[kern])

proc buildValueReadModule(): CirModule =
  ## kernel: the pointer local read as a value, assigned to the buffer slot
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let arrTy = tArray(4, f32)
  let bufTy = tStruct("Buf", fld("data", arrTy))
  let ptrTy = tPtr(caThread, bufTy)
  let kern = procD("kern_lower_value", @[
    param("dst", aptrD)], tVoid(), @[
    varD("tmp", bufTy, oconstrE(bufTy, kv("data",
      aconstrE(arrTy, fltLit(1.0, f32), fltLit(2.0, f32),
        fltLit(3.0, f32), fltLit(4.0, f32))))),
    varD("p", ptrTy),
    asgnS(sym("p", ptrTy), addrE(ptrTy, sym("tmp", bufTy))),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("p", ptrTy)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_value", @[bufTy], @[kern])

proc buildPendingReadModule(): CirModule =
  ## kernel: the pointer local read through a deref before its address-of binding
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let arrTy = tArray(4, f32)
  let bufTy = tStruct("Buf", fld("data", arrTy))
  let ptrTy = tPtr(caThread, bufTy)
  let kern = procD("kern_lower_pending", @[
    param("dst", aptrD)], tVoid(), @[
    varD("tmp", bufTy, oconstrE(bufTy, kv("data",
      aconstrE(arrTy, fltLit(1.0, f32), fltLit(2.0, f32),
        fltLit(3.0, f32), fltLit(4.0, f32))))),
    varD("p", ptrTy),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      dotE(derefE(sym("p", ptrTy)), "data")),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_pending", @[bufTy], @[kern])

proc buildConstNoInitModule(): CirModule =
  ## kernel: a const_mem local without an initializer, read in the body so the unused-local passes leave it for the hoist pass to refuse
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

proc verifiedAliasModule(): string =
  ## Verify message of the alias module after the pipeline ran over it.
  let m = buildAliasModule()
  discard glslPipeline(m)
  result = verifyMessage(m)

proc hoistedTrace(): string =
  ## Module render of the hoist module after the pipeline ran over it.
  let m = buildHoistModule()
  discard glslPipeline(m)
  result = renderModule(m)

const
  aliasPipe = glslPipeline(buildAliasModule())
  hoistPipe = glslPipeline(buildHoistModule())
  valueErr = glslPipeline(buildValueReadModule()).err
  pendingErr = glslPipeline(buildPendingReadModule()).err
  constNoInitErr = glslPipeline(buildConstNoInitModule()).err
  bareAliasErr = barePipeline(buildAliasModule()).err
  bareHoistErr = barePipeline(buildHoistModule()).err
  aliasVerified = verifiedAliasModule()
  hoistTraceText = hoistedTrace()

# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════

suite "GLSL lowering fixtures":

  test "pointer local dissolves into its pointee local":
    check aliasPipe.err == ""
    check aliasPipe.src.contains("tmp.data[2] = 9.0f;")
    check aliasPipe.src.contains("dst.x[0] = tmp.data[2];")
    # pointer local and its address-of binding are gone
    check not aliasPipe.src.contains(" p;")
    check not aliasPipe.src.contains(" p =")
    check validate("lower_alias", "kern_lower_alias", aliasPipe.src)

  test "dissolved module verifies clean":
    check aliasVerified == ""

  test "smem and const_mem locals hoist to program scope":
    check hoistPipe.err == ""
    check hoistPipe.src.contains("shared float scratch[8];")
    check hoistPipe.src.contains("const float scale = 2.0f;")
    # relocated declarations left the kernel body
    check hoistPipe.src.count("float scratch[8]") == 1
    check hoistPipe.src.count("float scale") == 1
    # module keeps them at program scope, in kernel-body order
    check hoistTraceText.contains("(globals")
    check hoistTraceText.contains("(var threadgroup scratch (array (f +32) 8) .)")
    check hoistTraceText.contains("(var constant scale (f +32) 2.0)")
    check validate("lower_hoist", "kern_lower_hoist", hoistPipe.src)

  test "loud refusal: pointer local used as a value":
    check valueErr.contains("pointer local p used as a value")

  test "loud refusal: pointer local read before its address-of binding":
    check pendingErr.contains("read before its address-of binding")

  test "loud refusal: const_mem local without an initializer":
    check constNoInitErr.contains("const_mem local c without an initializer")

  test "bare pipeline refuses the shapes the lowering pass dissolves":
    ## Refusal contract: without the GLSL lowering passes the printer refuses,
    ## naming the pass that owns the transform.
    check bareAliasErr.contains("the GLSL lowering pass dissolves pointer locals")
    check bareHoistErr.contains("the GLSL lowering pass hoists threadgroup" &
      " locals to program scope")

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
  ## Every site relocates to a program-scope `const` declaration.
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

proc verifiedSurvivorModule(): string =
  ## Verify message of the survivor module after the pipeline ran over it.
  let m = buildSurvivorModule()
  discard glslPipeline(m)
  result = verifyMessage(m)

const
  survivorPipe = glslPipeline(buildSurvivorModule())
  survivorBareErr = barePipeline(buildSurvivorModule()).err
  survivorVerified = verifiedSurvivorModule()

suite "GLSL lowering fixtures: const_mem survivors":

  test "nested-scope and device-fn const_mem hoist to program scope":
    check survivorPipe.err == ""
    # every relocation site emits at program scope, before any function
    check survivorPipe.src.contains("const float cmBias = 1.5f;")
    check survivorPipe.src.contains("const float cmMult = 2.0f;")
    check survivorPipe.src.contains("const float cmK = 3.0f;")
    check survivorPipe.src.find("const float cmK") <
      survivorPipe.src.find("helper_scale")
    # declarations left their bodies, the reads stay
    check survivorPipe.src.count("float cmBias") == 1
    check survivorPipe.src.count("float cmMult") == 1
    check survivorPipe.src.count("float cmK") == 1
    check survivorPipe.src.contains("dst.x[0] = cmBias;")
    check survivorVerified == ""
    check validate("lower_survivor", "kern_lower_survivor", survivorPipe.src)

  test "bare pipeline refuses a surviving const_mem local":
    ## Control for the transform: without the pass the const_mem declaration
    ## survives to the printer and the refusal names the owning pass.
    check survivorBareErr.contains("in constant space, the GLSL lowering" &
      " pass hoists const_mem locals to program scope")
