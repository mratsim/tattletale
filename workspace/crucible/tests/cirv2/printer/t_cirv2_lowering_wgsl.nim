# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## WGSL lowering fixtures: hand-built CIRv2 modules run through the registered
## WGSL pipeline (the registry `codegenWebGpu` builds), the per-target lowering
## transforms dissolve pointer locals, patch bool kernel parameter reads,
## rename double-underscore names and hoist program-scope declarations.
##
## Like the printer fixtures, the modules build and run at compile time:
## the pipeline and the printer are the same VM code the `webgpu:` macro runs.
##
## Without the WGSL lowering passes the emission fixtures fail:
##
## - the bare printer refuses the surviving shapes, naming the pass
## - the "bare pipeline" tests check that refusal and the raw output the pass replaces
##
## There is no offline WGSL validator (the printer fixtures say so too), the real check is the ceramic axpy suite on WebGPU.
##
## - Run: `nim c -r --outdir:build/tests tests/cirv2/printer/t_cirv2_lowering_wgsl.nim`

import std/[os, strutils, unittest]
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
import workspace/crucible/src/codegen/passes/passes_lowering_wgsl
import workspace/crucible/src/codegen/passes/passes_validations
import workspace/crucible/src/codegen/backends/printer_wgsl

proc wgslPipeline(m: CirModule): tuple[src: string, err: string] =
  ## WGSL pipeline from the registry `codegenWebGpu`, shared stages first:
  ##
  ## - per-target rename with the `wgsl_` double-underscore prefix
  ## - the WGSL lowering passes
  ## - validation
  ## - the printer
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("wgslReservedIdentifiers", WgslReserved, "wgsl_")
  reg.registerWgslLoweringPasses()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    return ("", perr)
  let (src, serr) = printWgslModule(m)
  return (src, serr)

proc barePipeline(m: CirModule): tuple[src: string, err: string] =
  ## Pipeline with the WGSL lowering passes skipped, what the printer alone
  ## does with the shapes the passes dissolve, patch and hoist.
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("wgslReservedIdentifiers", WgslReserved, "wgsl_")
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    return ("", perr)
  let (src, serr) = printWgslModule(m)
  return (src, serr)

proc noPrefixPipeline(m: CirModule): tuple[src: string, err: string] =
  ## Pipeline with the rename stripped of its per-target `wgsl_` prefix data,
  ## the WGSL lowering passes still running.
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("wgslReservedIdentifiers", WgslReserved)
  reg.registerWgslLoweringPasses()
  reg.registerValidationPostPasses()
  let perr = runPasses(m, reg)
  if perr.len > 0:
    return ("", perr)
  let (src, serr) = printWgslModule(m)
  return (src, serr)

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

proc buildStructAliasModule(): CirModule =
  ## kernel: a struct local with a device buffer member, the write goes
  ## through the dot chain, the field paths dissolve to the construction's leaf symbols
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let viewTy = tStruct("View", fld("data", aptrD))
  let kern = procD("kern_lower_structalias", @[
    param("dst", aptrD), param("a", f32)], tVoid(), @[
    varD("v", viewTy, oconstrE(viewTy, kv("data", sym("dst", aptrD)))),
    asgnS(patE(dotE(sym("v", viewTy), "data"), intLit(0, i32)),
      sym("a", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_structalias", @[viewTy], @[kern])

proc buildBoolParamModule(): CirModule =
  ## kernel: a bool parameter read in a conditional, passed into a device
  ## function whose own bool parameter stays a real bool
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let noteFn = procD("note_flag", @[param("b", tBool())], tVoid(), @[
    retS(),
  ])
  let kern = procD("kern_lower_boolparam", @[
    param("dst", aptrD), param("flag", tBool())], tVoid(), @[
    iteS(sym("flag", tBool()), @[
      asgnS(patE(sym("dst", aptrD), intLit(0, i32)), fltLit(1.0, f32)),
    ]),
    callS(sym("note_flag", tProcT(@[tBool()], tVoid())),
      sym("flag", tBool())),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_boolparam", @[], @[noteFn, kern])

proc buildMangledNameModule(): CirModule =
  ## kernel: a struct type, a device function, a local and a call site whose
  ## names use the double underscore WGSL reserves, every site gains the `wgsl_` prefix
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let pairTy = tStruct("__Int_pair", fld("lo", i32), fld("hi", i32))
  let helper = procD("__helper", @[param("v", f32)], f32, @[
    retS(mulE(f32, sym("v", f32), fltLit(2.0, f32))),
  ])
  let kern = procD("kern_lower_mangled", @[
    param("dst", aptrD), param("pair", pairTy)], tVoid(), @[
    varD("__tmp", f32, nil),
    asgnS(sym("__tmp", f32), callE(sym("__helper", tProcT(@[f32], f32)),
      convE(f32, dotE(sym("pair", pairTy), "hi")))),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("__tmp", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_mangled", @[pairTy], @[helper, kern])

proc buildHoistModule(): CirModule =
  ## kernel: a smem scratch array and a const_mem scale, written and read
  ## from the kernel body, both destined for module scope
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

proc buildStructWholeModule(): CirModule =
  ## kernel: the pointer-holding struct local used whole, assigned to a buffer slot
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let viewTy = tStruct("View", fld("data", aptrD))
  let kern = procD("kern_lower_structwhole", @[
    param("dst", aptrD)], tVoid(), @[
    varD("v", viewTy, oconstrE(viewTy, kv("data", sym("dst", aptrD)))),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("v", viewTy)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_structwhole", @[viewTy], @[kern])

proc buildSmemInitModule(): CirModule =
  ## kernel: a smem local with an initializer, kept alive by a body read so the hoist pass refuses it
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let scratchTy = tArray(8, f32)
  let kern = procD("kern_lower_smeminit", @[
    param("dst", aptrD)], tVoid(), @[
    varD("scratch", scratchTy, fltLit(0.0, f32), caThreadgroup),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)),
      atE(sym("scratch", scratchTy), intLit(0, i32))),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_smeminit", @[], @[kern])

proc buildConstNoInitModule(): CirModule =
  ## kernel: a const_mem local without an initializer, kept alive by a body read so the hoist pass refuses it
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
  discard wgslPipeline(m)
  result = verifyMessage(m)

proc hoistedTrace(): string =
  ## Module render of the hoist module after the pipeline ran over it.
  let m = buildHoistModule()
  discard wgslPipeline(m)
  result = renderModule(m)

const
  aliasPipe = wgslPipeline(buildAliasModule())
  structAliasPipe = wgslPipeline(buildStructAliasModule())
  boolPipe = wgslPipeline(buildBoolParamModule())
  mangledPipe = wgslPipeline(buildMangledNameModule())
  hoistPipe = wgslPipeline(buildHoistModule())
  valueErr = wgslPipeline(buildValueReadModule()).err
  structWholeErr = wgslPipeline(buildStructWholeModule()).err
  smemInitErr = wgslPipeline(buildSmemInitModule()).err
  constNoInitErr = wgslPipeline(buildConstNoInitModule()).err
  bareAliasErr = barePipeline(buildAliasModule()).err
  bareHoistErr = barePipeline(buildHoistModule()).err
  bareBoolSrc = barePipeline(buildBoolParamModule()).src
  noPrefixMangledSrc = noPrefixPipeline(buildMangledNameModule()).src
  aliasVerified = verifiedAliasModule()
  hoistTraceText = hoistedTrace()

# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════

suite "WGSL lowering fixtures":

  test "pointer local dissolves into its pointee local":
    check aliasPipe.err == ""
    check aliasPipe.src.contains("tmp.data[2] = 9.0;")
    check aliasPipe.src.contains("dst[0] = tmp.data[2];")
    # pointer local and its address-of binding are gone
    check not aliasPipe.src.contains(" p;")
    check not aliasPipe.src.contains(" p =")
    check not aliasPipe.src.contains("(&")
    check aliasVerified == ""

  test "pointer-holding struct local dissolves into its field leaves":
    check structAliasPipe.err == ""
    check structAliasPipe.src.contains("dst[0] = a;")
    check "struct View" notin structAliasPipe.src
    check "var v" notin structAliasPipe.src
    check "v." notin structAliasPipe.src

  test "bool kernel parameter reads patch to name != 0u":
    check boolPipe.err == ""
    # u32 storage declaration stays, the body reads the patched comparison
    check "@group(0) @binding(1) var<storage, read> flag: u32;" in boolPipe.src
    check "if ((flag != 0u)) {" in boolPipe.src
    # device function call site reads the patched comparison
    check "note_flag((flag != 0u));" in boolPipe.src
    # device function's own bool parameter stays a real bool
    check "fn note_flag(b: bool) -> void {" in boolPipe.src
    check "(b != 0u)" notin boolPipe.src

  test "double-underscore names rename at every site":
    check mangledPipe.err == ""
    # struct declaration, binding, device function, local, call site and body
    # reads all use the prefixed form
    check "struct wgsl___Int_pair {" in mangledPipe.src
    check "var<storage, read> pair: wgsl___Int_pair;" in mangledPipe.src
    check "fn wgsl___helper(v: f32) -> f32" in mangledPipe.src
    check "var wgsl___tmp: f32;" in mangledPipe.src
    check "wgsl___tmp = wgsl___helper(f32(pair.hi));" in mangledPipe.src
    check "dst[0] = wgsl___tmp;" in mangledPipe.src
    check "__Int" notin mangledPipe.src.replace("wgsl___Int", "")

  test "smem and const_mem locals hoist to module scope":
    check hoistPipe.err == ""
    check hoistPipe.src.contains("var<workgroup> scratch: array<f32, 8>;")
    check hoistPipe.src.contains("const scale: f32 = 2.0;")
    # relocated declarations left the kernel body, one module-scope home each
    check hoistPipe.src.count("var<workgroup>") == 1
    check hoistPipe.src.count("const scale") == 1
    # emitted after the bindings, before the device functions and the kernel
    check hoistPipe.src.find("var<workgroup>") > hoistPipe.src.find("@binding")
    check hoistPipe.src.find("var<workgroup>") < hoistPipe.src.find("fn kern_lower_hoist")
    # module keeps them at program scope, in kernel-body order
    check hoistTraceText.contains("(globals")
    check hoistTraceText.contains("(var threadgroup scratch (array (f +32) 8) .)")
    check hoistTraceText.contains("(var constant scale (f +32) 2.0)")

  test "loud refusal: pointer local used as a value":
    check valueErr.contains("pointer local p used as a value")

  test "loud refusal: pointer-holding struct local used whole":
    check structWholeErr.contains("struct local v used whole")

  test "loud refusal: smem local with an initializer":
    check smemInitErr.contains("smem local scratch with an initializer")

  test "loud refusal: const_mem local without an initializer":
    check constNoInitErr.contains("const_mem local c without an initializer")

  test "bare pipeline refuses the shapes the lowering pass dissolves and hoists":
    ## Refusal contract: without the WGSL lowering passes the printer refuses,
    ## naming the pass that owns the transform.
    check bareAliasErr.contains("the WGSL lowering pass dissolves pointer locals")
    check bareHoistErr.contains("the WGSL lowering pass hoists threadgroup" &
      " locals to program scope")

  test "bare pipeline emits the raw bool read the lowering pass patches":
    ## Control for the bool transform: the patch is pass work, the bare pipeline
    ## leaves the raw parameter read in the kernel body.
    check bareBoolSrc.contains("if (flag) {")
    check "!=" notin bareBoolSrc

  test "rename without the double-underscore prefix data leaves the raw name":
    ## Control for the rename transform: the `wgsl_` prefix is per-target data
    ## of the reserved-word rename pass, without it the raw name reaches WGSL.
    check noPrefixMangledSrc.contains("__Int_pair")
    check "wgsl___Int" notin noPrefixMangledSrc

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
  ## Every site relocates to a module-scope `const` declaration.
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
  discard wgslPipeline(m)
  result = verifyMessage(m)

const
  survivorPipe = wgslPipeline(buildSurvivorModule())
  survivorBareErr = barePipeline(buildSurvivorModule()).err
  survivorVerified = verifiedSurvivorModule()

suite "WGSL lowering fixtures: const_mem survivors":

  test "nested-scope and device-fn const_mem hoist to module scope":
    check survivorPipe.err == ""
    # every relocation site emits at module scope, before any function
    check survivorPipe.src.contains("const cmBias: f32 = 1.5;")
    check survivorPipe.src.contains("const cmMult: f32 = 2.0;")
    check survivorPipe.src.contains("const cmK: f32 = 3.0;")
    check survivorPipe.src.find("const cmK") < survivorPipe.src.find("fn helper_scale")
    # declarations left their bodies, the reads stay
    check survivorPipe.src.count("cmBias: f32") == 1
    check survivorPipe.src.count("cmMult: f32") == 1
    check survivorPipe.src.count("cmK: f32") == 1
    check survivorPipe.src.contains("dst[0] = cmBias;")
    check survivorVerified == ""

  test "bare pipeline refuses a surviving const_mem local":
    ## Control for the transform: without the pass the const_mem declaration
    ## survives to the printer and the refusal names the owning pass.
    check survivorBareErr.contains("in constant space, the WGSL lowering" &
      " pass hoists const_mem locals to program scope")
