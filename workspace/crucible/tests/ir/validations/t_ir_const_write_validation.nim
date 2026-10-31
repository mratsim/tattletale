## Constant-storage write validation
##
## Test that an assignment into constant storage is refused, with the proc
## and target named, while thread and threadgroup writes stay legal
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/validations/t_ir_const_write_validation.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_lowering_msl
import workspace/crucible/src/codegen/passes/passes_validations

proc runValidationStage(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerValidationPostPasses()
  result = m.runPasses(reg)

proc runFullMetalBundle(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerMslLoweringPasses()
  reg.registerValidationPostPasses()
  result = m.runPasses(reg)

static:
  let f32 = tFloat(32)
  let u32 = tUint(32)
  let devA = tAptr(caDevice, f32)

  # ═══ 1. a relocated program-scope constant, written from the kernel body ═══
  # Post-hoist shape of the MSL bundle: the `{.const_mem.}` local lives in `m.globals`, and the kernel body still assigns to that name.
  block:
    let kern = procD("k1", @[param("out", devA)], tVoid(), @[
      asgnS(sym("g", f32), fltLit(5.0, f32)),
      asgnS(patE(sym("out", devA), intLit(0, tInt(32))), sym("g", f32)),
    ])
    kern.isKernel = true
    let m = moduleD("probe", @[], @[kern])
    m.globals.add(varD("g", f32, fltLit(3.0, f32), caConstant))
    let err = m.runFullMetalBundle()
    doAssert err.contains("validateConstantWrites"),
      "the metal bundle must refuse the constant-storage write, got: [" & err & "]"
    doAssert err.contains("k1"), "the refusal must name the kernel, got: " & err
    doAssert err.contains("g"), "the refusal must name the target, got: " & err
    doAssert err.contains("constant storage"), "the refusal must name the storage, got: " & err

  # ═══ 2. the same shape through the validation stage alone ═══
  block:
    let kern = procD("k1", @[param("out", devA)], tVoid(), @[
      asgnS(sym("g", f32), fltLit(5.0, f32)),
    ])
    kern.isKernel = true
    let m = moduleD("probe", @[], @[kern])
    m.globals.add(varD("g", f32, fltLit(3.0, f32), caConstant))
    let err = m.runValidationStage()
    doAssert err.contains("assignment into constant storage"),
      "the validation stage must refuse the write, got: [" & err & "]"

  # ═══ 3. a surviving in-body const_mem local, written after declaration ═══
  # CUDA keeps const_mem locals in-body as const-space storage.
  block:
    let kern = procD("k2", @[param("out", devA)], tVoid(), @[
      varD("c", f32, fltLit(1.0, f32), caConstant),
      asgnS(sym("c", f32), fltLit(2.0, f32)),
      asgnS(patE(sym("out", devA), intLit(0, tInt(32))), sym("c", f32)),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runValidationStage()
    doAssert err.contains("assignment into constant storage"),
      "the validation stage must refuse the const-space local write, got: [" & err & "]"
    doAssert err.contains("k2") and err.contains("c"),
      "the refusal must name the kernel and the target, got: " & err

  # ═══ 4. a write through a constant-space pointer ═══
  block:
    let constP = tPtr(caConstant, f32)
    let kern = procD("k3", @[param("p", constP), param("out", devA)], tVoid(), @[
      asgnS(derefE(sym("p", constP)), fltLit(2.0, f32)),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runValidationStage()
    doAssert err.contains("assignment into constant storage"),
      "the validation stage must refuse the write through the constant pointer, got: [" & err & "]"

  # ═══ 5. a write through a constant-space array pointer ═══
  block:
    let constAp = tAptr(caConstant, u32)
    let kern = procD("k4", @[param("p", constAp), param("out", devA)], tVoid(), @[
      asgnS(patE(sym("p", constAp), intLit(0, tInt(32))), intLit(2, u32)),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runValidationStage()
    doAssert err.contains("assignment into constant storage"),
      "the validation stage must refuse the write through the constant array pointer, got: [" & err & "]"

  # ═══ 6. an indexed write into a constant global array ═══
  block:
    let arrT = tArray(4, f32)
    let kern = procD("k5", @[param("out", devA)], tVoid(), @[
      asgnS(atE(sym("g", arrT), intLit(0, tInt(32))), fltLit(5.0, f32)),
    ])
    kern.isKernel = true
    let m = moduleD("probe", @[], @[kern])
    m.globals.add(varD("g", arrT, nil, caConstant))
    let err = m.runValidationStage()
    doAssert err.contains("assignment into constant storage"),
      "the validation stage must refuse the indexed write into the constant array, got: [" & err & "]"

  # ═══ hazard negatives: mutable storage and constant reads stay legal ═══
  block:
    let arrT = tArray(4, f32)
    let kern = procD("k6", @[param("out", devA)], tVoid(), @[
      varD("priv", f32, fltLit(1.0, f32), caThread),
      varD("scratch", arrT, nil, caThreadgroup),
      asgnS(sym("priv", f32), fltLit(2.0, f32)),
      asgnS(atE(sym("scratch", arrT), intLit(0, tInt(32))), sym("priv", f32)),
      asgnS(patE(sym("out", devA), intLit(0, tInt(32))),
        atE(sym("g", arrT), intLit(0, tInt(32)))),
    ])
    kern.isKernel = true
    let m = moduleD("probe", @[], @[kern])
    m.globals.add(varD("g", arrT, nil, caConstant))
    let err = m.runValidationStage()
    doAssert err.len == 0,
      "writes to thread and threadgroup storage and constant reads must stay legal: [" & err & "]"

echo "[OK] t_ir_const_write_validation"
