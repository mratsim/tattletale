## Cross-space pointer-cast validation
##
## Test that a cross-space cast or conversion is refused:
## an unprovenanced integer may only cast to the default device space.
## The uint64 round-trip and derived same-space casts stay legal
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/validations/t_ir_space_cast_validation.nim

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
  let u64 = tUint(64)
  let devA = tAptr(caDevice, f32)
  let devP = tPtr(caDevice, f32)
  let tgP = tPtr(caThreadgroup, f32)
  let arrT = tArray(8, f32)

  # ═══ 1. the unprovenanced route through the full metal bundle ═══
  # An integer cast names no storage. A threadgroup claim from one prints
  # a threadgroup pointer the target cannot honor.
  block:
    let kern = procD("k1", @[param("out", devA)], tVoid(), @[
      varD("raw", u64, intLit(0, u64)),
      varD("p", tgP, castE(tgP, sym("raw", u64))),
      asgnS(patE(sym("out", devA), intLit(0, tInt(32))), derefE(sym("p", tgP))),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runFullMetalBundle()
    doAssert err.contains("validateAddressSpaceCasts"),
      "the metal bundle must refuse the unprovenanced threadgroup cast, got: [" & err & "]"
    doAssert err.contains("k1"), "the refusal must name the kernel, got: " & err
    doAssert err.contains("caThreadgroup"), "the refusal must name the target space, got: " & err

  # ═══ 2. the same shape through the validation stage alone ═══
  block:
    let kern = procD("k1", @[param("out", devA)], tVoid(), @[
      varD("p", tgP, castE(tgP, sym("raw", u64))),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runValidationStage()
    doAssert err.contains("no pointer provenance"),
      "the validation stage must refuse the unprovenanced cast, got: [" & err & "]"

  # ═══ 3. an unprovenanced integer casts to the default device space ═══
  # Address-route shape: a raw address cast to a device pointer.
  block:
    let kern = procD("k2", @[param("out", devA)], tVoid(), @[
      varD("p", devP, castE(devP, sym("raw", u64))),
      asgnS(patE(sym("out", devA), intLit(0, tInt(32))), derefE(sym("p", devP))),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runValidationStage()
    doAssert err.len == 0, "the unprovenanced device-space cast must stay legal: [" & err & "]"

  # ═══ 4. the uint64 round-trip re-derives the operand's space ═══
  # cast[uint64](p) + off keeps p's provenance, so the cast re-derives it.
  block:
    let offset = addE(u64, castE(u64, sym("p", devP)), intLit(4, u64))
    let kern = procD("k3", @[param("p", devP), param("out", devA)], tVoid(), @[
      varD("q", devP, castE(devP, offset)),
      asgnS(patE(sym("out", devA), intLit(0, tInt(32))), derefE(sym("q", devP))),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runValidationStage()
    doAssert err.len == 0, "the uint64 round-trip must stay legal: [" & err & "]"

  # ═══ 5. round-trip misuse: provenance naming another space refuses ═══
  block:
    let offset = addE(u64, castE(u64, sym("p", devP)), intLit(4, u64))
    let kern = procD("k4", @[param("p", devP), param("out", devA)], tVoid(), @[
      varD("q", tgP, castE(tgP, offset)),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runValidationStage()
    doAssert err.contains("cross-space cast"),
      "the validation stage must refuse the cross-space round-trip, got: [" & err & "]"
    doAssert err.contains("caDevice") and err.contains("caThreadgroup"),
      "the refusal must name both spaces, got: " & err

  # ═══ 6. same-space casts and conversions pass ═══
  block:
    let kern = procD("k5", @[param("p", devP), param("out", devA)], tVoid(), @[
      varD("q", devP, castE(devP, sym("p", devP))),
      varD("r", devP, convE(devP, sym("q", devP))),
      asgnS(patE(sym("out", devA), intLit(0, tInt(32))), derefE(sym("r", devP))),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runValidationStage()
    doAssert err.len == 0, "the same-space cast and conversion must stay legal: [" & err & "]"

  # ═══ 7. a conversion naming another space refuses ═══
  block:
    let kern = procD("k6", @[param("p", devP), param("out", devA)], tVoid(), @[
      varD("q", tgP, convE(tgP, sym("p", devP))),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runValidationStage()
    doAssert err.contains("cross-space conversion"),
      "the validation stage must refuse the cross-space conversion, got: [" & err & "]"

  # ═══ 8. a derived pointer keeps its source storage ═══
  # addr of a threadgroup local names threadgroup provenance, so a cast
  # targeting the same space passes.
  block:
    let tgAddr = addrE(tgP, atE(sym("scratch", arrT), intLit(0, tInt(32))))
    let kern = procD("k7", @[param("out", devA)], tVoid(), @[
      varD("scratch", arrT, nil, caThreadgroup),
      varD("q", tgP, castE(tgP, tgAddr)),
      asgnS(patE(sym("out", devA), intLit(0, tInt(32))), derefE(sym("q", tgP))),
    ])
    kern.isKernel = true
    let err = moduleD("probe", @[], @[kern]).runValidationStage()
    doAssert err.len == 0, "the derived same-space cast must stay legal: [" & err & "]"

echo "[OK] t_ir_space_cast_validation"
