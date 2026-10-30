## Hand-built CIRv2: the struct-layout pass packing rules, both rule-set variants.
##
## Test that both layout passes store member offsets, struct alignment, and padded size under their own bool width.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_layout_std430.nim

import std/tables
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/layout_std430

proc runLayout(m: CirModule, glsl, wgsl: bool): string =
  ## Runs the requested layout passes over `m` through the registry.
  var reg = PassRegistry.new()
  if glsl:
    reg.registerGlslLayoutPass()
  if wgsl:
    reg.registerWgslLayoutPass()
  result = m.runPasses(reg)

static:
  # index 0 packs 4 bytes, index 1 packs 8 and starts at offset 8
  # struct size rounds up to the 8-byte alignment, 16 bytes total
  let inner = tStruct("Inner", @[
    fld("a", tUint(32)),
    fld("b", tUint(64)),
  ])
  # a nested struct plus a trailing scalar: the scalar starts after the padded nested struct and the size pads to 24
  let outer = tStruct("Outer", @[
    fld("i", inner),
    fld("f", tFloat(32)),
  ])
  # two bools: the packing disagrees between the rule-set variants
  let boolPair = tStruct("BoolPair", @[
    fld("a", tBool()),
    fld("b", tBool()),
  ])
  # a vector packs its components back to back and its alignment equals
  # one component width, so the trailing scalar needs no padding before it
  let vecStruct = tStruct("VecStruct", @[
    fld("v", tVec(4, tFloat(32))),
    fld("w", tFloat(32)),
  ])
  # an array packs whole element copies
  let arrStruct = tStruct("ArrStruct", @[
    fld("arr", tArray(3, tFloat(32))),
  ])
  let kernelFn = procD("k", @[], tVoid(), @[retS()])
  kernelFn.isKernel = true
  let m = moduleD("probe",
    @[inner, outer, boolPair, vecStruct, arrStruct],
    @[kernelFn])

  block glslPacking:
    let err = m.runLayout(glsl = true, wgsl = false)
    doAssert err.len == 0, "the GLSL layout pass must succeed: " & err
    doAssert m.layouts["Inner"].memberOffsets == @[0, 8],
      "the 8-byte member must start at offset 8, got " &
        $m.layouts["Inner"].memberOffsets
    doAssert m.layouts["Inner"].size == 16,
      "the struct size must pad to the 8-byte alignment"
    doAssert m.layouts["Inner"].align == 8
    doAssert m.layouts["Outer"].memberOffsets == @[0, 16],
      "the scalar must start after the padded nested struct"
    doAssert m.layouts["Outer"].size == 24
    doAssert m.layouts["BoolPair"].memberOffsets == @[0, 1],
      "std430 packs a bool in 1 byte so the second bool starts at offset 1"
    doAssert m.layouts["BoolPair"].size == 2
    doAssert m.layouts["VecStruct"].memberOffsets == @[0, 16],
      "a vector's alignment is the component width so the trailing scalar " &
        "needs no padding before it"
    doAssert m.layouts["VecStruct"].size == 20
    doAssert m.layouts["ArrStruct"].size == 12
    # composite queries read the stored struct sizes
    doAssert typeByteSize(m.layouts, outer, std430BoolWidth) == 24
    doAssert typeByteSize(m.layouts, tArray(3, outer), std430BoolWidth) == 72
    doAssert typeByteSize(m.layouts, tVec(3, tFloat(32)), std430BoolWidth) == 12
    doAssert typeByteSize(m.layouts, tBool(), std430BoolWidth) == 1
    doAssert typeByteSize(m.layouts, tAptr(caDevice, tFloat(32)),
      std430BoolWidth) == 8

  block wgslPacking:
    let err = m.runLayout(glsl = false, wgsl = true)
    doAssert err.len == 0, "the WGSL layout pass must succeed: " & err
    doAssert m.layouts["BoolPair"].memberOffsets == @[0, 4],
      "the launch ABI packs a bool in 4 bytes so the second bool starts " &
        "at offset 4, got " & $m.layouts["BoolPair"].memberOffsets
    doAssert m.layouts["BoolPair"].size == 8
    doAssert m.layouts["BoolPair"].align == 4
    doAssert m.layouts["Inner"].memberOffsets == @[0, 8],
      "the bool width must not move the integer members"
    doAssert typeByteSize(m.layouts, tBool(), launchAbiBoolWidth) == 4
