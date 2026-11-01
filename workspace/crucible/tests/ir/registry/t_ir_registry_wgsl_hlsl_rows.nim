## WGSL and HLSL registry rows: genTargets lookup surface and values
##
## Test that the wgsl and hlsl registry rows pass the genTargets compile-time
## checks and return the WGSL and HLSL printers' values in both HLSL flavors
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/registry/t_ir_registry_wgsl_hlsl_rows.nim

import std/tables
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_rows

proc cell(row: ScalarRow, kind: CirTypeKind, width: int): ScalarCell =
  ## One (kind, width) cell of a registry-checked row.
  for c in row.cells:
    if c.kind == kind and c.width == width:
      return c

static:
  block:
    let scalar = wgslScalarRow()
    doAssert scalar.cells.len == ScalarCellAxis.len
    doAssert cell(scalar, ctVoid, 0).spelling == "void"
    doAssert cell(scalar, ctBool, 0).spelling == "bool"
    # WGSL has no 8/16-bit integer types: 32 and 64 bit are its only integer widths.
    doAssert scalarCellRefusal(scalar, ctInt, 8) == $rcCapability
    doAssert scalarCellRefusal(scalar, ctInt, 16) == $rcCapability
    doAssert scalarCellRefusal(scalar, ctUint, 8) == $rcCapability
    doAssert scalarCellRefusal(scalar, ctUint, 16) == $rcCapability
    doAssert cell(scalar, ctInt, 32).spelling == "i32"
    doAssert cell(scalar, ctInt, 64).spelling == "i64"
    doAssert cell(scalar, ctUint, 32).spelling == "u32"
    doAssert cell(scalar, ctUint, 64).spelling == "u64"
    doAssert cell(scalar, ctFloat, 16).spelling == "f16"
    doAssert cell(scalar, ctFloat, 32).spelling == "f32"
    doAssert cell(scalar, ctFloat, 64).spelling == "f64"

    let lit = wgslLitRow()
    doAssert lit.int.u32Suffix == "u"
    doAssert lit.int.u64Prefix == ""
    doAssert lit.int.u64Suffix == ""
    doAssert lit.int.i64Prefix == ""
    doAssert lit.int.i64Suffix == ""
    doAssert not lit.int.narrowCast
    doAssert lit.f16Prefix == ""
    doAssert lit.f16Suffix == "h"
    doAssert lit.f32Suffix == ""
    doAssert lit.f64Prefix == ""
    doAssert lit.f64Suffix == ""
    # nil literals refuse at the site
    doAssert lit.nilToken == ""
    doAssert lit.nonFinite.inf16 == "bitcast<f16>(0x7c00u)"
    doAssert lit.nonFinite.nan16 == "bitcast<f16>(0x7e00u)"
    doAssert lit.nonFinite.inf32 == "bitcast<f32>(0x7f800000u)"
    doAssert lit.nonFinite.negInf32 == "bitcast<f32>(0xff800000u)"
    doAssert lit.nonFinite.nan64 == "bitcast<f64>(0x7ff8000000000000u)"

    let kind = wgslKindRow()
    doAssert kind.loop.unrollDirective == "// {.unroll.} requested but not honored on WebGPU"
    doAssert kind.loop.infiniteHead == "loop {\n"
    doAssert kind.loop.braceSameLine
    # WGSL routes the ternary through select(f, t, cond)
    doAssert kind.ternary == tfSelectCall
    doAssert not kind.foldsIteElse
    doAssert not kind.emitOperandRendersExpr
    doAssert kind.asmForm == afRefuse
    doAssert kind.constDecl == cdfLet
    doAssert kind.constructor == cofCall
    doAssert kind.targetExprKinds == {eNil, eAt, ePat, eTernary}

    let space = wgslSpaceRow()
    for i in AddressSpace:
      doAssert space.ptrQualifier[i] == ""
      doAssert space.localPrefix[i] == ""
    doAssert space.kernelParamScalar == ""
    doAssert space.kernelParamPointer == ""
    doAssert not space.kernelParamPointerRestrict
    doAssert space.smemParam == ""
    # Address-space words sit inside the pointer type, storage pointers declare the access mode.
    doAssert space.ptrSpace[caThread] == "function"
    doAssert space.ptrSpace[caThreadgroup] == "workgroup"
    doAssert space.ptrSpace[caDevice] == "storage"
    doAssert space.ptrSpace[caConstant] == ""
    doAssert space.ptrAccessMode == "read_write"
    doAssert space.declVarPrefix == "var<workgroup>"
    doAssert space.bindingVarPrefix == "var<storage, read>"
    doAssert space.bindingBufferPrefix == "var<storage, read_write>"

    let conv = wgslConvRow()
    doAssert conv.builtinNames.len == 0
    doAssert conv.fpRefusals.len == 0

    let coord = wgslCoordRow()
    doAssert coord.cells.len == 6
    for c in coord.cells:
      # Refusal cell: threads_per_threadgroup states no read form
      if c.whole != cwfRefuse:
        doAssert c.componentCast == "i32"
      case c.kind
      of cbkThreadPositionInGrid:
        doAssert c.paramName == "global_id"
        doAssert c.paramType == "vec3<u32>"
        doAssert c.paramAnnotation == "global_invocation_id"
        doAssert c.whole == cwfVector
        doAssert c.componentForm == ccField
      of cbkThreadgroupPositionInGrid:
        doAssert c.paramName == "workgroup_id"
        doAssert c.paramAnnotation == "workgroup_id"
        doAssert c.whole == cwfVector
      of cbkThreadPositionInThreadgroup:
        doAssert c.paramName == "local_invocation_id"
        doAssert c.paramAnnotation == "local_invocation_id"
        doAssert c.whole == cwfVector
      of cbkThreadsPerThreadgroup:
        # Refusal cell: WGSL exposes the threadgroup size only through @workgroup_size.
        doAssert c.paramName == ""
        doAssert c.paramType == ""
        doAssert c.paramAnnotation == ""
        doAssert c.whole == cwfRefuse
      of cbkThreadgroupsPerGrid:
        doAssert c.paramName == "num_workgroups"
        doAssert c.paramAnnotation == "num_workgroups"
        doAssert c.whole == cwfVector
      of cbkThreadIndexInThreadgroup:
        # Scalar builtin: every read composes the whole form int(param)
        doAssert c.paramName == "local_invocation_index"
        doAssert c.paramType == "u32"
        doAssert c.paramAnnotation == "local_invocation_index"
        doAssert c.whole == cwfCast
        doAssert c.componentForm == ccWhole
      of cbkNone:
        doAssert false

    let sig = wgslSigRow()
    doAssert sig.entryLinkage == "fn "
    doAssert sig.deviceFnLinkage == "fn "
    # WGSL has no void type name, a void return emits no arrow
    doAssert sig.voidRet == ""
    doAssert sig.inlineLinkage == ""
    doAssert sig.entryAttr == "@compute "
    doAssert sig.workgroupAttr == "@workgroup_size($n, $n, $n)\n"
    doAssert sig.workgroupDefault == [64, 1, 1]
    doAssert sig.bindingPrefix == "@group(0) @binding($1)"
    doAssert sig.groupPrefix == ""
    # Kernel parameters become bindings and the signature takes only the coordinate parameters.
    doAssert not sig.entryTakesParams

    let vec = wgslVectorRule()
    doAssert vec.context == vcEverywhere
    doAssert vec.nameForm == vnfGenericElem
    doAssert vec.elems == @[
      (ctVoid, 0, "vec"), (ctBool, 0, "vec"),
      (ctInt, 32, "vec"), (ctInt, 64, "vec"),
      (ctUint, 32, "vec"), (ctUint, 64, "vec"),
      (ctFloat, 16, "vec"), (ctFloat, 32, "vec"), (ctFloat, 64, "vec")]
    doAssert vec.lanes == @[2, 3, 4]

  block:
    let scalar = hlslScalarRow()
    doAssert scalar.cells.len == ScalarCellAxis.len
    doAssert cell(scalar, ctVoid, 0).spelling == "void"
    doAssert cell(scalar, ctBool, 0).spelling == "bool"
    # 8/16-bit integers and int64_t are parse errors in the HLSL frontend
    doAssert scalarCellRefusal(scalar, ctInt, 8) == $rcParseError
    doAssert scalarCellRefusal(scalar, ctInt, 16) == $rcParseError
    doAssert scalarCellRefusal(scalar, ctInt, 64) == $rcParseError
    doAssert scalarCellRefusal(scalar, ctUint, 8) == $rcParseError
    doAssert scalarCellRefusal(scalar, ctUint, 16) == $rcParseError
    # uint64_t pulls the SPIR-V Int64 capability, float16 is outside the subset
    doAssert scalarCellRefusal(scalar, ctUint, 64) == $rcOutsideSubset
    doAssert scalarCellRefusal(scalar, ctFloat, 16) == $rcOutsideSubset
    doAssert cell(scalar, ctInt, 32).spelling == "int"
    doAssert cell(scalar, ctUint, 32).spelling == "uint"
    doAssert cell(scalar, ctFloat, 32).spelling == "float"
    doAssert cell(scalar, ctFloat, 64).spelling == "double"

    let lit = hlslLitRow()
    doAssert lit.int.u32Suffix == "u"
    doAssert lit.int.u64Suffix == "u"
    doAssert lit.int.i64Prefix == ""
    doAssert lit.int.i64Suffix == ""
    doAssert not lit.int.narrowCast
    doAssert lit.f16Suffix == "f"
    doAssert lit.f32Suffix == "f"
    doAssert lit.f64Suffix == "l"
    doAssert lit.nilToken == ""
    doAssert lit.nonFinite.inf16 == "asfloat(0x7f800000)"
    doAssert lit.nonFinite.nan16 == "asfloat(0x7fc00000)"
    doAssert lit.nonFinite.inf64 == "asdouble(0u, 0x7ff00000u)"
    doAssert lit.nonFinite.negInf64 == "asdouble(0u, 0xfff00000u)"
    doAssert lit.nonFinite.nan64 == "asdouble(0u, 0x7ff80000u)"

    let kind = hlslKindRow()
    doAssert kind.loop.unrollDirective == "[unroll]"
    doAssert kind.loop.infiniteHead == "for (;;)\n"
    # HLSL is the only braceless-head target, its brace prints on the next line
    doAssert not kind.loop.braceSameLine
    doAssert kind.ternary == tfQuestion
    doAssert kind.foldsIteElse
    doAssert kind.emitOperandRendersExpr
    doAssert kind.asmForm == afRefuse
    doAssert kind.constDecl == cdfConst
    doAssert kind.constructor == cofMixed
    doAssert kind.targetExprKinds == {eNil, eAt, ePat, eDeref, eAddr}

    let space = hlslSpaceRow()
    for i in AddressSpace:
      doAssert space.ptrQualifier[i] == ""
      doAssert space.localPrefix[i] == ""
    doAssert space.kernelParamScalar == ""
    doAssert space.kernelParamPointer == ""
    doAssert not space.kernelParamPointerRestrict
    doAssert space.smemParam == ""
    doAssert space.ptrAccessMode == ""
    doAssert space.declVarPrefix == ""
    doAssert space.bindingVarPrefix == ""
    doAssert space.bindingBufferPrefix == ""

    let conv = hlslConvRow()
    doAssert conv.builtinNames.len == 0
    doAssert conv.fpRefusals == @[
      "toFp16", "toFp32", "toBf16", "toBf32", "toFp8e4m3", "toFp8e5m2",
      "asFp16", "asU16"]

    let coord = hlslCoordRow()
    doAssert coord.cells.len == 6
    for c in coord.cells:
      doAssert c.paramAnnotation != "" or c.paramName == ""
      case c.kind
      of cbkThreadPositionInGrid:
        doAssert c.paramName == "gid"
        doAssert c.paramType == "uint3"
        doAssert c.paramAnnotation == "SV_DispatchThreadID"
        # whole-vector reads refuse, read the components one by one
        doAssert c.whole == cwfRefuse
        doAssert c.componentCast == "int"
        doAssert c.componentForm == ccField
      of cbkThreadgroupPositionInGrid:
        doAssert c.paramName == "groupId"
        doAssert c.paramAnnotation == "SV_GroupID"
        doAssert c.whole == cwfRefuse
      of cbkThreadPositionInThreadgroup:
        doAssert c.paramName == "groupThreadId"
        doAssert c.paramAnnotation == "SV_GroupThreadID"
        doAssert c.whole == cwfRefuse
      of cbkThreadsPerThreadgroup:
        # components substitute the baked workgroup dimensions at emission
        doAssert c.paramName == ""
        doAssert c.paramAnnotation == ""
        doAssert c.whole == cwfRefuse
        doAssert c.componentCast == ""
        doAssert c.componentForm == ccWorkgroupDim
      of cbkThreadgroupsPerGrid:
        # HLSL's lowering pass materializes the dispatch grid size
        doAssert c.paramName == ""
        doAssert c.paramAnnotation == ""
        doAssert c.whole == cwfRefuse
        doAssert c.componentCast == ""
      of cbkThreadIndexInThreadgroup:
        # Scalar builtin: every read composes the whole form int(gidx)
        doAssert c.paramName == "gidx"
        doAssert c.paramType == "uint"
        doAssert c.paramAnnotation == "SV_GroupIndex"
        doAssert c.whole == cwfCast
        doAssert c.componentForm == ccWhole
      of cbkNone:
        doAssert false

    let vec = hlslVectorRule()
    doAssert vec.context == vcEverywhere
    doAssert vec.nameForm == vnfPrefixed
    doAssert vec.elems == @[
      (ctBool, 0, "bool"), (ctInt, 32, "int"), (ctUint, 32, "uint"),
      (ctFloat, 32, "float")]
    doAssert vec.lanes == @[2, 3, 4]

    # Flavors change the binding section only and share one language body.
    let d3d12 = hlslFvD3D12SigRow()
    doAssert d3d12.entryLinkage == "void "
    doAssert d3d12.deviceFnLinkage == ""
    doAssert d3d12.inlineLinkage == "inline "
    doAssert d3d12.voidRet == "void"
    doAssert d3d12.workgroupAttr == "[numthreads($n, $n, $n)]\n"
    doAssert d3d12.workgroupDefault == [256, 1, 1]
    doAssert d3d12.entryAttr == ""
    doAssert d3d12.bindingPrefix == ": register($1, space0)"
    doAssert d3d12.groupPrefix == ""
    doAssert not d3d12.entryTakesParams
    let vulkan = hlslFvVulkanSigRow()
    for field in [
      (d3d12.entryLinkage, vulkan.entryLinkage),
      (d3d12.inlineLinkage, vulkan.inlineLinkage),
      (d3d12.voidRet, vulkan.voidRet),
      (d3d12.workgroupAttr, vulkan.workgroupAttr),
      (d3d12.entryAttr, vulkan.entryAttr),
      (d3d12.groupPrefix, vulkan.groupPrefix)]:
      doAssert field[0] == field[1]
    doAssert d3d12.workgroupDefault == vulkan.workgroupDefault
    doAssert d3d12.entryTakesParams == vulkan.entryTakesParams
    # fvVulkan binds through the push-constant struct, no register form exists
    doAssert vulkan.bindingPrefix == ""

