## WGSL and HLSL registry rows: genTargets lookup surface and values
##
## Test that the wgsl and hlsl registry rows pass the genTargets compile-time
## checks and return the WGSL and HLSL printers' values
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
    doAssert not cell(scalar, ctInt, 8).supported
    doAssert not cell(scalar, ctInt, 16).supported
    doAssert not cell(scalar, ctUint, 8).supported
    doAssert not cell(scalar, ctUint, 16).supported
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
    doAssert not kind.foldsIteElse
    doAssert not kind.emitOperandRendersExpr
    doAssert kind.targetExprKinds == {eNil, eAt, ePat, eTernary}

    let conv = wgslConvRow()
    doAssert conv.builtinNames.len == 0
    doAssert conv.fpRefusals.len == 0

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
    # 8/16-bit integers and int64_t are parse errors in the HLSL frontend,
    # uint64_t pulls the SPIR-V Int64 capability, float16 is outside the subset
    doAssert not cell(scalar, ctInt, 8).supported
    doAssert not cell(scalar, ctInt, 16).supported
    doAssert not cell(scalar, ctInt, 64).supported
    doAssert not cell(scalar, ctUint, 8).supported
    doAssert not cell(scalar, ctUint, 16).supported
    doAssert not cell(scalar, ctUint, 64).supported
    doAssert not cell(scalar, ctFloat, 16).supported
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
    doAssert kind.foldsIteElse
    doAssert kind.emitOperandRendersExpr
    doAssert kind.targetExprKinds == {eNil, eAt, ePat, eDeref, eAddr}

    let conv = hlslConvRow()
    doAssert conv.builtinNames.len == 0
    doAssert conv.fpRefusals == @[
      "toFp16", "toFp32", "toBf16", "toBf32", "toFp8e4m3", "toFp8e5m2",
      "asFp16", "asU16"]

    let vec = hlslVectorRule()
    doAssert vec.context == vcEverywhere
    doAssert vec.nameForm == vnfPrefixed
    doAssert vec.elems == @[
      (ctBool, 0, "bool"), (ctInt, 32, "int"), (ctUint, 32, "uint"),
      (ctFloat, 32, "float")]
    doAssert vec.lanes == @[2, 3, 4]

echo "[OK] t_ir_registry_wgsl_hlsl_rows"
