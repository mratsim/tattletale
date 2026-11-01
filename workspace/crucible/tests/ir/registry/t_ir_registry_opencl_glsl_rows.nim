## OpenCL C and GLSL registry rows: genTargets lookup surface and values
##
## Test that the opencl and glsl registry rows pass the genTargets compile-time
## checks and return the OpenCL C and GLSL printers' values
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/registry/t_ir_registry_opencl_glsl_rows.nim

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
    let scalar = openclScalarRow()
    doAssert scalar.cells.len == ScalarCellAxis.len
    doAssert cell(scalar, ctVoid, 0).spelling == "void"
    # bool prints as `int` for the launch ABI, matching its (ctInt, 32) name
    doAssert cell(scalar, ctBool, 0).spelling == "int"
    doAssert cell(scalar, ctInt, 8).spelling == "char"
    doAssert cell(scalar, ctInt, 16).spelling == "short"
    doAssert cell(scalar, ctInt, 32).spelling == "int"
    doAssert cell(scalar, ctInt, 64).spelling == "long"
    doAssert cell(scalar, ctUint, 8).spelling == "uchar"
    doAssert cell(scalar, ctUint, 16).spelling == "ushort"
    doAssert cell(scalar, ctUint, 32).spelling == "uint"
    doAssert cell(scalar, ctUint, 64).spelling == "ulong"
    doAssert cell(scalar, ctFloat, 16).spelling == "half"
    doAssert cell(scalar, ctFloat, 32).spelling == "float"
    doAssert cell(scalar, ctFloat, 64).spelling == "double"

    let lit = openclLitRow()
    doAssert lit.int.u64Suffix == "ULL"
    doAssert lit.int.u32Suffix == "U"
    doAssert lit.int.i64Suffix == "LL"
    doAssert lit.int.narrowCast
    doAssert lit.f16Prefix == "(half)"
    doAssert lit.f16Suffix == ""
    doAssert lit.f32Suffix == "f"
    doAssert lit.f64Prefix == ""
    doAssert lit.f64Suffix == ""
    doAssert lit.nilToken == "0"
    doAssert lit.nonFinite.inf16 == "(half)INFINITY"
    doAssert lit.nonFinite.nan16 == "(half)NAN"
    doAssert lit.nonFinite.inf32 == "INFINITY"
    doAssert lit.nonFinite.negInf64 == "((double)-INFINITY)"
    doAssert lit.nonFinite.nan64 == "((double)NAN)"

    let kind = openclKindRow()
    doAssert kind.loop.unrollDirective == "#pragma unroll"
    doAssert kind.loop.infiniteHead == "for (;;) {\n"
    doAssert kind.loop.braceSameLine
    doAssert not kind.foldsIteElse
    doAssert not kind.emitOperandRendersExpr
    doAssert kind.targetExprKinds == {}

    let conv = openclConvRow()
    doAssert conv.builtinNames.len == 0
    doAssert conv.fpRefusals.len == 0

    let vec = openclVectorRule()
    doAssert vec.context == vcEverywhere
    doAssert vec.elems.len == ScalarCellAxis.len
    for e in vec.elems:
      # vector name appends the lane count to the scalar name
      doAssert e.prefix == ""
    doAssert vec.lanes == @[2, 3, 4, 8, 16]

  block:
    let scalar = glslScalarRow()
    doAssert scalar.cells.len == ScalarCellAxis.len
    doAssert cell(scalar, ctVoid, 0).spelling == "void"
    doAssert cell(scalar, ctBool, 0).spelling == "bool"
    doAssert cell(scalar, ctInt, 8).spelling == "int8_t"
    doAssert cell(scalar, ctInt, 16).spelling == "int16_t"
    doAssert cell(scalar, ctInt, 32).spelling == "int"
    doAssert cell(scalar, ctInt, 64).spelling == "int64_t"
    doAssert cell(scalar, ctUint, 8).spelling == "uint8_t"
    doAssert cell(scalar, ctUint, 16).spelling == "uint16_t"
    doAssert cell(scalar, ctUint, 32).spelling == "uint"
    doAssert cell(scalar, ctUint, 64).spelling == "uint64_t"
    doAssert cell(scalar, ctFloat, 16).spelling == "float16_t"
    doAssert cell(scalar, ctFloat, 32).spelling == "float"
    doAssert cell(scalar, ctFloat, 64).spelling == "double"

    let lit = glslLitRow()
    doAssert lit.int.u64Prefix == "uint64_t("
    doAssert lit.int.u64Suffix == ")"
    doAssert lit.int.u32Suffix == "U"
    doAssert lit.int.i64Prefix == "int64_t("
    doAssert lit.int.i64Suffix == ")"
    doAssert lit.int.narrowCast
    doAssert lit.f16Prefix == "float16_t("
    doAssert lit.f16Suffix == ")"
    doAssert lit.f32Suffix == "f"
    doAssert lit.f64Prefix == "double("
    doAssert lit.f64Suffix == ")"
    doAssert lit.nilToken == ""
    doAssert lit.nonFinite.inf16 == "float16_t(uintBitsToFloat(0x7f800000u))"
    doAssert lit.nonFinite.nan16 == "float16_t(uintBitsToFloat(0x7fc00000u))"
    doAssert lit.nonFinite.inf32 == "uintBitsToFloat(0x7f800000u)"
    doAssert lit.nonFinite.negInf32 == "uintBitsToFloat(0xff800000u)"
    doAssert lit.nonFinite.nan64 == "double(uintBitsToFloat(0x7fc00000u))"

    let kind = glslKindRow()
    doAssert kind.loop.unrollDirective == "[[unroll]]"
    doAssert kind.loop.infiniteHead == "for (;;) {\n"
    doAssert kind.loop.braceSameLine
    doAssert not kind.foldsIteElse
    doAssert not kind.emitOperandRendersExpr
    doAssert kind.targetExprKinds == {eNil, eAt, ePat, eDeref, eAddr}

    let conv = glslConvRow()
    doAssert conv.builtinNames.len == 0
    doAssert conv.fpRefusals.len == 0

    let vec = glslVectorRule()
    doAssert vec.context == vcEverywhere
    doAssert vec.elems == @[
      (ctFloat, 16, "f16vec"), (ctFloat, 32, "vec"), (ctFloat, 64, "dvec"),
      (ctInt, 32, "ivec"), (ctUint, 32, "uvec"), (ctBool, 0, "bvec")]
    doAssert vec.lanes == @[2, 3, 4]

echo "[OK] t_ir_registry_opencl_glsl_rows"
