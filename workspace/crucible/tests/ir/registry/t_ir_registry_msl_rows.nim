## MSL registry rows: genTargets lookup surface and values
##
## Test that the msl registry rows pass the genTargets compile-time checks and return the MSL printer's values
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/registry/t_ir_registry_msl_rows.nim

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
  let scalar = mslScalarRow()
  doAssert scalar.cells.len == ScalarCellAxis.len
  doAssert cell(scalar, ctInt, 8).spelling == "char"
  doAssert cell(scalar, ctInt, 64).spelling == "long"
  doAssert cell(scalar, ctUint, 8).spelling == "uchar"
  doAssert cell(scalar, ctUint, 64).spelling == "ulong"
  doAssert cell(scalar, ctFloat, 16).spelling == "half"
  doAssert cell(scalar, ctBool, 0).spelling == "bool"
  doAssert not cell(scalar, ctFloat, 64).supported

  let lit = mslLitRow()
  doAssert lit.int.u64Suffix == "ULL"
  doAssert lit.int.u32Suffix == "U"
  doAssert lit.int.i64Suffix == "LL"
  doAssert lit.int.narrowCast
  doAssert lit.f16Suffix == "h"
  doAssert lit.f32Suffix == "f"
  doAssert lit.nilToken == "nullptr"
  doAssert lit.nonFinite.inf32 == "INFINITY"
  doAssert lit.nonFinite.nan64 == "NAN"

  let kind = mslKindRow()
  doAssert kind.loop.unrollDirective == "#pragma clang loop unroll(full)"
  doAssert kind.loop.infiniteHead == "for (;;) {\n"
  doAssert kind.loop.braceSameLine
  doAssert not kind.foldsIteElse
  doAssert not kind.emitOperandRendersExpr
  doAssert kind.targetExprKinds == {}

  let conv = mslConvRow()
  doAssert conv.builtinNames.len == 6
  doAssert conv.builtinNames["toFp16"] == "as_type<ushort>(half($1))"
  doAssert conv.builtinNames["toBf32"] == "float(as_type<bfloat>($1))"
  doAssert conv.builtinNames["asU16"] == "$1"
  doAssert conv.fpRefusals.len == 0

  let vec = mslVectorRule()
  doAssert vec.context == vcCoordinate
  doAssert vec.elems == @[(ctInt, 32, ""), (ctUint, 32, "")]
  doAssert vec.lanes == @[2, 3, 4]

echo "[OK] t_ir_registry_msl_rows"
