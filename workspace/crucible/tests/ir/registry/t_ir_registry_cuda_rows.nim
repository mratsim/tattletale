## CUDA registry rows: genTargets lookup surface and values
##
## Test that the cuda registry rows pass the genTargets compile-time checks and return the CUDA printer's values
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/registry/t_ir_registry_cuda_rows.nim

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
  let scalar = cudaScalarRow()
  doAssert scalar.cells.len == ScalarCellAxis.len
  doAssert cell(scalar, ctVoid, 0).spelling == "void"
  doAssert cell(scalar, ctBool, 0).spelling == "bool"
  doAssert cell(scalar, ctInt, 8).spelling == "signed char"
  doAssert cell(scalar, ctInt, 16).spelling == "short"
  doAssert cell(scalar, ctInt, 32).spelling == "int"
  doAssert cell(scalar, ctInt, 64).spelling == "long long"
  doAssert cell(scalar, ctUint, 8).spelling == "unsigned char"
  doAssert cell(scalar, ctUint, 16).spelling == "unsigned short"
  doAssert cell(scalar, ctUint, 32).spelling == "unsigned int"
  doAssert cell(scalar, ctUint, 64).spelling == "unsigned long long"
  doAssert not cell(scalar, ctFloat, 16).supported
  doAssert cell(scalar, ctFloat, 32).spelling == "float"
  doAssert cell(scalar, ctFloat, 64).spelling == "double"

  let lit = cudaLitRow()
  doAssert lit.int.u64Suffix == "ULL"
  doAssert lit.int.u32Suffix == "U"
  doAssert lit.int.i64Suffix == "LL"
  doAssert lit.int.narrowCast
  doAssert lit.f16Suffix == ""
  doAssert lit.f32Suffix == "f"
  doAssert lit.f64Suffix == ""
  doAssert lit.nilToken == "nullptr"
  doAssert lit.nonFinite.inf16 == "INFINITY"
  doAssert lit.nonFinite.nan32 == "NAN"
  doAssert lit.nonFinite.negInf64 == "-INFINITY"

  let kind = cudaKindRow()
  doAssert kind.loop.unrollDirective == "#pragma unroll"
  doAssert kind.loop.infiniteHead == "for (;;) {\n"
  doAssert kind.loop.braceSameLine
  doAssert not kind.foldsIteElse
  doAssert not kind.emitOperandRendersExpr
  doAssert kind.targetExprKinds == {}

  let conv = cudaConvRow()
  doAssert conv.builtinNames.len == 0
  doAssert conv.fpRefusals.len == 0

  let vec = cudaVectorRule()
  doAssert vec.context == vcNone
  doAssert vec.elems.len == 0
  doAssert vec.lanes.len == 0

echo "[OK] t_ir_registry_cuda_rows"
