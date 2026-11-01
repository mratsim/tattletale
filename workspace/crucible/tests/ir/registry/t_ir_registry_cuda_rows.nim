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
  doAssert cell(scalar, ctFloat, 16).supported == false
  doAssert scalarCellRefusal(scalar, ctFloat, 16) == "rcOutsideSubset"
  doAssert cell(scalar, ctFloat, 32).spelling == "float"
  doAssert cell(scalar, ctFloat, 64).spelling == "double"
  doAssert scalarCellRefusal(scalar, ctInt, 8) == ""

  # msl names the 8-bit and 64-bit signed widths `char` and `long`.
  # cuda names the same widths `signed char` and `long long`.
  # ScalarDivergences records that divergence, converging fails the compile.
  let msl = mslScalarRow()
  doAssert cell(msl, ctInt, 8).spelling == "char"
  doAssert cell(msl, ctInt, 64).spelling == "long"
  doAssert cell(scalar, ctInt, 8).spelling != cell(msl, ctInt, 8).spelling
  doAssert cell(scalar, ctInt, 64).spelling != cell(msl, ctInt, 64).spelling

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
  doAssert kind.ternary == tfQuestion
  doAssert not kind.foldsIteElse
  doAssert not kind.emitOperandRendersExpr
  doAssert kind.asmForm == afWrap
  doAssert kind.constDecl == cdfConstexpr
  doAssert kind.constructor == cofBraced
  doAssert kind.targetExprKinds == {}

  let space = cudaSpaceRow()
  for i in AddressSpace:
    doAssert space.ptrQualifier[i] == ""
    if i != caThreadgroup:
      doAssert space.localPrefix[i] == ""
  doAssert space.localPrefix[caThreadgroup] == "__shared__"
  doAssert space.kernelParamScalar == ""
  doAssert space.kernelParamPointer == ""
  doAssert space.smemParam == ""

  let conv = cudaConvRow()
  doAssert conv.builtinNames.len == 0
  doAssert conv.fpRefusals.len == 0

  let coord = cudaCoordRow()
  doAssert coord.cells.len == 6
  for c in coord.cells:
    doAssert c.paramName == ""
    doAssert c.paramType == ""
    doAssert c.componentCast == "int"
    case c.kind
    of cbkThreadPositionInGrid:
      doAssert c.whole == cwfParam
    of cbkThreadgroupPositionInGrid, cbkThreadPositionInThreadgroup,
        cbkThreadsPerThreadgroup, cbkThreadgroupsPerGrid:
      doAssert c.whole == cwfVector
    of cbkThreadIndexInThreadgroup:
      doAssert c.whole == cwfFlatIndex
    of cbkNone:
      doAssert false

  let sig = cudaSigRow()
  doAssert sig.entryLinkage == "extern \"C\" __global__ "
  doAssert sig.deviceFnLinkage == "__device__ "
  doAssert sig.inlineLinkage == "__forceinline__ __device__ "
  doAssert sig.voidRet == "void"
  doAssert sig.workgroupAttr == ""
  doAssert sig.workgroupDefault == [0, 0, 0]
  doAssert sig.entryAttr == ""
  doAssert sig.bindingPrefix == ""
  doAssert sig.groupPrefix == ""

  let vec = cudaVectorRule()
  doAssert vec.context == vcNone
  doAssert vec.elems.len == 0
  doAssert vec.lanes.len == 0

echo "[OK] t_ir_registry_cuda_rows"
