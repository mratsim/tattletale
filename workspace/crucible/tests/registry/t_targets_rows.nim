## Generated target rows: the printers' static target data.
##
## Test that every generated lookup resolves to the printer-observed value
## with the backend name tables of AGENTS.md as the independent truth.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/registry/t_targets_rows.nim
import std/tables
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/print_common

static:
  block: # scalar rows: the stated cells resolve, absent cells refuse
    for row in [mslScalarRow, cudaScalarRow, openclScalarRow, glslScalarRow,
        wgslScalarRow, hlslScalarRow]:
      var seen: seq[(CirTypeKind, int)]
      for c in row.cells:
        doAssert c.spelling.len > 0, "a stated cell names its type"
        doAssert (c.kind, c.width) notin seen, "each cell stated once"
        seen.add (c.kind, c.width)
    doAssert scalarName(mslScalarRow, tFloat(16)) == "half"
    doAssert scalarName(mslScalarRow, tInt(64)) == "long"
    doAssert scalarName(cudaScalarRow, tInt(64)) == "long long"
    doAssert scalarName(openclScalarRow, tBool()) == "int"
    doAssert scalarName(openclScalarRow, tFloat(32)) == "float"
    doAssert scalarName(wgslScalarRow, tInt(32)) == "i32"
    doAssert scalarName(hlslScalarRow, tUint(32)) == "uint"
    # absent cells refuse, scalarName returns no name and cDecl prints the refusal
    doAssert scalarName(wgslScalarRow, tInt(8)) == ""
    doAssert scalarName(hlslScalarRow, tFloat(16)) == ""
    doAssert scalarName(mslScalarRow, tFloat(64)) == ""
    doAssert scalarName(cudaScalarRow, tFloat(16)) == ""
    doAssert cDecl(mslScalarRow, mslVectorRule, "msl", tFloat(64), "x", false) ==
      "<msl: no f64, double has no Apple GPU support>"
    doAssert cDecl(cudaScalarRow, cudaVectorRule, "cuda", tFloat(16), "x", false) ==
      "<cuda: no f16, no native f16>"

  block: # lit rows: literal token forms
    doAssert wgslLitRow.int.u32Suffix == "u"
    doAssert mslLitRow.int.u64Suffix == "ULL"
    doAssert openclLitRow.f16Prefix == "(half)"
    doAssert openclLitRow.nilToken == "0"
    doAssert glslLitRow.f64Prefix == "double("
    doAssert hlslLitRow.f64Suffix == "l"
    doAssert mslLitRow.nonFinite.inf32 == "INFINITY"
    doAssert wgslLitRow.nonFinite.nan16 == "bitcast<f16>(0x7e00u)"

  block: # kind rows: walk policy
    doAssert mslKindRow.loop.unrollDirective == "#pragma clang loop unroll(full)"
    doAssert cudaKindRow.loop.unrollDirective == "#pragma unroll"
    doAssert hlslKindRow.loop.braceSameLine == false
    doAssert hlslKindRow.foldsIteElse
    doAssert hlslKindRow.emitOperandRendersExpr
    doAssert eTernary in wgslKindRow.targetExprKinds
    doAssert eDeref in glslKindRow.targetExprKinds

  block: # conv/math/refusals: the builtin name tables
    doAssert mslConvBuiltinNames["toFp16"] == "half($1)"
    doAssert openclConvBuiltinNames["asFp16"] == "as_half($1)"
    doAssert glslConvBuiltinNames["asU16"] == "float16BitsToUint16($1)"
    doAssert wgslConvBuiltinNames["toFp32"] == "f32($1)"
    doAssert cudaMathBuiltinNames[("abs", 32)] == "fabsf"
    doAssert cudaMathBuiltinNames[("exp2", 32)] == "exp2f"
    doAssert "toFp16" in hlslConvFpRefusals
    doAssert "asU16" in hlslConvFpRefusals

  block: # vector rules: each target's real coverage
    doAssert vectorName(glslVectorRule, glslScalarRow, tVec(4, tFloat(32))) == "vec4"
    doAssert vectorName(glslVectorRule, glslScalarRow, tVec(2, tUint(32))) == "uvec2"
    doAssert vectorName(wgslVectorRule, wgslScalarRow, tVec(4, tFloat(32))) == "vec4<f32>"
    doAssert vectorName(mslVectorRule, mslScalarRow, tVec(3, tInt(32))) == "int3"
    doAssert vectorName(cudaVectorRule, cudaScalarRow, tVec(4, tFloat(32))) == ""
    doAssert vectorName(hlslVectorRule, hlslScalarRow, tVec(2, tFloat(32))) == "float2"

  block: # bitcast rules: first-match coverage
    doAssert cudaCastRules[0].spelling == "__float_as_uint($1)"
    doAssert wgslCastRules[0].srcAny
    doAssert openclCastRules[0].spelling == "as_$2($1)"
    doAssert hlslCastRules[0].spelling == "asint($1)"
    doAssert hlslCastRules[^1].refusal.len > 0
    doAssert mslCastRules[0].sameWidth
