## genTargets: a scalar row may omit cells, refusal fires at the lookup.
##
## Test that an msl scalar row without the `f64` cell compiles and its f64
## lookup refuses with `cDecl`'s one-line message, row reason or bare form.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/registry/t_targets_absent_cell.nim
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_macros
import workspace/crucible/src/codegen/backends/print_common

genTargets:
  target msl:
    scalar void "void", bool "bool", i8 "char", i16 "short", i32 "int", i64 "long",
      u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "half", f32 "float",
      refused f64 "double has no Apple GPU support"
    lit u32Suffix "U", nilToken "nullptr"
    kind unrollDirective "#pragma unroll", braceSameLine true
    vector:
      elems i32, u32
      lanes 2, 3, 4
    bitcast:
      rule srcKinds {ctInt, ctUint, ctFloat}, srcWidths {8, 16, 32, 64},
        dstKinds {ctInt, ctUint, ctFloat}, sameWidth, "as_type<$2>($1)"

  target mslcore:
    scalar void "void", bool "bool", i8 "char", i16 "short", i32 "int", i64 "long",
      u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "half", f32 "float"
    lit u32Suffix "U", nilToken "nullptr"
    kind unrollDirective "#pragma unroll", braceSameLine true
    vector:
      elems i32, u32
      lanes 2, 3, 4
    bitcast:
      rule srcKinds {ctInt, ctUint, ctFloat}, srcWidths {8, 16, 32, 64},
        dstKinds {ctInt, ctUint, ctFloat}, sameWidth, "as_type<$2>($1)"

proc runTest() =
  doAssert getScalarName(mslScalarRow, tFloat(64)) == "",
    "an absent cell has no type name"
  doAssert getScalarName(mslScalarRow, tFloat(32)) == "float",
    "the stated cells keep their names"
  doAssert cDecl(mslScalarRow, mslVectorRule, "msl", tFloat(64), "x", false) ==
    "<msl: no f64, double has no Apple GPU support>",
    "the stated reason rides the one-line refusal"
  doAssert cDecl(mslcoreScalarRow, mslcoreVectorRule, "mslcore", tFloat(64), "x", false) ==
    "<mslcore: no f64>",
    "an omission without a stated reason still refuses, naming target and cell"
  doAssert cDecl(mslScalarRow, mslVectorRule, "msl", tFloat(32), "x", false) ==
    "float x",
    "a stated cell keeps the plain declaration form"

when isMainModule:
  runTest()
