## genTargets: a scalar row missing a (kind, width) cell fails the compile.
##
## Test that the macro checks every `ScalarCellAxis` cell is stated.
## msl's scalar row omits the `f64` cell and the check names it at the row.
## Expected compile failure: "genTargets: the scalar row of msl is missing cell(s) f64"
##
## Run:
##   nim check --hints:off --warnings:off \
##     --nimcache:nimcache/tests workspace/crucible/tests/registry/manual_targets_missing_cell.nim
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_macros

genTargets:
  target msl:
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
