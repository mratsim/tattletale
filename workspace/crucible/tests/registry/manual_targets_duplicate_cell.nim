## genTargets: a scalar cell stated twice fails the compile.
##
## Test that one (kind, width) cell appears exactly once per target.
## msl's scalar row states `i32` twice and the check names the cell
## at the repeat.
## Expected compile failure: "genTargets: scalar cell 'i32' appears twice in msl"
##
## Run:
##   nim check --hints:off --warnings:off \
##     --nimcache:nimcache/tests workspace/crucible/tests/registry/manual_targets_duplicate_cell.nim
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_macros

genTargets:
  target msl:
    scalar void "void", bool "bool", i8 "char", i16 "short", i32 "int", i32 "long",
      i64 "long", u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "half", f32 "float", f64 "double"
    lit u32Suffix "U", nilToken "nullptr"
    kind unrollDirective "#pragma unroll", braceSameLine true
    vector:
      elems i32, u32
      lanes 2, 3, 4
    bitcast:
      rule srcKinds {ctInt, ctUint, ctFloat}, srcWidths {8, 16, 32, 64},
        dstKinds {ctInt, ctUint, ctFloat}, sameWidth, "as_type<$2>($1)"
