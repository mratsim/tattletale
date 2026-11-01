## genTargets: an unknown row field fails the compile at its line.
##
## Test that a misspelled lit field fails the compile naming the field
## at the row. `f32sufix` is not a LitRow field (the schema name is `f32Suffix`).
## Expected compile failure: "genTargets: unknown field 'f32sufix' in the lit row of msl"
##
## Run:
##   nim check --hints:off --warnings:off \
##     --nimcache:nimcache/tests workspace/crucible/tests/registry/manual_targets_unknown_field.nim
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_macros

genTargets:
  target msl:
    scalar void "void", bool "bool", i8 "char", i16 "short", i32 "int", i64 "long",
      u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "half", f32 "float", f64 ""
    lit u32Suffix "U", f32sufix "f", nilToken "nullptr"
    kind unrollDirective "#pragma unroll", braceSameLine true
    vector:
      elems i32, u32
      lanes 2, 3, 4
    bitcast:
      rule srcKinds {ctInt, ctUint, ctFloat}, srcWidths {8, 16, 32, 64},
        dstKinds {ctInt, ctUint, ctFloat}, sameWidth, "as_type<$2>($1)"
