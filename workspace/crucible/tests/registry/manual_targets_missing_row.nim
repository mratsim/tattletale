## genTargets: a target key missing a required row family fails the compile.
##
## Test that an msl section without the `lit` family fails the compile
## naming the family at the target's section.
## Expected compile failure: "genTargets: target msl is missing row family 'lit'"
##
## Run:
##   nim check --hints:off --warnings:off \
##     --nimcache:nimcache/tests workspace/crucible/tests/registry/manual_targets_missing_row.nim
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_macros

genTargets:
  target msl:
    scalar void "void", bool "bool", i8 "char", i16 "short", i32 "int", i64 "long",
      u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "half", f32 "float", f64 ""
    kind unrollDirective "#pragma unroll", braceSameLine true
    vector:
      elems i32, u32
      lanes 2, 3, 4
    bitcast:
      rule srcKinds {ctInt, ctUint, ctFloat}, srcWidths {8, 16, 32, 64},
        dstKinds {ctInt, ctUint, ctFloat}, sameWidth, "as_type<$2>($1)"
