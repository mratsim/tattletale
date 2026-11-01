## genTargets: a type name drifted onto another cell fails the compile.
##
## Test that a shared type name denoting different cells across two targets fails the compile at the drifted cell.
## Here cuda names its 32-bit float `half` and msl names the 16-bit float.
## Expected compile failure: "genTargets: name 'half' denotes (ctFloat, 16) on msl but (ctFloat, 32) on cuda"
##
## Run:
##   nim check --hints:off --warnings:off \
##     --nimcache:nimcache/tests workspace/crucible/tests/registry/manual_targets_drifted_cell.nim
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/backends/targets_registry
import workspace/crucible/src/codegen/backends/targets_macros

genTargets:
  target msl:
    scalar void "void", bool "bool", i8 "char", i16 "short", i32 "int", i64 "long",
      u8 "uchar", u16 "ushort", u32 "uint", u64 "ulong", f16 "half", f32 "float", f64 "double"
    lit u32Suffix "U", nilToken "nullptr"
    kind unrollDirective "#pragma unroll", braceSameLine true
    vector:
      elems i32, u32
      lanes 2, 3, 4
    bitcast:
      rule srcKinds {ctInt, ctUint, ctFloat}, srcWidths {8, 16, 32, 64},
        dstKinds {ctInt, ctUint, ctFloat}, sameWidth, "as_type<$2>($1)"

  target cuda:
    scalar void "void", bool "bool", i8 "signed char", i16 "short", i32 "int", i64 "long long",
      u8 "unsigned char", u16 "unsigned short", u32 "unsigned int", u64 "unsigned long long",
      f32 "half", f64 "double"
    lit u32Suffix "U", nilToken "nullptr"
    kind unrollDirective "#pragma unroll", braceSameLine true
    vector ()
    bitcast:
      rule srcKinds {ctFloat}, srcWidths {32}, dstKinds {ctUint}, sameWidth, "__float_as_uint($1)"
