# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Lowering: reserved-keyword renaming, one common pass registered per target
## with the target's reserved set as data.
##
## An identifier colliding with a target reserved word takes the `_nim` suffix
## at every emitted site, so the printers never rename. A target whose language
## reserves the double-underscore identifier head additionally prefixes every
## `__`-prefixed name (WGSL: `wgsl_`), the suffix applies first:
##
## - fn names, call sites, params and their body references
## - locals (var/const) and their body references
## - struct type names, struct fields and their field accesses
## - object-construction field names
##
## The rename is name-based per site list. A local and a proc sharing a name
## are one symbol after the frontend's flat lowering, they rename together.

import std/[sets, strutils, tables]
import ../ir/cirv2_lang
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# The target reserved sets
# ═══════════════════════════════════════════════════════════════════════

const MetalReserved* = [
  # C++/MSL keyword table (MSL is C++14-based)
  "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
  "bool", "break", "case", "catch", "char", "char16_t", "char32_t", "class",
  "compl", "const", "const_cast", "constexpr", "continue", "decltype",
  "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
  "explicit", "export", "extern", "float", "for", "friend", "goto",
  "half", "if", "inline", "int", "long", "mutable", "namespace", "new",
  "noexcept", "not", "not_eq", "nullptr", "object_data", "operator", "or",
  "or_eq", "private", "protected", "public", "ray_data", "register",
  "reinterpret_cast", "return", "short", "signed", "sizeof", "static",
  "static_assert", "static_cast", "struct", "switch", "template", "this",
  "thread", "thread_local", "threadgroup", "threadgroup_imageblock", "throw",
  "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
  "vertex", "virtual", "void", "volatile", "wchar_t", "while", "xor",
  "xor_eq",
  # MSL address-space qualifiers and scalar type names
  "kernel", "device", "constant", "threadgroup", "uchar", "ushort", "ulong",
  # `metal` is the MSL namespace, not a keyword
  "metal",
].toHashSet()

const GlslReserved* = [
  "output", "input", "in", "out", "attribute", "uniform", "varying",
  "buffer", "shared", "layout", "main", "void", "return",
  "if", "else", "for", "while", "break", "continue", "struct",
  "const", "true", "false", "bool", "int", "uint", "float", "double",
  "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "uvec2", "uvec3",
  "uvec4", "bvec2", "bvec3", "bvec4", "dvec2", "dvec3", "dvec4",
  "mat2", "mat3", "mat4", "sampler", "texture", "image", "subroutine",
  "discard", "precise", "switch", "case", "default", "do", "goto",
  "long", "short", "signed", "unsigned", "sizeof", "static", "union",
  "enum", "class", "this", "template", "namespace", "using", "new",
  "delete", "operator", "not", "and", "or", "xor", "cast", "conv",
].toHashSet()

const OpenclReserved* = [
  "kernel", "__kernel", "global", "__global", "local", "__local",
  "constant", "__constant", "read_only", "write_only", "read_write"
].toHashSet()

const WgslReserved* = [
  "NULL", "Self", "abstract", "active", "alignas", "alignof", "alias", "as",
  "asm", "asm_fragment", "async", "attribute", "auto", "await", "become",
  "binding_array", "break", "case", "cast", "catch", "class", "co_await",
  "co_return", "co_yield", "coherent", "column_major", "common", "compile",
  "compile_fragment", "concept", "const", "const_assert", "const_cast",
  "consteval", "constexpr", "constinit", "continue", "continuing", "crate",
  "debugger", "decltype", "delete", "demote", "demote_to_helper",
  "diagnostic", "discard", "do", "dynamic_cast", "else", "enable", "enum",
  "explicit", "export", "extends", "extern", "external", "fallthrough",
  "false", "filter", "final", "finally", "fn", "for", "friend", "from",
  "fxgroup", "get", "goto", "groupshared", "highp", "impl", "implements",
  "import", "inline", "instanceof", "interface", "layout", "let", "loop",
  "lowp", "macro", "macro_rules", "match", "mediump", "meta", "mod",
  "module", "move", "mut", "mutable", "namespace", "new", "nil", "noexcept",
  "noinline", "nointerpolation", "noperspective", "null", "nullptr",
  "override", "of", "operator", "package", "packoffset", "partition", "pass",
  "patch", "pixelfragment", "precise", "precision", "premerge", "priv",
  "protected", "pub", "public", "readonly", "ref", "regardless", "register",
  "requires", "reinterpret_cast", "require", "resource", "restrict",
  "return", "self", "set", "shared", "sizeof", "smooth", "snorm", "static",
  "static_assert", "static_cast", "std", "struct", "subroutine", "super",
  "switch", "target", "template", "this", "thread_local", "throw", "trait",
  "true", "try", "type", "typedef", "typeof", "union", "unless", "unorm",
  "unsafe", "unsized", "use", "using", "var", "varying", "virtual",
  "volatile", "wgsl", "where", "while", "with", "writeonly", "yield",
  "bool", "f16", "f32", "f64", "i32", "i64", "u32", "u64",
  "vec2", "vec3", "vec4", "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3",
  "mat3x4", "mat4x2", "mat4x3", "mat4x4", "array", "atomic", "ptr",
  "sampler", "texture_1d", "texture_2d", "texture_3d", "texture_cube",
].toHashSet()

const HlslReserved* = [
  # statements and declarations
  "if", "else", "for", "while", "do", "switch", "case", "default",
  "break", "continue", "return", "discard", "struct", "class", "interface",
  "typedef", "namespace", "using", "cbuffer", "tbuffer", "register",
  "packoffset", "technique", "pass", "compile", "main",
  # storage qualifiers and modifiers
  "static", "const", "extern", "uniform", "volatile", "shared", "groupshared",
  "inline", "in", "out", "inout", "precise", "row_major", "column_major",
  # builtin scalar, vector and matrix type names
  "void", "bool", "int", "uint", "float", "double", "half", "fixed", "string",
  "int1", "int2", "int3", "int4", "uint1", "uint2", "uint3", "uint4",
  "float1", "float2", "float3", "float4", "bool1", "bool2", "bool3", "bool4",
  "half2", "half3", "half4", "double1", "double2", "double3", "double4",
  "matrix", "float2x2", "float3x3", "float4x4", "int2x2", "int3x3", "int4x4",
  "uint2x2", "uint3x3", "uint4x4", "bool2x2", "bool3x3", "bool4x4",
  # fixed-width type names: int64_t and uint64_t are rejected by validation,
  # remaining stdint names are plain parse errors in the HLSL frontend
  "int32_t", "uint32_t", "float32_t", "float64_t", "int64_t", "uint64_t",
  # resource types the printer emits and their siblings
  "ConstantBuffer", "TextureBuffer", "StructuredBuffer", "RWStructuredBuffer",
  "AppendStructuredBuffer", "ConsumeStructuredBuffer", "ByteAddressBuffer",
  "RWByteAddressBuffer", "Texture1D", "Texture2D", "Texture3D", "TextureCube",
  "RWTexture1D", "RWTexture2D", "RWTexture3D",
  "SamplerState", "SamplerComparisonState",
  # semantics names the printer emits and their siblings
  "numthreads", "SV_DispatchThreadID", "SV_GroupID", "SV_GroupThreadID",
  "SV_GroupIndex", "SV_Position", "SV_NumWorkgroups", "SV_Target",
  # literals
  "true", "false",
].toHashSet()

const CudaReserved* = [
  # execution-space qualifiers the CUDA-C++ emitter writes.
  # a user identifier of one of these names collides with an emitted qualifier
  "__global__", "__device__", "__shared__", "__constant__",
].toHashSet()

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

template registerRenameReservedIdentifiers*(reg: var PassRegistry, passName: string, reserved: untyped, doubleUnderscorePrefix: string = "") =
  ## Registers the reserved-keyword renaming for one target, the run order
  ## name-matching passes -> reserved renaming -> printing.
  ##
  ## Args:
  ## - `passName`: the registry key, one registration per target
  ## - `reserved`: the target's reserved words, one of `MetalReserved`,
  ##   `GlslReserved`, `OpenclReserved`, `WgslReserved`, `HlslReserved`,
  ##   `CudaReserved`
  ## - `doubleUnderscorePrefix`: the prefix every `__`-prefixed name gains,
  ##   "" for targets without a reserved double-underscore head (WGSL: `wgsl_`)
  reg.register(passName, pkTransform, phaseMain,
    "Renames identifiers colliding with backend reserved words: fn names, call sites, params, locals, struct fields, field accesses",
    run = proc(m: CirModule): string {.nimcall.} =
      renameReservedIdentifiers(m, reserved, doubleUnderscorePrefix)
  )

# ═══════════════════════════════════════════════════════════════════════
# The pass
# ═══════════════════════════════════════════════════════════════════════

func safeIdent(name: string, reserved: HashSet[string], doubleUnderscorePrefix: string): string =
  ## Returns `name` unchanged when it is neither reserved nor `__`-prefixed,
  ## else the renamed form:
  ##
  ## - a reserved word takes the `_nim` suffix
  ## - a surviving `__` head gains `doubleUnderscorePrefix` in front, the suffix applies first
  var safe = if name in reserved: name & "_nim" else: name
  if doubleUnderscorePrefix.len > 0 and safe.startsWith("__"):
    safe = doubleUnderscorePrefix & safe
  result = safe

proc renameFieldAccesses(e: CirExpr, fieldRenames, symRenames: Table[string, string]) =
  ## Renames field accesses and symbol references through the maps, in place.
  ## Expressions are refs, the rename applies to the shared node.
  if e.isNil:
    return
  case e.kind
  of eSym:
    if symRenames.hasKey(e.name):
      e.name = symRenames[e.name]
    return
  of eDot:
    if fieldRenames.hasKey(e.fieldName):
      e.fieldName = fieldRenames[e.fieldName]
    renameFieldAccesses(e.dotBase, fieldRenames, symRenames)
    return
  of eOconstr:
    # constructed field names take the struct fields' renames, the member values walk through the children below
    for f in mitems(e.kvFields):
      if fieldRenames.hasKey(f.name):
        f.name = fieldRenames[f.name]
  else:
    discard
  for ch in getExprChildren(e):
    renameFieldAccesses(ch, fieldRenames, symRenames)

proc renameStmts(ss: var seq[CirStmt], fieldRenames, symRenames: Table[string, string]) =
  ## Renames the identifiers of one statement list, nested bodies included.
  for s in mitems(ss):
    case s.kind
    of sVar, sConst:
      renameFieldAccesses(s.init, fieldRenames, symRenames)
    of sAsgn:
      renameFieldAccesses(s.lhs, fieldRenames, symRenames)
      renameFieldAccesses(s.rhs, fieldRenames, symRenames)
    of sIte:
      renameFieldAccesses(s.cond, fieldRenames, symRenames)
      renameStmts(s.thenBody, fieldRenames, symRenames)
      renameStmts(s.elseBody, fieldRenames, symRenames)
    of sLoop:
      renameStmts(s.loopBody, fieldRenames, symRenames)
    of sScope:
      renameStmts(s.scopeBody, fieldRenames, symRenames)
    of sAssume:
      renameFieldAccesses(s.assumeCond, fieldRenames, symRenames)
    of sRet:
      renameFieldAccesses(s.retVal, fieldRenames, symRenames)
    of sDiscard:
      renameFieldAccesses(s.discardVal, fieldRenames, symRenames)
    of sCall:
      renameFieldAccesses(s.callCallee, fieldRenames, symRenames)
      for a in mitems(s.callArgs):
        renameFieldAccesses(a, fieldRenames, symRenames)
    else:
      discard

proc renameReservedIdentifiers(m: CirModule, reserved: HashSet[string], doubleUnderscorePrefix: string = "") =
  ## Args:
  ## - `m`: the module, renamed in place
  ## - `reserved`: the target's reserved words
  ## - `doubleUnderscorePrefix`: the prefix every `__`-prefixed name gains, "" when none
  ##
  ## The `_nim` suffix marks every renamed identifier, declarations and uses
  ## rename through one map so the emitted source stays consistent.
  var fieldRenames: Table[string, string]
  for t in m.types:
    if t.kind == ctStruct:
      # type name renames once here, its uses inherit the renamed name
      let safeName = safeIdent(t.name, reserved, doubleUnderscorePrefix)
      if safeName != t.name:
        fieldRenames[t.name] = safeName
        t.name = safeName
      for f in mitems(t.fields):
        let safe = safeIdent(f.name, reserved, doubleUnderscorePrefix)
        if safe != f.name:
          fieldRenames[f.name] = safe
          f.name = safe
  # function names rename once at module level, one shared map
  # feeds the per-proc symbol walks so every call site renames with its callee
  var fnRenames: Table[string, string]
  for fn in m.procs:
    let safeFn = safeIdent(fn.name, reserved, doubleUnderscorePrefix)
    if safeFn != fn.name:
      fnRenames[fn.name] = safeFn
      fn.name = safeFn
  for fn in m.procs:
    var symRenames = fnRenames
    for p in mitems(fn.params):
      let safe = safeIdent(p.name, reserved, doubleUnderscorePrefix)
      if safe != p.name:
        symRenames[p.name] = safe
        p.name = safe
    # declare-then-rename: the walk reads the maps, the declarations feed them
    var decls: seq[tuple[name: string, safe: string]]
    for s in fn.body.mitems:
      case s.kind
      of sVar, sConst:
        let safe = safeIdent(s.name, reserved, doubleUnderscorePrefix)
        if safe != s.name:
          decls.add((s.name, safe))
          s.name = safe
      else:
        discard
    for d in decls:
      symRenames[d.name] = d.safe
    fn.body.renameStmts(fieldRenames, symRenames)
