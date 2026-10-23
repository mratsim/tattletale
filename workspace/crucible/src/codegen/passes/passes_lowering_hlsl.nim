# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            HLSL lowering passes
#
# ############################################################
#
# Lower the crucible IR onto the HLSL storage model.
# Registered by the HLSL pipeline only.
# The bundle matches the GLSL chain's shape minus two passes,
# the HLSL resource model has no use for them:
# - insertPointerDeref: the printer reads through gpuDeref on index bases,
#   explicit derefs are unnecessary IR
# - qualifyVarParams: `var` kernel parameters have no HLSL binding,
#   there is nothing to qualify
#
# DXC compiles multi-function HLSL, so device functions emit as functions,
# not spliced: no pass rewrites kernel bodies to inline callees.

import std / sets
import ./pass_registry
import ./passes_lowering
import ./passes_low_views
import ./passes_low_array_assign
import ./passes_low_address_spaces

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

const HlslReserved = [
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
  # the remaining stdint names are plain parse errors in the HLSL frontend
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

func registerHlslLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerArraysAndViews(decayArrays = false)
  reg.registerSharedLoweringPasses()
  reg.registerDecomposeArrayAssign()
  reg.registerRenameReservedIdentifiers("hlslReservedIdentifiers", HlslReserved)
  reg.registerInstantiateImplicitAddressSpaceGenerics()
