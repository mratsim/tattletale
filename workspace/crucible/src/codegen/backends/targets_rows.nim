# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Registry rows of the Metal (MSL), CUDA C++, OpenCL C, GLSL, WebGPU WGSL and HLSL targets.
##
## Plain-data restatements of the static target values the six printers state in code:
## msl_lang.nim, cuda_lang.nim, opencl_lang.nim, glsl_lang.nim, wgsl_lang.nim and hlsl_lang.nim.
##
## - HLSL keys its signature row per flavor (`fvD3D12`, `fvVulkan`), every other family
##   is target-wide: the flavor changes the binding section only
## - printers do not read the rows yet, registry consumer tests verify each row
##   against the printer behavior it restates

import std/tables
import ../ir/cirv2_lang
import ./targets_registry
import ./targets_macros

genTargets:
  target msl:
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "bool", supported: true),
      # `char` and `long` are MSL's 8-bit signed and 64-bit type names.
      # CUDA names the same widths `signed char` and `long long`, each row states its own names
      ScalarCell(kind: ctInt, width: 8, spelling: "char", supported: true),
      ScalarCell(kind: ctInt, width: 16, spelling: "short", supported: true),
      ScalarCell(kind: ctInt, width: 32, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 64, spelling: "long", supported: true),
      ScalarCell(kind: ctUint, width: 8, spelling: "uchar", supported: true),
      ScalarCell(kind: ctUint, width: 16, spelling: "ushort", supported: true),
      ScalarCell(kind: ctUint, width: 32, spelling: "uint", supported: true),
      ScalarCell(kind: ctUint, width: 64, spelling: "ulong", supported: true),
      ScalarCell(kind: ctFloat, width: 16, spelling: "half", supported: true),
      ScalarCell(kind: ctFloat, width: 32, spelling: "float", supported: true),
      ScalarCell(kind: ctFloat, width: 64, supported: false, reasonClass: rcCapability)])
    lit: LitRow(
      int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
      f16Suffix: "h",
      f32Suffix: "f",
      nilToken: "nullptr",
      nonFinite: FloatLitTokens(
        inf16: "INFINITY", negInf16: "-INFINITY", nan16: "NAN",
        inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
        inf64: "INFINITY", negInf64: "-INFINITY", nan64: "NAN"))
    kind: KindRow(
      loop: LoopStyle(
        unrollDirective: "#pragma clang loop unroll(full)",
        infiniteHead: "for (;;) {\n",
        braceSameLine: true),
      ternary: tfQuestion,
      asmForm: afSpliceRaw,
      constDecl: cdfConstexpr,
      constructor: cofBraced)
    space: SpaceRow(
      ptrQualifier: ["thread", "threadgroup", "constant", "device"],
      localPrefix: ["", "threadgroup", "", ""],
      kernelParamScalar: "constant",
      kernelParamPointer: "device",
      smemParam: "threadgroup")
    conv: ConvRow(
      builtinNames: {
        "toFp16": "as_type<ushort>(half($1))",
        "toFp32": "float(as_type<half>($1))",
        "toBf16": "as_type<ushort>((bfloat)($1))",
        "toBf32": "float(as_type<bfloat>($1))",
        "asFp16": "$1",
        "asU16": "$1",
      }.toTable())
    coord: CoordRow(cells: @[
      CoordCell(kind: cbkThreadPositionInGrid, paramType: "uint3",
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadgroupPositionInGrid, paramType: "uint3",
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadPositionInThreadgroup, paramType: "uint3",
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadsPerThreadgroup, paramType: "uint3",
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadgroupsPerGrid, paramType: "uint3",
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadIndexInThreadgroup, paramType: "uint",
        whole: cwfParam, componentCast: "int")])
    sig: SigRow(
      entryLinkage: "kernel ",
      inlineLinkage: "inline ",
      voidRet: "void",
      bindingPrefix: "[[buffer($1)]]")
    vector: VectorRule(
      context: vcCoordinate,
      elems: @[(ctInt, 32, ""), (ctUint, 32, "")],
      lanes: @[2, 3, 4])

  target cuda:
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "bool", supported: true),
      ScalarCell(kind: ctInt, width: 8, spelling: "signed char", supported: true),
      ScalarCell(kind: ctInt, width: 16, spelling: "short", supported: true),
      ScalarCell(kind: ctInt, width: 32, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 64, spelling: "long long", supported: true),
      ScalarCell(kind: ctUint, width: 8, spelling: "unsigned char", supported: true),
      ScalarCell(kind: ctUint, width: 16, spelling: "unsigned short", supported: true),
      ScalarCell(kind: ctUint, width: 32, spelling: "unsigned int", supported: true),
      ScalarCell(kind: ctUint, width: 64, spelling: "unsigned long long", supported: true),
      # CUDA names fp16 `__half` behind cuda_fp16.h. This printer emits no
      # includes and no fp16 type name, so the cell refuses.
      ScalarCell(kind: ctFloat, width: 16, supported: false,
        reasonClass: rcOutsideSubset),
      ScalarCell(kind: ctFloat, width: 32, spelling: "float", supported: true),
      ScalarCell(kind: ctFloat, width: 64, spelling: "double", supported: true)])
    lit: LitRow(
      int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
      f32Suffix: "f",
      nilToken: "nullptr",
      nonFinite: FloatLitTokens(
        inf16: "INFINITY", negInf16: "-INFINITY", nan16: "NAN",
        inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
        inf64: "INFINITY", negInf64: "-INFINITY", nan64: "NAN"))
    kind: KindRow(
      loop: LoopStyle(
        unrollDirective: "#pragma unroll",
        infiniteHead: "for (;;) {\n",
        braceSameLine: true),
      ternary: tfQuestion,
      asmForm: afWrap,
      constDecl: cdfConstexpr,
      constructor: cofBraced)
    # CUDA has one generic address space: pointers, kernel parameters,
    # plain locals print unqualified. Kernel-scope shared storage prefixes
    # `__shared__`. Program-scope `__constant__` storage is module assembly,
    # not a row context.
    space: SpaceRow(
      ptrQualifier: ["", "", "", ""],
      localPrefix: ["", "__shared__", "", ""],
      kernelParamScalar: "",
      kernelParamPointer: "",
      smemParam: "")
    conv: ConvRow()
    # CUDA's coordinate builtins are implicit variables (blockIdx, threadIdx, ...),
    # no attributed parameter exists. Component reads cast to int.
    # Whole-value reads:
    # - a vector builtin composes the braced init of the use-site struct type
    # - the grid position references the canonical name
    # - the thread index in threadgroup is the x-major flat index
    coord: CoordRow(cells: @[
      CoordCell(kind: cbkThreadPositionInGrid,
        whole: cwfParam, componentCast: "int"),
      CoordCell(kind: cbkThreadgroupPositionInGrid,
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadPositionInThreadgroup,
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadsPerThreadgroup,
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadgroupsPerGrid,
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadIndexInThreadgroup,
        whole: cwfFlatIndex, componentCast: "int")])
    sig: SigRow(
      entryLinkage: "extern \"C\" __global__ ",
      deviceFnLinkage: "__device__ ",
      inlineLinkage: "__forceinline__ __device__ ",
      voidRet: "void")
    vector: VectorRule(context: vcNone)

  target opencl:
    # OpenCL C is LP64 C: `char` and `long` name the 8-bit and 64-bit signed widths, as on msl.
    # Bool prints as `int` to match the launch ABI.
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 8, spelling: "char", supported: true),
      ScalarCell(kind: ctInt, width: 16, spelling: "short", supported: true),
      ScalarCell(kind: ctInt, width: 32, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 64, spelling: "long", supported: true),
      ScalarCell(kind: ctUint, width: 8, spelling: "uchar", supported: true),
      ScalarCell(kind: ctUint, width: 16, spelling: "ushort", supported: true),
      ScalarCell(kind: ctUint, width: 32, spelling: "uint", supported: true),
      ScalarCell(kind: ctUint, width: 64, spelling: "ulong", supported: true),
      ScalarCell(kind: ctFloat, width: 16, spelling: "half", supported: true),
      ScalarCell(kind: ctFloat, width: 32, spelling: "float", supported: true),
      ScalarCell(kind: ctFloat, width: 64, spelling: "double", supported: true)])
    lit: LitRow(
      int: IntLitTokens(u64Suffix: "ULL", u32Suffix: "U", i64Suffix: "LL", narrowCast: true),
      f16Prefix: "(half)",
      f32Suffix: "f",
      nilToken: "0",
      nonFinite: FloatLitTokens(
        inf16: "(half)INFINITY", negInf16: "(half)-INFINITY", nan16: "(half)NAN",
        inf32: "INFINITY", negInf32: "-INFINITY", nan32: "NAN",
        inf64: "((double)INFINITY)", negInf64: "((double)-INFINITY)", nan64: "((double)NAN)"))
    kind: KindRow(
      loop: LoopStyle(
        unrollDirective: "#pragma unroll",
        infiniteHead: "for (;;) {\n",
        braceSameLine: true),
      ternary: tfQuestion,
      asmForm: afRefuse,
      constDecl: cdfConst,
      constructor: cofBraced)
    # Pointer declarations lead the address-space qualifier, single underscore:
    # `__global`, `__local`, `__constant`. Private pointers stay unqualified.
    # `restrict` is a function-parameter qualifier, a kernel's device pointer parameter takes it.
    # Shared locals prefix `__local`, private locals prefix `__private`.
    # Both prefixes apply to non-pointer locals only, a pointer type includes its own qualifier.
    # Program-scope `__constant` storage is module assembly, not a row context.
    space: SpaceRow(
      ptrQualifier: ["", "__local", "__constant", "__global"],
      localPrefix: ["__private", "__local", "", ""],
      kernelParamScalar: "",
      kernelParamPointer: "",
      kernelParamPointerRestrict: true,
      smemParam: "")
    conv: ConvRow()
    # OpenCL C names vectors by appending the lane count to the scalar name
    # (`float4`, `uchar3`), legal lengths 2, 3, 4, 8, 16.
    # Coordinate whole-value reads refuse, components read as dimension calls.
    coord: CoordRow(cells: @[
      CoordCell(kind: cbkThreadPositionInGrid, paramName: "get_global_id",
        whole: cwfRefuse, componentCast: "int", componentForm: ccDimCall),
      CoordCell(kind: cbkThreadgroupPositionInGrid, paramName: "get_group_id",
        whole: cwfRefuse, componentCast: "int", componentForm: ccDimCall),
      CoordCell(kind: cbkThreadPositionInThreadgroup, paramName: "get_local_id",
        whole: cwfRefuse, componentCast: "int", componentForm: ccDimCall),
      CoordCell(kind: cbkThreadsPerThreadgroup, paramName: "get_local_size",
        whole: cwfRefuse, componentCast: "int", componentForm: ccDimCall),
      CoordCell(kind: cbkThreadgroupsPerGrid, paramName: "get_num_groups",
        whole: cwfRefuse, componentCast: "int", componentForm: ccDimCall),
      # x-major flat thread index, text the glue proc emits
      CoordCell(kind: cbkThreadIndexInThreadgroup,
        whole: cwfFlatIndex, componentCast: "int", componentForm: ccDimCall)])
    sig: SigRow(
      entryLinkage: "__kernel void ",
      deviceFnLinkage: "static inline ",
      inlineLinkage: "static inline ",
      voidRet: "void")
    vector: VectorRule(
      context: vcEverywhere,
      elems: @[
        (ctVoid, 0, ""), (ctBool, 0, ""),
        (ctInt, 8, ""), (ctInt, 16, ""), (ctInt, 32, ""), (ctInt, 64, ""),
        (ctUint, 8, ""), (ctUint, 16, ""), (ctUint, 32, ""), (ctUint, 64, ""),
        (ctFloat, 16, ""), (ctFloat, 32, ""), (ctFloat, 64, "")],
      lanes: @[2, 3, 4, 8, 16])

  target glsl:
    # 8, 16 and 64-bit names compile behind the explicit-arithmetic-types extensions the module emits.
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "bool", supported: true),
      ScalarCell(kind: ctInt, width: 8, spelling: "int8_t", supported: true),
      ScalarCell(kind: ctInt, width: 16, spelling: "int16_t", supported: true),
      ScalarCell(kind: ctInt, width: 32, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 64, spelling: "int64_t", supported: true),
      ScalarCell(kind: ctUint, width: 8, spelling: "uint8_t", supported: true),
      ScalarCell(kind: ctUint, width: 16, spelling: "uint16_t", supported: true),
      ScalarCell(kind: ctUint, width: 32, spelling: "uint", supported: true),
      ScalarCell(kind: ctUint, width: 64, spelling: "uint64_t", supported: true),
      ScalarCell(kind: ctFloat, width: 16, spelling: "float16_t", supported: true),
      ScalarCell(kind: ctFloat, width: 32, spelling: "float", supported: true),
      ScalarCell(kind: ctFloat, width: 64, spelling: "double", supported: true)])
    # Wide integer and float literals print as constructor calls, glslang rejects the `LL` suffix.
    # GLSL has no infinity or NaN literal, IEEE-754 bit patterns go through `uintBitsToFloat`.
    lit: LitRow(
      int: IntLitTokens(
        u64Prefix: "uint64_t(", u64Suffix: ")",
        u32Suffix: "U",
        i64Prefix: "int64_t(", i64Suffix: ")",
        narrowCast: true),
      f16Prefix: "float16_t(", f16Suffix: ")",
      f32Suffix: "f",
      f64Prefix: "double(", f64Suffix: ")",
      nonFinite: FloatLitTokens(
        inf16: "float16_t(uintBitsToFloat(0x7f800000u))",
        negInf16: "float16_t(uintBitsToFloat(0xff800000u))",
        nan16: "float16_t(uintBitsToFloat(0x7fc00000u))",
        inf32: "uintBitsToFloat(0x7f800000u)",
        negInf32: "uintBitsToFloat(0xff800000u)",
        nan32: "uintBitsToFloat(0x7fc00000u)",
        inf64: "double(uintBitsToFloat(0x7f800000u))",
        negInf64: "double(uintBitsToFloat(0xff800000u))",
        nan64: "double(uintBitsToFloat(0x7fc00000u))"))
    kind: KindRow(
      loop: LoopStyle(
        unrollDirective: "[[unroll]]",
        infiniteHead: "for (;;) {\n",
        braceSameLine: true),
      ternary: tfQuestion,
      asmForm: afRefuse,
      constDecl: cdfConst,
      constructor: cofCall,
      targetExprKinds: {eNil, eAt, ePat, eDeref, eAddr})
    # A buffer pointer prints as its reference-block handle, no space qualifier.
    # Locals print unqualified, every other address space refuses in the printer.
    # Kernel parameters pack into the push-constant block with no qualifier.
    # Layout-block prefixes are module assembly and pass territory, not address-space contexts.
    space: SpaceRow(
      ptrQualifier: ["", "", "", ""],
      localPrefix: ["", "", "", ""],
      kernelParamScalar: "",
      kernelParamPointer: "",
      smemParam: "")
    conv: ConvRow()
    # All six coordinate builtins are native GLSL names.
    # Whole-value reads of the vector builtins compose the int vector from int-cast components.
    # Flat thread index whole reads wrap the native name in the int cast.
    coord: CoordRow(cells: @[
      CoordCell(kind: cbkThreadPositionInGrid, paramName: "gl_GlobalInvocationID",
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadgroupPositionInGrid, paramName: "gl_WorkGroupID",
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadPositionInThreadgroup, paramName: "gl_LocalInvocationID",
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadsPerThreadgroup, paramName: "gl_WorkGroupSize",
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadgroupsPerGrid, paramName: "gl_NumWorkGroups",
        whole: cwfVector, componentCast: "int"),
      CoordCell(kind: cbkThreadIndexInThreadgroup, paramName: "gl_LocalInvocationIndex",
        whole: cwfCast, componentCast: "int")])
    sig: SigRow(
      entryLinkage: "void ",
      voidRet: "void",
      workgroupAttr: "layout(local_size_x = $n, local_size_y = $n, local_size_z = $n) in;\n",
      workgroupDefault: [256, 1, 1],
      entryTakesParams: false)
    vector: VectorRule(
      context: vcEverywhere,
      elems: @[
        (ctFloat, 16, "f16vec"), (ctFloat, 32, "vec"), (ctFloat, 64, "dvec"),
        (ctInt, 32, "ivec"), (ctUint, 32, "uvec"), (ctBool, 0, "bvec")],
      lanes: @[2, 3, 4])


  target wgsl:
    # WGSL has no 8/16-bit integer types: 32 and 64 bit are its only integer widths.
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "bool", supported: true),
      ScalarCell(kind: ctInt, width: 8, supported: false, reasonClass: rcCapability),
      ScalarCell(kind: ctInt, width: 16, supported: false, reasonClass: rcCapability),
      ScalarCell(kind: ctInt, width: 32, spelling: "i32", supported: true),
      ScalarCell(kind: ctInt, width: 64, spelling: "i64", supported: true),
      ScalarCell(kind: ctUint, width: 8, supported: false, reasonClass: rcCapability),
      ScalarCell(kind: ctUint, width: 16, supported: false, reasonClass: rcCapability),
      ScalarCell(kind: ctUint, width: 32, spelling: "u32", supported: true),
      ScalarCell(kind: ctUint, width: 64, spelling: "u64", supported: true),
      ScalarCell(kind: ctFloat, width: 16, spelling: "f16", supported: true),
      ScalarCell(kind: ctFloat, width: 32, spelling: "f32", supported: true),
      ScalarCell(kind: ctFloat, width: 64, spelling: "f64", supported: true)])
    # Bare integer and float literals are abstract and work in any typed context,
    # u32 gains the `u` suffix and f16 the `h`, f32 and f64 print bare.
    # WGSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
    # `bitcast` from the unsigned integer of the same width. Nil literals refuse
    # at the site.
    lit: LitRow(
      int: IntLitTokens(u32Suffix: "u"),
      f16Suffix: "h",
      nonFinite: FloatLitTokens(
        inf16: "bitcast<f16>(0x7c00u)", negInf16: "bitcast<f16>(0xfc00u)", nan16: "bitcast<f16>(0x7e00u)",
        inf32: "bitcast<f32>(0x7f800000u)",
        negInf32: "bitcast<f32>(0xff800000u)",
        nan32: "bitcast<f32>(0x7fc00000u)",
        inf64: "bitcast<f64>(0x7ff0000000000000u)",
        negInf64: "bitcast<f64>(0xfff0000000000000u)",
        nan64: "bitcast<f64>(0x7ff8000000000000u)"))
    kind: KindRow(
      loop: LoopStyle(
        # WGSL has no unroll directive: the drop surfaces in the artifact
        unrollDirective: "// {.unroll.} requested but not honored on WebGPU",
        infiniteHead: "loop {\n",
        braceSameLine: true),
      ternary: tfSelectCall,
      asmForm: afRefuse,
      constDecl: cdfLet,
      constructor: cofCall,
      targetExprKinds: {eNil, eAt, ePat, eTernary})
    # Address-space words sit inside the pointer type (`ptr<storage, T>`), no qualifier precedes the type.
    # Storage pointers declare the access mode, array pointers refuse the pointer form (the binding is the array).
    # Locals print unqualified `var`, pointer locals refuse at the site, and WGSL has no `<uniform>` class.
    space: SpaceRow(
      ptrQualifier: ["", "", "", ""],
      localPrefix: ["", "", "", ""],
      kernelParamScalar: "",
      kernelParamPointer: "",
      smemParam: "",
      ptrSpace: ["function", "workgroup", "", "storage"],
      ptrAccessMode: "read_write",
      declVarPrefix: "var<workgroup>",
      bindingVarPrefix: "var<storage, read>",
      bindingBufferPrefix: "var<storage, read_write>")
    conv: ConvRow()
    # One `@builtin(...)` kernel parameter exists per referenced coordinate builtin,
    # kernel bodies read the parameter name. Whole-value reads compose the i32
    # vector from int-cast components and the scalar thread index casts whole.
    # threads_per_threadgroup refuses: WGSL exposes the threadgroup size only
    # through the `@workgroup_size` attribute.
    coord: CoordRow(cells: @[
      CoordCell(kind: cbkThreadPositionInGrid,
        paramName: "global_id", paramType: "vec3<u32>",
        paramAnnotation: "global_invocation_id",
        whole: cwfVector, componentCast: "i32"),
      CoordCell(kind: cbkThreadgroupPositionInGrid,
        paramName: "workgroup_id", paramType: "vec3<u32>",
        paramAnnotation: "workgroup_id",
        whole: cwfVector, componentCast: "i32"),
      CoordCell(kind: cbkThreadPositionInThreadgroup,
        paramName: "local_invocation_id", paramType: "vec3<u32>",
        paramAnnotation: "local_invocation_id",
        whole: cwfVector, componentCast: "i32"),
      CoordCell(kind: cbkThreadsPerThreadgroup,
        whole: cwfRefuse),
      CoordCell(kind: cbkThreadgroupsPerGrid,
        paramName: "num_workgroups", paramType: "vec3<u32>",
        paramAnnotation: "num_workgroups",
        whole: cwfVector, componentCast: "i32"),
      CoordCell(kind: cbkThreadIndexInThreadgroup,
        paramName: "local_invocation_index", paramType: "u32",
        paramAnnotation: "local_invocation_index",
        whole: cwfCast, componentCast: "i32", componentForm: ccWhole)])
    sig: SigRow(
      entryLinkage: "fn ",
      deviceFnLinkage: "fn ",
      # WGSL has no void type name, a void return emits no arrow
      voidRet: "",
      entryAttr: "@compute ",
      workgroupAttr: "@workgroup_size($n, $n, $n)\n",
      workgroupDefault: [64, 1, 1],
      bindingPrefix: "@group(0) @binding($1)",
      entryTakesParams: false)
    vector: VectorRule(
      context: vcEverywhere,
      elems: @[
        (ctVoid, 0, "vec"), (ctBool, 0, "vec"),
        (ctInt, 32, "vec"), (ctInt, 64, "vec"),
        (ctUint, 32, "vec"), (ctUint, 64, "vec"),
        (ctFloat, 16, "vec"), (ctFloat, 32, "vec"), (ctFloat, 64, "vec")],
      lanes: @[2, 3, 4],
      nameForm: vnfGenericElem)

  target hlsl:
    # 8/16-bit integers and int64_t are parse errors in the HLSL frontend,
    # uint64_t pulls the SPIR-V Int64 capability, float16 is outside the subset.
    scalar: ScalarRow(cells: @[
      ScalarCell(kind: ctVoid, width: 0, spelling: "void", supported: true),
      ScalarCell(kind: ctBool, width: 0, spelling: "bool", supported: true),
      ScalarCell(kind: ctInt, width: 8, supported: false, reasonClass: rcParseError),
      ScalarCell(kind: ctInt, width: 16, supported: false, reasonClass: rcParseError),
      ScalarCell(kind: ctInt, width: 32, spelling: "int", supported: true),
      ScalarCell(kind: ctInt, width: 64, supported: false, reasonClass: rcParseError),
      ScalarCell(kind: ctUint, width: 8, supported: false, reasonClass: rcParseError),
      ScalarCell(kind: ctUint, width: 16, supported: false, reasonClass: rcParseError),
      ScalarCell(kind: ctUint, width: 32, spelling: "uint", supported: true),
      ScalarCell(kind: ctUint, width: 64, supported: false, reasonClass: rcOutsideSubset),
      ScalarCell(kind: ctFloat, width: 16, supported: false, reasonClass: rcOutsideSubset),
      ScalarCell(kind: ctFloat, width: 32, spelling: "float", supported: true),
      ScalarCell(kind: ctFloat, width: 64, spelling: "double", supported: true)])
    # HLSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
    # `asfloat` (32 bit) and `asdouble` (64 bit). Without the suffix the HLSL
    # frontend changes width or signedness. Nil literals refuse at the site.
    lit: LitRow(
      int: IntLitTokens(u32Suffix: "u", u64Suffix: "u"),
      f16Suffix: "f",
      f32Suffix: "f",
      f64Suffix: "l",
      nonFinite: FloatLitTokens(
        inf16: "asfloat(0x7f800000)", negInf16: "asfloat(0xff800000)", nan16: "asfloat(0x7fc00000)",
        inf32: "asfloat(0x7f800000)", negInf32: "asfloat(0xff800000)", nan32: "asfloat(0x7fc00000)",
        inf64: "asdouble(0u, 0x7ff00000u)",
        negInf64: "asdouble(0u, 0xfff00000u)",
        nan64: "asdouble(0u, 0x7ff80000u)"))
    kind: KindRow(
      loop: LoopStyle(
        unrollDirective: "[unroll]",
        infiniteHead: "for (;;)\n",
        braceSameLine: false),
      ternary: tfQuestion,
      foldsIteElse: true,
      emitOperandRendersExpr: true,
      asmForm: afRefuse,
      constDecl: cdfConst,
      constructor: cofMixed,
      targetExprKinds: {eNil, eAt, ePat, eDeref, eAddr})
    # HLSL has no address-space qualifiers: pointer, threadgroup and device-space
    # refuse at the site. Const_mem locals hoist to file-scope `static const` module assembly.
    # Kernel values pack into the push block.
    space: SpaceRow(
      ptrQualifier: ["", "", "", ""],
      localPrefix: ["", "", "", ""],
      kernelParamScalar: "",
      kernelParamPointer: "",
      smemParam: "")
    conv: ConvRow(
      fpRefusals: @[
        "toFp16", "toFp32", "toBf16", "toBf32", "toFp8e4m3", "toFp8e5m2",
        "asFp16", "asU16"])
    # Coordinate semantics are synthesized `main` inputs, exactly the ones the kernel
    # body reads. threads_per_threadgroup components substitute the baked workgroup
    # dimensions at emission, and the scalar thread index reads whole as `int(gidx)`.
    # threadgroups_per_grid refuses: the HLSL lowering pass materializes the dispatch grid size, and whole-vector reads refuse.
    coord: CoordRow(cells: @[
      CoordCell(kind: cbkThreadPositionInGrid,
        paramName: "gid", paramType: "uint3", paramAnnotation: "SV_DispatchThreadID",
        whole: cwfRefuse, componentCast: "int"),
      CoordCell(kind: cbkThreadgroupPositionInGrid,
        paramName: "groupId", paramType: "uint3", paramAnnotation: "SV_GroupID",
        whole: cwfRefuse, componentCast: "int"),
      CoordCell(kind: cbkThreadPositionInThreadgroup,
        paramName: "groupThreadId", paramType: "uint3", paramAnnotation: "SV_GroupThreadID",
        whole: cwfRefuse, componentCast: "int"),
      CoordCell(kind: cbkThreadsPerThreadgroup,
        whole: cwfRefuse, componentForm: ccWorkgroupDim),
      CoordCell(kind: cbkThreadgroupsPerGrid,
        whole: cwfRefuse),
      CoordCell(kind: cbkThreadIndexInThreadgroup,
        paramName: "gidx", paramType: "uint", paramAnnotation: "SV_GroupIndex",
        whole: cwfCast, componentCast: "int", componentForm: ccWhole)])
    vector: VectorRule(
      context: vcEverywhere,
      elems: @[
        (ctBool, 0, "bool"), (ctInt, 32, "int"), (ctUint, 32, "uint"),
        (ctFloat, 32, "float")],
      lanes: @[2, 3, 4])

  target hlsl(fvD3D12):
    sig: SigRow(
      entryLinkage: "void ",
      inlineLinkage: "inline ",
      voidRet: "void",
      workgroupAttr: "[numthreads($n, $n, $n)]\n",
      workgroupDefault: [256, 1, 1],
      bindingPrefix: ": register($1, space0)",
      entryTakesParams: false)

  target hlsl(fvVulkan):
    # Zero-descriptor flavor bindings go through the `[[vk::push_constant]]` struct,
    # member offsets lock at emission, no register form exists
    sig: SigRow(
      entryLinkage: "void ",
      inlineLinkage: "inline ",
      voidRet: "void",
      workgroupAttr: "[numthreads($n, $n, $n)]\n",
      workgroupDefault: [256, 1, 1],
      entryTakesParams: false)

## Cells that two targets name on purpose differently: msl names 8-bit signed
## `char` and 64-bit signed `long`, CUDA names the same widths `signed char`
## and `long long`. The ledger stores no names, so converging the two rows
## fails the compile.
const ScalarDivergences* = [
  (a: "msl", b: "cuda", kind: ctInt, width: 8),
  (a: "msl", b: "cuda", kind: ctInt, width: 64)]

static:
  checkScalarDivergences(
    [("msl", mslScalarRow()), ("cuda", cudaScalarRow())], ScalarDivergences)
