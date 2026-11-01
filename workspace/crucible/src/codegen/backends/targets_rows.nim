# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Registry rows of the CUDA C++ and Metal Shading Language (MSL) targets.
##
## Plain-data restatements of the static target values cuda_lang.nim and msl_lang.nim state in code.
## The printers do not read the rows yet.
## Registry consumer tests verify them against the printers' behavior.

import std/tables
import ../ir/cirv2_lang
import ./emit_common
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
      elems: @[(ctInt, 32), (ctUint, 32)],
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
