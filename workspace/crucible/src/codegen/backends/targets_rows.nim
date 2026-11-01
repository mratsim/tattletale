# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Registry rows of the Metal Shading Language (MSL) target.
##
## Plain-data restatements of the static target values msl_lang.nim states in code.
## The MSL printer does not read the rows yet.
## A registry consumer test verifies them against the printer's behavior.

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
