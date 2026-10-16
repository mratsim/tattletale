# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Kernel arg binding contract tests:
## - classification, buffer / bytes / static / whole-value chunk
## - layout agreement between the bridge's AST walk and the engines' typed walk
## - scalar byte widths, one lowering per backend ABI
## - the array rule: an array arg is static iff its element type is static, dynamic arrays cross raw
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests workspace/crucible/tests/runtime/test_kernel_args.nim

import std/[macros, unittest, strutils]
import workspace/crucible/src/runtime/kernel_args

type
  ConstInt[V: static int] = object
    ## fieldless: the value lives in the type

  StaticArr = array[2, ConstInt[128]]
  DynArr = array[3, int32]
  Args = tuple[x: DynArr]

  Mixed = object
    ## one arg of each classification: pointer, static subtrees, dynamic scalars, a static array, a dynamic array
    data: ptr UncheckedArray[uint32]
    shape: tuple[a: ConstInt[128], b: ConstInt[64]]
    stride: tuple[a: int32, b: int32]
    pad: StaticArr
    xs: DynArr

  PtrArg = object
    p: pointer
    n: int32

  Whole = object
    ## ptr-free composite: binds as one chunk in whole-value mode, fields
    ## packed flat in field order at the crucible-lowered widths
    m, n, k: int32
    strideA: int64
    scale: float32

  Inner = object
    ## ptr-free composite nested in a ptr holder: the exploded outer
    ## holds one chunk, only the buffer leaf explodes the outer
    m: int32
    strideA: int64

  Outer = object
    p: pointer
    inner: Inner

  StoreMask {.size: sizeof(uint32).} = enum
    smStoreNone, smStoreAll, smStoreDiag

  TestHandle = distinct pointer

# ── bridge-side helpers: the AST walk the descriptor records ────────────────

macro astLayoutOf(vals: varargs[typed]): KernelArgLayout =
  ## Returns the bridge-side layout of the launch values, built by the same
  ## reconstructKernelArg walk buildGlobalFn runs
  var layout: KernelArgLayout
  var params = nnkFormalParams.newTree(newEmptyNode())
  for v in vals:
    let t = v.getTypeInst()
    discard reconstructKernelArg("t", t, t.getTypeImpl(), params, layout)
  result = newLit(layout)

macro astIsStaticArg(v: typed): bool =
  newLit(isStaticKernelArg(v.getTypeInst()))

macro astArrayBridgeValue(v: typed): string =
  ## Returns the bridge-body value a static-array arg rebuilds from its type:
  ## an N-element bracket literal of element constructors
  let t = v.getTypeInst()
  let val = constantKernelValue(t)
  result = newLit($val.kind & ":" & $val.len & ":" & $val[0].kind)

macro irKindNode(T: typed): GpuTypeKind =
  ## Returns the codegen's own scalar lowering of T (toGpuTypeKind), the AST-side
  ## projection the typed chain is checked against
  var t = T.getTypeInst()
  if t.kind == nnkBracketExpr and t[0].typeKind == ntyTypeDesc:
    t = t[1]
  newLit(toGpuTypeKind(t.typeKind))

# ── independent expected layouts ────────────────────────────────────────────

# independent expected layout: data binds as a buffer, the two stride
# scalars pack as int32, xs crosses as one gtArray chunk of 3 elements
const mixedExpected = KernelArgLayout(args: @[
  KernelArg(kind: kaBuffer),
  KernelArg(kind: kaBytes, irKind: gtInt32),
  KernelArg(kind: kaBytes, irKind: gtInt32),
  KernelArg(kind: kaBytes, irKind: gtArray, count: 3),
])

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Kernel args - contract":

    test "static vs dynamic classification agrees in both domains":
      # static: fieldless ConstInt, tuples of them, arrays of them
      check isStaticKernelArg[ConstInt[128]]()
      check isStaticKernelArg[tuple[a: ConstInt[128], b: ConstInt[64]]]()
      check isStaticKernelArg[StaticArr]()
      check isStaticKernelArg[array[0, int32]]()   # no elements: vacuously static
      check astIsStaticArg(ConstInt[128]())
      check astIsStaticArg([ConstInt[128](), ConstInt[128]()])
      # dynamic: scalars, pointers, arrays of scalars
      check not isStaticKernelArg[int32]()
      check not isStaticKernelArg[ptr uint32]()
      check not isStaticKernelArg[DynArr]()
      check not isStaticKernelArg[Mixed]()
      check not astIsStaticArg(int32(1))
      check not astIsStaticArg(DynArr.default)

    test "bridge and engine walks record the same layout for one arg of each kind":
      const astL = astLayoutOf(Mixed.default)
      const typedL = kernelArgLayout[Mixed, tuple[], void](kagExploded, bkMetal)
      check astL == typedL
      check astL == mixedExpected

    test "layout agreement for scalar, pointer and array top-level values":
      const astL = astLayoutOf(int32(1), cast[ptr uint32](nil), DynArr.default,
                               StaticArr.default)
      const typedL = kernelArgLayout[int32, (ptr uint32, DynArr, StaticArr), void](kagExploded, bkMetal)
      check astL == typedL
      check astL.args.len == 3   # the static array records nothing
      check astL.args[0] == KernelArg(kind: kaBytes, irKind: gtInt32)
      check astL.args[1] == KernelArg(kind: kaBuffer)
      check astL.args[2] == KernelArg(kind: kaBytes, irKind: gtArray, count: 3)

    test "static arrays rebuild from the type as an N-element bracket literal":
      check astArrayBridgeValue(StaticArr.default) == "nnkBracket:2:nnkCall"
      check astArrayBridgeValue([ConstInt[7](), ConstInt[7](), ConstInt[7]()]) ==
        "nnkBracket:3:nnkCall"

    test "byte widths match the codegen's kernel ABI per backend":
      # the lowering the codegen applies to kernel params (toGpuTypeKind):
      # int/uint lower to 32-bit, bool to a 4-byte int except CUDA's C++ bool
      check kernelArgByteWidth(gtBool, bkMetal) == 4
      check kernelArgByteWidth(gtBool, bkOpenCL) == 4
      check kernelArgByteWidth(gtBool, bkVulkan) == 4
      check kernelArgByteWidth(gtBool, bkWGSL) == 4
      check kernelArgByteWidth(gtBool, bkCuda) == 1
      check kernelArgByteWidth(gtInt32, bkMetal) == 4
      check kernelArgByteWidth(gtUint32, bkCuda) == 4
      check kernelArgByteWidth(gtInt64, bkOpenCL) == 8
      check kernelArgByteWidth(gtUint64, bkVulkan) == 8
      check kernelArgByteWidth(gtFloat32, bkWGSL) == 4
      check kernelArgByteWidth(gtFloat64, bkMetal) == 8
      check kernelArgByteWidth(gtUint8, bkOpenCL) == 1
      check kernelArgByteWidth(gtInt16, bkMetal) == 2
      check kernelArgByteWidth(gtUint16, bkCuda) == 2
      check kernelArgByteWidth(gtFloat16, bkWGSL) == 2
      check kernelArgByteWidth(gtBf16, bkMetal) == 2
      expect AssertionError:
        discard kernelArgByteWidth(gtArray, bkMetal)

    test "the typed IR-kind chain matches the codegen's own lowering":
      check irKindNode(bool) == kernelArgIrKind[bool]()
      check irKindNode(int) == kernelArgIrKind[int]()
      check irKindNode(uint) == kernelArgIrKind[uint]()
      check irKindNode(uint8) == kernelArgIrKind[uint8]()
      check irKindNode(int16) == kernelArgIrKind[int16]()
      check irKindNode(uint16) == kernelArgIrKind[uint16]()
      check irKindNode(int32) == kernelArgIrKind[int32]()
      check irKindNode(uint32) == kernelArgIrKind[uint32]()
      check irKindNode(int64) == kernelArgIrKind[int64]()
      check irKindNode(uint64) == kernelArgIrKind[uint64]()
      check irKindNode(float32) == kernelArgIrKind[float32]()
      check irKindNode(float64) == kernelArgIrKind[float64]()

    test "the launch gate rejects a descriptor the engine binding disagrees with":
      const wrong = KernelArgLayout(args: @[
        KernelArg(kind: kaBytes, irKind: gtInt32)])
      check not compiles(checkedKernelArgLayout[Mixed, tuple[], void](wrong, bkMetal))
      # Mixed holds a pointer, whole-value mode explodes it, its ptr-free
      # stride tuple binds as one 8-byte chunk
      const own = checkedKernelArgLayout[Mixed, tuple[], void](KernelArgLayout(), bkMetal)
      check own == KernelArgLayout(args: @[
        KernelArg(kind: kaBuffer),
        KernelArg(kind: kaStruct, width: 8),
        KernelArg(kind: kaBytes, irKind: gtArray, count: 3)])

    test "the binder packs one raw chunk per dynamic arg in explosion order":
      var m = Mixed(
        stride: (11'i32, -7'i32),
        xs: [10'i32, 20, 30])
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[Mixed, tuple[], TestHandle](kagExploded, bkMetal)
      bindKernelArgs[TestHandle](b, l, bkMetal, m, (), ptrAsBuffer = true, argAlign = 1)
      check b.slots.len == 4
      check b.slots[0].isBuffer            # data
      check not b.slots[1].isBuffer and b.slots[1].width == 4
      check not b.slots[2].isBuffer and b.slots[2].width == 4
      check not b.slots[3].isBuffer and b.slots[3].width == 12   # xs, raw
      # int32 chunks and the raw array chunk, little-endian host order
      check b.bytes == @[byte 11, 0, 0, 0, 249, 255, 255, 255,
                         10, 0, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0]

    test "enums cross as their uint32 storage, both domains agree":
      const astL = astLayoutOf(smStoreAll)
      const typedL = kernelArgLayout[StoreMask, tuple[], void](kagExploded, bkMetal)
      check astL == typedL
      check astL.args == @[KernelArg(kind: kaBytes, irKind: gtUint32)]
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[StoreMask, tuple[], TestHandle](kagExploded, bkMetal)
      bindKernelArgs[TestHandle](b, l, bkMetal, smStoreAll, (),
                                 ptrAsBuffer = true, argAlign = 1)
      check b.slots.len == 1 and b.slots[0].width == 4
      check b.bytes == @[byte 1, 0, 0, 0]

    test "the binder packs Nim int and uint at the 32-bit lowered width":
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[int, (uint, int64), TestHandle](kagExploded, bkMetal)
      bindKernelArgs[TestHandle](b, l, bkMetal, 1_000_000_000, (7'u, 5'i64),
                                 ptrAsBuffer = true, argAlign = 1)
      check b.slots.len == 3
      check b.slots[0].width == 4 and b.slots[1].width == 4 and b.slots[2].width == 8
      check b.bytes == @[byte 0, 202, 154, 59, 7, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]

    test "the binder packs pointer leaves as raw bytes without ptrAsBuffer":
      var a = PtrArg(n: 4)
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[PtrArg, tuple[], TestHandle](kagExploded, bkMetal)
      bindKernelArgs[TestHandle](b, l, bkOpenCL, a, (), ptrAsBuffer = false, argAlign = 1)
      check b.slots.len == 2
      check not b.slots[0].isBuffer and b.slots[0].width == sizeof(pointer)
      check not b.slots[1].isBuffer and b.slots[1].width == 4
      check b.bytes.len == 12

    test "the binder skips static subtrees, the lockstep walk lands at the layout end":
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[Mixed, tuple[], TestHandle](kagExploded, bkMetal)
      bindKernelArgs[TestHandle](b, l, bkMetal, Mixed.default, (), ptrAsBuffer = true, argAlign = 1)
      # the stride scalars and the dynamic array cross, the static
      # subtrees (shape, pad) carry no bytes
      check b.bytes.len == 20

    test "whole-value direct runs bind a ptr-free composite as one chunk":
      # the granularity the launch path selects: a descriptor explodes,
      # an empty descriptor binds whole-value
      const exploded = checkedKernelArgLayout[Whole, tuple[], TestHandle](
        kernelArgLayout[Whole, tuple[], TestHandle](kagExploded, bkMetal), bkMetal)
      check exploded == kernelArgLayout[Whole, tuple[], TestHandle](kagExploded, bkMetal)
      const whole = checkedKernelArgLayout[Whole, tuple[], TestHandle](
        KernelArgLayout(), bkMetal)
      check whole == kernelArgLayout[Whole, tuple[], TestHandle](kagWholeValue, bkMetal)
      check whole.args == @[KernelArg(kind: kaStruct, width: 24)]
      # 3x int32 (4) + 1x int64 (8) + 1x float32 (4), flat field order
      var w = Whole(m: 1, n: 2, k: 3, strideA: -1'i64, scale: 2.5'f32)
      var b: KernelBinding[TestHandle]
      bindKernelArgs[TestHandle](b, whole, bkMetal, w, (),
                                 ptrAsBuffer = true, argAlign = 1)
      check b.slots.len == 1
      check not b.slots[0].isBuffer and b.slots[0].width == 24
      check b.bytes == @[byte 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0,
                         255, 255, 255, 255, 255, 255, 255, 255,
                         0, 0, 32, 64]

    test "a composite with a buffer leaf explodes even in whole-value mode":
      const whole = checkedKernelArgLayout[PtrArg, tuple[], TestHandle](
        KernelArgLayout(), bkMetal)
      check whole == kernelArgLayout[PtrArg, tuple[], TestHandle](kagExploded, bkMetal)

    test "whole-value mode chunks a ptr-free object nested in a ptr holder":
      # only the buffer leaf explodes the outer, the ptr-free inner binds
      # as one chunk
      const whole = checkedKernelArgLayout[Outer, tuple[], TestHandle](
        KernelArgLayout(), bkMetal)
      check whole.args == @[KernelArg(kind: kaBuffer),
                            KernelArg(kind: kaStruct, width: 12)]
      var o = Outer(inner: Inner(m: 5, strideA: 6'i64))
      var b: KernelBinding[TestHandle]
      bindKernelArgs[TestHandle](b, whole, bkMetal, o, (),
                                 ptrAsBuffer = true, argAlign = 1)
      check b.slots.len == 2
      check b.slots[0].isBuffer
      check not b.slots[1].isBuffer and b.slots[1].width == 12
      check b.bytes == @[byte 5, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0]

    test "a whole-value chunk aligns like any byte chunk":
      const l = kernelArgLayout[int32, (Whole,), TestHandle](kagWholeValue, bkWGSL)
      var w = Whole.default
      var b: KernelBinding[TestHandle]
      bindKernelArgs[TestHandle](b, l, bkWGSL, 7'i32, (w,),
                                 ptrAsBuffer = false, argAlign = 256)
      check b.slots.len == 2
      check b.slots[0].offset == 0 and b.slots[0].width == 4
      check b.slots[1].offset == 256 and b.slots[1].width == 24
      check b.bytes.len == 280

    test "the binder aligns byte chunks (WebGPU's 256-byte storage slots)":
      var a = PtrArg(n: 4)
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[PtrArg, tuple[], TestHandle](kagExploded, bkMetal)
      bindKernelArgs[TestHandle](b, l, bkWGSL, a, (), ptrAsBuffer = false, argAlign = 256)
      check b.slots.len == 2
      check b.slots[1].offset == 256          # scalar slot on the next 256-byte line
      check b.bytes.len == 260

    test "the declared layout binds fields at the declared member offsets":
      # the kernel declares int fields a, b, c, long field big, float
      # field f: `big` pads to its 8-byte alignment (offset 16), a flat
      # field-order chunk would write it at 12
      const declared = KernelArgLayout(args: @[
        KernelArg(kind: kaStruct, width: 32, index: 1, members: @[
          KernelArgMember(irKind: gtInt32, offset: 0, width: 4),
          KernelArgMember(irKind: gtInt32, offset: 4, width: 4),
          KernelArgMember(irKind: gtInt32, offset: 8, width: 4),
          KernelArgMember(irKind: gtInt64, offset: 16, width: 8),
          KernelArgMember(irKind: gtFloat32, offset: 24, width: 4)])])
      var g = Whole(m: 1, n: 2, k: 3, strideA: 4_000_000'i64, scale: 2.5'f32)
      var b: KernelBinding[TestHandle]
      bindKernelArgs[TestHandle](b, declared, bkMetal, g, (),
                                 ptrAsBuffer = true, argAlign = 1)
      check b.slots.len == 1
      check not b.slots[0].isBuffer and b.slots[0].width == 32
      check b.slots[0].index == 1
      # padding at [12..15] stays zero, each field sits at its declared offset
      # the declared chunk is 32 bytes: the struct pads to its 8-byte alignment
      check b.bytes == @[byte 1, 0, 0, 0,  2, 0, 0, 0,  3, 0, 0, 0,
                         0, 0, 0, 0,  0, 9, 61, 0,  0, 0, 0, 0,
                         0, 0, 32, 64,  0, 0, 0, 0]

    test "the declared layout rejects a member kind the value cannot bind":
      const declared = KernelArgLayout(args: @[
        KernelArg(kind: kaStruct, width: 8, index: 0, members: @[
          KernelArgMember(irKind: gtInt64, offset: 0, width: 8)])])
      var w = Whole(m: 1, n: 2, k: 3, strideA: 4, scale: 1)
      var b: KernelBinding[TestHandle]
      doAssertRaises Defect:
        bindKernelArgs[TestHandle](b, declared, bkMetal, w, (),
                                   ptrAsBuffer = true, argAlign = 1)

    test "the declared layout rejects a member-count mismatch":
      const declared = KernelArgLayout(args: @[
        KernelArg(kind: kaStruct, width: 4, index: 0, members: @[
          KernelArgMember(irKind: gtInt32, offset: 0, width: 4)])])
      var w = Whole(m: 1, n: 2, k: 3, strideA: 4, scale: 1)
      var b: KernelBinding[TestHandle]
      doAssertRaises Defect:
        bindKernelArgs[TestHandle](b, declared, bkMetal, w, (),
                                   ptrAsBuffer = true, argAlign = 1)

    test "a bool binds the declared 4-byte int slot and the 1-byte bool slot":
      for w in [4, 1]:
        var b: KernelBinding[TestHandle]
        var l: KernelArgLayout
        if w == 4:
          l = KernelArgLayout(args: @[KernelArg(kind: kaBytes, irKind: gtInt32,
                                                width: 4, index: 2)])
        else:
          l = KernelArgLayout(args: @[KernelArg(kind: kaBytes, irKind: gtBool,
                                                width: 1, index: 2)])
        bindKernelArgs[TestHandle](b, l, bkMetal, true, (),
                                   ptrAsBuffer = true, argAlign = 1)
        check b.slots.len == 1 and b.slots[0].width == w and b.slots[0].index == 2
        if w == 4:
          check b.bytes == @[byte 1, 0, 0, 0]
        else:
          check b.bytes == @[byte 1]

    test "a pointer value binds a declared struct entry as a buffer":
      # a constant reference accepts a handle, the kernel's pointee type
      # is not host-checkable on the host side
      const declared = KernelArgLayout(args: @[
        KernelArg(kind: kaStruct, width: 72, index: 1, members: @[
          KernelArgMember(irKind: gtInt32, offset: 0, width: 4)])])
      var b: KernelBinding[TestHandle]
      bindKernelArgs[TestHandle](b, declared, bkMetal, cast[pointer](0xDEAD'i64), (),
                                 ptrAsBuffer = true, argAlign = 1)
      check b.slots.len == 1
      check b.slots[0].isBuffer and b.slots[0].index == 1
      check b.bytes.len == 0

    test "the launch reconciles the value projection against the declared signature":
      # what the args tuple projects as vs the kernel's own declaration:
      # handles bind any declared entry, one scalar binds a one-member struct,
      # the result binds at the declared indices
      const given = KernelArgLayout(args: @[
        KernelArg(kind: kaBuffer),
        KernelArg(kind: kaBytes, irKind: gtInt32),
        KernelArg(kind: kaBytes, irKind: gtInt32)])
      const declared = KernelArgLayout(args: @[
        KernelArg(kind: kaBytes, irKind: gtUint32, width: 4, index: 0),
        KernelArg(kind: kaStruct, width: 16, index: 2, members: @[
          KernelArgMember(irKind: gtInt32, offset: 0, width: 4)]),
        KernelArg(kind: kaBytes, irKind: gtInt32, width: 4, index: 1)])
      # note the declared indices are not dense: Metal assigns unattributed
      # scalars after the explicit [[buffer(n)]] ones
      let launch = declaredLaunchLayout(given, declared)
      check launch.args.len == 3
      check launch.args[0].index == 0
      check launch.args[1].index == 2
      check launch.args[2].index == 1
      var b: KernelBinding[TestHandle]
      bindKernelArgs[TestHandle](b, launch, bkMetal,
                                 cast[TestHandle](nil), (7'i32, 5'i32),
                                 ptrAsBuffer = true, argAlign = 1)
      check b.slots.len == 3
      check b.slots[0].isBuffer and b.slots[0].index == 0
      check not b.slots[1].isBuffer and b.slots[1].index == 2 and
            b.slots[1].width == 4
      check b.slots[2].index == 1
      check b.bytes == @[byte 7, 0, 0, 0, 5, 0, 0, 0]

    test "the launch reconciliation fails loudly on a wrong arg count":
      const given = KernelArgLayout(args: @[KernelArg(kind: kaBuffer)])
      const declared = KernelArgLayout(args: @[
        KernelArg(kind: kaBytes, irKind: gtInt32, width: 4),
        KernelArg(kind: kaBytes, irKind: gtInt32, width: 4)])
      doAssertRaises Defect:
        discard declaredLaunchLayout(given, declared)

    test "the launch reconciliation fails loudly on a kind drift":
      const given = KernelArgLayout(args: @[
        KernelArg(kind: kaBytes, irKind: gtFloat32)])
      const declared = KernelArgLayout(args: @[
        KernelArg(kind: kaBytes, irKind: gtInt32, width: 4)])
      doAssertRaises Defect:
        discard declaredLaunchLayout(given, declared)

    test "the launch reconciliation rejects a scalar against a multi-member struct":
      const given = KernelArgLayout(args: @[
        KernelArg(kind: kaBytes, irKind: gtInt32)])
      const declared = KernelArgLayout(args: @[
        KernelArg(kind: kaStruct, width: 8, members: @[
          KernelArgMember(irKind: gtInt32, offset: 0, width: 4),
          KernelArgMember(irKind: gtInt32, offset: 4, width: 4)])])
      doAssertRaises Defect:
        discard declaredLaunchLayout(given, declared)

    test "the launch reconciliation rejects a composite against a declared scalar":
      const given = KernelArgLayout(args: @[
        KernelArg(kind: kaStruct, width: 24)])
      const declared = KernelArgLayout(args: @[
        KernelArg(kind: kaBytes, irKind: gtInt32, width: 4)])
      doAssertRaises Defect:
        discard declaredLaunchLayout(given, declared)

runTest()
