# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Kernel arg binding contract tests:
## - classification, buffer / bytes / static
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
      const typedL = kernelArgLayout[Mixed, tuple[], void]()
      check astL == typedL
      check astL == mixedExpected

    test "layout agreement for scalar, pointer and array top-level values":
      const astL = astLayoutOf(int32(1), cast[ptr uint32](nil), DynArr.default,
                               StaticArr.default)
      const typedL = kernelArgLayout[int32, (ptr uint32, DynArr, StaticArr), void]()
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
      const own = checkedKernelArgLayout[Mixed, tuple[], void](KernelArgLayout(), bkMetal)
      check own == mixedExpected

    test "the binder packs one raw chunk per dynamic arg in explosion order":
      var m = Mixed(
        stride: (11'i32, -7'i32),
        xs: [10'i32, 20, 30])
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[Mixed, tuple[], TestHandle]()
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
      const typedL = kernelArgLayout[StoreMask, tuple[], void]()
      check astL == typedL
      check astL.args == @[KernelArg(kind: kaBytes, irKind: gtUint32)]
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[StoreMask, tuple[], TestHandle]()
      bindKernelArgs[TestHandle](b, l, bkMetal, smStoreAll, (),
                                 ptrAsBuffer = true, argAlign = 1)
      check b.slots.len == 1 and b.slots[0].width == 4
      check b.bytes == @[byte 1, 0, 0, 0]

    test "the binder packs Nim int and uint at the 32-bit lowered width":
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[int, (uint, int64), TestHandle]()
      bindKernelArgs[TestHandle](b, l, bkMetal, 1_000_000_000, (7'u, 5'i64),
                                 ptrAsBuffer = true, argAlign = 1)
      check b.slots.len == 3
      check b.slots[0].width == 4 and b.slots[1].width == 4 and b.slots[2].width == 8
      check b.bytes == @[byte 0, 202, 154, 59, 7, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]

    test "the binder packs pointer leaves as raw bytes without ptrAsBuffer":
      var a = PtrArg(n: 4)
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[PtrArg, tuple[], TestHandle]()
      bindKernelArgs[TestHandle](b, l, bkOpenCL, a, (), ptrAsBuffer = false, argAlign = 1)
      check b.slots.len == 2
      check not b.slots[0].isBuffer and b.slots[0].width == sizeof(pointer)
      check not b.slots[1].isBuffer and b.slots[1].width == 4
      check b.bytes.len == 12

    test "the binder skips static subtrees, the lockstep walk lands at the layout end":
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[Mixed, tuple[], TestHandle]()
      bindKernelArgs[TestHandle](b, l, bkMetal, Mixed.default, (), ptrAsBuffer = true, argAlign = 1)
      # the stride scalars and the dynamic array cross, the static
      # subtrees (shape, pad) carry no bytes
      check b.bytes.len == 20

    test "the binder aligns byte chunks (WebGPU's 256-byte storage slots)":
      var a = PtrArg(n: 4)
      var b: KernelBinding[TestHandle]
      const l = kernelArgLayout[PtrArg, tuple[], TestHandle]()
      bindKernelArgs[TestHandle](b, l, bkWGSL, a, (), ptrAsBuffer = false, argAlign = 256)
      check b.slots.len == 2
      check b.slots[1].offset == 256          # scalar slot on the next 256-byte line
      check b.bytes.len == 260

runTest()
