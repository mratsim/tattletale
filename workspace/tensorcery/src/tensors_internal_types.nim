# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import std/macros
import std/typetraits
import workspace/crucible
import workspace/ceramic
import workspace/ceramic/src/macros/static_for
import workspace/iface
import tensors_public_types
import workspace/design_by_contract

export tensors_public_types

type Span*[T] = object
  p*: openArray[T]

iface *AnyTensorImpl:
    func dtype(): Dtype
    func shape(): Span[int]
    func strides(): Span[int]
    func size(): int
    func isMaterialized(): bool

iface *TensorNode:
    proc getImpl(): AnyTensorImpl
    func opName(): string
    func size(): int

type
  TensorImpl*[DeviceMem; Rank: static int] = ref object
    data*: ref DeviceMem
    offset*: int
    shape*: array[Rank, int]
    stride*: array[Rank, int]

  TensorBridge*[DeviceMem; L: Layout] = object
    data: ref DeviceMem
    offset: int
    layout: L

  PosNode*[C] = ref object of RootRef
    core*: C
    pos*: int

proc allocData*[E: HwEngine, H, T; Rank: static int](engine: E, impl: TensorImpl[DeviceMem[H, T], Rank]) {.inline.} =
  impl.data.new()
  impl.data[] = engine.allocDevice(T, impl.size())

template getLayoutType*(rank: static int): typedesc =
  when rank == 1:
    Layout[(int,), (int,)]
  elif rank == 2:
    Layout[(int, int), (int, int)]
  elif rank == 3:
    Layout[(int, int, int), (int, int, int)]
  elif rank == 4:
    Layout[(int, int, int, int), (int, int, int, int)]
  elif rank == 5:
    Layout[(int, int, int, int, int), (int, int, int, int, int)]
  else:
    {.error: "getLayoutType: rank " & $rank & " unsupported".}

func bridge*[DeviceMem; Rank: static int](t: TensorImpl[DeviceMem, Rank]): lent TensorBridge[DeviceMem, getLayoutType(Rank)] {.inline.} =
  static: doAssert sizeof(t[]) == sizeof(TensorBridge[DeviceMem, getLayoutType(Rank)])
  return cast[ptr TensorBridge[DeviceMem, getLayoutType(Rank)]](t)[]

iface *AnyEngine:
    proc newTensorNode(dtype: Dtype, shape: openArray[int]): TensorNode
    proc newPlaceholderNode(dtype: Dtype, shape: openArray[int]): TensorNode
    proc upload(node: TensorNode, src: openArray[float32])
    proc upload(node: TensorNode, src: openArray[bfloat16])
    proc download(dst: var openArray[float32], node: TensorNode)
    proc download(dst: var openArray[bfloat16], node: TensorNode)
    proc download(dst: var openArray[float32], node: TensorNode, offsetElems: int)
    proc download(dst: var openArray[bfloat16], node: TensorNode, offsetElems: int)
    proc eval(roots: var openArray[TensorNode])

func rowMajorStrides*[Rank: static int](strides: var array[Rank, int], shape: array[Rank, int]) {.inline.} =
  var acc = 1
  for j in 1 ..< Rank + 1:
    strides[Rank - j] = acc
    acc *= shape[Rank - j]
