# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

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
  p*: ptr UncheckedArray[T]
  len*: int

template toSpan*(s: tuple): Span[int] =
  Span[int](p: cast[ptr UncheckedArray[int]](unsafeAddr s), len: tupleLen(s))

template toSpan*[T](s: seq[T]): Span[T] =
  Span[T](p: cast[ptr UncheckedArray[T]](unsafeAddr s[0]), len: s.len)

template toSpan*[T](s: openArray[T]): Span[T] =
  Span[T](p: cast[ptr UncheckedArray[T]](unsafeAddr s[0]), len: s.len)

template `[]`*[T](s: Span[T], i: int): T =
  s.p[i]

iterator items[T](s: Span[T]): T =
  for i in 0 ..< s.len:
    yield s.p[i]

func `==`*[T](a, b: Span[T]): bool =
  if a.len != b.len:
    return false
  for i in 0 ..< a.len:
    if a[i] != b[i]:
      return false
  true


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
  TensorImpl*[DeviceMem; L: Layout] = ref object
    data*: ref DeviceMem
    offset*: int
    layout*: L

  TensorBridge*[DeviceMem; L: Layout] = object
    data {.cursor.}: DeviceMem
    offset: int
    layout: L

  PosNode*[C] = ref object of RootRef
    core*: C
    pos*: int

func bridge*[DeviceMem, L](t: TensorImpl[DeviceMem, L]): TensorBridge[DeviceMem, L] {.inline.} =
  TensorBridge[DeviceMem, L](data: t.data[], offset: t.offset, layout: t.layout)

iface *AnyEngine:
    proc newTensorNode(dtype: Dtype, shape: Span[int]): TensorNode
    proc newPlaceholderNode(dtype: Dtype, shape: Span[int]): TensorNode
    proc upload(node: TensorNode, src: openArray[float32])
    proc upload(node: TensorNode, src: openArray[bfloat16])
    proc download(dst: var openArray[float32], node: TensorNode)
    proc download(dst: var openArray[bfloat16], node: TensorNode)
    proc download(dst: var openArray[float32], node: TensorNode, offsetElems: int)
    proc download(dst: var openArray[bfloat16], node: TensorNode, offsetElems: int)
    proc eval(roots: var Span[TensorNode])

func atAxis*[Sh: tuple](s: Sh, axis: int): int {.inline.} =
  staticFor i, 0, tupleLen(Sh):
    if i == axis:
      result = s[i]

func withAxis*[Sh: tuple](s: Sh, axis: int, val: int): Sh {.inline.} =
  staticFor i, 0, tupleLen(Sh):
    if i == axis:
      result[i] = val
    else:
      result[i] = s[i]

func rowMajorStrides*[Sh: tuple](shape: Sh): Sh =
  var acc = 1
  staticFor j, 0, tupleLen(Sh):
    result[tupleLen(Sh) - 1 - j] = acc
    acc *= shape[tupleLen(Sh) - 1 - j]

func fillShape*[Sh: tuple](shape: var Sh, s: Span[int]) {.contracts.} =
  {.precondition: s.len == tupleLen(Sh).}
  staticFor i, 0, tupleLen(Sh):
    shape[i] = s[i]
