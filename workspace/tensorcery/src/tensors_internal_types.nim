# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import std/typetraits
import workspace/crucible/src/builtins/builtins_gpu_types
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
    func offset(): int
    func capacity(): int

iface *TensorNode:
    proc getImpl(): AnyTensorImpl
    func opName(): string

type
  TensorImpl*[Mem, L] = ref object
    data*: ref Mem
    offset*: int
    layout*: L
    capacity*: int

  TensorBridge*[DevMem, Sh, St] = object
    data {.cursor.}: DevMem
    offset: int
    layout: Layout[Sh, St]

  PosNode*[C] = ref object of RootRef
    core*: C
    pos*: int

func initTensorBridge*[DevMem, Sh, St](data: DevMem, offset: int, layout: Layout[Sh, St]): TensorBridge[DevMem, Sh, St] =
  ## Kernel ABI of a materialized tensor, assembled at launch sites
  TensorBridge[DevMem, Sh, St](data: data, offset: offset, layout: layout)

iface *AnyEngine:
    proc newTensorNode(dtype: Dtype, shape: Span[int]): TensorNode
    proc newViewNode(dtype: Dtype, shape: Span[int]): TensorNode
    proc newViewImpl(src: AnyTensorImpl, shape, strides: Span[int], offset: int): TensorNode
    proc upload(node: TensorNode, src: openArray[float32])
    proc upload(node: TensorNode, src: openArray[bfloat16])
    proc download(dst: var openArray[float32], node: TensorNode)
    proc download(dst: var openArray[bfloat16], node: TensorNode)
    proc download(dst: var openArray[float32], node: TensorNode, offsetElems: int)
    proc download(dst: var openArray[bfloat16], node: TensorNode, offsetElems: int)
    proc downloadItem(dst: var float32, node: TensorNode, offsetElems: int)
    proc downloadItem(dst: var bfloat16, node: TensorNode, offsetElems: int)
    proc eval(roots: var Span[TensorNode])

func getNumElems*(shape: Span[int]): int =
  result = 1
  for i in 0 ..< shape.len:
    result *= shape[i]

func getNumElems*[Sh: tuple](shape: Sh): int =
  result = 1
  staticFor i, 0, tupleLen(Sh):
    result *= shape[i]

func atAxis*[Sh: tuple](s: Sh, axis: int): int =
  staticFor i, 0, tupleLen(Sh):
    if i == axis:
      result = s[i]

func withAxis*[Sh: tuple](s: Sh, axis: int, val: int): Sh =
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
