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
import workspace/design_by_contract
import ./tensors_public_types

export tensors_public_types

type Span*[T] = object
  p*: openArray[T]

iface *AnyTensorImpl:
    func dtype(): Dtype {.inline.}
    func shape(): Span[int] {.inline.}
    func strides(): Span[int] {.inline.}
    func size(): int {.inline.}
    func offset(): int {.inline.}
    func isMaterialized(): bool {.inline.}

iface *TensorNode:
    proc getImpl(): AnyTensorImpl {.inline.}
    func opName(): string {.inline.}
    func size(): int {.inline.}
    proc isNil(): bool {.inline.}

type NilNode = ref object of RootRef
proc getImpl(n: NilNode): AnyTensorImpl {.inline.} = AnyTensorImpl()
func opName(n: NilNode): string {.inline.} = "Nil"
func size(n: NilNode): int {.inline.} = 0
proc NilTensorNode*(): TensorNode {.inline.} = NilNode(nil).to(TensorNode)

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

proc allocDataIn*[E: HwEngine, H, T; Rank: static int](engine: E, impl: TensorImpl[DeviceMem[H, T], Rank]) {.inline.} =
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
    proc newTensorNode(dtype: Dtype, shape: openArray[int]): TensorNode {.inline.}
    proc newPlaceholderNode(dtype: Dtype, shape: openArray[int]): TensorNode {.inline.}
    proc newViewNode(src: TensorNode, shape: openArray[int], strides: openArray[int], offset: int): TensorNode {.inline.}
    proc upload(node: TensorNode, src: openArray[float32]) {.inline.}
    proc upload(node: TensorNode, src: openArray[bfloat16]) {.inline.}
    proc download1(dst: pointer, node: TensorNode) {.inline.}
    proc eval(roots: var openArray[TensorNode]) {.inline.}

func rowMajorStrides*[Rank: static int](strides: var array[Rank, int], shape: array[Rank, int]) {.inline.} =
  var acc = 1
  for j in 1 ..< Rank + 1:
    strides[Rank - j] = acc
    acc *= shape[Rank - j]

# ═══════════════════════════════════════════════════════════════════════
#  2D dispatch: one body over (element type, rank), registry-driven
# ═══════════════════════════════════════════════════════════════════════

var ElemTypes {.compileTime.}: seq[tuple[kind, elemType: NimNode]]

macro registerElemType(kind, elemType: untyped) =
  ## Declares the element type backing a Dtype value for match dispatch.
  ElemTypes.add (kind, elemType)

func replaceNode(ast, toReplace, replaceBy: NimNode): NimNode =
  func inspect(node: NimNode): NimNode =
    case node.kind
    of {nnkIdent, nnkSym}:
      if node.eqIdent(toReplace):
        return replaceBy
      return node
    of nnkEmpty, nnkLiterals:
      return node
    else:
      var rTree = node.kind.newTree()
      for child in node:
        rTree.add inspect(child)
      return rTree
  result = inspect(ast)

macro match*(elemBinding, dtypeValue, rankBinding, rankValue: untyped, body: untyped) =
  ## Dispatches runtime dtype and rank to compile-time type and Rank
  ##
  ## Usage:
  ##   match(ElemType, dtype, Rank, shape.len):
  ##     engine.newPlaceholderImpl(ElemType, Rank, shape).to(TensorNode)
  var dtypeCase = nnkCaseStmt.newTree(dtypeValue)
  for registered in ElemTypes:
    var rankCase = nnkCaseStmt.newTree(rankValue)
    for rank in 1 .. 5:
      rankCase.add nnkOfBranch.newTree(
        newLit(rank),
        body
          .replaceNode(elemBinding, registered.elemType.copyNimTree())
          .replaceNode(rankBinding, newLit(rank)))
    rankCase.add nnkElse.newTree(quote do:
      raise ValueError.newException("Tensorcery: unsupported rank " & $`rankValue`))
    dtypeCase.add nnkOfBranch.newTree(registered.kind, nnkStmtList.newTree(rankCase))
  dtypeCase.add nnkElse.newTree(quote do:
    raise ValueError.newException("Tensorcery: unsupported dtype " & $`dtypeValue`))
  result = nnkStmtList.newTree(dtypeCase)

registerElemType(kFp16, float16)
registerElemType(kFp32, float32)
registerElemType(kBf16, bfloat16)
