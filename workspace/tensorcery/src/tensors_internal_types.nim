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

# ############################################################
#
#              Types and interfaces
#
# ############################################################

type Span*[T] = object
  p*: openArray[T]

iface *AnyTensorImpl:
    func dtype(): Dtype {.inline.}
    func shape(): Span[int] {.inline.}
    func strides(): Span[int] {.inline.}
    func size(): int {.inline.}
    func offset(): int {.inline.}
    func isPreAllocated(): bool {.inline.}

iface *TensorNode:
    proc getImpl(): AnyTensorImpl {.inline.}
    func opName(): string {.inline.}
    proc isNil(): bool {.inline.}

iface *AnyEngine:
    proc newTensorNode(dtype: Dtype, shape: openArray[int]): TensorNode {.inline.}
    proc newPlaceholderNode(dtype: Dtype, shape: openArray[int]): TensorNode {.inline.}
    proc newViewNode(src: TensorNode, shape: openArray[int], strides: openArray[int], offset: int, srcOpName: string): TensorNode {.inline.}
    proc upload(node: TensorNode, src: openArray[float32]) {.inline.}
    proc upload(node: TensorNode, src: openArray[bfloat16]) {.inline.}
    proc download1(dst: pointer, node: TensorNode) {.inline.}
    proc eval(roots: var openArray[TensorNode]) {.inline.}

const MaxRank* = 5

type
  LayoutBasic* = object
    ## Runtime-rank layout: the first `rank` entries of `shape` and `strides`
    ## are active and the array tail pads the rest. Memory-bound op kernels
    ## consume it. Ceramic's layout algebra never sees it.
    shape*, strides*: array[MaxRank, int]
    rank*: int

  TensorImpl*[DeviceMem; L: Layout | LayoutBasic] = ref object
    ## One impl type for every rank. The layout lives in the type slot:
    ## - a ceramic `Layout` is the static-rank tier
    ## - `LayoutBasic` is the runtime-rank tier the factories produce
    data*: ref DeviceMem
    offset*: int
    layout*: L

  TensorBridge*[DeviceMem; L: Layout | LayoutBasic] = object
    ## Host-side device-function argument holding the TensorImpl fields.
    ## - the frontend derives the kernel's view type from it
    ## - DeviceMem fields launch as device buffer addresses
    data: ref DeviceMem
    offset: int
    layout: L

  ViewNode* = ref object
    r*: AnyTensorImpl
    src*: TensorNode
    srcOpName*: string

  NilNode* = ref object

proc initTensorBridge*[DeviceMem, L](data: ref DeviceMem, offset: int, layout: L): TensorBridge[DeviceMem, L] {.inline.} =
  ## Cross-module constructor: the TensorBridge fields are module-private.
  TensorBridge[DeviceMem, L](data: data, offset: offset, layout: layout)

# ═══════════════════════════════════════════════════════════════════════
#  Layout
# ═══════════════════════════════════════════════════════════════════════

func rowMajorStrides*[Rank: static int](strides: var array[Rank, int], shape: array[Rank, int]) {.inline.} =
  var acc = 1
  for j in 1 ..< Rank + 1:
    strides[Rank - j] = acc
    acc *= shape[Rank - j]

func toLayoutBasic*(shape: openArray[int]): LayoutBasic =
  ## Dense layout over `shape` with ascending strides. Entries beyond the rank are zero.
  if shape.len > MaxRank:
    raise ValueError.newException("Tensorcery: unsupported rank " & $shape.len)
  result.rank = shape.len
  var acc = 1
  for j in 1 ..< shape.len + 1:
    result.shape[shape.len - j] = shape[shape.len - j]
    result.strides[shape.len - j] = acc
    acc *= shape[shape.len - j]

func toLayoutBasic*(shape, strides: openArray[int]): LayoutBasic =
  ## Layout over `shape` and `strides`, entries past rank zero.
  if shape.len > MaxRank:
    raise ValueError.newException("Tensorcery: unsupported rank " & $shape.len)
  result.rank = shape.len
  for i in 0 ..< shape.len:
    result.shape[i] = shape[i]
    result.strides[i] = strides[i]

# ═══════════════════════════════════════════════════════════════════════
#  Tensor and element types
# ═══════════════════════════════════════════════════════════════════════

func isPreAllocated*(t: TensorImpl): bool {.inline.} =
  not t.data.isNil()

template getTensorImplType*(engine: HwEngine, T: typedesc): typedesc =
  TensorImpl[engine.getMemHandleType(T), LayoutBasic]

proc allocDataIn*[E: HwEngine, H, T](engine: E, impl: TensorImpl[DeviceMem[H, T], LayoutBasic]) {.inline.} =
  impl.data.new()
  impl.data[] = engine.allocDevice(T, impl.size())

func bridge*[DeviceMem; L: Layout | LayoutBasic](t: TensorImpl[DeviceMem, L]): TensorBridge[DeviceMem, L] {.inline.} =
  if t.isNil():
    TensorBridge[DeviceMem, L]()
  else:
    static: doAssert sizeof(t[]) == sizeof(TensorBridge[DeviceMem, L])
    cast[ptr TensorBridge[DeviceMem, L]](t)[]

# ═══════════════════════════════════════════════════════════════════════
#  Graph
# ═══════════════════════════════════════════════════════════════════════

proc isPreAllocated*(n: TensorNode): bool {.inline.} =
  n.getImpl().isPreAllocated()

proc size*(n: TensorNode): int {.inline.} =
  n.getImpl().size()

proc getImpl(n: NilNode): AnyTensorImpl {.inline.} = AnyTensorImpl()
func opName(n: NilNode): string {.inline.} = "Nil"
proc NilTensorNode*(): TensorNode {.inline.} = NilNode(nil).to(TensorNode)

func getImpl*(n: ViewNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: ViewNode): string {.inline.} =
  n.srcOpName

# ═══════════════════════════════════════════════════════════════════════
#  Dispatch: one body over the element type, registry-driven
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

macro match*(elemBinding, dtypeValue: untyped, body: untyped) =
  ## Dispatches runtime dtype to compile-time element type
  ##
  ## Usage:
  ##   match(ElemType, dtype):
  ##     engine.newPlaceholderImpl(ElemType, shape).to(TensorNode)
  var dtypeCase = nnkCaseStmt.newTree(dtypeValue)
  for registered in ElemTypes:
    dtypeCase.add nnkOfBranch.newTree(registered.kind,
      body.replaceNode(elemBinding, registered.elemType.copyNimTree()))
  dtypeCase.add nnkElse.newTree(quote do:
    raise ValueError.newException("Tensorcery: unsupported dtype " & $`dtypeValue`))
  result = nnkStmtList.newTree(dtypeCase)

registerElemType(kFp16, float16)
registerElemType(kFp32, float32)
registerElemType(kBf16, bfloat16)
