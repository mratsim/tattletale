# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/assertions
import std/macros
import std/typetraits

import workspace/crucible
import workspace/crucible/src/builtins/builtins_gpu_types
export builtins_gpu_types
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/ceramic/src/macros/static_for
import workspace/math
import ../vendor/iface/iface
export iface.to

# ############################################################
#
#                   Tensor datatypes
#
# ############################################################
#
#
# Design:
#
# Tensorcery is lazy/pull-based, evaluation is on-demand.
# A graph is built and only on explicit .eval() or conversions (string, writing to disk)
# is the compute scheduled.
#
# This allows batching compute all-at-once, reducing the CPU<->GPU synchronization overhead.
# Furthermore, when backprop is added, the graph can be reused for backward propagation.

# ═══════════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════════
type Dtype* = enum
    kFp16, kBf16, kFp32

func toDtype(_: typedesc[float32]): Dtype =
  kFp32

func toDtype(_: typedesc[bfloat16]): Dtype =
  kBf16

type Span*[T] = object
  p*: ptr UncheckedArray[T]
  len*: int

template toSpan*(s: tuple): Span[int] =
  Span[int](p: cast[ptr UncheckedArray[int]](unsafeAddr s), len: tupleLen(s))

template toSpan*[T](s: seq[T]): Span[T] =
  Span[T](p: cast[ptr UncheckedArray[T]](unsafeAddr s[0]), len: s.len)

template toSpan*[T](s: openArray[T]): Span[T] =
  Span[T](p: cast[ptr UncheckedArray[T]](unsafeAddr s[0]), len: s.len)

template toOpenArray[T](bytes: openArray[byte], _: typedesc[T]): openArray[T] =
  ## Erased copy boundary byte stream read back as T elements,
  ## length derives from the byte count
  toOpenArray(cast[ptr UncheckedArray[T]](unsafeAddr bytes[0]), 0, bytes.len div sizeof(T) - 1)

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
    proc dtype(): Dtype
    proc shape(): Span[int]
    proc strides(): Span[int]
    proc offset(): int

iface *TensorNode:
    proc eval()
    proc getImpl(): AnyTensorImpl

type
  TensorImpl*[Mem, L] = ref object
    data*: ref Mem
    offset*: int
    layout*: L

  TensorBridge*[DevMem, Sh, St] = object
    ## Launch-side view of a device tensor. Host-side form of the kernel
    ## face's `DeviceView`:
    ##
    ## - same fields as `DeviceView`, in the same order
    ## - the DeviceMem leaf contributes the element type the kernel binds
    ## - `.cursor.` borrows the impl's buffer, so a bridge copies without
    ##   touching the move-only handle
    data* {.cursor.}: DevMem
    offset*: int
    layout*: Layout[Sh, St]

  ViewNode* = ref object
    ## Lazy shared-buffer view: `v` aliases the source impl's buffer cell,
    ## eval materializes the view's storage through `src`.
    src: TensorNode
    v: AnyTensorImpl


iface *AnyEngine:
    proc newTensorNode(dtype: Dtype, shape: Span[int]): TensorNode
    proc makeViewImpl(src: AnyTensorImpl, shape, strides: Span[int], offset: int): AnyTensorImpl
    proc copyToDeviceFromHost(node: TensorNode, src: openArray[byte])
    proc copyToHostFromDevice(dst: var openArray[byte], node: TensorNode)
    proc eval(roots: Span[TensorNode])

type Tensor* = object
  engine*: AnyEngine
  node*: TensorNode

# ═══════════════════════════════════════════════════════════════════════
#  Layout helpers
# ═══════════════════════════════════════════════════════════════════════

func getNumElems*[Sh: tuple](shape: Sh): int =
  result = 1
  staticFor i, 0, tupleLen(Sh):
    result *= shape[i]

func rowMajorStrides[Sh: tuple](shape: Sh): Sh =
  var acc = 1
  staticFor j, 0, tupleLen(Sh):
    result[tupleLen(Sh) - 1 - j] = acc
    acc *= shape[tupleLen(Sh) - 1 - j]

func isDenseRowMajor[Sh: tuple, St: tuple](shape: Sh, strides: St): bool =
  ## Contiguous ascending storage: the last stride is 1 and each earlier
  ## stride is the product of the later strides and shapes.
  result = strides[tupleLen(St) - 1] == 1
  staticFor i, 0, tupleLen(Sh) - 1:
    if strides[i] != strides[i + 1] * shape[i + 1]:
      return false

func fillShape[Sh: tuple](shape: var Sh, s: Span[int]) =
  if s.len != tupleLen(Sh):
    raise ValueError.newException(
      "rank " & $s.len & " does not match the context rank " & $tupleLen(Sh))
  staticFor i, 0, tupleLen(Sh):
    shape[i] = s[i]

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - AnyTensorImpl implementation
# ═══════════════════════════════════════════════════════════════════════

proc gatherToHost[E: HwEngine, Sh: tuple, St: tuple, Mem, T](
    engine: E, t: TensorImpl[Mem, Layout[Sh, St]], dst: var openArray[T]) =
  ## Reads back a strided or offset view.
  ## The engine DMA is flat over the whole buffer.
  ## The view's storage window copies into a host scratch first:
  ## - window: offset + cosize(layout) elements from the buffer base
  ## - dst[i] reads the scratch at the layout address of element i
  ##   (elements ordered by ascending flat index)
  ## - elemAddr(i) = offset + sum_a crd_a(i) * stride_a
  ##   crd decomposes so the last axis varies fastest
  let numel = t.layout.shape.getNumElems()
  var scratch = newSeq[T](t.offset + t.layout.cosize().toInt())
  engine.copyToHostFromDevice(scratch, t.data[])
  for i in 0 ..< numel:
    var rem = i
    var elemAddr = t.offset
    # decomposition with the last axis varying fastest
    staticFor a, 0, tupleLen(Sh):
      const ax = tupleLen(Sh) - 1 - a
      elemAddr += (rem mod t.layout.shape[ax]) * t.layout.stride[ax]
      rem = rem div t.layout.shape[ax]
    dst[i] = scratch[elemAddr]

proc shape*[Sh: tuple, Mem](t: TensorImpl[Mem, Layout[Sh, Sh]]): Span[int] =
  toSpan(t.layout.shape)

proc strides*[Sh: tuple, Mem](t: TensorImpl[Mem, Layout[Sh, Sh]]): Span[int] =
  toSpan(t.layout.stride)

proc offset*[Sh: tuple, Mem](t: TensorImpl[Mem, Layout[Sh, Sh]]): int =
  t.offset

proc dtype*[Sh: tuple, H, T](t: TensorImpl[DeviceMem[H, T], Layout[Sh, Sh]]): Dtype =
  toDtype(T)

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - TensorNode implementation (materialized sources)
# ═══════════════════════════════════════════════════════════════════════

proc eval*[Mem, L: Layout](t: TensorImpl[Mem, L]) =
  discard

proc getImpl*[Mem, L: Layout](t: TensorImpl[Mem, L]): AnyTensorImpl =
  t.to(AnyTensorImpl)

proc eval*(x: AnyTensorImpl) =
  discard

proc getImpl*(x: AnyTensorImpl): AnyTensorImpl =
  x

# ═══════════════════════════════════════════════════════════════════════
#  Nodes - TensorNode implementation
# ═══════════════════════════════════════════════════════════════════════

proc eval*(n: ViewNode) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getImpl*(n: ViewNode): AnyTensorImpl =
  n.v

# ═══════════════════════════════════════════════════════════════════════
#  The erased engine
# ═══════════════════════════════════════════════════════════════════════

proc toAnyEngine*[E: HwEngine](engine: E): AnyEngine =
  engine.to(AnyEngine)

proc makeTensorImpl[E: HwEngine, Sh: tuple, St: tuple](engine: E, dtype: Dtype, layout: Layout[Sh, St]): AnyTensorImpl =
  case dtype
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    var t = TensorImpl[Mem, Layout[Sh, St]](data: nil, offset: 0, layout: layout)
    t.data.new()
    t.data[] = engine.allocDevice(float32, layout.shape.getNumElems())
    t.to(AnyTensorImpl)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    var t = TensorImpl[Mem, Layout[Sh, St]](data: nil, offset: 0, layout: layout)
    t.data.new()
    t.data[] = engine.allocDevice(bfloat16, layout.shape.getNumElems())
    t.to(AnyTensorImpl)
  else:
    raise ValueError.newException("makeTensorImpl: dtype " & $dtype & " unsupported")

proc makeTensorImpl[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): AnyTensorImpl =
  case shape.len
  of 1:
    var sh: (int,)
    fillShape(sh, shape)
    makeTensorImpl(engine, dtype, make_layout(sh, rowMajorStrides(sh)))
  of 2:
    var sh: (int, int)
    fillShape(sh, shape)
    makeTensorImpl(engine, dtype, make_layout(sh, rowMajorStrides(sh)))
  of 3:
    var sh: (int, int, int)
    fillShape(sh, shape)
    makeTensorImpl(engine, dtype, make_layout(sh, rowMajorStrides(sh)))
  of 4:
    var sh: (int, int, int, int)
    fillShape(sh, shape)
    makeTensorImpl(engine, dtype, make_layout(sh, rowMajorStrides(sh)))
  of 5:
    var sh: (int, int, int, int, int)
    fillShape(sh, shape)
    makeTensorImpl(engine, dtype, make_layout(sh, rowMajorStrides(sh)))
  else:
    raise ValueError.newException("makeTensorImpl: rank " & $shape.len & " unsupported")

proc makeTensorImpl[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): AnyTensorImpl =
  makeTensorImpl(engine, dtype, shape.toSpan())

proc makeViewImpl[E: HwEngine, Sh: tuple, St: tuple](
    engine: E, src: AnyTensorImpl, layout: Layout[Sh, St], offset: int): AnyTensorImpl =
  ## Shared-buffer view over `src`: aliases the source's `ref DeviceMem` cell
  ## with a new layout and element offset, no allocation and no copy.
  ## Contract: the view stays inside the source's storage window,
  ## offset + cosize(layout) must not exceed the source buffer length.
  case src.dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    let s = src.to(TensorImpl[Mem, Layout[Sh, St]])
    TensorImpl[Mem, Layout[Sh, St]](data: s.data, offset: offset, layout: layout).to(AnyTensorImpl)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    let s = src.to(TensorImpl[Mem, Layout[Sh, St]])
    TensorImpl[Mem, Layout[Sh, St]](data: s.data, offset: offset, layout: layout).to(AnyTensorImpl)
  else:
    raise ValueError.newException("makeViewImpl: dtype " & $src.dtype() & " unsupported")

proc makeViewImpl*[E: HwEngine](engine: E, src: AnyTensorImpl, shape, strides: Span[int], offset: int): AnyTensorImpl =
  case shape.len
  of 1:
    var sh: (int,)
    var st: (int,)
    fillShape(sh, shape)
    fillShape(st, strides)
    makeViewImpl(engine, src, make_layout(sh, st), offset)
  of 2:
    var sh: (int, int)
    var st: (int, int)
    fillShape(sh, shape)
    fillShape(st, strides)
    makeViewImpl(engine, src, make_layout(sh, st), offset)
  of 3:
    var sh: (int, int, int)
    var st: (int, int, int)
    fillShape(sh, shape)
    fillShape(st, strides)
    makeViewImpl(engine, src, make_layout(sh, st), offset)
  of 4:
    var sh: (int, int, int, int)
    var st: (int, int, int, int)
    fillShape(sh, shape)
    fillShape(st, strides)
    makeViewImpl(engine, src, make_layout(sh, st), offset)
  of 5:
    var sh: (int, int, int, int, int)
    var st: (int, int, int, int, int)
    fillShape(sh, shape)
    fillShape(st, strides)
    makeViewImpl(engine, src, make_layout(sh, st), offset)
  else:
    raise ValueError.newException("makeViewImpl: rank " & $shape.len & " unsupported")

proc newTensorNode*[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): TensorNode =
  makeTensorImpl(engine, dtype, shape).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Eval - pull walk, enqueue into one batch, submit
# ═══════════════════════════════════════════════════════════════════════

var ctgEnqueueHandlers {.compileTime.}: seq[NimNode]

macro registerEnqueueNode*(enqueue: typed) =
  ## Registers an op file's enqueue handler with the eval walk.
  ## Called once, at the op module's top level.
  ##
  ## Handler contract:
  ## - signature `proc[E: HwEngine, Sh: tuple](engine: E, batch: Batch[E], n: TensorNode): bool`
  ## - walks the node's children through `evalNode`, then enqueues the node's
  ##   kernels into the batch
  ## - returns true when the node is the op's own, false for a node the op
  ##   does not own, so the handlers of several ops compose
  ##
  ## The handler resolves the launch's static engine, rank and element types
  ## inside its own generic body. Nothing outside the op file knows them.
  ctgEnqueueHandlers.add(enqueue)

macro dispatchEnqueueNodes(E, Sh: typed, engine, batch, n: typed): untyped =
  ## Expands the registered enqueue handlers over one node. The first handler
  ## that owns the node enqueues its kernels and ends the walk branch.
  result = nnkStmtList.newTree()
  for h in ctgEnqueueHandlers:
    let call = nnkCall.newTree(
      nnkBracketExpr.newTree(h, E, Sh), engine, batch, n)
    result.add nnkIfStmt.newTree(
      nnkElifBranch.newTree(call, nnkReturnStmt.newTree(newEmptyNode())))

proc evalWalk[E: HwEngine, Sh: tuple](engine: E, batch: Batch[E], n: TensorNode) =
  let asView = n.to(ViewNode)
  if not asView.isNil:
    # views alias their source's buffer cell, materialize the source first
    evalWalk[E, Sh](engine, batch, asView.src)
    return
  # op nodes enqueue through the handlers their op files registered.
  # A node no handler owns is a materialized impl, nothing to enqueue.
  dispatchEnqueueNodes(E, Sh, engine, batch, n)

proc evalNode*[E: HwEngine](engine: E, batch: Batch[E], n: TensorNode) =
  ## Walks one node at the node's own rank, dispatching the rank tuple.
  ## Op enqueue handlers recurse into a node's children through this proc.
  case n.getImpl().shape().len
  of 1: evalWalk[E, (int,)](engine, batch, n)
  of 2: evalWalk[E, (int, int)](engine, batch, n)
  of 3: evalWalk[E, (int, int, int)](engine, batch, n)
  of 4: evalWalk[E, (int, int, int, int)](engine, batch, n)
  of 5: evalWalk[E, (int, int, int, int, int)](engine, batch, n)
  else:
    raise ValueError.newException(
      "eval: rank " & $n.getImpl().shape().len & " unsupported")

proc eval*[E: HwEngine](engine: E, roots: Span[TensorNode]) =
  if roots.len == 0:
    return
  var batch = engine.recordBatch()
  for i in 0 ..< roots.len:
    evalNode(engine, batch, roots[i])
  batch.submit()

# ═══════════════════════════════════════════════════════════════════════
#  Ops - free functions over TensorNode, the engine materializes
# ═══════════════════════════════════════════════════════════════════════

proc transpose*(t: Tensor): Tensor =
  ## View of `t` with every axis reversed.
  ## Args:
  ## - t: the source tensor, any rank
  ## Matches MLX transpose with no axes argument.
  ## Shared buffer, no allocation and no copy.
  let impl = t.node.getImpl()
  let s = impl.shape()
  let st = impl.strides()
  var rs = newSeq[int](s.len)
  var rt = newSeq[int](s.len)
  for i in 0 ..< s.len:
    rs[i] = s[s.len - 1 - i]
    rt[i] = st[st.len - 1 - i]
  let v = t.engine.makeViewImpl(impl, rs.toSpan(), rt.toSpan(), 0)
  Tensor(engine: t.engine, node: ViewNode(src: t.node, v: v).to(TensorNode))

proc narrow*(t: Tensor, axis: int, start: int, width: int): Tensor =
  ## Window view of `t`, other axes unchanged.
  ## Args:
  ## - axis: the narrowed dimension
  ## - start, width: the window inside that dimension
  ## The element offset shifts by start * stride[axis].
  ## Shared buffer, no allocation and no copy.
  let impl = t.node.getImpl()
  let s = impl.shape()
  let st = impl.strides()
  if axis < 0 or axis >= s.len:
    raise ValueError.newException("narrow: axis " & $axis & " outside the rank " & $s.len)
  if start < 0 or width < 0 or start + width > s[axis]:
    raise ValueError.newException(
      "narrow: window [" & $start & ", " & $(start + width) & ") outside dimension " & $s[axis])
  var ns = newSeq[int](s.len)
  for i in 0 ..< s.len:
    ns[i] = s[i]
  ns[axis] = width
  let offset = impl.offset() + start * st[axis]
  let v = t.engine.makeViewImpl(impl, ns.toSpan(), st, offset)
  Tensor(engine: t.engine, node: ViewNode(src: t.node, v: v).to(TensorNode))

proc tn_split*(engine: AnyEngine, x: TensorNode, numParts: int, axis: int = -1): seq[TensorNode] =
  let ix = x.getImpl()
  let s = ix.shape()
  let st = ix.strides()
  let ax = if axis == -1: s.len - 1 else: axis
  if ax < 0 or ax >= s.len:
    raise ValueError.newException(
      "tn_split: axis " & $axis & " outside the rank " & $s.len)
  if s[ax] mod numParts != 0:
    raise ValueError.newException(
      "tn_split: dimension " & $s[ax] & " not divisible by " & $numParts)
  # parts are shared-buffer views over the source impl:
  # - layout: the source layout narrowed along the split axis
  # - offset: part i starts at i * partLen * stride[axis] elements
  var partShape = newSeq[int](s.len)
  for i in 0 ..< s.len:
    partShape[i] = s[i]
  partShape[ax] = s[ax] div numParts
  for i in 0 ..< numParts:
    let offset = i * partShape[ax] * st[ax]
    let v = engine.makeViewImpl(ix, partShape.toSpan(), st, offset)
    result.add(ViewNode(src: x, v: v).to(TensorNode))

# ═══════════════════════════════════════════════════════════════════════
#  Tensor - the erased user face
# ═══════════════════════════════════════════════════════════════════════

proc newTensor*(engine: AnyEngine, shape: openArray[int], dtype: Dtype): Tensor =
  if shape.len == 0:
    raise ValueError.newException("newTensor: empty shape")
  Tensor(engine: engine,
         node: engine.newTensorNode(dtype, shape.toSpan()))

template erasedUpload[E: HwEngine](engine: E, impl: AnyTensorImpl, src: openArray[byte], T: typedesc) =
  # dense host upload through the typed impl, one rank branch per tuple shape
  type Mem = typeof(engine.allocDevice(T, 0))
  case impl.shape().len
  of 1:
    let t = impl.to(TensorImpl[Mem, Layout[(int,), (int,)]])
    let numel = t.layout.shape.getNumElems()
    if src.len != numel * sizeof(T):
      raise ValueError.newException(
        "copyToDeviceFromHost: expected " & $(numel * sizeof(T)) & " bytes, got " & $src.len)
    if t.offset != 0 or not isDenseRowMajor(t.layout.shape, t.layout.stride):
      raise ValueError.newException(
        "copyToDeviceFromHost: the engine upload is a flat write, only dense impls accept it")
    engine.copyToDeviceFromHost(t.data[], src.toOpenArray(T))
  of 2:
    let t = impl.to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
    let numel = t.layout.shape.getNumElems()
    if src.len != numel * sizeof(T):
      raise ValueError.newException(
        "copyToDeviceFromHost: expected " & $(numel * sizeof(T)) & " bytes, got " & $src.len)
    if t.offset != 0 or not isDenseRowMajor(t.layout.shape, t.layout.stride):
      raise ValueError.newException(
        "copyToDeviceFromHost: the engine upload is a flat write, only dense impls accept it")
    engine.copyToDeviceFromHost(t.data[], src.toOpenArray(T))
  of 3:
    let t = impl.to(TensorImpl[Mem, Layout[(int, int, int), (int, int, int)]])
    let numel = t.layout.shape.getNumElems()
    if src.len != numel * sizeof(T):
      raise ValueError.newException(
        "copyToDeviceFromHost: expected " & $(numel * sizeof(T)) & " bytes, got " & $src.len)
    if t.offset != 0 or not isDenseRowMajor(t.layout.shape, t.layout.stride):
      raise ValueError.newException(
        "copyToDeviceFromHost: the engine upload is a flat write, only dense impls accept it")
    engine.copyToDeviceFromHost(t.data[], src.toOpenArray(T))
  of 4:
    let t = impl.to(TensorImpl[Mem, Layout[(int, int, int, int), (int, int, int, int)]])
    let numel = t.layout.shape.getNumElems()
    if src.len != numel * sizeof(T):
      raise ValueError.newException(
        "copyToDeviceFromHost: expected " & $(numel * sizeof(T)) & " bytes, got " & $src.len)
    if t.offset != 0 or not isDenseRowMajor(t.layout.shape, t.layout.stride):
      raise ValueError.newException(
        "copyToDeviceFromHost: the engine upload is a flat write, only dense impls accept it")
    engine.copyToDeviceFromHost(t.data[], src.toOpenArray(T))
  of 5:
    let t = impl.to(TensorImpl[Mem, Layout[(int, int, int, int, int), (int, int, int, int, int)]])
    let numel = t.layout.shape.getNumElems()
    if src.len != numel * sizeof(T):
      raise ValueError.newException(
        "copyToDeviceFromHost: expected " & $(numel * sizeof(T)) & " bytes, got " & $src.len)
    if t.offset != 0 or not isDenseRowMajor(t.layout.shape, t.layout.stride):
      raise ValueError.newException(
        "copyToDeviceFromHost: the engine upload is a flat write, only dense impls accept it")
    engine.copyToDeviceFromHost(t.data[], src.toOpenArray(T))
  else:
    raise ValueError.newException("copyToDeviceFromHost: rank " & $impl.shape().len & " unsupported")

template erasedDownload[E: HwEngine](engine: E, impl: AnyTensorImpl, dst: var openArray[byte], T: typedesc) =
  # dense device download through the typed impl, one rank branch per tuple shape
  type Mem = typeof(engine.allocDevice(T, 0))
  case impl.shape().len
  of 1:
    let t = impl.to(TensorImpl[Mem, Layout[(int,), (int,)]])
    let numel = t.layout.shape.getNumElems()
    if dst.len != numel * sizeof(T):
      raise ValueError.newException(
        "copyToHostFromDevice: expected " & $(numel * sizeof(T)) & " bytes, got " & $dst.len)
    if t.offset == 0 and isDenseRowMajor(t.layout.shape, t.layout.stride):
      engine.copyToHostFromDevice(dst.toOpenArray(T), t.data[])
    else:
      gatherToHost(engine, t, dst.toOpenArray(T))
  of 2:
    let t = impl.to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
    let numel = t.layout.shape.getNumElems()
    if dst.len != numel * sizeof(T):
      raise ValueError.newException(
        "copyToHostFromDevice: expected " & $(numel * sizeof(T)) & " bytes, got " & $dst.len)
    if t.offset == 0 and isDenseRowMajor(t.layout.shape, t.layout.stride):
      engine.copyToHostFromDevice(dst.toOpenArray(T), t.data[])
    else:
      gatherToHost(engine, t, dst.toOpenArray(T))
  of 3:
    let t = impl.to(TensorImpl[Mem, Layout[(int, int, int), (int, int, int)]])
    let numel = t.layout.shape.getNumElems()
    if dst.len != numel * sizeof(T):
      raise ValueError.newException(
        "copyToHostFromDevice: expected " & $(numel * sizeof(T)) & " bytes, got " & $dst.len)
    if t.offset == 0 and isDenseRowMajor(t.layout.shape, t.layout.stride):
      engine.copyToHostFromDevice(dst.toOpenArray(T), t.data[])
    else:
      gatherToHost(engine, t, dst.toOpenArray(T))
  of 4:
    let t = impl.to(TensorImpl[Mem, Layout[(int, int, int, int), (int, int, int, int)]])
    let numel = t.layout.shape.getNumElems()
    if dst.len != numel * sizeof(T):
      raise ValueError.newException(
        "copyToHostFromDevice: expected " & $(numel * sizeof(T)) & " bytes, got " & $dst.len)
    if t.offset == 0 and isDenseRowMajor(t.layout.shape, t.layout.stride):
      engine.copyToHostFromDevice(dst.toOpenArray(T), t.data[])
    else:
      gatherToHost(engine, t, dst.toOpenArray(T))
  of 5:
    let t = impl.to(TensorImpl[Mem, Layout[(int, int, int, int, int), (int, int, int, int, int)]])
    let numel = t.layout.shape.getNumElems()
    if dst.len != numel * sizeof(T):
      raise ValueError.newException(
        "copyToHostFromDevice: expected " & $(numel * sizeof(T)) & " bytes, got " & $dst.len)
    if t.offset == 0 and isDenseRowMajor(t.layout.shape, t.layout.stride):
      engine.copyToHostFromDevice(dst.toOpenArray(T), t.data[])
    else:
      gatherToHost(engine, t, dst.toOpenArray(T))
  else:
    raise ValueError.newException("copyToHostFromDevice: rank " & $impl.shape().len & " unsupported")

proc copyToDeviceFromHost*[E: HwEngine](engine: E, node: TensorNode, src: openArray[byte]) =
  let impl = node.getImpl()
  case impl.dtype()
  of kFp32: erasedUpload(engine, impl, src, float32)
  of kBf16: erasedUpload(engine, impl, src, bfloat16)
  else:
    raise ValueError.newException("copyToDeviceFromHost: dtype " & $impl.dtype() & " unsupported")

proc copyToHostFromDevice*[E: HwEngine](engine: E, dst: var openArray[byte], node: TensorNode) =
  let impl = node.getImpl()
  case impl.dtype()
  of kFp32: erasedDownload(engine, impl, dst, float32)
  of kBf16: erasedDownload(engine, impl, dst, bfloat16)
  else:
    raise ValueError.newException("copyToHostFromDevice: dtype " & $impl.dtype() & " unsupported")

proc copyToDeviceFromHost*[T](t: Tensor, src: openArray[T]) =
  t.engine.copyToDeviceFromHost(t.node,
    toOpenArray(cast[ptr UncheckedArray[byte]](unsafeAddr src[0]), 0, src.len * sizeof(T) - 1))

proc copyToHostFromDevice*[T](dst: var openArray[T], t: Tensor) =
  t.engine.copyToHostFromDevice(
    toOpenArray(cast[ptr UncheckedArray[byte]](addr dst[0]), 0, dst.len * sizeof(T) - 1),
    t.node)

proc split*(t: Tensor, numParts: int, axis: int = -1): seq[Tensor] =
  for n in t.engine.tn_split(t.node, numParts, axis):
    result.add(Tensor(engine: t.engine, node: n))

proc arange*(engine: AnyEngine, n: int, dtype: Dtype = kBf16): Tensor =
  ## Rank-1 tensor of `n` elements 0, 1, ..., n-1.
  ## Args:
  ## - n: the element count
  ## - dtype: the element type, kFp16 raises
  ## Values compute on the host and upload in one DMA. The op never
  ## reads the device back.
  case dtype
  of kFp32:
    var host = newSeq[float32](n)
    for i in 0 ..< n:
      host[i] = float32(i)
    result = newTensor(engine, @[n], dtype)
    result.copyToDeviceFromHost(host)
  of kBf16:
    var host = newSeq[bfloat16](n)
    for i in 0 ..< n:
      host[i] = float32(i).to(bfloat16)
    result = newTensor(engine, @[n], dtype)
    result.copyToDeviceFromHost(host)
  else:
    raise ValueError.newException("arange: dtype " & $dtype & " unsupported")

proc full*(engine: AnyEngine, shape: openArray[int], value: float32, dtype: Dtype = kBf16): Tensor =
  ## Tensor of `shape` filled with `value`.
  ## Args:
  ## - shape: the tensor shape, any supported rank
  ## - value: the fill scalar, rounded to the element type
  ## - dtype: the element type, kFp16 raises
  ## Values compute on the host and upload in one DMA. The op never
  ## reads the device back.
  var numel = 1
  for i in 0 ..< shape.len:
    numel *= shape[i]
  case dtype
  of kFp32:
    var host = newSeq[float32](numel)
    for i in 0 ..< numel:
      host[i] = value
    result = newTensor(engine, shape, dtype)
    result.copyToDeviceFromHost(host)
  of kBf16:
    var host = newSeq[bfloat16](numel)
    let q = value.to(bfloat16)
    for i in 0 ..< numel:
      host[i] = q
    result = newTensor(engine, shape, dtype)
    result.copyToDeviceFromHost(host)
  else:
    raise ValueError.newException("full: dtype " & $dtype & " unsupported")

proc item*(t: Tensor): float32 =
  ## Scalar value of a single-element tensor, one device readback.
  ## Raises on any other shape.
  let s = t.node.getImpl().shape()
  if s.len != 1 or s[0] != 1:
    raise ValueError.newException(
      "item: a single-element tensor is required, got rank " & $s.len)
  case t.node.getImpl().dtype()
  of kFp32:
    var got: array[1, float32]
    copyToHostFromDevice(got, t)
    result = got[0]
  of kBf16:
    var got: array[1, bfloat16]
    copyToHostFromDevice(got, t)
    result = got[0].to(float32)
  else:
    raise ValueError.newException("item: dtype " & $t.node.getImpl().dtype() & " unsupported")

proc prune*(t: var Tensor) =
  ## Rebinds `t` to a materialized leaf carrying the node's impl.
  ## Precondition: the node's graph evaluated, the impl buffers hold data.
  ## The upstream graph becomes unreachable from `t`. Its impls collect
  ## under ORC once no other tensor references them.
  t.node = t.node.getImpl().to(TensorNode)

proc eval*(ts: varargs[Tensor]) =
  var roots = newSeq[TensorNode](ts.len)
  for i in 0 ..< ts.len:
    roots[i] = ts[i].node
  ts[0].engine.eval(roots.toSpan())

# Per-backend engine handles, constructed on first use.
# Each instance sits in a raw pointer: no destructor, no eager driver
# library load in binaries that import this module without the backend.
var
  gMetalEngineInstance: pointer
  gOpenCLEngineInstance: pointer
  gVulkanEngineInstance: pointer
  gWebgpuEngineInstance: pointer
  gCudaEngineInstance: pointer

proc gMetalEngine*(): MetalEngine =
  if gMetalEngineInstance.isNil:
    gMetalEngineInstance = cast[pointer](newEngine(runMetal))
  cast[MetalEngine](gMetalEngineInstance)

proc gOpenCLEngine*(): OpenCLEngine =
  if gOpenCLEngineInstance.isNil:
    gOpenCLEngineInstance = cast[pointer](newEngine(runOpenCL))
  cast[OpenCLEngine](gOpenCLEngineInstance)

proc gVulkanEngine*(): VulkanEngine =
  if gVulkanEngineInstance.isNil:
    gVulkanEngineInstance = cast[pointer](newEngine(runVulkan))
  cast[VulkanEngine](gVulkanEngineInstance)

proc gWebgpuEngine*(): WebgpuEngine =
  if gWebgpuEngineInstance.isNil:
    gWebgpuEngineInstance = cast[pointer](newEngine(runWebGPU))
  cast[WebgpuEngine](gWebgpuEngineInstance)

proc gCudaEngine*(): CudaEngine =
  if gCudaEngineInstance.isNil:
    gCudaEngineInstance = cast[pointer](newEngine(runCuda))
  cast[CudaEngine](gCudaEngineInstance)
