# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/assertions
import std/typetraits

import workspace/crucible
import workspace/crucible/src/builtins/builtins_gpu_types
export builtins_gpu_types
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/ceramic/src/macros/static_for
import workspace/math
import ../vendor/iface/iface
import ./kernels/k_add
import ./kernels/k_divmod

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

template toSpan(s: tuple): Span[int] =
  Span[int](p: cast[ptr UncheckedArray[int]](unsafeAddr s), len: tupleLen(s))

template toSpan[T](s: seq[T]): Span[T] =
  Span[T](p: cast[ptr UncheckedArray[T]](unsafeAddr s[0]), len: s.len)

template toSpan[T](s: openArray[T]): Span[T] =
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

iface *TensorNode:
    proc eval()
    proc getImpl(): AnyTensorImpl

type
  TensorImpl*[Mem, L] = ref object
    data*: ref Mem
    offset*: int
    layout*: L

  TensorBridge[DevMem, Sh, St] = object
    data {.cursor.}: DevMem
    layout: Layout[Sh, St]

  PosNode[C] = ref object of RootRef
    core: C
    pos: int

  AddNode = ref object
    r: AnyTensorImpl
    a, b: TensorNode

  DivModCore = ref object
    co: array[2, AnyTensorImpl]
    x, y: TensorNode

  ChunkSplitCore = ref object
    r: seq[AnyTensorImpl]
    a: TensorNode
    axis: int


iface *AnyEngine:
    proc newTensorNode(dtype: Dtype, shape: Span[int]): TensorNode
    proc copyToDeviceFromHost(node: TensorNode, src: openArray[byte])
    proc copyToHostFromDevice(dst: var openArray[byte], node: TensorNode)
    proc eval(roots: Span[TensorNode])

type Tensor* = object
  engine*: AnyEngine
  node*: TensorNode

# ═══════════════════════════════════════════════════════════════════════
#  Layout helpers
# ═══════════════════════════════════════════════════════════════════════

func getNumElems[Sh: tuple](shape: Sh): int =
  result = 1
  staticFor i, 0, tupleLen(Sh):
    result *= shape[i]

func rowMajorStrides[Sh: tuple](shape: Sh): Sh =
  var acc = 1
  staticFor j, 0, tupleLen(Sh):
    result[tupleLen(Sh) - 1 - j] = acc
    acc *= shape[tupleLen(Sh) - 1 - j]

func fillShape[Sh: tuple](shape: var Sh, s: Span[int]) =
  if s.len != tupleLen(Sh):
    raise ValueError.newException(
      "rank " & $s.len & " does not match the context rank " & $tupleLen(Sh))
  staticFor i, 0, tupleLen(Sh):
    shape[i] = s[i]

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - AnyTensorImpl implementation
# ═══════════════════════════════════════════════════════════════════════

proc shape*[Sh: tuple, Mem](t: TensorImpl[Mem, Layout[Sh, Sh]]): Span[int] =
  toSpan(t.layout.shape)

proc strides*[Sh: tuple, Mem](t: TensorImpl[Mem, Layout[Sh, Sh]]): Span[int] =
  toSpan(t.layout.stride)

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

proc eval*(n: AddNode) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getImpl*(n: AddNode): AnyTensorImpl =
  n.r

proc eval*(n: DivModCore) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getOutput*(n: DivModCore, i: int): AnyTensorImpl =
  n.co[i]

proc eval*(n: ChunkSplitCore) =
  raiseAssert "eval is not implemented yet: graph layer only"

proc getOutput*(n: ChunkSplitCore, i: int): AnyTensorImpl =
  n.r[i]

proc eval*[C](n: PosNode[C]) =
  n.core.eval()

proc getImpl*[C](n: PosNode[C]): AnyTensorImpl =
  n.core.getOutput(n.pos)

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
  of 2:
    var sh: (int, int)
    fillShape(sh, shape)
    makeTensorImpl(engine, dtype, make_layout(sh, rowMajorStrides(sh)))
  else:
    raise ValueError.newException("makeTensorImpl: rank " & $shape.len & " unsupported")

proc makeTensorImpl[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): AnyTensorImpl =
  makeTensorImpl(engine, dtype, shape.toSpan())

proc newTensorNode*[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): TensorNode =
  makeTensorImpl(engine, dtype, shape).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Eval - pull walk, enqueue into one batch, submit
# ═══════════════════════════════════════════════════════════════════════

proc evalWalk[E: HwEngine, Sh: tuple, Mem, T](
    engine: E, batch: Batch[E], n: TensorNode) =
  let asAdd = n.to(AddNode)
  if not asAdd.isNil:
    let rz = asAdd.r.to(TensorImpl[Mem, Layout[Sh, Sh]])
    evalWalk[E, Sh, Mem, T](engine, batch, asAdd.a)
    evalWalk[E, Sh, Mem, T](engine, batch, asAdd.b)
    let
      xa = asAdd.a.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
      xb = asAdd.b.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
      numel = rz.layout.shape.getNumElems()
    let
      zv = TensorBridge[Mem, Sh, Sh](data: rz.data[], layout: rz.layout)
      xv = TensorBridge[Mem, Sh, Sh](data: xa.data[], layout: xa.layout)
      yv = TensorBridge[Mem, Sh, Sh](data: xb.data[], layout: xb.layout)
    batch.enqueue<<(ceil_div(numel, 128), 32)>> k_add(zv, xv, yv)
    return
  let asDiv = n.to(PosNode[DivModCore])
  if not asDiv.isNil:
    let core = asDiv.core
    evalWalk[E, Sh, Mem, T](engine, batch, core.x)
    evalWalk[E, Sh, Mem, T](engine, batch, core.y)
    let
      xx = core.x.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
      xy = core.y.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
      qz = core.co[0].to(TensorImpl[Mem, Layout[Sh, Sh]])
      rz = core.co[1].to(TensorImpl[Mem, Layout[Sh, Sh]])
      numel = qz.layout.shape.getNumElems()
    let
      qv = TensorBridge[Mem, Sh, Sh](data: qz.data[], layout: qz.layout)
      rv = TensorBridge[Mem, Sh, Sh](data: rz.data[], layout: rz.layout)
      xv = TensorBridge[Mem, Sh, Sh](data: xx.data[], layout: xx.layout)
      yv = TensorBridge[Mem, Sh, Sh](data: xy.data[], layout: xy.layout)
    batch.enqueue<<(ceil_div(numel, 128), 32)>> k_divmod(qv, rv, xv, yv)
    return
  let asChunk = n.to(PosNode[ChunkSplitCore])
  if not asChunk.isNil:
    let core = asChunk.core
    evalWalk[E, Sh, Mem, T](engine, batch, core.a)
    let src = core.a.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
    for i in 0 ..< core.r.len:
      let v = core.r[i].to(TensorImpl[Mem, Layout[Sh, Sh]])
      v.data = src.data
    return
  if not n.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]]).isNil:
    return
  raise ValueError.newException("eval: unsupported node in the graph")

proc eval*[E: HwEngine](engine: E, roots: Span[TensorNode]) =
  if roots.len == 0:
    return
  let impl = roots[0].getImpl()
  var batch = engine.recordBatch()
  case impl.dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    case impl.shape().len
    of 2:
      for i in 0 ..< roots.len:
        evalWalk[E, (int, int), Mem, float32](engine, batch, roots[i])
    else:
      raise ValueError.newException("eval: rank " & $impl.shape().len & " unsupported")
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    case impl.shape().len
    of 2:
      for i in 0 ..< roots.len:
        evalWalk[E, (int, int), Mem, bfloat16](engine, batch, roots[i])
    else:
      raise ValueError.newException("eval: rank " & $impl.shape().len & " unsupported")
  else:
    raise ValueError.newException("eval: dtype unsupported")
  batch.submit()

# ═══════════════════════════════════════════════════════════════════════
#  Ops - free functions over TensorNode, the engine materializes
# ═══════════════════════════════════════════════════════════════════════

proc tn_add*(engine: AnyEngine, a, b: TensorNode): TensorNode =
  let ia = a.getImpl()
  let ib = b.getImpl()
  if ia.shape() != ib.shape():
    raise ValueError.newException("tn_add: shape mismatch")
  if ia.dtype() != ib.dtype():
    raise ValueError.newException("tn_add: dtype mismatch")
  AddNode(a: a, b: b, r: engine.newTensorNode(ia.dtype(), ia.shape()).getImpl()).to(TensorNode)

proc tn_divmod*(engine: AnyEngine, x, y: TensorNode): tuple[q, rem: TensorNode] =
  let ix = x.getImpl()
  let iy = y.getImpl()
  if ix.shape() != iy.shape():
    raise ValueError.newException("tn_divmod: shape mismatch")
  if ix.dtype() != iy.dtype():
    raise ValueError.newException("tn_divmod: dtype mismatch")
  let core = DivModCore(
    x: x, y: y,
    co: [engine.newTensorNode(ix.dtype(), ix.shape()).getImpl(),
         engine.newTensorNode(ix.dtype(), ix.shape()).getImpl()])
  result[0] = PosNode[DivModCore](core: core, pos: 0).to(TensorNode)
  result[1] = PosNode[DivModCore](core: core, pos: 1).to(TensorNode)

proc tn_chunk*(engine: AnyEngine, x: TensorNode, axis: int, k: int): seq[TensorNode] =
  let ix = x.getImpl()
  let s = ix.shape()
  if axis != 0:
    # erased impls carry contiguous layouts only, a split on an inner
    # axis needs the inherited strides of the source layout
    raise ValueError.newException(
      "tn_chunk: axis " & $axis & " unsupported, only the outer axis splits densely")
  if s[axis] mod k != 0:
    raise ValueError.newException(
      "tn_chunk: dimension " & $s[axis] & " not divisible by " & $k)
  var partShape = newSeq[int](s.len)
  for i in 0 ..< s.len:
    partShape[i] = s[i]
  partShape[axis] = s[axis] div k
  let core = ChunkSplitCore(a: x, axis: axis)
  for i in 0 ..< k:
    core.r.add(engine.newTensorNode(ix.dtype(), partShape.toSpan()).getImpl())
  for i in 0 ..< k:
    result.add(PosNode[ChunkSplitCore](core: core, pos: i).to(TensorNode))

# ═══════════════════════════════════════════════════════════════════════
#  Tensor - the erased user face
# ═══════════════════════════════════════════════════════════════════════

proc newTensor*(engine: AnyEngine, shape: openArray[int], dtype: Dtype): Tensor =
  if shape.len == 0:
    raise ValueError.newException("newTensor: empty shape")
  Tensor(engine: engine,
         node: engine.newTensorNode(dtype, shape.toSpan()))

proc copyToDeviceFromHost*[E: HwEngine](engine: E, node: TensorNode, src: openArray[byte]) =
  let impl = node.getImpl()
  case impl.dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    case impl.shape().len
    of 2:
      let t = impl.to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
      let numel = t.layout.shape.getNumElems()
      if src.len != numel * sizeof(float32):
        raise ValueError.newException(
          "copyToDeviceFromHost: expected " & $(numel * sizeof(float32)) & " bytes, got " & $src.len)
      engine.copyToDeviceFromHost(t.data[],
        src.toOpenArray(float32))
    else:
      raise ValueError.newException("copyToDeviceFromHost: rank " & $impl.shape().len & " unsupported")
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    case impl.shape().len
    of 2:
      let t = impl.to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
      let numel = t.layout.shape.getNumElems()
      if src.len != numel * sizeof(bfloat16):
        raise ValueError.newException(
          "copyToDeviceFromHost: expected " & $(numel * sizeof(bfloat16)) & " bytes, got " & $src.len)
      engine.copyToDeviceFromHost(t.data[],
        src.toOpenArray(bfloat16))
    else:
      raise ValueError.newException("copyToDeviceFromHost: rank " & $impl.shape().len & " unsupported")
  else:
    raise ValueError.newException("copyToDeviceFromHost: dtype " & $impl.dtype() & " unsupported")

proc copyToHostFromDevice*[E: HwEngine](engine: E, dst: var openArray[byte], node: TensorNode) =
  let impl = node.getImpl()
  case impl.dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    case impl.shape().len
    of 2:
      let t = impl.to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
      let numel = t.layout.shape.getNumElems()
      if dst.len != numel * sizeof(float32):
        raise ValueError.newException(
          "copyToHostFromDevice: expected " & $(numel * sizeof(float32)) & " bytes, got " & $dst.len)
      engine.copyToHostFromDevice(dst.toOpenArray(float32), t.data[])
    else:
      raise ValueError.newException("copyToHostFromDevice: rank " & $impl.shape().len & " unsupported")
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    case impl.shape().len
    of 2:
      let t = impl.to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
      let numel = t.layout.shape.getNumElems()
      if dst.len != numel * sizeof(bfloat16):
        raise ValueError.newException(
          "copyToHostFromDevice: expected " & $(numel * sizeof(bfloat16)) & " bytes, got " & $dst.len)
      engine.copyToHostFromDevice(dst.toOpenArray(bfloat16), t.data[])
    else:
      raise ValueError.newException("copyToHostFromDevice: rank " & $impl.shape().len & " unsupported")
  else:
    raise ValueError.newException("copyToHostFromDevice: dtype " & $impl.dtype() & " unsupported")

proc copyToDeviceFromHost*[T](t: Tensor, src: openArray[T]) =
  t.engine.copyToDeviceFromHost(t.node,
    toOpenArray(cast[ptr UncheckedArray[byte]](unsafeAddr src[0]), 0, src.len * sizeof(T) - 1))

proc copyToHostFromDevice*[T](dst: var openArray[T], t: Tensor) =
  t.engine.copyToHostFromDevice(
    toOpenArray(cast[ptr UncheckedArray[byte]](addr dst[0]), 0, dst.len * sizeof(T) - 1),
    t.node)

proc `+`*(a, b: Tensor): Tensor =
  Tensor(engine: a.engine, node: tn_add(a.engine, a.node, b.node))

proc divmod*(a, b: Tensor): tuple[q, rem: Tensor] =
  let (qn, rn) = tn_divmod(a.engine, a.node, b.node)
  (Tensor(engine: a.engine, node: qn), Tensor(engine: a.engine, node: rn))

proc chunk*(t: Tensor, axis: int, k: int): seq[Tensor] =
  for n in t.engine.tn_chunk(t.node, axis, k):
    result.add(Tensor(engine: t.engine, node: n))

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
