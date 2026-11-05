# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Typed tensor implementations behind the erased AnyTensorImpl face.
##
## TensorImpl[T, Hw] stores the producing op id, tensor id, grad flag, runtime shape, strides,
## element offset and the backend storage for element type T.
## The erased face has no type parameters, calls dispatch through the generated vtable.
##
## Backend responsibilities:
## - each op packs its blob once at enqueue and appends one record entry
## - the no-grad set runs the same kernels, grad flag off, for backward()
## - shape ops are pure metadata: a new view over the same storage, one record
##   entry per op that every backend drain skips
##
## iface initializes each global vtable lazily on first use, single-threaded use only.

import pkg/iface

import
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery/src/raw/dtypes,
  workspace/tensorcery/src/ctx,
  workspace/tensorcery/src/kernels/k_shape,
  workspace/tensorcery/src/kernels/k_sgd_adam

iface *AnyTensorImpl:
  proc getTid(): int
  proc getRequiresGrad(): bool
  proc getNumel(): int
  proc getShape(): SmallSeq[4, int32]
  proc getStrides(): SmallSeq[4, int32]
  proc add(other: AnyTensorImpl): AnyTensorImpl
  proc mul(other: AnyTensorImpl): AnyTensorImpl
  proc mulScalar(scalar: float64): AnyTensorImpl
  proc addNoGrad(other: AnyTensorImpl): AnyTensorImpl
  proc mulNoGrad(other: AnyTensorImpl): AnyTensorImpl
  proc mulScalarNoGrad(scalar: float64): AnyTensorImpl
  proc onesNoGrad(): AnyTensorImpl
  proc reshape(shape: SmallSeq[4, int32]): AnyTensorImpl
  proc permuteDims(perm: SmallSeq[4, int32]): AnyTensorImpl
  proc sliceRange(start, count: SmallSeq[4, int32]): AnyTensorImpl
  proc aliasAs(other: AnyTensorImpl): AnyTensorImpl
  proc aliasStrided(shape, strides: SmallSeq[4, int32]): AnyTensorImpl
  proc gather(indices: AnyTensorImpl): AnyTensorImpl
  proc layernorm(w, b: AnyTensorImpl, eps: float64): AnyTensorImpl
  proc gelu(): AnyTensorImpl
  proc softmax(): AnyTensorImpl
  proc crossEntropy(targets: AnyTensorImpl): AnyTensorImpl
  proc where(cond, other: AnyTensorImpl): AnyTensorImpl
  proc sgdUpdate(grad: AnyTensorImpl, lr: float64): void
  proc adamUpdate(grad, m, v: AnyTensorImpl, pack: AdamScalars): void
  proc softmaxBwdNoGrad(g, p: AnyTensorImpl): AnyTensorImpl
  proc geluBwdNoGrad(g: AnyTensorImpl): AnyTensorImpl
  proc layernormBwdDxNoGrad(g, w: AnyTensorImpl, eps: float64): AnyTensorImpl
  proc layernormBwdWNoGrad(g, x: AnyTensorImpl, eps: float64): AnyTensorImpl
  proc layernormBwdBNoGrad(g: AnyTensorImpl): AnyTensorImpl
  proc crossEntropyBwdNoGrad(g, targets: AnyTensorImpl): AnyTensorImpl
  proc gatherBwdNoGrad(g, idx: AnyTensorImpl): AnyTensorImpl
  proc sliceBwdNoGrad(g, view: AnyTensorImpl, start: SmallSeq[4, int32]): AnyTensorImpl
  proc readbackBytes(): seq[byte]

type DeviceHandle* = distinct uint64
  ## Opaque 8-byte device-buffer handle, owned by the device backend.

type SharedStorage*[T] = ref object
  ## CPU storage one tensor and every shape-op view over it hold together.
  ##
  ## ORC shares no seq between two owners:
  ##
  ## - a borrowed constructor argument deep-copies the buffer
  ## - a moved-out argument empties the source
  ##
  ## Reads materialize through `seqNo`, which names the op id that last wrote
  ## the buffer. A read of a pending write evals the record first.
  data*: seq[T]
  seqNo*: int

type SharedMemory* = ref object
  ## Metal storage one tensor and every shape-op view over it hold together.
  ##
  ## - `numel`: the allocation's element count in elements. The slice backward
  ##   scatters over it, and a view's parent can hold fewer elements than the storage
  ## - `seqNo`: the op id that last wrote the buffer. A read of a pending
  ##   write evals the record first
  handle*: DeviceHandle
  seqNo*: int
  numel*: int

type TensorImpl*[T; Hw: static BackendKind] = ref object
  ## Typed tensor state behind one entry of the erased face.
  ctx*: Ctx
  tid*: int       ## tensor id, unique within the context, referenced by record entries
  requiresGrad*: bool
  offset*: int    ## element offset of the view origin into the storage
  shape*: SmallSeq[4, int32]
  strides*: SmallSeq[4, int32]
  when Hw == bkCpu:
    storage*: SharedStorage[T]
  elif Hw == bkMetal:
    mem*: SharedMemory

func numel*(shape: openArray[int]): int =
  ## Returns the element count as a product over all shape sizes.
  result = 1
  for dim in shape:
    result *= dim

func toArray4*(v: SmallSeq[4, int32]): array[4, int32] =
  ## Copies the axes into a fixed 4-array, missing axes zeroed.
  for i in 0 ..< int(v.len):
    result[i] = v[i]

func numel32*(shape: SmallSeq[4, int32]): int =
  ## Returns the element count as a product over all shape sizes.
  result = 1
  for i in 0 ..< int(shape.len):
    result *= int(shape[i])

proc contiguousStrides*(shape: SmallSeq[4, int32]): SmallSeq[4, int32] =
  ## Returns the contiguous strides of an already rank-checked shape, one step per dimension in elements.
  var stride = 1
  var strides: array[4, int32]
  for i in countdown(int(shape.len) - 1, 0):
    strides[i] = int32(stride)
    stride *= int(shape[i])
  result = SmallSeq[4, int32].initFrom(strides.toOpenArray(0, int(shape.len) - 1))

proc contiguousMeta*(shape: openArray[int]): tuple[shape, strides: SmallSeq[4, int32]] =
  ## Rank-checked contiguous layout, last dimension varying fastest.
  ##
  ## Raises AssertionDefect when `shape` spans more than 4 dimensions.
  doAssert shape.len in 1 .. 4,
      "rank " & $shape.len & " construction refused, tensorcery supports rank 1 to 4"

  var extents: array[4, int32]
  for i in 0 ..< shape.len:
    extents[i] = int32(shape[i])
  result.shape = SmallSeq[4, int32].initFrom(extents.toOpenArray(0, shape.len - 1))

  result.strides = contiguousStrides(result.shape)

proc appendDim*(shape: SmallSeq[4, int32], dim: int32): SmallSeq[4, int32] =
  ## Appends one axis to a rank-checked shape, yielding the gather output shape.
  let rank = int(shape.len)
  doAssert rank < 4, "gather: output rank " & $(rank + 1) & " refused, tensorcery supports rank 1 to 4"
  var dims: array[4, int32]
  for i in 0 ..< rank:
    dims[i] = shape[i]
  dims[rank] = dim
  result = SmallSeq[4, int32].initFrom(dims.toOpenArray(0, rank))

proc dropLastDim*(shape: SmallSeq[4, int32]): SmallSeq[4, int32] =
  ## Removes the last axis of a rank-checked shape, yielding the cross entropy loss shape.
  let rank = int(shape.len)
  doAssert rank > 1, "cross entropy: a rank 1 logit tensor has no class axis"
  result = SmallSeq[4, int32].initFrom(
    toArray4(shape).toOpenArray(0, rank - 2))

type ViewImage* = object
  ## Padded address image of one strided view: offset, shape and strides
  ## for the divmod chain, plus the last-axis stride for the inner loop.
  off*: int32
  shape*, strides*: array[4, int32]
  step*: int32

proc viewImage*(offset: int, shape, strides: SmallSeq[4, int32]): ViewImage =
  ## Padded address image of one strided view over its storage:
  ##
  ## - shapes pad with 1s, keeping the divmod chain safe
  ## - strides pad with 0s, so padded axes contribute nothing
  ## - the step is the last real axis stride, never a padded 0
  result.off = int32(offset)
  for i in 0 ..< 4:
    if i < int(shape.len):
      result.shape[i] = shape[i]
      result.strides[i] = strides[i]
    else:
      result.shape[i] = 1
      result.strides[i] = 0
  result.step = int32(strides[int(shape.len) - 1])

proc isContiguous*(shape, strides: SmallSeq[4, int32]): bool =
  ## Returns true when `strides` is the contiguous layout of `shape`.
  strides == contiguousStrides(shape)

proc stridedBytes*[T](base: ptr T, shape, strides: SmallSeq[4, int32]): seq[byte] =
  ## Gathers a strided view into a fresh byte image in logical order, last axis varying fastest.
  ##
  ## Contract:
  ## - `base` points at the view's element 0. Callers apply the view offset before the call
  ## - a contiguous view copies in one memcpy, an arbitrary view gathers per element
  ##   through the coordinate divmod chain
  let n = numel32(shape)
  let width = sizeof(T)
  result = newSeq[byte](n * width)
  if n == 0:
    return
  if isContiguous(shape, strides):
    copyMem(result[0].addr, base, n * width)
    return
  let arr = cast[ptr UncheckedArray[T]](base)
  var rank = int(shape.len)
  var coords: array[4, int]
  for i in 0 ..< n:
    var t = i
    for d in countdown(rank - 1, 0):
      coords[d] = t mod int(shape[d])
      t = t div int(shape[d])
    var src = 0
    for d in 0 ..< rank:
      src += coords[d] * int(strides[d])
    copyMem(result[i * width].addr, arr[src].addr, width)

proc toTensorImpl*[T; Hw: static BackendKind](
    ctx: Ctx, data: openArray[T], shape: openArray[int], requiresGrad = false
): TensorImpl[T, Hw] =
  ## Builds a TensorImpl from host data over `shape`, contiguous, last dimension varying fastest.
  ##
  ## Raises AssertionDefect when `shape` spans more than 4 dimensions or when `data` does not hold exactly the element count of `shape`.
  when Hw != bkCpu:
    {.error: "device storage is allocated by the device backend, construct device tensors through the Tensor face".}

  let meta = contiguousMeta(shape)
  let count = numel(shape)

  doAssert data.len == count,
      "data holds " & $data.len & " elements, shape " & $shape & " needs " & $count

  result = TensorImpl[T, Hw](ctx: ctx, tid: ctx.newTid(), requiresGrad: requiresGrad)
  result.shape = meta.shape
  result.strides = meta.strides
  result.storage = SharedStorage[T](data: @data, seqNo: 0)

proc getTid*[T; Hw: static BackendKind](s: TensorImpl[T, Hw]): int =
  ## Returns the tensor id that record entries and the tape registry reference.
  s.tid

proc getRequiresGrad*[T; Hw: static BackendKind](s: TensorImpl[T, Hw]): bool =
  ## Returns the grad flag: true when backward() accumulates a gradient for the tensor.
  s.requiresGrad

proc getNumel*[T; Hw: static BackendKind](s: TensorImpl[T, Hw]): int =
  ## Returns the element count as a product over all shape sizes.
  result = 1
  for i in 0 ..< int(s.shape.len):
    result *= int(s.shape[i])

proc getShape*[T; Hw: static BackendKind](s: TensorImpl[T, Hw]): SmallSeq[4, int32] =
  s.shape

proc getStrides*[T; Hw: static BackendKind](s: TensorImpl[T, Hw]): SmallSeq[4, int32] =
  s.strides

# Shape-op views: pure metadata over the same storage.
# Each op records one entry for the backward walk, no dispatch, no submission.
# One proc set per backend specialization: storage differs by field, validation matches.

proc shapeEntry[T](s: TensorImpl[T, bkCpu], kernel: KernelId, img: ShapeBlob, outImpl: TensorImpl[T, bkCpu]) =
  discard s.ctx.enqueue(kernel, packShapeBlob(int(img.rank), img.a, img.b),
      LaunchCfg(), bkCpu, outImpl.tid, s.tid)

proc shapeEntry[T](s: TensorImpl[T, bkMetal], kernel: KernelId, img: ShapeBlob, outImpl: TensorImpl[T, bkMetal]) =
  discard s.ctx.enqueue(kernel, packShapeBlob(int(img.rank), img.a, img.b),
      LaunchCfg(), bkMetal, outImpl.tid, s.tid)

proc reshape*[T](s: TensorImpl[T, bkCpu], shape: SmallSeq[4, int32]): AnyTensorImpl =
  ## Erased reshape: a new contiguous view over the same storage.
  ## Raises AssertionDefect when the element count changes.
  doAssert numel32(shape) == s.getNumel(),
      "reshape: element count mismatch, shape " & $s.shape & " holds " & $s.getNumel() &
      " elements, " & $shape & " needs " & $numel32(shape)
  doAssert isContiguous(s.shape, s.strides),
      "reshape: non-contiguous view refused, a pure-metadata reshape needs contiguous storage"
  let outImpl = TensorImpl[T, bkCpu](
    ctx: s.ctx, tid: s.ctx.newTid(), requiresGrad: s.requiresGrad,
    offset: s.offset, shape: shape, strides: contiguousStrides(shape), storage: s.storage)
  shapeEntry(s, opReshape, ShapeBlob(rank: int32(shape.len), a: toArray4(shape)), outImpl)
  outImpl.to(AnyTensorImpl)

proc reshape*[T](s: TensorImpl[T, bkMetal], shape: SmallSeq[4, int32]): AnyTensorImpl =
  ## Erased reshape: a new contiguous view over the same device storage.
  ## Raises AssertionDefect when the element count changes.
  doAssert numel32(shape) == s.getNumel(),
      "reshape: element count mismatch, shape " & $s.shape & " holds " & $s.getNumel() &
      " elements, " & $shape & " needs " & $numel32(shape)
  doAssert isContiguous(s.shape, s.strides),
      "reshape: non-contiguous view refused, a pure-metadata reshape needs contiguous storage"
  let outImpl = TensorImpl[T, bkMetal](
    ctx: s.ctx, tid: s.ctx.newTid(), requiresGrad: s.requiresGrad,
    offset: s.offset, shape: shape, strides: contiguousStrides(shape), mem: s.mem)
  shapeEntry(s, opReshape, ShapeBlob(rank: int32(shape.len), a: toArray4(shape)), outImpl)
  outImpl.to(AnyTensorImpl)

proc permuteDims*[T](s: TensorImpl[T, bkCpu], perm: SmallSeq[4, int32]): AnyTensorImpl =
  ## Erased dimension permutation: output axis i reads source axis perm[i].
  ## Raises AssertionDefect when `perm` is not a permutation of the axes.
  let rank = int(s.shape.len)
  doAssert int(perm.len) == rank,
      "permute: permutation holds " & $perm.len & " axes, the tensor has " & $rank
  var seen: array[4, bool]
  var newShape, newStrides: array[4, int32]
  for i in 0 ..< rank:
    let src = int(perm[i])
    doAssert src in 0 ..< rank and not seen[src],
        "permute: " & $perm & " is not a permutation of " & $rank & " axes"
    seen[src] = true
    newShape[i] = s.shape[src]
    newStrides[i] = s.strides[src]
  let outShape = SmallSeq[4, int32].initFrom(newShape.toOpenArray(0, rank - 1))
  let outStrides = SmallSeq[4, int32].initFrom(newStrides.toOpenArray(0, rank - 1))
  let outImpl = TensorImpl[T, bkCpu](
    ctx: s.ctx, tid: s.ctx.newTid(), requiresGrad: s.requiresGrad,
    offset: s.offset, shape: outShape, strides: outStrides, storage: s.storage)
  shapeEntry(s, opPermute, ShapeBlob(rank: int32(rank), a: toArray4(perm)), outImpl)
  outImpl.to(AnyTensorImpl)

proc permuteDims*[T](s: TensorImpl[T, bkMetal], perm: SmallSeq[4, int32]): AnyTensorImpl =
  ## Erased dimension permutation: output axis i reads source axis perm[i].
  ## Raises AssertionDefect when `perm` is not a permutation of the axes.
  let rank = int(s.shape.len)
  doAssert int(perm.len) == rank,
      "permute: permutation holds " & $perm.len & " axes, the tensor has " & $rank
  var seen: array[4, bool]
  var newShape, newStrides: array[4, int32]
  for i in 0 ..< rank:
    let src = int(perm[i])
    doAssert src in 0 ..< rank and not seen[src],
        "permute: " & $perm & " is not a permutation of " & $rank & " axes"
    seen[src] = true
    newShape[i] = s.shape[src]
    newStrides[i] = s.strides[src]
  let outShape = SmallSeq[4, int32].initFrom(newShape.toOpenArray(0, rank - 1))
  let outStrides = SmallSeq[4, int32].initFrom(newStrides.toOpenArray(0, rank - 1))
  let outImpl = TensorImpl[T, bkMetal](
    ctx: s.ctx, tid: s.ctx.newTid(), requiresGrad: s.requiresGrad,
    offset: s.offset, shape: outShape, strides: outStrides, mem: s.mem)
  shapeEntry(s, opPermute, ShapeBlob(rank: int32(rank), a: toArray4(perm)), outImpl)
  outImpl.to(AnyTensorImpl)

proc sliceRange*[T](s: TensorImpl[T, bkCpu], start, count: SmallSeq[4, int32]): AnyTensorImpl =
  ## Erased range selection: axis i keeps `count[i]` elements from `start[i]`.
  ## Raises AssertionDefect when a selection is empty or reaches beyond the axis end.
  let rank = int(s.shape.len)
  var offset = s.offset
  var newShape, newStrides: array[4, int32]
  for i in 0 ..< rank:
    let dim = int(s.shape[i])
    let lo = int(start[i])
    let n = int(count[i])
    doAssert n >= 1 and lo >= 0 and lo + n <= dim,
        "slice: axis " & $i & " selection [" & $lo & ", " & $(lo + n) & ") of " & $dim &
        " elements refused"
    offset += lo * int(s.strides[i])
    newShape[i] = count[i]
    newStrides[i] = s.strides[i]
  let outShape = SmallSeq[4, int32].initFrom(newShape.toOpenArray(0, rank - 1))
  let outStrides = SmallSeq[4, int32].initFrom(newStrides.toOpenArray(0, rank - 1))
  let outImpl = TensorImpl[T, bkCpu](
    ctx: s.ctx, tid: s.ctx.newTid(), requiresGrad: s.requiresGrad,
    offset: offset, shape: outShape, strides: outStrides, storage: s.storage)
  shapeEntry(s, opSlice, ShapeBlob(rank: int32(rank), a: toArray4(start), b: toArray4(count)), outImpl)
  outImpl.to(AnyTensorImpl)

proc sliceRange*[T](s: TensorImpl[T, bkMetal], start, count: SmallSeq[4, int32]): AnyTensorImpl =
  ## Erased range selection: axis i keeps `count[i]` elements from `start[i]`.
  ## Raises AssertionDefect when a selection is empty or reaches beyond the axis end.
  let rank = int(s.shape.len)
  var offset = s.offset
  var newShape, newStrides: array[4, int32]
  for i in 0 ..< rank:
    let dim = int(s.shape[i])
    let lo = int(start[i])
    let n = int(count[i])
    doAssert n >= 1 and lo >= 0 and lo + n <= dim,
        "slice: axis " & $i & " selection [" & $lo & ", " & $(lo + n) & ") of " & $dim &
        " elements refused"
    offset += lo * int(s.strides[i])
    newShape[i] = count[i]
    newStrides[i] = s.strides[i]
  let outShape = SmallSeq[4, int32].initFrom(newShape.toOpenArray(0, rank - 1))
  let outStrides = SmallSeq[4, int32].initFrom(newStrides.toOpenArray(0, rank - 1))
  let outImpl = TensorImpl[T, bkMetal](
    ctx: s.ctx, tid: s.ctx.newTid(), requiresGrad: s.requiresGrad,
    offset: offset, shape: outShape, strides: outStrides, mem: s.mem)
  shapeEntry(s, opSlice, ShapeBlob(rank: int32(rank), a: toArray4(start), b: toArray4(count)), outImpl)
  outImpl.to(AnyTensorImpl)

proc aliasAs*[T](s: TensorImpl[T, bkCpu], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased metadata view: `s`'s storage reinterpreted over `other`'s shape and strides.
  ## The backward walk builds shape-op cotangents with it, no op records.
  let outImpl = TensorImpl[T, bkCpu](
    ctx: s.ctx, tid: s.ctx.newTid(), requiresGrad: s.requiresGrad,
    offset: s.offset, shape: other.getShape(), strides: other.getStrides(), storage: s.storage)
  outImpl.to(AnyTensorImpl)

proc aliasStrided*[T](s: TensorImpl[T, bkCpu], shape, strides: SmallSeq[4, int32]): AnyTensorImpl =
  ## Erased metadata view: `s`'s storage reinterpreted over `shape` and `strides`.
  ## The backward walk builds shape-op cotangents with it, no op records.
  let outImpl = TensorImpl[T, bkCpu](
    ctx: s.ctx, tid: s.ctx.newTid(), requiresGrad: s.requiresGrad,
    offset: s.offset, shape: shape, strides: strides, storage: s.storage)
  outImpl.to(AnyTensorImpl)

proc aliasStrided*[T](s: TensorImpl[T, bkMetal], shape, strides: SmallSeq[4, int32]): AnyTensorImpl =
  ## Erased metadata view: `s`'s device storage reinterpreted over `shape` and `strides`.
  ## The backward walk builds shape-op cotangents with it, no op records.
  let outImpl = TensorImpl[T, bkMetal](
    ctx: s.ctx, tid: s.ctx.newTid(), requiresGrad: s.requiresGrad,
    offset: s.offset, shape: shape, strides: strides, mem: s.mem)
  outImpl.to(AnyTensorImpl)

proc aliasAs*[T](s: TensorImpl[T, bkMetal], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased metadata view: `s`'s device storage reinterpreted over `other`'s shape and strides.
  ## The backward walk builds shape-op cotangents with it, no op records.
  let outImpl = TensorImpl[T, bkMetal](
    ctx: s.ctx, tid: s.ctx.newTid(), requiresGrad: s.requiresGrad,
    offset: s.offset, shape: other.getShape(), strides: other.getStrides(), mem: s.mem)
  outImpl.to(AnyTensorImpl)
