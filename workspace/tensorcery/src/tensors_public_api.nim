# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/iface
import tensors_internals
import tensors_public_types
import compute_graph/ops_2i1o/ops_add
import compute_graph/ops_2i1o/ops_matmul
import compute_graph/ops_2i1o/ops_silu_mul
import compute_graph/ops_1i1o/ops_rmsnorm
import compute_graph/ops_1iNo/ops_split
import compute_graph/ops_Ni1o/ops_embed_gather
import compute_graph/ops_Ni1o/ops_rope
import compute_graph/ops_Ni1o/ops_pool_write
import compute_graph/ops_Ni1o/ops_attention
import compute_graph/views
import tensors_globals
import std/math
import workspace/design_by_contract

export tensors_public_types
export bfloat16
export ops_attention.JaggedKvMeta

# #######################################################################
#
#                            Core Types
#
# #######################################################################

type Tensor* = ref object
  engine: AnyEngine
  node: TensorNode

# #######################################################################
#
#                            Construction
#
# #######################################################################

proc toTensor*[T: float32 | bfloat16](data: openArray[T], shape: openArray[int],
    run: static Runtime = runMetal, dtype: typedesc = T): Tensor {.contracts.} =
  {.precondition: data.len == getNumElems(shape.toSpan()).}
  {.precondition: nonNegativeDims(shape).}
  when T is dtype:
    let engine = gEngine(run)
    result = Tensor(engine: engine,
                    node: engine.newTensorNode(toDtype(T), shape.toSpan()))
    if data.len > 0:
      engine.upload(result.node, data)
  else:
    var converted: seq[dtype]
    converted.newSeq(data.len)
    for i in 0 ..< data.len:
      converted[i] = data[i].to(dtype)
    result = toTensor(converted, shape, run, dtype)

# #######################################################################
proc upload*[T: float32 | bfloat16](t: Tensor, data: openArray[T]) {.contracts.} =
  ## Writes `data` over the tensor's own storage in ascending flat order.
  ##
  ## Args:
  ## - t: a materialized tensor with its origin at the buffer base
  ## - data: one element per tensor position, in toTensor order
  let impl = t.node.getImpl()
  {.precondition: impl.offset() == 0.}
  {.precondition: impl.dtype() == toDtype(T).}
  {.precondition: data.len == getNumElems(impl.shape()).}
  t.engine.upload(t.node, data)

#
#                            Evaluation
#
# #######################################################################

proc eval*(ts: varargs[Tensor]) =
  var roots = newSeq[TensorNode](ts.len)
  for i in 0 ..< ts.len:
    roots[i] = ts[i].node
  var span = roots.toSpan()
  ts[0].engine.eval(span)
  # op nodes leave the walk materialized
  for i in 0 ..< ts.len:
    ts[i].node = roots[i]

proc shape*(t: Tensor): Span[int] =
  t.node.getImpl().shape()

proc strides*(t: Tensor): Span[int] =
  t.node.getImpl().strides()

proc dtype*(t: Tensor): Dtype =
  t.node.getImpl().dtype()

func allIdxInShape(impl: AnyTensorImpl, idx: varargs[int]): bool =
  let shape = impl.shape()
  for i in 0 ..< idx.len:
    if idx[i] < 0 or idx[i] >= shape[i]:
      return false
  true

proc item*[T: float32 | bfloat16](t: Tensor, idx: varargs[int], _: typedesc[T]): T {.contracts.} =
  let impl = t.node.getImpl()
  {.precondition: idx.len == impl.shape().len.}
  {.precondition: impl.dtype() == toDtype(T).}
  {.precondition: allIdxInShape(impl, idx).}
  var roots = newSeq[TensorNode](1)
  roots[0] = t.node
  var span = roots.toSpan()
  t.engine.eval(span)
  t.node = roots[0]
  t.engine.downloadItem(result, t.node, crd2Offset(impl, idx))

proc toSeq*[T: float32 | bfloat16](t: Tensor, _: typedesc[T]): seq[T] {.contracts.} =
  ## Reads the tensor's elements back to the host.
  ##
  ## Args:
  ## - T: the element type, must match the tensor's dtype
  ##
  ## Returns:
  ## - the elements in ascending flat index order, matching toTensor
  ##
  ## eval (like item) -> one flat DMA over the layout's reachable window -> host gather by layout
  ##
  ## Strided and offset views read back without per-element round-trips.
  let impl = t.node.getImpl()
  {.precondition: impl.dtype() == toDtype(T).}
  var roots = newSeq[TensorNode](1)
  roots[0] = t.node
  var span = roots.toSpan()
  t.engine.eval(span)
  t.node = roots[0]
  result = newSeq[T](getNumElems(impl.shape()))
  if result.len > 0:
    t.engine.gatherToHost(result, t.node)

# #######################################################################
#
#                            Host factories
#
# #######################################################################
#
# Construction-time host ops: the values compute on the host, then upload in one DMA.
# Each tensor materializes at construction like toTensor's inputs.
# Names match the MLX vocabulary: zeros, ones, full, zeros_like,
# ones_like, full_like, arange, eye and linspace. MLX's trailing stream
# parameter is not ported: the engine binds via run, or via the like-source's engine for the like-faces.

proc likeTensor[T: float32 | bfloat16](t: Tensor, value: T): Tensor =
  ## Dense tensor of t's shape and dtype, every element `value`
  let engine = t.engine
  let shape = t.shape()
  var host = newSeq[T](getNumElems(shape))
  for i in 0 ..< host.len:
    host[i] = value
  result = Tensor(engine: engine, node: engine.newTensorNode(toDtype(T), shape))
  if host.len > 0:
    engine.upload(result.node, host)

proc zeros*(shape: openArray[int], run: static Runtime = runMetal, T: typedesc[float32 | bfloat16] = bfloat16): Tensor {.contracts.} =
  ## Tensor of `shape` filled with zeros.
  ##
  ## Args:
  ## - shape: the tensor shape, rank 1 to 5
  ## - run: the engine runtime
  ## - T: the element type
  {.precondition: shape.len in 1 .. 5.}
  {.precondition: nonNegativeDims(shape).}
  var host = newSeq[T](getNumElems(shape.toSpan()))
  toTensor(host, shape, run, T)

proc ones*(shape: openArray[int], run: static Runtime = runMetal, T: typedesc[float32 | bfloat16] = bfloat16): Tensor {.contracts.} =
  ## Tensor of `shape` filled with ones.
  ##
  ## Args:
  ## - shape: the tensor shape, rank 1 to 5
  ## - run: the engine runtime
  ## - T: the element type
  {.precondition: shape.len in 1 .. 5.}
  {.precondition: nonNegativeDims(shape).}
  var host = newSeq[T](getNumElems(shape.toSpan()))
  for i in 0 ..< host.len:
    host[i] = 1'f32.to(T)
  toTensor(host, shape, run, T)

proc full*(shape: openArray[int], value: float32, run: static Runtime = runMetal,
    T: typedesc[float32 | bfloat16] = bfloat16): Tensor {.contracts.} =
  ## Tensor of `shape` filled with `value`, rounded to the element type.
  ##
  ## Args:
  ## - shape: the tensor shape, rank 1 to 5
  ## - value: the fill scalar
  ## - run: the engine runtime
  ## - T: the element type
  {.precondition: shape.len in 1 .. 5.}
  {.precondition: nonNegativeDims(shape).}
  var host = newSeq[T](getNumElems(shape.toSpan()))
  for i in 0 ..< host.len:
    host[i] = value.to(T)
  toTensor(host, shape, run, T)

proc zeros_like*(t: Tensor): Tensor =
  ## Tensor of t's shape and dtype filled with zeros, a fresh dense buffer.
  case t.dtype()
  of kFp32: likeTensor(t, 0'f32)
  of kBf16: likeTensor(t, 0'f32.to(bfloat16))
  else: raise ValueError.newException("zeros_like: dtype " & $t.dtype() & " unsupported")

proc ones_like*(t: Tensor): Tensor =
  ## Tensor of t's shape and dtype filled with ones, a fresh dense buffer.
  case t.dtype()
  of kFp32: likeTensor(t, 1'f32)
  of kBf16: likeTensor(t, 1'f32.to(bfloat16))
  else: raise ValueError.newException("ones_like: dtype " & $t.dtype() & " unsupported")

proc full_like*(t: Tensor, value: float32): Tensor =
  ## Tensor of t's shape and dtype filled with `value`,
  ## rounded to the element type, a fresh dense buffer.
  case t.dtype()
  of kFp32: likeTensor(t, value)
  of kBf16: likeTensor(t, value.to(bfloat16))
  else: raise ValueError.newException("full_like: dtype " & $t.dtype() & " unsupported")

proc empty*(shape: openArray[int], run: static Runtime = runMetal, T: typedesc[float32 | bfloat16] = bfloat16): Tensor {.contracts.} =
  ## Tensor of `shape` over fresh device storage, the values uninitialized.
  ##
  ## Args:
  ## - shape: the tensor shape, rank 1 to 5
  ## - run: the engine runtime
  ## - T: the element type
  ##
  ## The caller writes every element before the first read: the paged KV
  ## pool allocates its slabs this way, pages are written before they are
  ## read and the allocation does no zero fill.
  {.precondition: shape.len in 1 .. 5.}
  {.precondition: nonNegativeDims(shape).}
  let engine = gEngine(run)
  Tensor(engine: engine, node: engine.newTensorNode(toDtype(T), shape.toSpan()))

proc arange*(start, stop: float32, step: float32 = 1'f32,
    run: static Runtime = runMetal,
    T: typedesc[float32 | bfloat16] = bfloat16): Tensor {.contracts.} =
  ## Rank-1 tensor of start, start+step, ... up to but excluding stop.
  ##
  ## Args:
  ## - start, stop, step: the sequence bounds, step must be nonzero
  ##
  ## The element count is ceil((stop - start) / step), clamped to 0.
  {.precondition: step != 0'f32.}
  let count = max(0, ceil((stop - start) / step).int)
  var host = newSeq[T](count)
  for i in 0 ..< count:
    host[i] = (start + step * float32(i)).to(T)
  toTensor(host, @[count], run, T)

proc eye*(n: int, run: static Runtime = runMetal, T: typedesc[float32 | bfloat16] = bfloat16): Tensor {.contracts.} =
  ## Rank-2 (n, n) identity: ones on the diagonal, zeros elsewhere.
  {.precondition: n >= 0.}
  var host = newSeq[T](n * n)
  for i in 0 ..< n:
    host[i * n + i] = 1'f32.to(T)
  toTensor(host, @[n, n], run, T)

proc linspace*(start, stop: float32, num: int,
    run: static Runtime = runMetal,
    T: typedesc[float32 | bfloat16] = bfloat16): Tensor {.contracts.} =
  ## Rank-1 tensor of num evenly spaced values from start to stop,
  ## both endpoints included.
  ##
  ## Args:
  ## - start, stop: the interval bounds, both included
  ## - num: the element count, at least 2
  ##
  ## The last element is the stop exactly. The others are start + i*step
  ## with step = (stop - start) / (num - 1).
  {.precondition: num >= 2.}
  var host = newSeq[T](num)
  let step = (stop - start) / float32(num - 1)
  for i in 0 ..< num - 1:
    host[i] = (start + step * float32(i)).to(T)
  host[num - 1] = stop.to(T)
  toTensor(host, @[num], run, T)

# #######################################################################
#
#                            Ops
#
# #######################################################################

proc `+`*(a, b: Tensor): Tensor =
  Tensor(engine: a.engine, node: tn_add(a.engine, a.node, b.node))

proc split*(t: Tensor, axis: int, k: int): seq[Tensor] =
  for n in t.engine.tn_split(t.node, axis, k):
    result.add(Tensor(engine: t.engine, node: n))

proc view*(t: Tensor, shape, strides: openArray[int], offset: int): Tensor =
  ## Layout view of `t` over the same buffer: any rank, any strides.
  ##
  ## Args:
  ## - shape, strides: the view layout, in elements
  ## - offset: the view origin's element offset from the tensor's origin,
  ##   including the tensor's own offset
  ##
  ## The general case of narrow, transpose, permute, reshape and expand.
  ## The view builds shape, strides and offset over one shared buffer,
  ## no allocation and no copy.
  ##
  ## A stride-0 axis is allowed here and on device: the serializer packs
  ## the static shape leaves, so the kernel reads the true layout.
  Tensor(engine: t.engine, node: tn_view(t.engine, t.node, shape, strides, offset))

proc reshape*(t: Tensor, shape: openArray[int]): Tensor =
  ## View of `t` with a new shape, same element order.
  ##
  ## Args:
  ## - shape: the new shape, its element count must match the source's
  ##
  ## Requires a dense contiguous source: the new layout is contiguous over
  ## the same buffer window, and a strided source would need a copy op.
  ## The tensor's offset is part of the new view, which allocates nothing.
  Tensor(engine: t.engine, node: tn_reshape(t.engine, t.node, shape))

proc transpose*(t: Tensor): Tensor =
  ## View of `t` with every axis reversed, matching MLX transpose with no axes argument.
  ##
  ## Returns:
  ## - a view of `t` over the same buffer with the tensor's offset, no allocation, no copy
  Tensor(engine: t.engine, node: tn_transpose(t.engine, t.node))

proc permute*(t: Tensor, axes: openArray[int]): Tensor =
  ## View of `t` with the axes reordered: result axis i reads source axis axes[i].
  ##
  ## Args:
  ## - axes: the source axis each result axis reads, one per source axis
  ##
  ## Returns:
  ## - a view of `t` over the same buffer with the tensor's offset, no allocation, no copy
  Tensor(engine: t.engine, node: tn_permute(t.engine, t.node, axes))

proc narrow*(t: Tensor, axis, start, width: int): Tensor =
  ## Window view of `t` with the other axes unchanged.
  ##
  ## Args:
  ## - axis: the narrowed dimension
  ## - start, width: the window inside that dimension
  ##
  ## The element offset shifts by start·stride[axis] down the narrowed axis.
  ## A narrowed tensor keeps its offset under further views and splits.
  ## The view allocates nothing.
  Tensor(engine: t.engine, node: tn_narrow(t.engine, t.node, axis, start, width))

proc expand*(t: Tensor, shape: openArray[int]): Tensor =
  ## Broadcast view of `t`: a size-1 axis stretches to any target size at stride 0, and every other axis keeps its size and stride.
  ##
  ## Args:
  ## - shape: the target shape, one target axis per source axis
  ##
  ## A stride-0 axis reads correctly everywhere: on the host (item, toSeq),
  ## and ops consume the broadcast view per position on the device.
  Tensor(engine: t.engine, node: tn_expand(t.engine, t.node, shape))

proc linear*(x, w: Tensor): Tensor =
  ## y = x·wᵀ: the Linear projection,
  ## (tokens, inFeatures) × (outFeatures, inFeatures)ᵀ → (tokens, outFeatures).
  ##
  ## Args:
  ## - x: the input tensor of shape (tokens, inFeatures)
  ## - w: the weight tensor of shape (outFeatures, inFeatures), stored so
  ##   that the output's n-th column reads w's n-th weight vector
  ##
  ## Returns:
  ## - a tensor of shape (tokens, outFeatures) with the operands' dtype:
  ##   bfloat16 operands accumulate in float32, then round once into the bfloat16 output
  Tensor(engine: x.engine, node: tn_matmul(x.engine, x.node, w.node))

proc silu_mul*(gate, up: Tensor): Tensor =
  ## out = silu(gate)·up elementwise: the gated FFN activation.
  ##
  ## Args:
  ## - gate: the operand feeding silu (x·sigmoid(x)), shape (tokens, cols)
  ## - up: the multiplier operand, shape (tokens, cols)
  ##
  ## Returns:
  ## - a tensor of the operands' shape and dtype: the math runs in float32
  ##   with one rounding at the store
  Tensor(engine: gate.engine, node: tn_silu_mul(gate.engine, gate.node, up.node))

proc rmsnorm*(x, w: Tensor, eps: float32 = 1e-6'f32): Tensor =
  ## y = x·rsqrt(mean(x²) + eps)·w over the last axis: the RmsNorm layer.
  ##
  ## Args:
  ## - x: the input tensor of shape (tokens, width)
  ## - w: the weight tensor of shape (width,), one slot per width position
  ## - eps: the stability constant added to the mean of squares,
  ##   the Qwen3 reference default
  ##
  ## Returns:
  ## - a tensor of x's shape and dtype: the math runs in float32 with one
  ##   rounding at the store
  Tensor(engine: x.engine, node: tn_rmsnorm(x.engine, x.node, w.node, eps))

proc rope*(q, cos, sin: Tensor): Tensor =
  ## Applies the NEOX RoPE rotation over q's last axis.
  ##
  ## Args:
  ## - q: the input tensor of shape (tokens, heads, dim) in bfloat16,
  ##   last axis even
  ## - cos, sin: the half-table tensors, shape (tokens, dim div 2), f32.
  ##   Entry (t, i) pairs with the columns (i, i + dim div 2) of token t.
  ##   The table entries do not depend on the head
  ##
  ## Returns:
  ## - a tensor of q's shape and dtype. The rotation runs in f32 with one
  ##   rounding at the store
  Tensor(engine: q.engine, node: tn_rope(q.engine, q.node, cos.node, sin.node))

proc embed_gather*(w: Tensor, inputIds: openArray[int]): Tensor =
  ## Gathers one weight entry per token id.
  ##
  ## Args:
  ## - w: the weight tensor, shape (vocab, hidden)
  ## - inputIds: token ids in [0, vocab), each id selects one weight entry
  ##
  ## Returns:
  ## - a tensor of shape (numTokens, hidden), token t's vector copied from
  ##   the weight entry ids[t]
  Tensor(engine: w.engine, node: tn_embed_gather(w.engine, w.node, inputIds))

proc pool_write*(poolK, poolV, kSrc, vSrc: Tensor, layer: int, offset, seqLen, cachedTokens: int, pageSlots: openArray[int]): Tensor =
  ## Writes post-rope K/V entries into pool pages, in graph.
  ##
  ## Args:
  ## - poolK, poolV: (num_pages, num_layers, PAGE_SIZE, kv_heads, head_dim) pool slabs
  ##   `layer` selects the slab layer the write targets
  ## - kSrc, vSrc: (1, seq_len, kv_heads, head_dim) post-rope entries
  ## - offset: the global token position of src entry 0
  ## - seqLen: entries to write, at least one
  ##
  ## Write range:
  ##
  ## - cachedTokens: entries already cached from copy-on-write forks
  ## - positions before max(0, cachedTokens - offset) stay untouched
  ##
  ## Page table:
  ##
  ## - pageSlots: the pool slot of each sequence page, in page order
  ##
  ## Returns: a tensor aliasing the K pool slab
  ##
  ## - eval enqueues the write before any node reading the pool
  ## - one launch writes both pools
  ## - reading the result reads the written pool
  Tensor(engine: poolK.engine, node: tn_pool_write(poolK.engine, poolK.node,
    poolV.node, kSrc.node, vSrc.node, layer, offset, seqLen, cachedTokens, pageSlots))

proc jagged_attention*(q, poolK, poolV: Tensor, qOffsets: openArray[int],
    kv: JaggedKvMeta, scale: float32, layer, H, Nkv: int,
    windowSize: int = 0): Tensor =
  ## Runs jagged attention over the page pool slabs, in graph.
  ##
  ## Args:
  ## - q: (q_total, H, D) post-rope queries in bfloat16
  ## - poolK, poolV: (num_pages, num_layers, page_size, Nkv, D) pool slabs
  ##   in bfloat16, `layer` selects the slab layer the attention reads
  ## - qOffsets, kv: segment q starts (batchSize + 1 entries) and the jagged KV metadata
  ## - scale, H, Nkv: the softmax scale and the head counts, H a multiple of Nkv
  ##
  ## Returns:
  ## - a tensor of shape (q_total, H, D) in bfloat16, fully overwritten.
  ##   The op instantiates the Qwen3 statics (page_size 256, head dim 128)
  ##   and refuses any other geometry. windowSize must be <= 0. A windowed dispatch
  ##   needs its own op instantiation
  Tensor(engine: q.engine, node: tn_jagged_attention(q.engine, q.node,
    poolK.node, poolV.node, qOffsets, kv, scale, layer, H, Nkv, windowSize))
