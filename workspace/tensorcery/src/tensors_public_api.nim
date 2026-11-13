# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import std/[macros, macrocache]
import workspace/crucible
import workspace/iface
import workspace/design_by_contract
import ./tensors_internals
import ./tensors_public_types
import ./tensors_globals
import ./compute_graph/ops_2i1o/ops_add
import ./compute_graph/ops_1iNo/ops_split
import ./compute_graph/ops_metadata/ops_swapaxes
import ./compute_graph/ops_metadata/ops_views
import ./compute_graph/ops_0i1o/ops_zeros
import ./compute_graph/ops_0i1o/ops_ones
import ./compute_graph/ops_0i1o/ops_full
import ./compute_graph/ops_0i1o/ops_eye
import ./compute_graph/ops_0i1o/ops_arange
import ./compute_graph/ops_reductions/ops_allClose
import ./compute_graph/ops_neuralnets/ops_rmsnorm
import ./compute_graph/ops_neuralnets/ops_linear
import ./compute_graph/ops_blas/ops_gemm
import ./compute_graph/eval

export tensors_public_types
export bfloat16

# Export internal symbols to workaround visibility limitations
export tensors_internal_types
export tensors_internals
export eval

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

func product(a: openArray[int]): int {.inline.} =
  result = 1
  for dim in a:
    result *= dim

proc toTensor*[T](data: openArray[T], shape: openArray[int], run: static Runtime, dtype: typedesc = T): Tensor {.contracts.} =
  {.precondition: data.len == product(shape).}
  when T is dtype:
    let engine = gEngine(run)
    result = Tensor(engine: engine,
                    node: engine.newTensorNode(toDtype(T), shape))
    if data.len > 0:
      engine.upload(result.node, data)
  else:
    var converted: seq[dtype]
    converted.newSeq(data.len)
    for i in 0 ..< data.len:
      converted[i] = data[i].to(dtype)
    result = converted.toTensor(shape, run, dtype)

proc toTensor*[T](data: openArray[T], run: static Runtime, shape: varargs[int]): Tensor {.inline.} =
  data.toTensor(shape, run, T)

# #######################################################################
#
#                            Factories
#
# #######################################################################

proc zeros*(shape: openArray[int], run: static Runtime, dtype: Dtype = kFp32): Tensor {.inline, hasKernel: "Zeros".} =
  Tensor(engine: gEngine(run), node: gEngine(run).tn_zeros(dtype, shape))

proc ones*(shape: openArray[int], run: static Runtime, dtype: Dtype = kFp32): Tensor {.inline, hasKernel: "Ones".} =
  Tensor(engine: gEngine(run), node: gEngine(run).tn_ones(dtype, shape))

proc full*(shape: openArray[int], vals: float32, run: static Runtime, dtype: Dtype = kFp32): Tensor {.inline, hasKernel: "Full".} =
  Tensor(engine: gEngine(run), node: gEngine(run).tn_full(vals, dtype, shape))

proc eye*(n: int, run: static Runtime, dtype: Dtype = kFp32, k: int = 0): Tensor {.inline, hasKernel: "Eye".} =
  let engine = gEngine(run)
  Tensor(engine: engine, node: engine.tn_reshape(engine.tn_eye(n, n, k, dtype), [n, n]))

proc eye*(n, m: int, run: static Runtime, dtype: Dtype = kFp32, k: int = 0): Tensor {.inline, hasKernel: "Eye".} =
  let engine = gEngine(run)
  Tensor(engine: engine, node: engine.tn_reshape(engine.tn_eye(n, m, k, dtype), [n, m]))

proc arange*(stop: float32, run: static Runtime, dtype: Dtype = kFp32): Tensor {.inline, hasKernel: "Arange".} =
  let engine = gEngine(run)
  Tensor(engine: engine, node: engine.tn_arange(0.0'f32, stop, 1.0'f32, dtype))

proc arange*(start, stop: float32, run: static Runtime, step = 1.0'f32, dtype: Dtype = kFp32): Tensor {.inline, hasKernel: "Arange".} =
  let engine = gEngine(run)
  Tensor(engine: engine, node: engine.tn_arange(start, stop, step, dtype))

proc zeros_like*(a: Tensor): Tensor {.inline, hasKernel: "Zeros".} =
  let engine = a.engine
  let impl = a.node.getImpl()
  Tensor(engine: engine, node: engine.tn_zeros(impl.dtype(), impl.shape().p))

proc zeros_like*(a: Tensor, dtype: Dtype): Tensor {.inline, hasKernel: "Zeros".} =
  let engine = a.engine
  let impl = a.node.getImpl()
  Tensor(engine: engine, node: engine.tn_zeros(dtype, impl.shape().p))

proc ones_like*(a: Tensor): Tensor {.inline, hasKernel: "Ones".} =
  let engine = a.engine
  let impl = a.node.getImpl()
  Tensor(engine: engine, node: engine.tn_ones(impl.dtype(), impl.shape().p))

proc ones_like*(a: Tensor, dtype: Dtype): Tensor {.inline, hasKernel: "Ones".} =
  let engine = a.engine
  let impl = a.node.getImpl()
  Tensor(engine: engine, node: engine.tn_ones(dtype, impl.shape().p))

proc full_like*(a: Tensor, vals: float32): Tensor {.inline, hasKernel: "Full".} =
  let engine = a.engine
  let impl = a.node.getImpl()
  Tensor(engine: engine, node: engine.tn_full(vals, impl.dtype(), impl.shape().p))

proc full_like*(a: Tensor, vals: float32, dtype: Dtype): Tensor {.inline, hasKernel: "Full".} =
  let engine = a.engine
  let impl = a.node.getImpl()
  Tensor(engine: engine, node: engine.tn_full(vals, dtype, impl.shape().p))

# #######################################################################
#
#                              Metadata
#
# #######################################################################

template shape*(t: Tensor): openArray[int] =
  t.node.getImpl().shape().p

template strides*(t: Tensor): openArray[int] =
  t.node.getImpl().strides().p

proc dtype*(t: Tensor): Dtype {.inline.} =
  t.node.getImpl().dtype()

proc size*(t: Tensor): int {.inline.} =
  t.node.size()

template item*[T](t: Tensor, _: typedesc[T]): T =
  ## Downloads the single element of `t` as `T`, evaluating its graph.
  ##
  ## - a template: the dispatch expands per eval call site, so ops marked
  ##   after an earlier eval still dispatch here
  ## - `t` is bound once: a template re-evaluates its argument at every use
  ##   and a temporary argument would read back a never-evaluated node
  let self = t
  let impl = self.node.getImpl()
  precondition(impl.dtype() == toDtype(T))
  var node = [self.node]
  eval(ctgSoleBackend(), node)
  self.node = node[0]
  var read: T
  self.engine.download1(addr read, self.node)
  read

# #######################################################################
#
#                            Evaluation
#
# #######################################################################

template eval*(ts: varargs[Tensor]) =
  ## Enqueues and submits the graph rooted at the tensors' nodes.
  ##
  ## - backend: read from ctgBackendSelection at the call site, construction
  ##   sites recorded it there
  var roots = newSeq[TensorNode](ts.len)
  for i in 0 ..< ts.len:
    roots[i] = ts[i].node
  eval(ctgSoleBackend(), roots)
  for i in 0 ..< ts.len:
    ts[i].node = roots[i]

# #######################################################################
#
#                       Arithmetic Operators
#
# #######################################################################

# Binary (Tensor + Tensor)
# ----------------------------------------------------------

proc `+`*(a, b: Tensor): Tensor {.inline, hasKernel: "Add".} =
  Tensor(engine: a.engine, node: tn_add(a.engine, a.node, b.node))

# #######################################################################
#
#                          Linear Algebra
#
# #######################################################################

proc gemm*(alpha = 1.0'f32, a, b: Tensor, beta = 0.0'f32, c: Tensor = nil): Tensor {.inline, hasKernel: "Gemm".} =
  ## D = alpha·A·B + beta·C over the last two axes, C nil requires beta = 0
  let cn = if c.isNil(): NilTensorNode() else: c.node
  Tensor(engine: a.engine, node: tn_gemm(a.engine, alpha, a.node, b.node, beta, cn))

proc matmul*(a, b: Tensor): Tensor {.inline, hasKernel: "Gemm".} =
  ## A·B over the last two axes
  gemmImpl(1.0'f32, a, b, 0.0'f32)

# #######################################################################
#
#                         Views / Shapeshifting
#
# #######################################################################

proc swapaxes*(t: Tensor, axis1, axis2: int): Tensor {.inline.} =
  Tensor(engine: t.engine, node: tn_swapaxes(t.engine, t.node, axis1, axis2))

proc reshape*(t: Tensor, shape: openArray[int]): Tensor {.inline.} =
  Tensor(engine: t.engine, node: tn_reshape(t.engine, t.node, shape))

proc transpose*(t: Tensor): Tensor {.inline.} =
  Tensor(engine: t.engine, node: tn_transpose(t.engine, t.node))

proc transpose*(t: Tensor, axes: openArray[int]): Tensor {.inline.} =
  Tensor(engine: t.engine, node: tn_transpose(t.engine, t.node, axes))

proc broadcast_to*(t: Tensor, shape: openArray[int]): Tensor {.inline.} =
  Tensor(engine: t.engine, node: tn_broadcastTo(t.engine, t.node, shape))

proc squeeze*(t: Tensor, axis: int): Tensor {.inline.} =
  Tensor(engine: t.engine, node: tn_squeeze(t.engine, t.node, axis))

proc expand_dims*(t: Tensor, axis: int): Tensor {.inline.} =
  Tensor(engine: t.engine, node: tn_expandDims(t.engine, t.node, axis))

proc narrow*(t: Tensor, axis, start, width: int): Tensor {.inline.} =
  Tensor(engine: t.engine, node: tn_narrow(t.engine, t.node, axis, start, width))

proc split*(t: Tensor, axis: int, k: int): seq[Tensor] {.inline.} =
  ## Splits `t` along `axis` into `k` equal parts, views into `t`'s buffer.
  var parts: seq[Tensor]
  for n in t.engine.tn_split(t.node, axis, k):
    parts.add(Tensor(engine: t.engine, node: n))
  parts

# #######################################################################
#
#                            Comparison
#
# #######################################################################

macro allClose*(tested, reference: Tensor, rtol = 1e-5'f32, atol = 1e-8'f32, equal_nan = false): bool =
  ## True when every element satisfies |tested - reference| <= atol + rtol * |reference|.
  ## Same-sign infinities are equal, NaN equal only with equal_nan
  ##
  ## - a macro, not a hasKernel forward: the eval inside `item` must expand
  ##   per call site, an impl proc body sem-checks once at definition
  CacheCounter("allClose").inc()
  quote do:
    Tensor(engine: `tested`.engine,
        node: tn_allClose(`tested`.engine, `tested`.node, `reference`.node, `atol`, `rtol`, `equal_nan`)
      ).item(float32) == 0.0'f32

# #######################################################################
#
#                     Neural Network Functional API
#
# #######################################################################

# Normalization
# ----------------------------------------------------------

proc rms_norm*(x: Tensor, weight: Tensor = nil, eps = 1e-5'f32): Tensor {.hasKernel: "RmsNorm".} =
  let w = if weight.isNil(): NilTensorNode() else: weight.node
  Tensor(engine: x.engine, node: tn_rmsnorm(x.engine, x.node, w, eps))

# Layers
# ----------------------------------------------------------

proc linear*(x, weight: Tensor, bias: Tensor = nil): Tensor {.hasKernel: "Linear".} =
  ## o = x·Wᵀ + b: (M, K) × (out, K) → (M, out), bias nil when the layer has none
  let bn = if bias.isNil(): NilTensorNode() else: bias.node
  Tensor(engine: x.engine, node: tn_linear(x.engine, x.node, weight.node, bn))
