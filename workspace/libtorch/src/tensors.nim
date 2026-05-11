# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/macros,
  # Internal
  workspace/libtorch/src/raw_libtorch as F,
  workspace/libtorch/src/raw/torch_tensors_sugar,
  workspace/libtorch/src/raw/support/[ast_utils, indexing_macros],
  workspace/libtorch/src/vecs/vecs

# Export Nim-friendly types (no C++ types leak past this boundary)
export F.ScalarKind, F.DeviceKind, F.Device, F.TensorOptions,
       F.Scalar, F.SomeTorchType, F.TorchComplex
# Indexing sugar
export F.`_`, F.ellipsis, F.`...`

# #######################################################################
#
#                            Core Type
#
# #######################################################################

type Tensor* = ref object
  raw: TorchTensor

# Construction helpers

proc placementNew[T](p: ptr T): ptr T {.importcpp: "(new (#) '*0(@))", nodecl, discardable.}
  ## Default-construct an object at the given memory location via placement-new.

proc wrapTorchTensor(a: sink TorchTensor): Tensor {.inline.} =
  new result
  discard placementNew(result.raw.addr)
  `=sink`(result.raw, a)

# #######################################################################
#
#                          Array → Tensor
#
# #######################################################################

func toTensor*[T: SomeTorchType](oa: openarray[T]): Tensor {.inline.} =
  ## Convert a flat Nim array/seq to a 1D Tensor (owning copy).
  wrapTorchTensor(toTorchTensor(oa))

func toTensor*[T: seq | array](oa: openarray[T]): Tensor {.inline.} =
  ## Convert a nested Nim array/seq to a multi-dimensional Tensor (owning copy).
  wrapTorchTensor(toTorchTensor(oa))

# #######################################################################
#
#                              Factory
#
# #######################################################################

# empty

func empty*(size: varargs[int]): Tensor {.inline.} =
  wrapTorchTensor(F.empty(asTorchView(size)))

func empty*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.empty(asTorchView(size), toScalarKind(T)))

func empty*(size: varargs[int], scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.empty(asTorchView(size), scalarKind))

func empty*(size: varargs[int], device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.empty(asTorchView(size), device))

func empty*(size: varargs[int], options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.empty(asTorchView(size), options))

# zeros

func zeros*(size: varargs[int]): Tensor {.inline.} =
  wrapTorchTensor(F.zeros(asTorchView(size)))

func zeros*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.zeros(asTorchView(size), toScalarKind(T)))

func zeros*(size: varargs[int], scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.zeros(asTorchView(size), scalarKind))

func zeros*(size: varargs[int], device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.zeros(asTorchView(size), device))

func zeros*(size: varargs[int], options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.zeros(asTorchView(size), options))

# ones

func ones*(size: varargs[int]): Tensor {.inline.} =
  wrapTorchTensor(F.ones(asTorchView(size)))

func ones*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.ones(asTorchView(size), toScalarKind(T)))

func ones*(size: varargs[int], scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.ones(asTorchView(size), scalarKind))

func ones*(size: varargs[int], device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.ones(asTorchView(size), device))

func ones*(size: varargs[int], options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.ones(asTorchView(size), options))

# full

func full*(size: varargs[int], fillValue: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.full(asTorchView(size), fillValue))

func full*(size: varargs[int], fillValue: Scalar, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.full(asTorchView(size), fillValue, toScalarKind(T)))

func full*(size: varargs[int], fillValue: Scalar, scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.full(asTorchView(size), fillValue, scalarKind))

func full*(size: varargs[int], fillValue: Scalar, device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.full(asTorchView(size), fillValue, device))

func full*(size: varargs[int], fillValue: Scalar, options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.full(asTorchView(size), fillValue, options))

# rand

func rand*(size: varargs[int]): Tensor {.inline.} =
  wrapTorchTensor(F.rand(asTorchView(size)))

func rand*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.rand(asTorchView(size), toScalarKind(T)))

func rand*(size: varargs[int], scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.rand(asTorchView(size), scalarKind))

func rand*(size: varargs[int], device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.rand(asTorchView(size), device))

func rand*(size: varargs[int], options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.rand(asTorchView(size), options))

# randn

func randn*(size: varargs[int]): Tensor {.inline.} =
  wrapTorchTensor(F.randn(asTorchView(size)))

func randn*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.randn(asTorchView(size), toScalarKind(T)))

func randn*(size: varargs[int], scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.randn(asTorchView(size), scalarKind))

func randn*(size: varargs[int], device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.randn(asTorchView(size), device))

func randn*(size: varargs[int], options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.randn(asTorchView(size), options))

# rand_like

func rand_like*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.rand_like(a.raw))

func rand_like*(a: Tensor, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.rand_like(a.raw, toScalarKind(T)))

func rand_like*(a: Tensor, scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.rand_like(a.raw, scalarKind))

func rand_like*(a: Tensor, device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.rand_like(a.raw, device))

func rand_like*(a: Tensor, options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.rand_like(a.raw, options))

# eye

func eye*(n: int): Tensor {.inline.} =
  wrapTorchTensor(F.eye(n))

func eye*(n: int, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.eye(n, toScalarKind(T)))

func eye*(n: int, scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.eye(n, scalarKind))

func eye*(n: int, device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.eye(n, device))

func eye*(n: int, options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.eye(n, options))

# arange

func arange*(stop: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.arange(stop))

func arange*(stop: Scalar, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.arange(stop, toScalarKind(T)))

func arange*(stop: Scalar, scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.arange(stop, scalarKind))

func arange*(stop: Scalar, device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.arange(stop, device))

func arange*(stop: Scalar, options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.arange(stop, options))

func arange*(start, stop: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.arange(start, stop))

func arange*(start, stop: Scalar, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.arange(start, stop, toScalarKind(T)))

func arange*(start, stop: Scalar, scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.arange(start, stop, scalarKind))

func arange*(start, stop: Scalar, device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.arange(start, stop, device))

func arange*(start, stop: Scalar, options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.arange(start, stop, options))

func arange*(start, stop, step: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.arange(start, stop, step))

func arange*(start, stop, step: Scalar, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.arange(start, stop, step, toScalarKind(T)))

func arange*(start, stop, step: Scalar, scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.arange(start, stop, step, scalarKind))

func arange*(start, stop, step: Scalar, device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.arange(start, stop, step, device))

func arange*(start, stop, step: Scalar, options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.arange(start, stop, step, options))

# linspace

func linspace*(start, stop: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.linspace(start, stop))

func linspace*(start, stop: Scalar, steps: int): Tensor {.inline.} =
  wrapTorchTensor(F.linspace(start, stop, steps))

func linspace*(start, stop: Scalar, steps: int, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.linspace(start, stop, steps, toScalarKind(T)))

func linspace*(start, stop: Scalar, steps: int, scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.linspace(start, stop, steps, scalarKind))

func linspace*(start, stop: Scalar, steps: int, device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.linspace(start, stop, steps, device))

func linspace*(start, stop: Scalar, steps: int, options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.linspace(start, stop, steps, options))

# logspace

func logspace*(start, stop: Scalar, steps: int): Tensor {.inline.} =
  wrapTorchTensor(F.logspace(start, stop, steps))

func logspace*(start, stop: Scalar, steps: int, base: int): Tensor {.inline.} =
  wrapTorchTensor(F.logspace(start, stop, steps, base))

func logspace*(start, stop: Scalar, steps: int, base: int, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.logspace(start, stop, steps, base, toScalarKind(T)))

func logspace*(start, stop: Scalar, steps: int, base: int, scalarKind: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.logspace(start, stop, steps, base, scalarKind))

func logspace*(start, stop: Scalar, steps: int, base: int, device: DeviceKind): Tensor {.inline.} =
  wrapTorchTensor(F.logspace(start, stop, steps, base, device))

func logspace*(start, stop: Scalar, steps: int, base: int, options: TensorOptions): Tensor {.inline.} =
  wrapTorchTensor(F.logspace(start, stop, steps, base, options))

# randint

func randint*(start, stopEx: int, size: varargs[int]): Tensor {.inline.} =
  wrapTorchTensor(F.randint(start, stopEx, asTorchView(size)))

func randint*(start, stopEx: int, size: varargs[int], T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.randint(start, stopEx, asTorchView(size), toScalarKind(T)))

# from_blob

func from_blob*(data: pointer, sizes: varargs[int], T: typedesc[SomeTorchType]): Tensor {.inline.} =
  ## Create a non-owning Tensor view from a data pointer.
  ## ⚠ `data` MUST remain valid for the lifetime of this Tensor.
  wrapTorchTensor(F.from_blob(data, asTorchView(sizes), toScalarKind(T)))

func from_blob*(data: pointer, sizes: varargs[int], scalarKind: ScalarKind): Tensor {.inline.} =
  ## Create a non-owning Tensor view from a data pointer.
  ## ⚠ `data` MUST remain valid for the lifetime of this Tensor.
  wrapTorchTensor(F.from_blob(data, asTorchView(sizes), scalarKind))

func from_blob*(data: pointer, sizes: varargs[int], device: DeviceKind): Tensor {.inline.} =
  ## Create a non-owning Tensor view from a data pointer.
  ## ⚠ `data` MUST remain valid for the lifetime of this Tensor.
  wrapTorchTensor(F.from_blob(data, asTorchView(sizes), device))

func from_blob*(data: pointer, sizes: varargs[int], options: TensorOptions): Tensor {.inline.} =
  ## Create a non-owning Tensor view from a data pointer.
  ## ⚠ `data` MUST remain valid for the lifetime of this Tensor.
  wrapTorchTensor(F.from_blob(data, asTorchView(sizes), options))

func from_blob*(data: pointer, sizes, strides: varargs[int], T: typedesc[SomeTorchType]): Tensor {.inline.} =
  ## Create a non-owning Tensor view from a data pointer with explicit strides.
  ## ⚠ `data` MUST remain valid for the lifetime of this Tensor.
  wrapTorchTensor(F.from_blob(data, asTorchView(sizes), asTorchView(strides), toScalarKind(T)))

func from_blob*(data: pointer, sizes, strides: varargs[int], scalarKind: ScalarKind): Tensor {.inline.} =
  ## Create a non-owning Tensor view from a data pointer with explicit strides.
  ## ⚠ `data` MUST remain valid for the lifetime of this Tensor.
  wrapTorchTensor(F.from_blob(data, asTorchView(sizes), asTorchView(strides), scalarKind))

func from_blob*(data: pointer, sizes, strides: varargs[int], device: DeviceKind): Tensor {.inline.} =
  ## Create a non-owning Tensor view from a data pointer with explicit strides.
  ## ⚠ `data` MUST remain valid for the lifetime of this Tensor.
  wrapTorchTensor(F.from_blob(data, asTorchView(sizes), asTorchView(strides), device))

func from_blob*(data: pointer, sizes, strides: varargs[int], options: TensorOptions): Tensor {.inline.} =
  ## Create a non-owning Tensor view from a data pointer with explicit strides.
  ## ⚠ `data` MUST remain valid for the lifetime of this Tensor.
  wrapTorchTensor(F.from_blob(data, asTorchView(sizes), asTorchView(strides), options))

# clone

func clone*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.clone(a.raw))

# #######################################################################
#
#                         Methods / Shapeshifting
#
# #######################################################################

# Shape manipulation

func reshape*(a: Tensor, size: varargs[int]): Tensor {.inline.} =
  wrapTorchTensor(F.reshape(a.raw, asTorchView(size)))

func view*(a: Tensor, size: varargs[int]): Tensor {.inline.} =
  wrapTorchTensor(F.view(a.raw, asTorchView(size)))

func permute*(a: Tensor, dims: varargs[int]): Tensor {.inline.} =
  wrapTorchTensor(F.permute(a.raw, asTorchView(dims)))

func expand*(a: Tensor, size: varargs[int], implicit: bool = false): Tensor {.inline.} =
  wrapTorchTensor(F.expand(a.raw, asTorchView(size), implicit))

func transpose*(a: Tensor, dim0, dim1: int64): Tensor {.inline.} =
  wrapTorchTensor(F.transpose(a.raw, dim0, dim1))

func t*(a: Tensor): Tensor {.inline.} =
  ## Transposes a 2D tensor. Equivalent to ``transpose(0, 1)``.
  wrapTorchTensor(F.t(a.raw))

func repeat_interleave*(a: Tensor, repeats: int, dim: int = -1): Tensor {.inline.} =
  wrapTorchTensor(F.repeat_interleave(a.raw, repeats, dim))

func narrow*(a: Tensor, dim: int, start: int, length: int): Tensor {.inline.} =
  wrapTorchTensor(F.narrow(a.raw, dim, start, length))

func flip*(a: Tensor, dims: varargs[int]): Tensor {.inline.} =
  wrapTorchTensor(F.flip(a.raw, asTorchView(dims)))

func squeeze*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.squeeze(a.raw))

func squeeze*(a: Tensor, axis: int): Tensor {.inline.} =
  wrapTorchTensor(F.squeeze(a.raw, axis))

func unsqueeze*(a: Tensor, axis: int): Tensor {.inline.} =
  wrapTorchTensor(F.unsqueeze(a.raw, axis))

# Backend / dtype

func to*(a: Tensor, device: DeviceKind, non_blocking: bool = false, copy: bool = false): Tensor {.inline.} =
  wrapTorchTensor(F.to(a.raw, device, non_blocking, copy))

func to*(a: Tensor, dtype: ScalarKind): Tensor {.inline.} =
  wrapTorchTensor(F.to(a.raw, dtype))

func to*(a: Tensor, device: DeviceKind, dtype: ScalarKind, non_blocking: bool = false, copy: bool = false): Tensor {.inline.} =
  wrapTorchTensor(F.to(a.raw, device, dtype, non_blocking, copy))

func to*(a: Tensor, device: Device, non_blocking: bool = false, copy: bool = false): Tensor {.inline.} =
  wrapTorchTensor(F.to(a.raw, device, non_blocking, copy))

func contiguous*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.contiguous(a.raw))

func toSparse*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.toSparse(a.raw))

func toSparse*(a: Tensor, sparseDim: int): Tensor {.inline.} =
  wrapTorchTensor(F.toSparse(a.raw, sparseDim))

# Complex

func view_as_real*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.view_as_real(a.raw))

func view_as_complex*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.view_as_complex(a.raw))

# Device transfers

func cpu*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.cpu(a.raw))

func cuda*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.cuda(a.raw))

func hip*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.hip(a.raw))

func vulkan*(a: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.vulkan(a.raw))

# #######################################################################
#
#                            Metadata
#
# #######################################################################

func dim*(a: Tensor): int {.inline.} =
  F.dim(a.raw)

func sizes*(a: Tensor): IntArrayRef {.inline.} =
  F.sizes(a.raw)

func shape*(a: Tensor): IntArrayRef {.inline.} =
  F.shape(a.raw)

func strides*(a: Tensor): IntArrayRef {.inline.} =
  F.strides(a.raw)

func ndimension*(a: Tensor): int {.inline.} =
  F.ndimension(a.raw)

func nbytes*(a: Tensor): uint {.inline.} =
  F.nbytes(a.raw)

func numel*(a: Tensor): int {.inline.} =
  F.numel(a.raw)

func size*(a: Tensor, axis: int): int {.inline.} =
  F.size(a.raw, axis)

func itemsize*(a: Tensor): uint {.inline.} =
  F.itemsize(a.raw)

func element_size*(a: Tensor): int {.inline.} =
  F.element_size(a.raw)

func scalarType*(a: Tensor): ScalarKind {.inline.} =
  F.scalarType(a.raw)

func get_device*(a: Tensor): int {.inline.} =
  F.get_device(a.raw)

func isDefined*(a: Tensor): bool {.inline.} =
  F.isDefined(a.raw)

# Backend checks

func is_cuda*(a: Tensor): bool {.inline.} =
  F.is_cuda(a.raw)

func is_hip*(a: Tensor): bool {.inline.} =
  F.is_hip(a.raw)

func is_sparse*(a: Tensor): bool {.inline.} =
  F.is_sparse(a.raw)

func is_mkldnn*(a: Tensor): bool {.inline.} =
  F.is_mkldnn(a.raw)

func is_vulkan*(a: Tensor): bool {.inline.} =
  F.is_vulkan(a.raw)

func is_quantized*(a: Tensor): bool {.inline.} =
  F.is_quantized(a.raw)

func is_meta*(a: Tensor): bool {.inline.} =
  F.is_meta(a.raw)

func has_storage*(a: Tensor): bool {.inline.} =
  F.has_storage(a.raw)

# Reference checks

func is_same*(a: Tensor, b: Tensor): bool {.inline.} =
  F.is_same(a.raw, b.raw)

func is_alias_of*(a: Tensor, b: Tensor): bool {.inline.} =
  F.is_alias_of(a.raw, b.raw)

# #######################################################################
#
#                          Math Unary
#
# #######################################################################

func abs*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.abs(a.raw))
func absolute*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.absolute(a.raw))
func angle*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.angle(a.raw))
func sgn*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.sgn(a.raw))
func conj*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.conj(a.raw))
func acos*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.acos(a.raw))
func arccos*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.arccos(a.raw))
func acosh*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.acosh(a.raw))
func arccosh*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.arccosh(a.raw))
func asinh*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.asinh(a.raw))
func arcsinh*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.arcsinh(a.raw))
func atanh*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.atanh(a.raw))
func arctanh*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.arctanh(a.raw))
func asin*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.asin(a.raw))
func arcsin*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.arcsin(a.raw))
func atan*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.atan(a.raw))
func arctan*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.arctan(a.raw))
func cos*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.cos(a.raw))
func sin*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.sin(a.raw))
func tan*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.tan(a.raw))
func exp*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.exp(a.raw))
func exp2*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.exp2(a.raw))
func log*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.log(a.raw))
func log2*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.log2(a.raw))
func log10*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.log10(a.raw))
func erf*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.erf(a.raw))
func erfc*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.erfc(a.raw))
func reciprocal*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.reciprocal(a.raw))
func neg*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.neg(a.raw))
func square*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.square(a.raw))
func sqrt*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.sqrt(a.raw))

# With scalar params

func clamp*(a: Tensor, minVal, maxVal: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.clamp(a.raw, minVal, maxVal))

func clampMin*(a: Tensor, minVal: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.clampMin(a.raw, minVal))

func clampMax*(a: Tensor, maxVal: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.clampMax(a.raw, maxVal))

# #######################################################################
#
#                          Math Binary
#
# #######################################################################

func dot*(a: Tensor, other: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.dot(a.raw, other.raw))

func pow*(a: Tensor, exponent: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.pow(a.raw, exponent.raw))

func pow*(a: Tensor, exponent: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.pow(a.raw, exponent))

# #######################################################################
#
#                         Linear Algebra
#
# #######################################################################

func add*(a: Tensor, other: Tensor, alpha: Scalar = 1): Tensor {.inline.} =
  wrapTorchTensor(F.add(a.raw, other.raw, alpha))

func add*(a: Tensor, other: Scalar, alpha: Scalar = 1): Tensor {.inline.} =
  wrapTorchTensor(F.add(a.raw, other, alpha))

func addmv*(a: Tensor, mat: Tensor, vec: Tensor, beta: Scalar = 1, alpha: Scalar = 1): Tensor {.inline.} =
  wrapTorchTensor(F.addmv(a.raw, mat.raw, vec.raw, beta, alpha))

func addmm*(a: Tensor, mat1: Tensor, mat2: Tensor, beta: Scalar = 1, alpha: Scalar = 1): Tensor {.inline.} =
  wrapTorchTensor(F.addmm(a.raw, mat1.raw, mat2.raw, beta, alpha))

func mm*(a: Tensor, other: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.mm(a.raw, other.raw))

func matmul*(a: Tensor, other: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.matmul(a.raw, other.raw))

func bmm*(a: Tensor, other: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.bmm(a.raw, other.raw))

func luSolve*(a: Tensor, data: Tensor, pivots: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.luSolve(a.raw, data.raw, pivots.raw))

# #######################################################################
#
#                         Comparison
#
# #######################################################################

func equal*(a: Tensor, b: Tensor): bool {.inline.} =
  ## Checks if two tensors have the same shape and all elements are equal.
  ## For floating-point tensors, prefer allClose() for tolerance-based comparison.
  F.equal(a.raw, b.raw)

func eq*(a: Tensor, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.eq(a.raw, b.raw))

func `==.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
  eq(a, b)

func `<.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.lt(a.raw, b.raw))

func `>.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.gt(a.raw, b.raw))

func `<=.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.le(a.raw, b.raw))

func `>=.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.ge(a.raw, b.raw))

func `!=.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.ne(a.raw, b.raw))

func `<.`*(a: Tensor, b: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.lt(a.raw, b))

func `>.`*(a: Tensor, b: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.gt(a.raw, b))

func `<=.`*(a: Tensor, b: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.le(a.raw, b))

func `>=.`*(a: Tensor, b: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.ge(a.raw, b))

func `!=.`*(a: Tensor, b: Scalar): Tensor {.inline.} =
  wrapTorchTensor(F.ne(a.raw, b))

func `<.`*(a: Scalar, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.lt(a, b.raw))

func `>.`*(a: Scalar, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.gt(a, b.raw))

func `<=.`*(a: Scalar, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.le(a, b.raw))

func `>=.`*(a: Scalar, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.ge(a, b.raw))

func `!=.`*(a: Scalar, b: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.ne(a, b.raw))

func allClose*(a: Tensor, b: Tensor, rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false): bool {.inline.} =
  F.allClose(a.raw, b.raw, rtol, abstol, equalNan)

# #######################################################################
#
#                        Arithmetic Operators
#
# #######################################################################

# Binary (Tensor + Tensor)

func `+`*(a: Tensor, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`+`(a.raw, b.raw))
func `-`*(a: Tensor, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`-`(a.raw, b.raw))
func `*`*(a: Tensor, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`*`(a.raw, b.raw))
func `/`*(a: Tensor, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`/`(a.raw, b.raw))
func `%`*(a: Tensor, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`%`(a.raw, b.raw))
func `^`*(a: Tensor, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`^`(a.raw, b.raw))
func `**`*(a: Tensor, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`**`(a.raw, b.raw))

# Scalar mixed

func `+`*(a: SomeNumber, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`+`(a, b.raw))
func `+`*(a: Tensor, b: SomeNumber): Tensor {.inline.} = wrapTorchTensor(F.`+`(a.raw, b))
func `-`*(a: Tensor, b: SomeNumber): Tensor {.inline.} = wrapTorchTensor(F.`-`(a.raw, b))
func `*`*(a: SomeNumber, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`*`(a, b.raw))
func `*`*(a: Tensor, b: SomeNumber): Tensor {.inline.} = wrapTorchTensor(F.`*`(a.raw, b))
func `/`*(a: Tensor, b: SomeNumber): Tensor {.inline.} = wrapTorchTensor(F.`/`(a.raw, b))
func `%`*(a: SomeNumber, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`%`(a, b.raw))
func `%`*(a: Tensor, b: SomeNumber): Tensor {.inline.} = wrapTorchTensor(F.`%`(a.raw, b))
func `^`*(a: Tensor, b: Scalar): Tensor {.inline.} = wrapTorchTensor(F.`^`(a.raw, b))
func `**`*(a: Tensor, b: Scalar): Tensor {.inline.} = wrapTorchTensor(F.`**`(a.raw, b))

# Unary

func `-`*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`-`(a.raw))
func `not`*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`not`(a.raw))

# Bitwise (returns Tensor)

func `and`*(a: Tensor, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`and`(a.raw, b.raw))
func `or`*(a: Tensor, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`or`(a.raw, b.raw))
func `xor`*(a: Tensor, b: Tensor): Tensor {.inline.} = wrapTorchTensor(F.`xor`(a.raw, b.raw))

# In-place (var Tensor)

proc `+=`*(a: var Tensor, b: Tensor) {.inline.} = F.`+=`(a.raw, b.raw)
proc `+=`*(a: var Tensor, s: Scalar) {.inline.} = F.`+=`(a.raw, s)
proc `-=`*(a: var Tensor, b: Tensor) {.inline.} = F.`-=`(a.raw, b.raw)
proc `-=`*(a: var Tensor, s: Scalar) {.inline.} = F.`-=`(a.raw, s)
proc `*=`*(a: var Tensor, b: Tensor) {.inline.} = F.`*=`(a.raw, b.raw)
proc `*=`*(a: var Tensor, s: Scalar) {.inline.} = F.`*=`(a.raw, s)
proc `/=`*(a: var Tensor, b: Tensor) {.inline.} = F.`/=`(a.raw, b.raw)
proc `/=`*(a: var Tensor, s: Scalar) {.inline.} = F.`/=`(a.raw, s)
proc bitand_mut*(a: var Tensor, b: Tensor) {.inline.} = F.bitand_mut(a.raw, b.raw)
proc bitor_mut*(a: var Tensor, b: Tensor) {.inline.} = F.bitor_mut(a.raw, b.raw)
proc bitxor_mut*(a: var Tensor, b: Tensor) {.inline.} = F.bitxor_mut(a.raw, b.raw)
proc assign*(a: var Tensor, other: Tensor) {.inline.} = a.raw = other.raw

# #######################################################################
#
#                        Reductions (Scalar)
#
# #######################################################################

func sum*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.sum(a.raw))
func sum*(a: Tensor, T: typedesc[SomeTorchType]): Tensor {.inline.} = wrapTorchTensor(F.sum(a.raw, toScalarKind(T)))
func sum*(a: Tensor, axis: int, keepdim: bool = false): Tensor {.inline.} = wrapTorchTensor(F.sum(a.raw, axis, keepdim))
func sum*(a: Tensor, axis: int, keepdim: bool = false, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.sum(a.raw, axis, keepdim, toScalarKind(T)))

func mean*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.mean(a.raw))
func mean*(a: Tensor, T: typedesc[SomeTorchType]): Tensor {.inline.} = wrapTorchTensor(F.mean(a.raw, toScalarKind(T)))
func mean*(a: Tensor, axis: int, keepdim: bool = false): Tensor {.inline.} = wrapTorchTensor(F.mean(a.raw, axis, keepdim))
func mean*(a: Tensor, axis: int, keepdim: bool = false, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.mean(a.raw, axis, keepdim, toScalarKind(T)))

func prod*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.prod(a.raw))
func prod*(a: Tensor, T: typedesc[SomeTorchType]): Tensor {.inline.} = wrapTorchTensor(F.prod(a.raw, toScalarKind(T)))
func prod*(a: Tensor, axis: int, keepdim: bool = false): Tensor {.inline.} = wrapTorchTensor(F.prod(a.raw, axis, keepdim))
func prod*(a: Tensor, axis: int, keepdim: bool = false, T: typedesc[SomeTorchType]): Tensor {.inline.} =
  wrapTorchTensor(F.prod(a.raw, axis, keepdim, toScalarKind(T)))

func min*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.min(a.raw))
func max*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.max(a.raw))

func all*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.all(a.raw))
func all*(a: Tensor, axis: int): Tensor {.inline.} = wrapTorchTensor(F.all(a.raw, axis))
func all*(a: Tensor, axis: int, keepdim: bool): Tensor {.inline.} = wrapTorchTensor(F.all(a.raw, axis, keepdim))

func any*(a: Tensor, axis: int): Tensor {.inline.} = wrapTorchTensor(F.any(a.raw, axis))
func any*(a: Tensor, axis: int, keepdim: bool): Tensor {.inline.} = wrapTorchTensor(F.any(a.raw, axis, keepdim))

func argmax*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.argmax(a.raw))
func argmax*(a: Tensor, axis: int, keepdim: bool = false): Tensor {.inline.} = wrapTorchTensor(F.argmax(a.raw, axis, keepdim))
func argmin*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.argmin(a.raw))
func argmin*(a: Tensor, axis: int, keepdim: bool = false): Tensor {.inline.} = wrapTorchTensor(F.argmin(a.raw, axis, keepdim))

func variance*(a: Tensor, unbiased: bool = true): Tensor {.inline.} = wrapTorchTensor(F.variance(a.raw, unbiased))
func variance*(a: Tensor, axis: int, unbiased: bool = true, keepdim: bool = false): Tensor {.inline.} =
  wrapTorchTensor(F.variance(a.raw, axis, unbiased, keepdim))

func stddev*(a: Tensor, unbiased: bool = true): Tensor {.inline.} = wrapTorchTensor(F.stddev(a.raw, unbiased))
func stddev*(a: Tensor, axis: int, unbiased: bool = true, keepdim: bool = false): Tensor {.inline.} =
  wrapTorchTensor(F.stddev(a.raw, axis, unbiased, keepdim))

# #######################################################################
#
#                       Reductions (Tuple)
#
# #######################################################################

func min*(a: Tensor, axis: int, keepdim: bool = false): tuple[values: Tensor, indices: Tensor] {.inline.} =
  let raw = F.min(a.raw, axis, keepdim)
  (values: wrapTorchTensor(get(raw, 0)), indices: wrapTorchTensor(get(raw, 1)))

func max*(a: Tensor, axis: int, keepdim: bool = false): tuple[values: Tensor, indices: Tensor] {.inline.} =
  let raw = F.max(a.raw, axis, keepdim)
  (values: wrapTorchTensor(get(raw, 0)), indices: wrapTorchTensor(get(raw, 1)))

func sort*(a: Tensor, axis: int = -1, descending: bool = false): tuple[values: Tensor, indices: Tensor] {.inline.} =
  let raw = F.sort(a.raw, axis, descending)
  (values: wrapTorchTensor(get(raw, 0)), indices: wrapTorchTensor(get(raw, 1)))

func argsort*(a: Tensor, axis: int = -1, descending: bool = false): Tensor {.inline.} =
  wrapTorchTensor(F.argsort(a.raw, axis, descending))

func qr*(a: Tensor, some: bool = true): tuple[q: Tensor, r: Tensor] {.inline.} =
  let raw = F.qr(a.raw, some)
  (q: wrapTorchTensor(get(raw, 0)), r: wrapTorchTensor(get(raw, 1)))

# #######################################################################
#
#                          Algorithms
#
# #######################################################################

func cat*(tensors: varargs[Tensor], axis: int = 0): Tensor {.inline.} =
  var raws = new(Vec[TorchTensor], tensors.len)
  for i, t in tensors:
    raws[i] = t.raw
  wrapTorchTensor(F.cat(raws.asTorchView(), axis))

func stack*(tensors: varargs[Tensor], dim: int = 0): Tensor {.inline.} =
  var raws = new(Vec[TorchTensor], tensors.len)
  for i, t in tensors:
    raws[i] = t.raw
  wrapTorchTensor(F.stack(raws.asTorchView(), dim))

func chunk*(a: Tensor, chunks: int, dim: int = 0): seq[Tensor] {.inline.} =
  var raw = F.chunk(a.raw, chunks, dim)
  result = newSeq[Tensor](len(raw))
  for i in 0..<len(raw):
    result[i] = wrapTorchTensor(move raw[i])

func unbind*(a: Tensor, dim: int = 0): seq[Tensor] {.inline.} =
  var raw = F.unbind(a.raw, dim)
  result = newSeq[Tensor](len(raw))
  for i in 0..<len(raw):
    result[i] = wrapTorchTensor(move raw[i])

# #######################################################################
#
#                        Fancy Indexing
#
# #######################################################################

func index_select*(a: Tensor, axis: int, indices: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.index_select(a.raw, axis, indices.raw))

func masked_select*(a: Tensor, mask: Tensor): Tensor {.inline.} =
  wrapTorchTensor(F.masked_select(a.raw, mask.raw))

# #######################################################################
#
#                        In-place Mutation
#
# #######################################################################

proc random_mut*(a: var Tensor, start, stopEx: int) {.inline.} =
  F.random_mut(a.raw, start, stopEx)

proc index_fill_mut*(a: var Tensor, mask: Tensor, value: Scalar) {.inline.} =
  F.index_fill_mut(a.raw, mask.raw, value)

proc index_fill_mut*(a: var Tensor, mask: Tensor, value: Tensor) {.inline.} =
  F.index_fill_mut(a.raw, mask.raw, value.raw)

proc masked_fill_mut*(a: var Tensor, mask: Tensor, value: Scalar) {.inline.} =
  F.masked_fill_mut(a.raw, mask.raw, value)

proc masked_fill_mut*(a: var Tensor, mask: Tensor, value: Tensor) {.inline.} =
  F.masked_fill_mut(a.raw, mask.raw, value.raw)

# #######################################################################
#
#                           Indexing
#
# #######################################################################

func item*(t: Tensor, T: typedesc): T {.inline.} =
  t.raw.item(T)

template `[]`*(t: Tensor{call}, args: varargs[untyped]): untyped =
  let tmp = t
  wrapTorchTensor(tmp.raw[args])

template `[]`*(t: Tensor{`let`|`var`|`const`|lvalue}, args: varargs[untyped]): untyped =
  wrapTorchTensor(t.raw[args])

macro `[]=`*(t: var Tensor, args: varargs[untyped]): untyped =
  var tmp = args
  let valAST = tmp.pop()
  let new_args = getAST(desugarSlices(tmp))
  result = quote do:
    let tmpVal = `valAST`
    when tmpVal is Tensor:
      slice_typed_dispatch_mut(`t`.raw, `new_args`, tmpVal.raw)
    else:
      slice_typed_dispatch_mut(`t`.raw, `new_args`, tmpVal)

# #######################################################################
#
#                           Data Access
#
# #######################################################################

func data_ptr*(a: Tensor): pointer {.inline.} =
  F.data_ptr(a.raw)

func data_ptr*(a: Tensor, T: typedesc): ptr UncheckedArray[T] {.inline.} =
  F.data_ptr(a.raw, T)

# #######################################################################
#
#                              FFT
#
# #######################################################################

# 1-D
func fft*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.fft(a.raw))
func fft*(a: Tensor, n: int, dim: int = -1): Tensor {.inline.} = wrapTorchTensor(F.fft(a.raw, n, dim))
func ifft*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.ifft(a.raw))
func rfft*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.rfft(a.raw))
func irfft*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.irfft(a.raw))
func hfft*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.hfft(a.raw))
func ihfft*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.ihfft(a.raw))

# N-D
func fft2*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.fft2(a.raw))
func fft2*(a: Tensor, s: varargs[int]): Tensor {.inline.} = wrapTorchTensor(F.fft2(a.raw, asTorchView(s)))
func ifft2*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.ifft2(a.raw))
func ifft2*(a: Tensor, s: varargs[int]): Tensor {.inline.} = wrapTorchTensor(F.ifft2(a.raw, asTorchView(s)))
func fftn*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.fftn(a.raw))
func fftn*(a: Tensor, s: varargs[int]): Tensor {.inline.} = wrapTorchTensor(F.fftn(a.raw, asTorchView(s)))
func ifftn*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.ifftn(a.raw))
func ifftn*(a: Tensor, s: varargs[int]): Tensor {.inline.} = wrapTorchTensor(F.ifftn(a.raw, asTorchView(s)))
func rfft2*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.rfft2(a.raw))
func irfft2*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.irfft2(a.raw))
func rfftn*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.rfftn(a.raw))
func irfftn*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.irfftn(a.raw))

# Shift
func fftshift*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.fftshift(a.raw))
func fftshift*(a: Tensor, dim: varargs[int]): Tensor {.inline.} = wrapTorchTensor(F.fftshift(a.raw, asTorchView(dim)))
func ifftshift*(a: Tensor): Tensor {.inline.} = wrapTorchTensor(F.ifftshift(a.raw))
func ifftshift*(a: Tensor, dim: varargs[int]): Tensor {.inline.} = wrapTorchTensor(F.ifftshift(a.raw, asTorchView(dim)))

# #######################################################################
#
#                         Debugging
#
# #######################################################################

proc `$`*(t: Tensor): string =
  {.emit:
    """
    std::ostringstream stream;
    stream << `t.raw`;
    result = "Tensor\\n" + stream.str();
    """
  .}

proc print*(t: Tensor) {.sideeffect, inline.} =
  F.print(t.raw)
