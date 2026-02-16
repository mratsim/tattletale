# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/libtorch/src/abi/torch_tensors
import workspace/libtorch/src/torch_tensors_sugar

static: doAssert sizeof(int) == 8, "Only 64-bit OSes are supported"

# Overload templates for openArray[int] -> IntArrayRef automatic conversion
# -----------------------------------------------------------------------
#
# These templates provide overloads that automatically convert Nim sequences and arrays
# to Torch ArrayRef using asTorchView(). This eliminates the need to explicitly call
# asTorchView() in user code.
#
# Usage:
#   Instead of: tensor.reshape(@[3, 4].asTorchView())
#   Write:      tensor.reshape(@[3, 4])
#   Or:         tensor.reshape([3, 4])
#   Or:         tensor.reshape(3, 4)
#
# The template automatically converts the varargs to ArrayRef.

# Constructors
# -----------------------------------------------------------------------

template from_blob*(
  data: pointer,
  sizes: openArray[int],
  options: TensorOptions
): TorchTensor =
  from_blob(data, asTorchView(sizes), options)

template from_blob*(
  data: pointer,
  sizes: openArray[int],
  scalarKind: ScalarKind
): TorchTensor =
  from_blob(data, asTorchView(sizes), scalarKind)

template from_blob*(data: pointer, sizes: openArray[int], device: DeviceKind): TorchTensor =
  from_blob(data, asTorchView(sizes), device)

template from_blob*(data: pointer, sizes, strides: openArray[int], options: TensorOptions): TorchTensor =
  from_blob(data, asTorchView(sizes), asTorchView(strides), options)

template from_blob*(data: pointer, sizes, strides: openArray[int], scalarKind: ScalarKind): TorchTensor =
  from_blob(data, asTorchView(sizes), asTorchView(strides), scalarKind)

template from_blob*(data: pointer, sizes, strides: openArray[int], device: DeviceKind): TorchTensor =
  from_blob(data, asTorchView(sizes), asTorchView(strides), device)

template empty*(size: varargs[int]): TorchTensor =
  empty(asTorchView(size))

template empty*(size: varargs[int], options: TensorOptions): TorchTensor =
  empty(asTorchView(size), options)

template empty*(size: varargs[int], scalarKind: ScalarKind): TorchTensor =
  empty(asTorchView(size), scalarKind)

template empty*(size: varargs[int], device: DeviceKind): TorchTensor =
  empty(asTorchView(size), device)

# Random sampling
# -----------------------------------------------------------------------

template randint*(start, stopEx: int, size: openArray[int]): TorchTensor =
  randint(start, stopEx, asTorchView(size))

template rand*(size: openArray[int], options: TensorOptions): TorchTensor =
  rand(asTorchView(size), options)

template rand*(size: openArray[int], options: DeviceKind): TorchTensor =
  rand(asTorchView(size), options)

template rand*(size: openArray[int], options: Device): TorchTensor =
  rand(asTorchView(size), options)

template rand*(size: openArray[int], options: ScalarKind): TorchTensor =
  rand(asTorchView(size), options)

template rand*(size: openArray[int]): TorchTensor =
  rand(asTorchView(size))

# Shapeshifting
# -----------------------------------------------------------------------

func reshape*(self: TorchTensor, sizes: varargs[int]): TorchTensor {.inline.} =
  reshape(self, asTorchView(sizes))

func view*(self: TorchTensor, size: varargs[int]): TorchTensor {.inline.} =
  view(self, asTorchView(size))

func permute*(self: TorchTensor, dims: varargs[int]): TorchTensor {.inline.} =
  permute(self, asTorchView(dims))

func expand*(self: TorchTensor, size: varargs[int]): TorchTensor {.inline.} =
  expand(self, asTorchView(size))

func flip*(self: TorchTensor, dims: varargs[int]): TorchTensor {.inline.} =
  flip(self, asTorchView(dims))

# Functions.h
# -----------------------------------------------------------------------

template zeros*(dim: varargs[int]): TorchTensor =
  zeros(asTorchView(dim))

template zeros*(dim: varargs[int], options: TensorOptions): TorchTensor =
  zeros(asTorchView(dim), options)

template zeros*(dim: varargs[int], scalarKind: ScalarKind): TorchTensor =
  zeros(asTorchView(dim), scalarKind)

template zeros*(dim: varargs[int], device: DeviceKind): TorchTensor =
  zeros(asTorchView(dim), device)

template ones*(dim: varargs[int]): TorchTensor =
  ones(asTorchView(dim))

template ones*(dim: varargs[int], options: TensorOptions): TorchTensor =
  ones(asTorchView(dim), options)

template ones*(dim: varargs[int], scalarKind: ScalarKind): TorchTensor =
  ones(asTorchView(dim), scalarKind)

template ones*(dim: varargs[int], device: DeviceKind): TorchTensor =
  ones(asTorchView(dim), device)

template full*(size: varargs[int], fill_value: Scalar): TorchTensor =
  full(asTorchView(size), fill_value)

template full*(size: varargs[int], fill_value: Scalar, options: TensorOptions): TorchTensor =
  full(asTorchView(size), fill_value, options)

template full*(size: openArray[int], fill_value: Scalar, scalarKind: ScalarKind): TorchTensor =
  full(asTorchView(size), fill_value, scalarKind)

template full*(size: openArray[int], fill_value: Scalar, device: DeviceKind): TorchTensor =
  full(asTorchView(size), fill_value, device)

template randn*(size: openArray[int]): TorchTensor =
  randn(asTorchView(size))

template randn*(size: openArray[int], options: TensorOptions): TorchTensor =
  randn(asTorchView(size), options)

template randn*(size: openArray[int], scalarKind: ScalarKind): TorchTensor =
  randn(asTorchView(size), scalarKind)

template randn*(size: openArray[int], device: DeviceKind): TorchTensor =
  randn(asTorchView(size), device)

# aggregate
# -----------------------------------------------------------------------

template sum*(self: TorchTensor, axis: openArray[int], keepdim: bool = false): TorchTensor =
  sum(self, asTorchView(axis), keepdim)

template sum*(self: TorchTensor, axis: openArray[int], keepdim: bool = false, dtype: ScalarKind): TorchTensor =
  sum(self, asTorchView(axis), keepdim, dtype)

template mean*(self: TorchTensor, axis: openArray[int], keepdim: bool = false): TorchTensor =
  mean(self, asTorchView(axis), keepdim)

template mean*(self: TorchTensor, axis: openArray[int], keepdim: bool = false, dtype: ScalarKind): TorchTensor =
  mean(self, asTorchView(axis), keepdim, dtype)

template variance*(self: TorchTensor, axis: openArray[int], unbiased: bool = true, keepdim: bool = false): TorchTensor =
  variance(self, asTorchView(axis), unbiased, keepdim)

template stddev*(self: TorchTensor, axis: openArray[int], unbiased: bool = true, keepdim: bool = false): TorchTensor =
  stddev(self, asTorchView(axis), unbiased, keepdim)

# algorithms
# -----------------------------------------------------------------------

func cat*(tensors: varargs[TorchTensor], axis = 0): TorchTensor {.inline.} =
  cat(asTorchView(tensors), axis)

func stack*(tensors: varargs[TorchTensor], dim = 0): TorchTensor {.inline.} =
  stack(asTorchView(tensors), dim)

# FFT
# -----------------------------------------------------------------------

template fftshift*(self: TorchTensor, dim: openArray[int]): TorchTensor =
  fftshift(self, asTorchView(dim))

template ifftshift*(self: TorchTensor, dim: openArray[int]): TorchTensor =
  ifftshift(self, asTorchView(dim))

template fft2*(self: TorchTensor, s: openArray[int]): TorchTensor =
  fft2(self, asTorchView(s))

template ifft2*(self: TorchTensor, s: openArray[int]): TorchTensor =
  ifft2(self, asTorchView(s))

template fftn*(self: TorchTensor, s: openArray[int]): TorchTensor =
  fftn(self, asTorchView(s))

template ifftn*(self: TorchTensor, s: openArray[int]): TorchTensor =
  ifftn(self, asTorchView(s))

template rfft2*(self: TorchTensor, s: openArray[int]): TorchTensor =
  rfft2(self, asTorchView(s))

template irfft2*(self: TorchTensor, s: openArray[int]): TorchTensor =
  irfft2(self, asTorchView(s))

template rfftn*(self: TorchTensor, s: openArray[int]): TorchTensor =
  rfftn(self, asTorchView(s))

template irfftn*(self: TorchTensor, s: openArray[int]): TorchTensor =
  irfftn(self, asTorchView(s))
