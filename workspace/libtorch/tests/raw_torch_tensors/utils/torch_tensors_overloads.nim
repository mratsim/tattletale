# Tattletale
# Copyright (c) 2026 Mamy AndrÃ©-Ratsimbazafy
# Licensed under MIT or Apache v2

import workspace/libtorch/src/raw_libtorch as F

static: doAssert sizeof(int) == 8, "Only 64-bit OSes are supported"

# Constructor factories
# -----------------------------------------------------------------------

template zeros*(dim: varargs[int], scalarKind: F.ScalarKind): F.TorchTensor =
  F.zeros(F.asTorchView(dim), scalarKind)

template ones*(dim: varargs[int], scalarKind: F.ScalarKind): F.TorchTensor =
  F.ones(F.asTorchView(dim), scalarKind)

template full*(size: varargs[int], fill_value: F.Scalar, scalarKind: F.ScalarKind): F.TorchTensor =
  F.full(F.asTorchView(size), fill_value, scalarKind)

template rand*(size: varargs[int], scalarKind: F.ScalarKind): F.TorchTensor =
  F.rand(F.asTorchView(size), scalarKind)

template randn*(size: varargs[int], scalarKind: F.ScalarKind): F.TorchTensor =
  F.randn(F.asTorchView(size), scalarKind)

# Aggregate
# -----------------------------------------------------------------------

func cat*(tensors: varargs[F.TorchTensor], axis = 0): F.TorchTensor {.inline.} =
  F.cat(F.asTorchView(tensors), axis)

# Shape methods
# -----------------------------------------------------------------------

func reshape*(self: F.TorchTensor, sizes: varargs[int]): F.TorchTensor {.inline.} =
  F.reshape(self, F.asTorchView(sizes))

func flip*(self: F.TorchTensor, dims: varargs[int]): F.TorchTensor {.inline.} =
  F.flip(self, F.asTorchView(dims))

