## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU-suitable copy kernels: divmod-based flat-index iteration.
##
## These use `dst(i) = src(i)` which calls `crd2idx` per element
## (divmod for flat→coord decomposition). Acceptable on GPU where
## divmod is relatively cheap and warp divergence from wheel-winding
## would be catastrophic.
##
## On CPU, use `kernel_copy_cpu` (`copySameShape_cpu`/`copyPermuted_cpu`)
## which avoids divmod entirely via contiguity-fused copyMem.

import ./int_tuples
import ./layouts
import ./tensors

{.experimental: "callOperator".}

template copyFrom*[T, ShA, StA, ShB, StB](
    dst: var TensorView[T, ShB, StB];
    src: TensorView[T, ShA, StA]) =
  ## Copy every logical element from src to dst.
  ## Uses flat-index iteration (`dst(i) = src(i)`) which calls crd2idx
  ## per element — acceptable on GPU, slow on CPU.
  for i in 0 ..< size(dst):
    dst(i) = src(i)

template copyFrom*[T, ShA, StA, ShB, StB](
    dst: var Tensor[T, ShB, StB];
    src: TensorView[T, ShA, StA] or Tensor[T, ShA, StA]) =
  ## Owning-tensor dst form — the fragment tensors (make_fragment_A/B,
  ## make_tensor/make_tensor_like). The flat-index `dst(i) = src(i)` is
  ## coordinate semantics: crd2idx decodes `i` through each tensor's own
  ## shape (mode order) then maps through its own strides, so the
  ## fragment (V = atom register order, stride-1) receives the element at
  ## the same logical coordinate as src, whatever src's layout (row-major
  ## included). The fragment's physical order follows the fragment's
  ## layout: V fastest, matching gemm_fragment's data[k·VA+i] read.
  for i in 0 ..< size(dst):
    dst(i) = src(i)

template copyFromIf*[T, ShA, StA, ShB, StB](
    dst: var TensorView[T, ShB, StB];
    src: TensorView[T, ShA, StA];
    predicate: typed;
    defaultVal: T) =
  ## Copy elements where predicate(i) is true, fill rest with defaultVal.
  for i in 0 ..< size(dst):
    if predicate(i):
      dst(i) = src(i)
    else:
      dst(i) = defaultVal
