# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  std/complex,
  # Internal
  workspace/libtorch/src/std_cpp,
  workspace/libtorch/src/c10,
  workspace/libtorch/vendor/libtorch

# (Almost) raw bindings to PyTorch Tensors
# -----------------------------------------------------------------------
#
# This provides almost raw bindings to PyTorch tensors.
#
# "Nimification" (camelCase), ergonomic indexing and interoperability with Nim types is left to the "high-level" bindings.
# This should ease searching PyTorch and libtorch documentation,
# and make C++ tutorials easily applicable.
#
# Nonetheless some slight modifications were given to the raw bindings:
# - `&=`, `|=` and `^=` have been renamed bitand, bitor, bitxor
# - `[]` and `[]=` are not exported as index and index_put are more flexible
#   and we want to leave those symbols available for Numpy-like ergonomic indexing.
# - Nim's `index_fill_mut` and `masked_fill_mut` are mapped to the in-place
#   C++ `index_fill_` and `masked_fill_`.
#   The original out-of-place versions are doing clone+in-place mutation

# C++ interop
# -----------------------------------------------------------------------

{.push cdecl, header: TorchHeader.}

# #######################################################################
#
#                         Context
#
# #######################################################################

type Torch* = object

# Random Number Generation
# -----------------------------------------------------------------------

proc manual_seed*(_: type Torch, seed: uint64) {.sideeffect, importcpp: "torch::manual_seed(@)".}
  ## Set torch random number generator seed

# Backends
# -----------------------------------------------------------------------

# `hasCuda` is defined in `Context.h`, but not available here?
proc hasCuda*(_: type Torch): bool {.sideeffect, importcpp: "torch::hasCuda()".}
  ## Returns true if libtorch was compiled with CUDA support

proc deviceCount*(_: type Torch): csize_t {.sideeffect, importcpp: "torch::cuda::device_count()".}
  ## Returns true if libtorch was compiled with CUDA support

proc cuda_is_available*(_: type Torch): bool {.sideeffect, importcpp: "torch::cuda::is_available()".}
  ## Returns true if libtorch was compiled with CUDA support
  ## and at least one CUDA device is available

proc cudnn_is_available*(_: type Torch): bool {.sideeffect, importcpp: "torch::cuda::cudnn_is_available()".}
  ## Returns true if libtorch was compiled with CUDA and CuDNN support
  ## and at least one CUDA device is available

# #######################################################################
#
#                         Tensor Metadata
#
# #######################################################################

# Backend Device
# -----------------------------------------------------------------------
# libtorch/include/c10/core/DeviceType.h
# libtorch/include/c10/core/Device.h

type
  DeviceIndex = int16

  DeviceKind* {.importc: "c10::DeviceType", size: sizeof(int16).} = enum
    kCPU = 0
    kCUDA = 1
    kMKLDNN = 2
    kOpenGL = 3
    kOpenCL = 4
    kIDEEP = 5
    kHIP = 6
    kFPGA = 7
    kMSNPU = 8
    kXLA = 9
    kVulkan = 10

  Device* {.importc: "c10::Device", bycopy.} = object
    kind: DeviceKind
    index: DeviceIndex

func init*(T: type Device, kind: DeviceKind): T {.constructor, importcpp: "torch::Device(#)".}

# Datatypes
# -----------------------------------------------------------------------
# libtorch/include/torch/csrc/api/include/torch/types.h
# libtorch/include/c10/core/ScalarType.h

type
  ScalarKind* {.importc: "torch::ScalarType", size: sizeof(int8).} = enum
    kUint8 = 0 # kByte
    kInt8 = 1 # kChar
    kInt16 = 2 # kShort
    kInt32 = 3 # kInt
    kInt64 = 4 # kLong
    kFloat16 = 5 # kHalf
    kFloat32 = 6 # kFloat
    kFloat64 = 7 # kDouble
    kComplexF16 = 8 # kComplexHalf
    kComplexF32 = 9 # kComplexFloat
    kComplexF64 = 10 # kComplexDouble
    kBool = 11
    kQint8 = 12 # Quantized int8
    kQuint8 = 13 # Quantized uint8
    kQint32 = 14 # Quantized int32
    kBfloat16 = 15 # Brain float16

  SomeTorchType* = uint8 | byte or SomeSignedInt or SomeUnsignedInt or SomeFloat or Complex[float32] or Complex[float64]
  ## Torch Tensor type mapped to Nim type

# TensorOptions
# -----------------------------------------------------------------------
# libtorch/include/c10/core/TensorOptions.h

type TensorOptions* {.importcpp: "torch::TensorOptions", bycopy.} = object

func init*(T: type TensorOptions): TensorOptions {.constructor, importcpp: "torch::TensorOptions".}

# Scalars
# -----------------------------------------------------------------------
# Scalars are defined in libtorch/include/c10/core/Scalar.h
# as tagged unions of double, int64, complex
# And C++ types are implicitly convertible to Scalar
#
# Hence in Nim we don't need to care about Scalar or defined converters
# (except maybe for complex)
type Scalar* = SomeNumber or bool or TorchComplex

# TensorAccessors
# -----------------------------------------------------------------------
# libtorch/include/ATen/core/TensorAccessors.h
#
# Tensor accessors gives "medium-level" access to a Tensor raw-data
# - Compared to low-level "data_ptr" they take care of striding and shape
# - Compared to high-level functions they don't provide any parallelism.

# #######################################################################
#
#                            Tensors
#
# #######################################################################

# Tensors
# -----------------------------------------------------------------------

type TorchTensor* {.importcpp: "torch::Tensor", cppNonPod, bycopy.} = object

# Strings & Debugging
# -----------------------------------------------------------------------

proc print*(self: TorchTensor) {.sideeffect, importcpp: "torch::print(@)".}

# Metadata
# -----------------------------------------------------------------------

func dim*(self: TorchTensor): int64 {.importcpp: "#.dim()".} ## Number of dimensions
func reset*(self: var TorchTensor) {.importcpp: "#.reset()".}
func is_same*(self, other: TorchTensor): bool {.importcpp: "#.is_same(#)".}
  ## Reference equality
  ## Do the tensors use the same memory.

func sizes*(self: TorchTensor): IntArrayRef {.importcpp: "#.sizes()".} ## This is Arraymancer and Numpy "shape"

func strides*(self: TorchTensor): IntArrayRef {.importcpp: "#.strides()".}

func ndimension*(self: TorchTensor): int64 {.importcpp: "#.ndimension()".} ## This is Arraymancer rank
func nbytes*(self: TorchTensor): uint {.importcpp: "#.nbytes()".} ## Bytes-size of the Tensor
func numel*(self: TorchTensor): int64 {.importcpp: "#.numel()".} ## This is Arraymancer and Numpy "size"

func size*(self: TorchTensor, axis: int64): int64 {.importcpp: "#.size(#)".}
func itemsize*(self: TorchTensor): uint {.importcpp: "#.itemsize()".}
func element_size*(self: TorchTensor): int64 {.importcpp: "#.element_size()".}

# Accessors
# -----------------------------------------------------------------------

func data_ptr*(self: TorchTensor, T: typedesc): ptr UncheckedArray[T] {.importcpp: "#.data_ptr<'2>(#)".}
  ## Gives raw access to a tensor data of type T.
  ##
  ## This is a very low-level procedure. You need to take care
  ## of the tensor shape and strides yourself.
  ##
  ## It is recommended to use this only on contiguous tensors
  ## (freshly created or freshly cloned) and to avoid
  ## sliced tensors.

# Backend
# -----------------------------------------------------------------------

func has_storage*(self: TorchTensor): bool {.importcpp: "#.has_storage()".}
func get_device*(self: TorchTensor): int64 {.importcpp: "#.get_device()".}
func is_cuda*(self: TorchTensor): bool {.importcpp: "#.is_cuda()".}
func is_hip*(self: TorchTensor): bool {.importcpp: "#.is_hip()".}
func is_sparse*(self: TorchTensor): bool {.importcpp: "#.is_sparse()".}
func is_mkldnn*(self: TorchTensor): bool {.importcpp: "#.is_mkldnn()".}
func is_vulkan*(self: TorchTensor): bool {.importcpp: "#.is_vulkan()".}
func is_quantized*(self: TorchTensor): bool {.importcpp: "#.is_quantized()".}
func is_meta*(self: TorchTensor): bool {.importcpp: "#.is_meta()".}

func cpu*(self: TorchTensor): TorchTensor {.importcpp: "#.cpu()".}
func cuda*(self: TorchTensor): TorchTensor {.importcpp: "#.cuda()".}
func hip*(self: TorchTensor): TorchTensor {.importcpp: "#.hip()".}
func vulkan*(self: TorchTensor): TorchTensor {.importcpp: "#.vulkan()".}
func to*(self: TorchTensor, device: DeviceKind): TorchTensor {.importcpp: "#.to(#)".}
func to*(self: TorchTensor, device: Device): TorchTensor {.importcpp: "#.to(#)".}

# dtype
# -----------------------------------------------------------------------

func to*(self: TorchTensor, dtype: ScalarKind): TorchTensor {.importcpp: "#.to(#)".}
func scalarType*(self: TorchTensor): ScalarKind {.importcpp: "#.scalar_type()".}

# Constructors
# -----------------------------------------------------------------------

# DeviceType and ScalarType are auto-convertible to TensorOptions

func init*(T: type TorchTensor): TorchTensor {.constructor, importcpp: "torch::Tensor".}
# Default empty constructor
func initRawTensor*(): TorchTensor {.constructor, importcpp: "torch::Tensor".}
# Move / Copy constructor ?
func initRawTensor*(t: TorchTensor): TorchTensor {.constructor, importcpp: "torch::Tensor(@)".}

func from_blob*(
  data: pointer, sizes: IntArrayRef, options: TensorOptions
): TorchTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(
  data: pointer, sizes: IntArrayRef, scalarKind: ScalarKind
): TorchTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, device: DeviceKind): TorchTensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes: int64, options: TensorOptions): TorchTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int64, device: DeviceKind): TorchTensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes, strides: IntArrayRef, options: TensorOptions): TorchTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes, strides: IntArrayRef, device: DeviceKind): TorchTensor {.importcpp: "torch::from_blob(@)".}

func empty*(size: IntArrayRef, options: TensorOptions): TorchTensor {.importcpp: "torch::empty(@)".}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually
  ##
  ## The output tensor will be row major (C contiguous)
func empty*(size: IntArrayRef, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::empty(@)".}
func empty*(size: IntArrayRef, device: DeviceKind): TorchTensor {.importcpp: "torch::empty(@)".}
  ## Create an uninitialized tensor of shape `size`
  ## The tensor data must be filled manually.
  ##
  ## If device is NOT on CPU make sure to use specialized
  ## copy operations. For example to update on Cuda devices
  ## use cudaMemcpy not a.data[i] = 123
  ##
  ## The output tensor will be row major (C contiguous)

func clone*(self: TorchTensor): TorchTensor {.importcpp: "#.clone()".}

# TODO : Test this
func view_as_real*(self: TorchTensor): TorchTensor {.importcpp: "#.view_as_real()".}
func view_as_complex*(self: TorchTensor): TorchTensor {.importcpp: "#.view_as_complex()".}

# Random sampling
# -----------------------------------------------------------------------
func random_mut*(self: var TorchTensor, start, stopEx: int64) {.importcpp: "#.random_(@)".}
func randint*(start, stopEx: int64): TorchTensor {.varargs, importcpp: "torch::randint(#, #, {@})".}
func randint*(start, stopEx: int64, size: IntArrayRef): TorchTensor {.importcpp: "torch::randint(@)".}

func rand_like*(self: TorchTensor, options: TensorOptions): TorchTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: TorchTensor, options: ScalarKind): TorchTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: TorchTensor, options: DeviceKind): TorchTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: TorchTensor, options: Device): TorchTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(self: TorchTensor): TorchTensor {.importcpp: "torch::rand_like(@)".}

func rand*(size: IntArrayRef, options: TensorOptions): TorchTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef, options: DeviceKind): TorchTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef, options: Device): TorchTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef, options: ScalarKind): TorchTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef): TorchTensor {.importcpp: "torch::rand(@)".}

# Indexing
# -----------------------------------------------------------------------
# libtorch/include/ATen/TensorIndexing.h
# and https://pytorch.org/cppdocs/notes/tensor_indexing.html

func item*(self: TorchTensor, T: typedesc): T {.importcpp: "#.item<'0>()".}
  ## Extract the scalar from a 0-dimensional tensor
func item*(self: TorchTensor, T: typedesc[Complex32]): TorchComplex[float32] {.importcpp: "#.item<c10::complex<float>>()".}
func item*(
  self: TorchTensor, T: typedesc[Complex64]
): TorchComplex[float64] {.importcpp: "#.item<c10::complex<double>>()".}

# Bounds checking for raw tensors
func check_index*(t: TorchTensor, idx: varargs[int]) {.inline.} =
  ## Check raw tensor indexing bounds
  when compileOption("boundChecks"):
    let ndim = t.ndimension
    if unlikely(idx.len != ndim):
      raise newException(
        IndexDefect,
        "Error Out-of-bounds access." & " Index must match Tensor rank! Expected: " & $ndim & ", got: " & $(idx.len) &
          " elements",
      )
    let sizes = t.sizes()
    for i in 0 ..< idx.len:
      let dim_size: int64 = getAt(sizes, i)
      if unlikely(not (0 <= idx[i] and idx[i] < dim_size)):
        raise newException(
          IndexDefect,
          "Error Out-of-bounds access." & " Index [" & $idx & "] " & " must be in range of Tensor dimensions " &
            $t.sizes(),
        )

# Unsure what those corresponds to in Python
# func `[]`*(self: Tensor, index: Scalar): Tensor {.importcpp: "#[#]".}
# func `[]`*(self: Tensor, index: Tensor): Tensor {.importcpp: "#[#]".}
# func `[]`*(self: Tensor, index: int64): Tensor {.importcpp: "#[#]".}

func index*(self: TorchTensor): TorchTensor {.varargs, importcpp: "#.index({@})".}
  ## Tensor indexing. It is recommended
  ## to Nimify this in a high-level wrapper.
  ## `tensor.index(indexers)`

# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*(self: var TorchTensor, i0: auto, val: Scalar or TorchTensor) {.importcpp: "#.index_put_({#}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(self: var TorchTensor, i0, i1: auto, val: Scalar or TorchTensor) {.importcpp: "#.index_put_({#, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(
  self: var TorchTensor, i0, i1, i2: auto, val: Scalar or TorchTensor
) {.importcpp: "#.index_put_({#, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(
  self: var TorchTensor, i0, i1, i2, i3: auto, val: Scalar or TorchTensor
) {.importcpp: "#.index_put_({#, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(
  self: var TorchTensor, i0, i1, i2, i3, i4: auto, val: Scalar or TorchTensor
) {.importcpp: "#.index_put_({#, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(
  self: var TorchTensor, i0, i1, i2, i3, i4, i5: auto, val: Scalar or TorchTensor
) {.importcpp: "#.index_put_({#, #, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.

# Fancy Indexing
# -----------------------------------------------------------------------
# TODO -> separate the FFI from the Nim Raw API to add IndexDefect when compileOptions("boundsCheck")
func index_select*(self: TorchTensor, axis: int64, indices: TorchTensor): TorchTensor {.importcpp: "#.index_select(@)".}
func masked_select*(self: TorchTensor, mask: TorchTensor): TorchTensor {.importcpp: "#.masked_select(@)".}

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.

func index_fill_mut*(self: var TorchTensor, mask: TorchTensor, value: Scalar or TorchTensor) {.importcpp: "#.index_fill_(@)".}
func masked_fill_mut*(
  self: var TorchTensor, mask: TorchTensor, value: Scalar or TorchTensor
) {.importcpp: "#.masked_fill_(@)".}

# Shapeshifting
# -----------------------------------------------------------------------

func reshape*(self: TorchTensor, sizes: IntArrayRef): TorchTensor {.importcpp: "#.reshape({@})".}
func view*(self: TorchTensor, size: IntArrayRef): TorchTensor {.importcpp: "#.reshape({@})".}

func transpose*(self: TorchTensor, dim0, dim1: int64): TorchTensor {.importcpp: "#.transpose(@)".}
  ## Swaps two dimensions. Returns a tensor that is a transposed version of input.
  ## The given dimensions dim0 and dim1 are swapped.

func t*(self: TorchTensor): TorchTensor {.importcpp: "#.t()".}
  ## Transposes a 2D tensor. Equivalent to transpose(0, 1).
  ## This function is only supported for 2D tensors.

func permute*(self: TorchTensor, dims: IntArrayRef): TorchTensor {.importcpp: "#.permute(@)".}
  ## Returns a view of the original tensor with its dimensions permuted.

# Automatic Differentiation
# -----------------------------------------------------------------------

func backward*(self: var TorchTensor) {.importcpp: "#.backward()".}
func detach*(self: TorchTensor): TorchTensor {.importcpp: "#.detach()".}
  ## Detach tensor from computation graph (stop gradient tracking).
  ##
  ## Returns a new tensor that shares the same storage but is completely
  ## disconnected from the autograd history. The returned tensor will never
  ## require gradients, even if the input tensor did.
  ##
  ## Use cases:
  ## - Inference/evaluation: prevent gradient computation to save memory
  ## - Break gradient flow: stop backpropagation at specific points
  ## - Mix requires_grad tensors: safely use a trained tensor without gradients
  ##
  ## Example:
  ##   var x = randn(@[3, 3])  # Training tensor with gradients
  ##   let y = x.detach()       # Same data, no gradients tracked
  ##   # Operations on y won't affect x's gradient computation

# Low-level slicing API
# -----------------------------------------------------------------------

type
  TorchSlice* {.importcpp: "torch::indexing::Slice", bycopy.} = object
  # libtorch/include/ATen/TensorIndexing.h
  TensorIndexType* {.size: sizeof(cint), bycopy, importcpp: "torch::indexing::TensorIndexType".} = enum
    ## This is passed to torchSlice functions
    IndexNone = 0
    IndexEllipsis = 1
    IndexInteger = 2
    IndexBoolean = 3
    IndexSlice = 4
    IndexTensor = 5

# The None used in Torch isn't actually the enum but a c10::nullopt
let None* {.importcpp: "torch::indexing::None".}: Nullopt_t

type EllipsisIndexType* {.importcpp: "torch::indexing::EllipsisIndexType".} = object

let Ellipsis* {.importcpp: "torch::indexing::Ellipsis".}: EllipsisIndexType # SomeSlicer* = TensorIndexType|SomeSignedInt

proc SliceSpan*(): TorchSlice {.importcpp: "at::indexing::Slice()".}
  ## This is passed to the "index" function
  ## This is Python ":", span / whole dimension

func torchSlice*() {.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(start: Nullopt_t | SomeSignedInt): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(
  start: Nullopt_t | SomeSignedInt, stop: Nullopt_t | SomeSignedInt
): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}
func torchSlice*(
  start: Nullopt_t | SomeSignedInt, stop: Nullopt_t | SomeSignedInt, step: Nullopt_t | SomeSignedInt
): TorchSlice {.importcpp: "torch::indexing::Slice(@)", constructor.}

func start*(s: TorchSlice): int64 {.importcpp: "#.start()".}
func stop*(s: TorchSlice): int64 {.importcpp: "#.stop()".}
func step*(s: TorchSlice): int64 {.importcpp: "#.step()".}

# Operators. We expose PyTorch convention for `div` and `mod` instead of Nim's
# -----------------------------------------------------------------------
func assign*(self: var TorchTensor, other: TorchTensor) {.importcpp: "# = #".}

func `not`*(self: TorchTensor): TorchTensor {.importcpp: "(~#)".}
func `-`*(self: TorchTensor): TorchTensor {.importcpp: "(-#)".}

func `+`*(self: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "(# + #)".}
func `-`*(self: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "(# - #)".}
func `*`*(self: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "(# * #)".}
func `%`*(self: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "#.remainder(#)".}

func `*`*(a: SomeNumber, b: TorchTensor): TorchTensor {.importcpp: "(# * #)".}
func `*`*(self: TorchTensor, b: SomeNumber): TorchTensor {.importcpp: "(# * #)".}

func `+`*(a: SomeNumber, b: TorchTensor): TorchTensor {.importcpp: "(# + #)".}
func `+`*(self: TorchTensor, b: SomeNumber): TorchTensor {.importcpp: "(# + #)".}

func `%`*(a: SomeNumber, b: TorchTensor): TorchTensor {.importcpp: "#.remainder(#)".}
func `%`*(self: TorchTensor, b: SomeNumber): TorchTensor {.importcpp: "#.remainder(#)".}

func `+=`*(self: var TorchTensor, b: TorchTensor) {.importcpp: "(# += #)".}
func `+=`*(self: var TorchTensor, s: Scalar) {.importcpp: "(# += #)".}
func `-=`*(self: var TorchTensor, b: TorchTensor) {.importcpp: "(# -= #)".}
func `-=`*(self: var TorchTensor, s: Scalar) {.importcpp: "(# -= #)".}
func `*=`*(self: var TorchTensor, b: TorchTensor) {.importcpp: "(# *= #)".}
func `*=`*(self: var TorchTensor, s: Scalar) {.importcpp: "(# *= #)".}
func `/=`*(self: var TorchTensor, b: TorchTensor) {.importcpp: "(# /= #)".}
func `/=`*(self: var TorchTensor, s: Scalar) {.importcpp: "(# /= #)".}

func `and`*(self: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "#.bitwise_and(#)".} ## bitwise `and`.
func `or`*(self: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "#.bitwise_or(#)".} ## bitwise `or`.
func `xor`*(self: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "#.bitwise_xor(#)".} ## bitwise `xor`.

func bitand_mut*(self: var TorchTensor, s: TorchTensor) {.importcpp: "#.bitwise_and_(#)".} ## In-place bitwise `and`.
func bitor_mut*(self: var TorchTensor, s: TorchTensor) {.importcpp: "#.bitwise_or_(#)".} ## In-place bitwise `or`.
func bitxor_mut*(self: var TorchTensor, s: TorchTensor) {.importcpp: "#.bitwise_xor_(#)".} ## In-place bitwise `xor`.

func eq*(a, b: TorchTensor): TorchTensor {.importcpp: "#.eq(#)".} ## Equality of each tensor values
func equal*(a, b: TorchTensor): bool {.importcpp: "#.equal(#)".}
template `==`*(a, b: TorchTensor): bool =
  a.equal(b)

# Functions.h
# -----------------------------------------------------------------------

func contiguous*(self: TorchTensor): TorchTensor {.importcpp: "#.contiguous(@)".}
func toType*(self: TorchTensor, dtype: ScalarKind): TorchTensor {.importcpp: "#.toType(@)".}
func toSparse*(self: TorchTensor): TorchTensor {.importcpp: "#.to_sparse()".}
func toSparse*(self: TorchTensor, sparseDim: int64): TorchTensor {.importcpp: "#.to_sparse(@)".}

func eye*(n: int64): TorchTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, options: TensorOptions): TorchTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int64, device: DeviceKind): TorchTensor {.importcpp: "torch::eye(@)".}

func zeros*(dim: int64): TorchTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef): TorchTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, options: TensorOptions): TorchTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, device: DeviceKind): TorchTensor {.importcpp: "torch::zeros(@)".}

func ones*(dim: int64): TorchTensor {.importcpp: "torch::ones(@)".} ## Create a tensor filled with ones (scalar value 1)
func ones*(dim: IntArrayRef): TorchTensor {.importcpp: "torch::ones(@)".}
  ## Create a tensor filled with ones (scalar value 1)
func ones*(dim: IntArrayRef, options: TensorOptions): TorchTensor {.importcpp: "torch::ones(@)".}
  ## Create a tensor filled with ones with specific dtype/device options
func ones*(dim: IntArrayRef, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::ones(@)".}
  ## Create a tensor filled with ones with specific dtype
func ones*(dim: IntArrayRef, device: DeviceKind): TorchTensor {.importcpp: "torch::ones(@)".}
  ## Create a tensor filled with ones on specific device (CPU/CUDA)

func full*(size: IntArrayRef, fill_value: Scalar): TorchTensor {.importcpp: "torch::full(@)".}
  ## Create a tensor filled with a specific value.
  ## Useful for creating constant tensors (e.g., all 3.14, all -1, etc.)
func full*(size: IntArrayRef, fill_value: Scalar, options: TensorOptions): TorchTensor {.importcpp: "torch::full(@)".}
  ## Create a tensor filled with a specific value with dtype/device options
func full*(size: IntArrayRef, fill_value: Scalar, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::full(@)".}
  ## Create a tensor filled with a specific value with specific dtype
func full*(size: IntArrayRef, fill_value: Scalar, device: DeviceKind): TorchTensor {.importcpp: "torch::full(@)".}
  ## Create a tensor filled with a specific value on specific device

func randn*(size: IntArrayRef): TorchTensor {.importcpp: "torch::randn(@)".}
  ## Create a tensor with values from standard normal distribution (mean=0, std=1).
  ## Critical for neural network weight initialization (Xavier/He initialization).
  ## Values are sampled from N(0,1) independently for each element.
func randn*(size: IntArrayRef, options: TensorOptions): TorchTensor {.importcpp: "torch::randn(@)".}
  ## Create a normal random tensor with specific dtype/device options
func randn*(size: IntArrayRef, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::randn(@)".}
  ## Create a normal random tensor with specific dtype
func randn*(size: IntArrayRef, device: DeviceKind): TorchTensor {.importcpp: "torch::randn(@)".}
  ## Create a normal random tensor on specific device (CPU/CUDA)

func linspace*(start, stop: Scalar, steps: int64, options: TensorOptions): TorchTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: ScalarKind): TorchTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: DeviceKind): TorchTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64, options: Device): TorchTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int64): TorchTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar): TorchTensor {.importcpp: "torch::linspace(@)".}

func logspace*(
  start, stop: Scalar, steps, base: int64, options: TensorOptions
): TorchTensor {.importcpp: "torch::logspace(@)".}
func logspace*(
  start, stop: Scalar, steps, base: int64, options: ScalarKind
): TorchTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: DeviceKind) {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64, options: Device): TorchTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int64): TorchTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps: int64): TorchTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar): TorchTensor {.importcpp: "torch::logspace(@)".}

func arange*(stop: Scalar, options: TensorOptions): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: ScalarKind): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: DeviceKind): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar, options: Device): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(stop: Scalar): TorchTensor {.importcpp: "torch::arange(@)".}

func arange*(start, stop: Scalar, options: TensorOptions): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: ScalarKind): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: DeviceKind): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar, options: Device): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop: Scalar): TorchTensor {.importcpp: "torch::arange(@)".}

func arange*(start, stop, step: Scalar, options: TensorOptions): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: ScalarKind): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: DeviceKind): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar, options: Device): TorchTensor {.importcpp: "torch::arange(@)".}
func arange*(start, stop, step: Scalar): TorchTensor {.importcpp: "torch::arange(@)".}

# Operations
# -----------------------------------------------------------------------
func add*(self: TorchTensor, other: TorchTensor, alpha: Scalar = 1): TorchTensor {.importcpp: "#.add(@)".}
func add*(self: TorchTensor, other: Scalar, alpha: Scalar = 1): TorchTensor {.importcpp: "#.add(@)".}
func addmv*(
  self: TorchTensor, mat: TorchTensor, vec: TorchTensor, beta: Scalar = 1, alpha: Scalar = 1
): TorchTensor {.importcpp: "#.addmv(@)".}
func addmm*(t, mat1, mat2: TorchTensor, beta: Scalar = 1, alpha: Scalar = 1): TorchTensor {.importcpp: "#.addmm(@)".}
func mm*(t, other: TorchTensor): TorchTensor {.importcpp: "#.mm(@)".}
func matmul*(t, other: TorchTensor): TorchTensor {.importcpp: "#.matmul(@)".}
func bmm*(t, other: TorchTensor): TorchTensor {.importcpp: "#.bmm(@)".}

func luSolve*(t, data, pivots: TorchTensor): TorchTensor {.importcpp: "#.lu_solve(@)".}

func qr*(self: TorchTensor, some: bool = true): CppTuple2[TorchTensor, TorchTensor] {.importcpp: "#.qr(@)".}
  ## Returns a tuple:
  ## - Q of shape (∗,m,k)
  ## - R of shape (∗,k,n)
  ## with k=min(m,n) if some is true otherwise k=m
  ##
  ## The QR decomposition is batched over dimension(s) *
  ## t = QR

# addr?
func all*(self: TorchTensor, axis: int64): TorchTensor {.importcpp: "#.all(@)".}
func all*(self: TorchTensor, axis: int64, keepdim: bool): TorchTensor {.importcpp: "#.all(@)".}
func allClose*(
  t, other: TorchTensor, rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false
): bool {.importcpp: "#.allclose(@)".}
func any*(self: TorchTensor, axis: int64): TorchTensor {.importcpp: "#.any(@)".}
func any*(self: TorchTensor, axis: int64, keepdim: bool): TorchTensor {.importcpp: "#.any(@)".}
func argmax*(self: TorchTensor): TorchTensor {.importcpp: "#.argmax()".}
func argmax*(self: TorchTensor, axis: int64, keepdim: bool = false): TorchTensor {.importcpp: "#.argmax(@)".}
func argmin*(self: TorchTensor): TorchTensor {.importcpp: "#.argmin()".}
func argmin*(self: TorchTensor, axis: int64, keepdim: bool = false): TorchTensor {.importcpp: "#.argmin(@)".}

# aggregate
# -----------------------------------------------------------------------

# sum needs wrapper procs/templates to allow for using nim arrays and single axis.
func sum*(self: TorchTensor): TorchTensor {.importcpp: "#.sum()".}
func sum*(self: TorchTensor, dtype: ScalarKind): TorchTensor {.importcpp: "#.sum(@)".}
func sum*(self: TorchTensor, axis: int64, keepdim: bool = false): TorchTensor {.importcpp: "#.sum(@)".}
func sum*(self: TorchTensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): TorchTensor {.importcpp: "#.sum(@)".}
func sum*(self: TorchTensor, axis: IntArrayRef, keepdim: bool = false): TorchTensor {.importcpp: "#.sum(@)".}
func sum*(
  self: TorchTensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind
): TorchTensor {.importcpp: "#.sum(@)".}

# mean as well
func mean*(self: TorchTensor): TorchTensor {.importcpp: "#.mean()".}
func mean*(self: TorchTensor, dtype: ScalarKind): TorchTensor {.importcpp: "#.mean(@)".}
func mean*(self: TorchTensor, axis: int64, keepdim: bool = false): TorchTensor {.importcpp: "#.mean(@)".}
func mean*(self: TorchTensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): TorchTensor {.importcpp: "#.mean(@)".}
func mean*(self: TorchTensor, axis: IntArrayRef, keepdim: bool = false): TorchTensor {.importcpp: "#.mean(@)".}
func mean*(
  self: TorchTensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind
): TorchTensor {.importcpp: "#.mean(@)".}

# median requires std::tuple

func prod*(self: TorchTensor): TorchTensor {.importcpp: "#.prod()".}
func prod*(self: TorchTensor, dtype: ScalarKind): TorchTensor {.importcpp: "#.prod(@)".}
func prod*(self: TorchTensor, axis: int64, keepdim: bool = false): TorchTensor {.importcpp: "#.prod(@)".}
func prod*(self: TorchTensor, axis: int64, keepdim: bool = false, dtype: ScalarKind): TorchTensor {.importcpp: "#.prod(@)".}

func min*(self: TorchTensor): TorchTensor {.importcpp: "#.min()".}
func min*(
  self: TorchTensor, axis: int64, keepdim: bool = false
): CppTuple2[TorchTensor, TorchTensor] {.importcpp: "torch::min(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the minimum values and their index in the specified axis

func max*(self: TorchTensor): TorchTensor {.importcpp: "#.max()".}
func max*(
  self: TorchTensor, axis: int64, keepdim: bool = false
): CppTuple2[TorchTensor, TorchTensor] {.importcpp: "torch::max(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the maximum values and their index in the specified axis

func variance*(self: TorchTensor, unbiased: bool = true): TorchTensor {.importcpp: "#.var(@)".}
  # can't use `var` because of keyword.
func variance*(
  self: TorchTensor, axis: int64, unbiased: bool = true, keepdim: bool = false
): TorchTensor {.importcpp: "#.var(@)".}
func variance*(
  self: TorchTensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false
): TorchTensor {.importcpp: "#.var(@)".}

func stddev*(self: TorchTensor, unbiased: bool = true): TorchTensor {.importcpp: "#.std(@)".}
func stddev*(
  self: TorchTensor, axis: int64, unbiased: bool = true, keepdim: bool = false
): TorchTensor {.importcpp: "#.std(@)".}
func stddev*(
  self: TorchTensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false
): TorchTensor {.importcpp: "#.std(@)".}

# algorithms:
# -----------------------------------------------------------------------
func sort*(
  self: TorchTensor, axis: int64 = -1, descending: bool = false
): CppTuple2[TorchTensor, TorchTensor] {.importcpp: "#.sort(@)".}
  ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
  ## If dim is not given, the last dimension of the input is chosen (dim=-1).
  ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
  ## where originalIndices is the original index of each values (before sorting)
func argsort*(self: TorchTensor, axis: int64 = -1, descending: bool = false): TorchTensor {.importcpp: "#.argsort(@)".}

func cat*(tensors: ArrayRef[TorchTensor], axis: int64 = 0): TorchTensor {.importcpp: "torch::cat(@)".}
func stack*(tensors: ArrayRef[TorchTensor], dim: int64 = 0): TorchTensor {.importcpp: "torch::stack(@)".}
  ## Stack tensors along a NEW dimension (unlike cat which concatenates along existing dim).
  ## All tensors must have the same shape.
  ## Example: stack([2x3, 2x3, 2x3], dim=0) -> 3x2x3
  ##          stack([2x3, 2x3], dim=1) -> 2x2x3
func flip*(self: TorchTensor, dims: IntArrayRef): TorchTensor {.importcpp: "#.flip(@)".}

# math
# -----------------------------------------------------------------------
func abs*(self: TorchTensor): TorchTensor {.importcpp: "#.abs()".}
func absolute*(self: TorchTensor): TorchTensor {.importcpp: "#.absolute()".}
func angle*(self: TorchTensor): TorchTensor {.importcpp: "#.angle()".}
func sgn*(self: TorchTensor): TorchTensor {.importcpp: "#.sgn()".}
func conj*(self: TorchTensor): TorchTensor {.importcpp: "#.conj()".}
func acos*(self: TorchTensor): TorchTensor {.importcpp: "#.acos()".}
func arccos*(self: TorchTensor): TorchTensor {.importcpp: "#.arccos()".}
func acosh*(self: TorchTensor): TorchTensor {.importcpp: "#.acosh()".}
func arccosh*(self: TorchTensor): TorchTensor {.importcpp: "#.arccosh()".}
func asinh*(self: TorchTensor): TorchTensor {.importcpp: "#.asinh()".}
func arcsinh*(self: TorchTensor): TorchTensor {.importcpp: "#.arcsinh()".}
func atanh*(self: TorchTensor): TorchTensor {.importcpp: "#.atanh()".}
func arctanh*(self: TorchTensor): TorchTensor {.importcpp: "#.arctanh()".}
func asin*(self: TorchTensor): TorchTensor {.importcpp: "#.asin()".}
func arcsin*(self: TorchTensor): TorchTensor {.importcpp: "#.arcsin()".}
func atan*(self: TorchTensor): TorchTensor {.importcpp: "#.atan()".}
func arctan*(self: TorchTensor): TorchTensor {.importcpp: "#.arctan()".}
func cos*(self: TorchTensor): TorchTensor {.importcpp: "#.cos()".}
func sin*(self: TorchTensor): TorchTensor {.importcpp: "#.sin()".}
func tan*(self: TorchTensor): TorchTensor {.importcpp: "#.tan()".}
func exp*(self: TorchTensor): TorchTensor {.importcpp: "#.exp()".}
func exp2*(self: TorchTensor): TorchTensor {.importcpp: "#.exp2()".}
func log*(self: TorchTensor): TorchTensor {.importcpp: "#.log()".}
  ## Natural logarithm (base e). log(exp(x)) = x
  ## Returns NaN for negative inputs, -Inf for 0
func log2*(self: TorchTensor): TorchTensor {.importcpp: "#.log2()".}
  ## Base-2 logarithm. Useful for information theory (entropy, bits)
func log10*(self: TorchTensor): TorchTensor {.importcpp: "#.log10()".}
  ## Base-10 logarithm. Useful for decibels and scientific notation
func erf*(self: TorchTensor): TorchTensor {.importcpp: "#.erf()".}
func erfc*(self: TorchTensor): TorchTensor {.importcpp: "#.erfc()".}
func reciprocal*(self: TorchTensor): TorchTensor {.importcpp: "#.reciprocal()".}
func neg*(self: TorchTensor): TorchTensor {.importcpp: "#.neg()".}
func clamp*(self: TorchTensor, min, max: Scalar): TorchTensor {.importcpp: "#.clamp(@)".}
func clampMin*(self: TorchTensor, min: Scalar): TorchTensor {.importcpp: "#.clamp_min(@)".}
func clampMax*(self: TorchTensor, max: Scalar): TorchTensor {.importcpp: "#.clamp_max(@)".}

func dot*(self: TorchTensor, other: TorchTensor): TorchTensor {.importcpp: "#.dot(@)".}

func squeeze*(self: TorchTensor): TorchTensor {.importcpp: "#.squeeze()".}
func squeeze*(self: TorchTensor, axis: int64): TorchTensor {.importcpp: "#.squeeze(@)".}
func unsqueeze*(self: TorchTensor, axis: int64): TorchTensor {.importcpp: "#.unsqueeze(@)".}
func square*(self: TorchTensor): TorchTensor {.importcpp: "#.square()".}
func sqrt*(self: TorchTensor): TorchTensor {.importcpp: "#.sqrt()".}
func pow*(self: TorchTensor, exponent: TorchTensor): TorchTensor {.importcpp: "#.pow(@)".}
func pow*(self: TorchTensor, exponent: Scalar): TorchTensor {.importcpp: "#.pow(@)".}

# FFT
# -----------------------------------------------------------------------
func fftshift*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_fftshift(@)".}
func fftshift*(self: TorchTensor, dim: IntArrayRef): TorchTensor {.importcpp: "torch::fft_ifftshift(@)".}
func ifftshift*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_fftshift(@)".}
func ifftshift*(self: TorchTensor, dim: IntArrayRef): TorchTensor {.importcpp: "torch::fft_ifftshift(@)".}

func fft*(self: TorchTensor, n: int64, dim: int64, norm: CppString): TorchTensor {.importcpp: "torch::fft_fft(@)".}
func fft*(self: TorchTensor, n: int64, dim: int64 = -1): TorchTensor {.importcpp: "torch::fft_fft(@)".}
## Compute the 1-D Fourier transform
## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
func fft*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_fft(@)".}
## Compute the 1-D Fourier transform

func ifft*(self: TorchTensor, n: int64, dim: int64 = -1, norm: CppString): TorchTensor {.importcpp: "torch::fft_ifft(@)".}
## Compute the 1-D Fourier transform
## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
func ifft*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_ifft(@)".}
## Compute the 1-D Fourier transform

func fft2*(
  self: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func fft2*(self: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func fft2*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform

func ifft2*(
  self: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func ifft2*(self: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func ifft2*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform

func fftn*(
  self: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func fftn*(self: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func fftn*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform

func ifftn*(
  self: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func ifftn*(self: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func ifftn*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform

func rfft*(self: TorchTensor, n: int64, dim: int64 = -1, norm: CppString): TorchTensor {.importcpp: "torch::fft_rfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.
func rfft*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_rfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.

func irfft*(self: TorchTensor, n: int64, dim: int64 = -1, norm: CppString): TorchTensor {.importcpp: "torch::fft_irfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.
func irfft*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_irfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.

func rfft2*(
  self: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func rfft2*(self: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func rfft2*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform

func irfft2*(
  self: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func irfft2*(self: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func irfft2*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform

func rfftn*(
  self: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func rfftn*(self: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func rfftn*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform

func irfftn*(
  self: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func irfftn*(self: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func irfftn*(self: TorchTensor): TorchTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform

func hfft*(self: TorchTensor, n: int64, dim: int64 = -1, norm: CppString): TorchTensor {.importcpp: "torch::hfft(@)".}
## Computes the 1 dimensional FFT of a onesided Hermitian signal.
func hfft*(self: TorchTensor): TorchTensor {.importcpp: "torch::hfft(@)".}
## Computes the 1 dimensional FFT of a onesided Hermitian signal.
func ihfft*(self: TorchTensor, n: int64, dim: int64 = -1, norm: CppString): TorchTensor {.importcpp: "torch::ihfft(@)".}
## Computes the inverse FFT of a real-valued Fourier domain signal.
func ihfft*(self: TorchTensor): TorchTensor {.importcpp: "torch::ihfft(@)".}
## Computes the inverse FFT of a real-valued Fourier domain signal.

#func convolution*(self: Tensor, weight: Tensor, bias: Tensor, stride, padding, dilation: int64, transposed: bool, outputPadding: int64, groups: int64): Tensor {.importcpp: "torch::convolution(@)".}

{.pop.}
