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
  workspace/libtorch/vendor/libtorch,
  workspace/libtorch/src/abi/c10,
  workspace/libtorch/src/abi/std_cpp

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

# Torch returns int64, but Nim uses int. It's more convenient
# to pass int everywhere since Torch only supports 64-bit OSes
static: doAssert sizeof(int) == 8, "Only 64-bit OSes are supported"


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

proc print*(a: TorchTensor) {.sideeffect, importcpp: "torch::print(@)".}

# Metadata
# -----------------------------------------------------------------------

func dim*(a: TorchTensor): int {.importcpp: "#.dim()".} ## Number of dimensions
func reset*(a: var TorchTensor) {.importcpp: "#.reset()".}
func is_same*(self, other: TorchTensor): bool {.importcpp: "#.is_same(#)".}
  ## Reference equality
  ## Do the tensors use the same memory.

func sizes*(a: TorchTensor): IntArrayRef {.importcpp: "#.sizes()".} ## This is Arraymancer and Numpy "shape"

func strides*(a: TorchTensor): IntArrayRef {.importcpp: "#.strides()".}

func ndimension*(a: TorchTensor): int {.importcpp: "#.ndimension()".} ## This is Arraymancer rank
func nbytes*(a: TorchTensor): uint {.importcpp: "#.nbytes()".} ## Bytes-size of the Tensor
func numel*(a: TorchTensor): int {.importcpp: "#.numel()".} ## This is Arraymancer and Numpy "size"

func size*(a: TorchTensor, axis: int): int {.importcpp: "#.size(#)".}
func defined*(a: TorchTensor): bool {.importcpp: "#.defined()".}
func itemsize*(a: TorchTensor): uint {.importcpp: "#.itemsize()".}
func element_size*(a: TorchTensor): int {.importcpp: "#.element_size()".}

# Accessors
# -----------------------------------------------------------------------

func data_ptr*(a: TorchTensor, T: typedesc): ptr UncheckedArray[T] {.importcpp: "#.data_ptr<'2>(#)".}
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

func has_storage*(a: TorchTensor): bool {.importcpp: "#.has_storage()".}
func get_device*(a: TorchTensor): int {.importcpp: "#.get_device()".}
func is_cuda*(a: TorchTensor): bool {.importcpp: "#.is_cuda()".}
func is_hip*(a: TorchTensor): bool {.importcpp: "#.is_hip()".}
func is_sparse*(a: TorchTensor): bool {.importcpp: "#.is_sparse()".}
func is_mkldnn*(a: TorchTensor): bool {.importcpp: "#.is_mkldnn()".}
func is_vulkan*(a: TorchTensor): bool {.importcpp: "#.is_vulkan()".}
func is_quantized*(a: TorchTensor): bool {.importcpp: "#.is_quantized()".}
func is_meta*(a: TorchTensor): bool {.importcpp: "#.is_meta()".}

func cpu*(a: TorchTensor): TorchTensor {.importcpp: "#.cpu()".}
func cuda*(a: TorchTensor): TorchTensor {.importcpp: "#.cuda()".}
func hip*(a: TorchTensor): TorchTensor {.importcpp: "#.hip()".}
func vulkan*(a: TorchTensor): TorchTensor {.importcpp: "#.vulkan()".}
func to*(a: TorchTensor, device: DeviceKind): TorchTensor {.importcpp: "#.to(#)".}
func to*(a: TorchTensor, device: Device): TorchTensor {.importcpp: "#.to(#)".}

# dtype
# -----------------------------------------------------------------------

func to*(a: TorchTensor, dtype: ScalarKind): TorchTensor {.importcpp: "#.to(#)".}
func scalarType*(a: TorchTensor): ScalarKind {.importcpp: "#.scalar_type()".}

# Constructors
# -----------------------------------------------------------------------

# Note: DeviceType and ScalarType are auto-convertible to TensorOptions

# We don't expose a default empty constructor, they cause ICE "Error: internal error: expr(skType); unknown symbol"
# And they are completely unidiomatic.
#
# I'm unsure about a Move / Copy constructor since torch::Tensor uses intrusive reference counting.

func from_blob*(
  data: pointer, sizes: IntArrayRef, options: TensorOptions
): TorchTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(
  data: pointer, sizes: IntArrayRef, scalarKind: ScalarKind
): TorchTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: IntArrayRef, device: DeviceKind): TorchTensor {.importcpp: "torch::from_blob(@)".}

func from_blob*(data: pointer, sizes: int, options: TensorOptions): TorchTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::from_blob(@)".}
func from_blob*(data: pointer, sizes: int, device: DeviceKind): TorchTensor {.importcpp: "torch::from_blob(@)".}

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

func clone*(a: TorchTensor): TorchTensor {.importcpp: "#.clone()".}

# TODO : Test this
func view_as_real*(a: TorchTensor): TorchTensor {.importcpp: "#.view_as_real()".}
func view_as_complex*(a: TorchTensor): TorchTensor {.importcpp: "#.view_as_complex()".}

# Random sampling
# -----------------------------------------------------------------------
func random_mut*(a: var TorchTensor, start, stopEx: int) {.importcpp: "#.random_(@)".}
func randint*(start, stopEx: int): TorchTensor {.varargs, importcpp: "torch::randint(#, #, {@})".}
func randint*(start, stopEx: int, size: IntArrayRef): TorchTensor {.importcpp: "torch::randint(@)".}

func rand_like*(a: TorchTensor, options: TensorOptions): TorchTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(a: TorchTensor, options: ScalarKind): TorchTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(a: TorchTensor, options: DeviceKind): TorchTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(a: TorchTensor, options: Device): TorchTensor {.importcpp: "torch::rand_like(@)".}
func rand_like*(a: TorchTensor): TorchTensor {.importcpp: "torch::rand_like(@)".}

func rand*(size: IntArrayRef, options: TensorOptions): TorchTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef, options: DeviceKind): TorchTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef, options: Device): TorchTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef, options: ScalarKind): TorchTensor {.importcpp: "torch::rand(@)".}
func rand*(size: IntArrayRef): TorchTensor {.importcpp: "torch::rand(@)".}

# Indexing
# -----------------------------------------------------------------------
# libtorch/include/ATen/TensorIndexing.h
# and https://pytorch.org/cppdocs/notes/tensor_indexing.html

func item*(a: TorchTensor, T: typedesc): T {.importcpp: "#.item<'0>()".}
  ## Extract the scalar from a 0-dimensional tensor
func item*(a: TorchTensor, T: typedesc[Complex32]): TorchComplex[float32] {.importcpp: "#.item<c10::complex<float>>()".}
func item*(
  a: TorchTensor, T: typedesc[Complex64]
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
      let dim_size: int = getAt(sizes, i)
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

func index*(a: TorchTensor): TorchTensor {.varargs, importcpp: "#.index({@})".}
  ## Tensor indexing. It is recommended
  ## to Nimify this in a high-level wrapper.
  ## `tensor.index(indexers)`

# We can't use the construct `#.index_put_({@}, #)`
# so hardcode sizes,
# 6d seems reasonable, that would be a batch of 3D videos (videoID/batchID, Time, Color Channel, Height, Width, Depth)
# If you need more you likely aren't indexing individual values.

func index_put*(a: var TorchTensor, i0: auto, val: Scalar or TorchTensor) {.importcpp: "#.index_put_({#}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(a: var TorchTensor, i0, i1: auto, val: Scalar or TorchTensor) {.importcpp: "#.index_put_({#, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(
  a: var TorchTensor, i0, i1, i2: auto, val: Scalar or TorchTensor
) {.importcpp: "#.index_put_({#, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(
  a: var TorchTensor, i0, i1, i2, i3: auto, val: Scalar or TorchTensor
) {.importcpp: "#.index_put_({#, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(
  a: var TorchTensor, i0, i1, i2, i3, i4: auto, val: Scalar or TorchTensor
) {.importcpp: "#.index_put_({#, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.
func index_put*(
  a: var TorchTensor, i0, i1, i2, i3, i4, i5: auto, val: Scalar or TorchTensor
) {.importcpp: "#.index_put_({#, #, #, #, #, #}, #)".}
  ## Tensor mutation at index. It is recommended
  ## to Nimify this in a high-level wrapper.

# Fancy Indexing
# -----------------------------------------------------------------------
# TODO -> separate the FFI from the Nim Raw API to add IndexDefect when compileOptions("boundsCheck")
func index_select*(a: TorchTensor, axis: int, indices: TorchTensor): TorchTensor {.importcpp: "#.index_select(@)".}
func masked_select*(a: TorchTensor, mask: TorchTensor): TorchTensor {.importcpp: "#.masked_select(@)".}

# PyTorch exposes in-place `index_fill_` and `masked_fill_`
# and out-of-place `index_fill` and `masked_fill`
# that does in-place + clone
# we only exposes the in-place version.

func index_fill_mut*(a: var TorchTensor, mask: TorchTensor, value: Scalar or TorchTensor) {.importcpp: "#.index_fill_(@)".}
func masked_fill_mut*(
  a: var TorchTensor, mask: TorchTensor, value: Scalar or TorchTensor
) {.importcpp: "#.masked_fill_(@)".}

# Shapeshifting
# -----------------------------------------------------------------------

func reshape*(a: TorchTensor, sizes: IntArrayRef): TorchTensor {.importcpp: "#.reshape({@})".}
func view*(a: TorchTensor, size: IntArrayRef): TorchTensor {.importcpp: "#.reshape({@})".}

func transpose*(a: TorchTensor, dim0, dim1: int64): TorchTensor {.importcpp: "#.transpose(@)".}
  ## Swaps two dimensions. Returns a tensor that is a transposed version of input.
  ## The given dimensions dim0 and dim1 are swapped.

func repeat_interleave*(a: TorchTensor, repeats: int, dim: int = -1): TorchTensor {.importcpp: "at::repeat_interleave(@)".}
  ## Repeats elements along a dimension.
  ##
  ## C++ signature:
  ##   inline at::Tensor at::repeat_interleave(
    ##     const at::Tensor &self,
    ##     int64_t repeats,
    ##     ::std::optional<int64_t> dim = ::std::nullopt
    ##   )

func t*(a: TorchTensor): TorchTensor {.importcpp: "#.t()".}
  ## Transposes a 2D tensor. Equivalent to transpose(0, 1).
  ## This function is only supported for 2D tensors.

func permute*(a: TorchTensor, dims: IntArrayRef): TorchTensor {.importcpp: "#.permute(@)".}
  ## Returns a view of the original tensor with its dimensions permuted.

func narrow*(a: TorchTensor, dim: int, start: int, length: int): TorchTensor {.importcpp: "#.narrow(@)".}
  ## Slices the tensor at the given dimension.
  ## Equivalent to PyTorch's narrow operation.

func chunk*(a: TorchTensor, chunks: int, dim: int = 0): CppVector[TorchTensor] {.importcpp: "at::chunk(@)".}
  ## Splits the tensor into chunks along a given dimension.
  ##
  ## C++ signature:
  ##   inline ::std::vector<at::Tensor> at::chunk(
  ##     const at::Tensor &self,
  ##     int64_t chunks,
  ##     int64_t dim = 0
  ##   )
  ##
  ## Each chunk will be a view of the original tensor.
  ## If the tensor size along the given dimension is not divisible by chunks,
  ## the last chunk will be smaller.
  ##
  ## Returns a vector of tensors.

func unbind*(a: TorchTensor, dim: int = 0): CppVector[TorchTensor] {.importcpp: "at::unbind(@)".}
  ## Removes a dimension and returns a tuple of all slices along that dimension.
  ##
  ## C++ signature:
  ##   inline ::std::vector<at::Tensor> at::unbind(
  ##     const at::Tensor &self,
  ##     int64_t dim = 0
  ##   )
  ##
  ## Returns a tuple of tensors where the dim-th dimension is removed.
  ## Equivalent to: tensor.unbind(dim) == tuple(tensor[i] for i in range(tensor.size(dim)))

func expand*(a: TorchTensor, sizes: IntArrayRef, implicit: bool = false): TorchTensor {.importcpp: "#.expand(@)".}
  ## Returns a view of the input tensor with size expanded to a larger size.
  ##
  ## C++ signature:
  ##   Tensor expand(const Tensor& self, IntArrayRef size, bool implicit = false)
  ##
  ## Passing -1 as the size for a dimension means not changing the size of that dimension.
  ## The dimensions must match between the original tensor and the required tensor.

func expand*(a: TorchTensor, sizes: openArray[int64], implicit: bool = false): TorchTensor {.importcpp: "#.expand(@)".}
  ## Overload for runtime sizes passed as openArray[int64].

# Automatic Differentiation
# -----------------------------------------------------------------------

func backward*(a: var TorchTensor) {.importcpp: "#.backward()".}
func detach*(a: TorchTensor): TorchTensor {.importcpp: "#.detach()".}
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

func start*(s: TorchSlice): int {.importcpp: "#.start()".}
func stop*(s: TorchSlice): int {.importcpp: "#.stop()".}
func step*(s: TorchSlice): int {.importcpp: "#.step()".}

# Operators. We expose PyTorch convention for `div` and `mod` instead of Nim's
# -----------------------------------------------------------------------
func assign*(a: var TorchTensor, other: TorchTensor) {.importcpp: "# = #".}

func `not`*(a: TorchTensor): TorchTensor {.importcpp: "(~#)".}
func `-`*(a: TorchTensor): TorchTensor {.importcpp: "(-#)".}

func `+`*(a: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "(# + #)".}
func `-`*(a: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "(# - #)".}
func `*`*(a: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "(# * #)".}
func `%`*(a: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "#.remainder(#)".}

func `*`*(a: SomeNumber, b: TorchTensor): TorchTensor {.importcpp: "(# * #)".}
func `*`*(a: TorchTensor, b: SomeNumber): TorchTensor {.importcpp: "(# * #)".}

func `+`*(a: SomeNumber, b: TorchTensor): TorchTensor {.importcpp: "(# + #)".}
func `+`*(a: TorchTensor, b: SomeNumber): TorchTensor {.importcpp: "(# + #)".}

proc `/`*(a, b: TorchTensor): TorchTensor {.importcpp: "(# / #)".}
proc `/`*(a: TorchTensor, b: SomeNumber): TorchTensor {.importcpp: "(# / #)".}

# func `%`*(a: SomeNumber, b: TorchTensor): TorchTensor {.importcpp: "#.remainder(#)".}
# func `%`*(a: TorchTensor, b: SomeNumber): TorchTensor {.importcpp: "#.remainder(#)".}

func `%`*(a: SomeNumber, b: TorchTensor): TorchTensor {.importcpp: "(# % #)".}
  ## Note: this uses the Python remainder behavior, not C/C++
func `%`*(a: TorchTensor, b: SomeNumber): TorchTensor {.importcpp: "(# % #)".}
  ## Note: this uses the Python remainder behavior, not C/C++

func `+=`*(a: var TorchTensor, b: TorchTensor) {.importcpp: "(# += #)".}
func `+=`*(a: var TorchTensor, s: Scalar) {.importcpp: "(# += #)".}
func `-=`*(a: var TorchTensor, b: TorchTensor) {.importcpp: "(# -= #)".}
func `-=`*(a: var TorchTensor, s: Scalar) {.importcpp: "(# -= #)".}
func `*=`*(a: var TorchTensor, b: TorchTensor) {.importcpp: "(# *= #)".}
func `*=`*(a: var TorchTensor, s: Scalar) {.importcpp: "(# *= #)".}
func `/=`*(a: var TorchTensor, b: TorchTensor) {.importcpp: "(# /= #)".}
func `/=`*(a: var TorchTensor, s: Scalar) {.importcpp: "(# /= #)".}

func `and`*(a: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "#.bitwise_and(#)".} ## bitwise `and`.
func `or`*(a: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "#.bitwise_or(#)".} ## bitwise `or`.
func `xor`*(a: TorchTensor, b: TorchTensor): TorchTensor {.importcpp: "#.bitwise_xor(#)".} ## bitwise `xor`.

func bitand_mut*(a: var TorchTensor, s: TorchTensor) {.importcpp: "#.bitwise_and_(#)".} ## In-place bitwise `and`.
func bitor_mut*(a: var TorchTensor, s: TorchTensor) {.importcpp: "#.bitwise_or_(#)".} ## In-place bitwise `or`.
func bitxor_mut*(a: var TorchTensor, s: TorchTensor) {.importcpp: "#.bitwise_xor_(#)".} ## In-place bitwise `xor`.

func eq*(a, b: TorchTensor): TorchTensor {.importcpp: "#.eq(#)".} ## Equality of each tensor values
func equal*(a, b: TorchTensor): bool {.importcpp: "#.equal(#)".}
template `==`*(a, b: TorchTensor): bool =
  a.equal(b)

# Functions.h
# -----------------------------------------------------------------------

func contiguous*(a: TorchTensor): TorchTensor {.importcpp: "#.contiguous(@)".}
func toType*(a: TorchTensor, dtype: ScalarKind): TorchTensor {.importcpp: "#.toType(@)".}
func toSparse*(a: TorchTensor): TorchTensor {.importcpp: "#.to_sparse()".}
func toSparse*(a: TorchTensor, sparseDim: int): TorchTensor {.importcpp: "#.to_sparse(@)".}

func eye*(n: int): TorchTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int, options: TensorOptions): TorchTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::eye(@)".}
func eye*(n: int, device: DeviceKind): TorchTensor {.importcpp: "torch::eye(@)".}

func zeros*(dim: int): TorchTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef): TorchTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, options: TensorOptions): TorchTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, scalarKind: ScalarKind): TorchTensor {.importcpp: "torch::zeros(@)".}
func zeros*(dim: IntArrayRef, device: DeviceKind): TorchTensor {.importcpp: "torch::zeros(@)".}

func ones*(dim: int): TorchTensor {.importcpp: "torch::ones(@)".} ## Create a tensor filled with ones (scalar value 1)
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

func linspace*(start, stop: Scalar, steps: int, options: TensorOptions): TorchTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int, options: ScalarKind): TorchTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int, options: DeviceKind): TorchTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int, options: Device): TorchTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar, steps: int): TorchTensor {.importcpp: "torch::linspace(@)".}
func linspace*(start, stop: Scalar): TorchTensor {.importcpp: "torch::linspace(@)".}

func logspace*(
  start, stop: Scalar, steps, base: int, options: TensorOptions
): TorchTensor {.importcpp: "torch::logspace(@)".}
func logspace*(
  start, stop: Scalar, steps, base: int, options: ScalarKind
): TorchTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int, options: DeviceKind) {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int, options: Device): TorchTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps, base: int): TorchTensor {.importcpp: "torch::logspace(@)".}
func logspace*(start, stop: Scalar, steps: int): TorchTensor {.importcpp: "torch::logspace(@)".}
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
func add*(a: TorchTensor, other: TorchTensor, alpha: Scalar = 1): TorchTensor {.importcpp: "#.add(@)".}
func add*(a: TorchTensor, other: Scalar, alpha: Scalar = 1): TorchTensor {.importcpp: "#.add(@)".}
func addmv*(
  a: TorchTensor, mat: TorchTensor, vec: TorchTensor, beta: Scalar = 1, alpha: Scalar = 1
): TorchTensor {.importcpp: "#.addmv(@)".}
func addmm*(t, mat1, mat2: TorchTensor, beta: Scalar = 1, alpha: Scalar = 1): TorchTensor {.importcpp: "#.addmm(@)".}
func mm*(t, other: TorchTensor): TorchTensor {.importcpp: "#.mm(@)".}
func matmul*(t, other: TorchTensor): TorchTensor {.importcpp: "#.matmul(@)".}
func bmm*(t, other: TorchTensor): TorchTensor {.importcpp: "#.bmm(@)".}

func luSolve*(t, data, pivots: TorchTensor): TorchTensor {.importcpp: "#.lu_solve(@)".}

func qr*(a: TorchTensor, some: bool = true): CppTuple2[TorchTensor, TorchTensor] {.importcpp: "#.qr(@)".}
  ## Returns a tuple:
  ## - Q of shape (∗,m,k)
  ## - R of shape (∗,k,n)
  ## with k=min(m,n) if some is true otherwise k=m
  ##
  ## The QR decomposition is batched over dimension(s) *
  ## t = QR

# addr?
func all*(a: TorchTensor, axis: int): TorchTensor {.importcpp: "#.all(@)".}
func all*(a: TorchTensor, axis: int, keepdim: bool): TorchTensor {.importcpp: "#.all(@)".}
func allClose*(
  t, other: TorchTensor, rtol: float64 = 1e-5, abstol: float64 = 1e-8, equalNan: bool = false
): bool {.importcpp: "#.allclose(@)".}
func any*(a: TorchTensor, axis: int): TorchTensor {.importcpp: "#.any(@)".}
func any*(a: TorchTensor, axis: int, keepdim: bool): TorchTensor {.importcpp: "#.any(@)".}
func argmax*(a: TorchTensor): TorchTensor {.importcpp: "#.argmax()".}
func argmax*(a: TorchTensor, axis: int, keepdim: bool = false): TorchTensor {.importcpp: "#.argmax(@)".}
func argmin*(a: TorchTensor): TorchTensor {.importcpp: "#.argmin()".}
func argmin*(a: TorchTensor, axis: int, keepdim: bool = false): TorchTensor {.importcpp: "#.argmin(@)".}

# aggregate
# -----------------------------------------------------------------------

# sum needs wrapper procs/templates to allow for using nim arrays and single axis.
func sum*(a: TorchTensor): TorchTensor {.importcpp: "#.sum()".}
func sum*(a: TorchTensor, dtype: ScalarKind): TorchTensor {.importcpp: "#.sum(@)".}
func sum*(a: TorchTensor, axis: int, keepdim: bool = false): TorchTensor {.importcpp: "#.sum(@)".}
func sum*(a: TorchTensor, axis: int, keepdim: bool = false, dtype: ScalarKind): TorchTensor {.importcpp: "#.sum(@)".}
func sum*(a: TorchTensor, axis: IntArrayRef, keepdim: bool = false): TorchTensor {.importcpp: "#.sum(@)".}
func sum*(
  a: TorchTensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind
): TorchTensor {.importcpp: "#.sum(@)".}

# mean as well
func mean*(a: TorchTensor): TorchTensor {.importcpp: "#.mean()".}
func mean*(a: TorchTensor, dtype: ScalarKind): TorchTensor {.importcpp: "#.mean(@)".}
func mean*(a: TorchTensor, axis: int, keepdim: bool = false): TorchTensor {.importcpp: "#.mean(@)".}
func mean*(a: TorchTensor, axis: int, keepdim: bool = false, dtype: ScalarKind): TorchTensor {.importcpp: "#.mean(@)".}
func mean*(a: TorchTensor, axis: IntArrayRef, keepdim: bool = false): TorchTensor {.importcpp: "#.mean(@)".}
func mean*(
  a: TorchTensor, axis: IntArrayRef, keepdim: bool = false, dtype: ScalarKind
): TorchTensor {.importcpp: "#.mean(@)".}

# median requires std::tuple

func prod*(a: TorchTensor): TorchTensor {.importcpp: "#.prod()".}
func prod*(a: TorchTensor, dtype: ScalarKind): TorchTensor {.importcpp: "#.prod(@)".}
func prod*(a: TorchTensor, axis: int, keepdim: bool = false): TorchTensor {.importcpp: "#.prod(@)".}
func prod*(a: TorchTensor, axis: int, keepdim: bool = false, dtype: ScalarKind): TorchTensor {.importcpp: "#.prod(@)".}

func min*(a: TorchTensor): TorchTensor {.importcpp: "#.min()".}
func min*(
  a: TorchTensor, axis: int, keepdim: bool = false
): CppTuple2[TorchTensor, TorchTensor] {.importcpp: "torch::min(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the minimum values and their index in the specified axis

func max*(a: TorchTensor): TorchTensor {.importcpp: "#.max()".}
func max*(
  a: TorchTensor, axis: int, keepdim: bool = false
): CppTuple2[TorchTensor, TorchTensor] {.importcpp: "torch::max(@)".}
  ## Returns a tuple (values, indices) of type (TensorT, TensorInt64)
  ## of the maximum values and their index in the specified axis

func variance*(a: TorchTensor, unbiased: bool = true): TorchTensor {.importcpp: "#.var(@)".}
  # can't use `var` because of keyword.
func variance*(
  a: TorchTensor, axis: int, unbiased: bool = true, keepdim: bool = false
): TorchTensor {.importcpp: "#.var(@)".}
func variance*(
  a: TorchTensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false
): TorchTensor {.importcpp: "#.var(@)".}

func stddev*(a: TorchTensor, unbiased: bool = true): TorchTensor {.importcpp: "#.std(@)".}
func stddev*(
  a: TorchTensor, axis: int, unbiased: bool = true, keepdim: bool = false
): TorchTensor {.importcpp: "#.std(@)".}
func stddev*(
  a: TorchTensor, axis: IntArrayRef, unbiased: bool = true, keepdim: bool = false
): TorchTensor {.importcpp: "#.std(@)".}

# algorithms:
# -----------------------------------------------------------------------
func sort*(
  a: TorchTensor, axis: int = -1, descending: bool = false
): CppTuple2[TorchTensor, TorchTensor] {.importcpp: "#.sort(@)".}
  ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
  ## If dim is not given, the last dimension of the input is chosen (dim=-1).
  ## Returns (values, originalIndices) or type (TensorT, TensorInt64)
  ## where originalIndices is the original index of each values (before sorting)
func argsort*(a: TorchTensor, axis: int = -1, descending: bool = false): TorchTensor {.importcpp: "#.argsort(@)".}

func cat*(tensors: ArrayRef[TorchTensor], axis: int = 0): TorchTensor {.importcpp: "torch::cat(@)".}
func stack*(tensors: ArrayRef[TorchTensor], dim: int = 0): TorchTensor {.importcpp: "torch::stack(@)".}
  ## Stack tensors along a NEW dimension (unlike cat which concatenates along existing dim).
  ## All tensors must have the same shape.
  ## Example: stack([2x3, 2x3, 2x3], dim=0) -> 3x2x3
  ##          stack([2x3, 2x3], dim=1) -> 2x2x3
func flip*(a: TorchTensor, dims: IntArrayRef): TorchTensor {.importcpp: "#.flip(@)".}

# math
# -----------------------------------------------------------------------
func abs*(a: TorchTensor): TorchTensor {.importcpp: "#.abs()".}
func absolute*(a: TorchTensor): TorchTensor {.importcpp: "#.absolute()".}
func angle*(a: TorchTensor): TorchTensor {.importcpp: "#.angle()".}
func sgn*(a: TorchTensor): TorchTensor {.importcpp: "#.sgn()".}
func conj*(a: TorchTensor): TorchTensor {.importcpp: "#.conj()".}
func acos*(a: TorchTensor): TorchTensor {.importcpp: "#.acos()".}
func arccos*(a: TorchTensor): TorchTensor {.importcpp: "#.arccos()".}
func acosh*(a: TorchTensor): TorchTensor {.importcpp: "#.acosh()".}
func arccosh*(a: TorchTensor): TorchTensor {.importcpp: "#.arccosh()".}
func asinh*(a: TorchTensor): TorchTensor {.importcpp: "#.asinh()".}
func arcsinh*(a: TorchTensor): TorchTensor {.importcpp: "#.arcsinh()".}
func atanh*(a: TorchTensor): TorchTensor {.importcpp: "#.atanh()".}
func arctanh*(a: TorchTensor): TorchTensor {.importcpp: "#.arctanh()".}
func asin*(a: TorchTensor): TorchTensor {.importcpp: "#.asin()".}
func arcsin*(a: TorchTensor): TorchTensor {.importcpp: "#.arcsin()".}
func atan*(a: TorchTensor): TorchTensor {.importcpp: "#.atan()".}
func arctan*(a: TorchTensor): TorchTensor {.importcpp: "#.arctan()".}
func cos*(a: TorchTensor): TorchTensor {.importcpp: "#.cos()".}
func sin*(a: TorchTensor): TorchTensor {.importcpp: "#.sin()".}
func tan*(a: TorchTensor): TorchTensor {.importcpp: "#.tan()".}
func exp*(a: TorchTensor): TorchTensor {.importcpp: "#.exp()".}
func exp2*(a: TorchTensor): TorchTensor {.importcpp: "#.exp2()".}
func log*(a: TorchTensor): TorchTensor {.importcpp: "#.log()".}
  ## Natural logarithm (base e). log(exp(x)) = x
  ## Returns NaN for negative inputs, -Inf for 0
func log2*(a: TorchTensor): TorchTensor {.importcpp: "#.log2()".}
  ## Base-2 logarithm. Useful for information theory (entropy, bits)
func log10*(a: TorchTensor): TorchTensor {.importcpp: "#.log10()".}
  ## Base-10 logarithm. Useful for decibels and scientific notation
func erf*(a: TorchTensor): TorchTensor {.importcpp: "#.erf()".}
func erfc*(a: TorchTensor): TorchTensor {.importcpp: "#.erfc()".}
func reciprocal*(a: TorchTensor): TorchTensor {.importcpp: "#.reciprocal()".}
func neg*(a: TorchTensor): TorchTensor {.importcpp: "#.neg()".}
func clamp*(a: TorchTensor, min, max: Scalar): TorchTensor {.importcpp: "#.clamp(@)".}
func clampMin*(a: TorchTensor, min: Scalar): TorchTensor {.importcpp: "#.clamp_min(@)".}
func clampMax*(a: TorchTensor, max: Scalar): TorchTensor {.importcpp: "#.clamp_max(@)".}

func dot*(a: TorchTensor, other: TorchTensor): TorchTensor {.importcpp: "#.dot(@)".}

func squeeze*(a: TorchTensor): TorchTensor {.importcpp: "#.squeeze()".}
func squeeze*(a: TorchTensor, axis: int): TorchTensor {.importcpp: "#.squeeze(@)".}
func unsqueeze*(a: TorchTensor, axis: int): TorchTensor {.importcpp: "#.unsqueeze(@)".}
func square*(a: TorchTensor): TorchTensor {.importcpp: "#.square()".}
func sqrt*(a: TorchTensor): TorchTensor {.importcpp: "#.sqrt()".}
func pow*(a: TorchTensor, exponent: TorchTensor): TorchTensor {.importcpp: "#.pow(@)".}
func pow*(a: TorchTensor, exponent: Scalar): TorchTensor {.importcpp: "#.pow(@)".}

# FFT
# -----------------------------------------------------------------------
func fftshift*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_fftshift(@)".}
func fftshift*(a: TorchTensor, dim: IntArrayRef): TorchTensor {.importcpp: "torch::fft_ifftshift(@)".}
func ifftshift*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_fftshift(@)".}
func ifftshift*(a: TorchTensor, dim: IntArrayRef): TorchTensor {.importcpp: "torch::fft_ifftshift(@)".}

func fft*(a: TorchTensor, n: int, dim: int, norm: CppString): TorchTensor {.importcpp: "torch::fft_fft(@)".}
func fft*(a: TorchTensor, n: int, dim: int = -1): TorchTensor {.importcpp: "torch::fft_fft(@)".}
## Compute the 1-D Fourier transform
## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
func fft*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_fft(@)".}
## Compute the 1-D Fourier transform

func ifft*(a: TorchTensor, n: int, dim: int = -1, norm: CppString): TorchTensor {.importcpp: "torch::fft_ifft(@)".}
## Compute the 1-D Fourier transform
## ``n`` represent Signal length. If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
func ifft*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_ifft(@)".}
## Compute the 1-D Fourier transform

func fft2*(
  a: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func fft2*(a: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func fft2*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_fft2(@)".}
## Compute the 2-D Fourier transform

func ifft2*(
  a: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func ifft2*(a: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func ifft2*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_ifft2(@)".}
## Compute the 2-D Inverse Fourier transform

func fftn*(
  a: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func fftn*(a: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func fftn*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_fftn(@)".}
## Compute the N-D Fourier transform

func ifftn*(
  a: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func ifftn*(a: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func ifftn*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_ifftn(@)".}
## Compute the N-D Inverse Fourier transform

func rfft*(a: TorchTensor, n: int, dim: int = -1, norm: CppString): TorchTensor {.importcpp: "torch::fft_rfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.
func rfft*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_rfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.

func irfft*(a: TorchTensor, n: int, dim: int = -1, norm: CppString): TorchTensor {.importcpp: "torch::fft_irfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.
func irfft*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_irfft(@)".}
## Computes the one dimensional Fourier transform of real-valued input.

func rfft2*(
  a: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func rfft2*(a: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func rfft2*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_rfft2(@)".}
## Compute the N-D Fourier transform

func irfft2*(
  a: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func irfft2*(a: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func irfft2*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_irfft2(@)".}
## Compute the N-D Inverse Fourier transform

func rfftn*(
  a: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##    * "forward" - normalize by 1/n
##    * "backward" - no normalization
##    * "ortho" - normalize by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func rfftn*(a: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func rfftn*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_rfftn(@)".}
## Compute the N-D Fourier transform

func irfftn*(
  a: TorchTensor, s: IntArrayRef, dim: IntArrayRef, norm: CppString
): TorchTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
## ``norm`` can be :
##   * "forward" - no normalization
##   * "backward" - normalization by 1/n
##   * "ortho" - normalization by 1/sqrt(n)
## With n the logical FFT size: ``n = prod(s)``.
func irfftn*(a: TorchTensor, s: IntArrayRef): TorchTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform
## ``s`` represents signal size. If given, each dimension dim[i] will either be zero padded or trimmed to the length s[i] before computing the FFT.
func irfftn*(a: TorchTensor): TorchTensor {.importcpp: "torch::fft_irfftn(@)".}
## Compute the N-D Inverse Fourier transform

func hfft*(a: TorchTensor, n: int, dim: int = -1, norm: CppString): TorchTensor {.importcpp: "torch::hfft(@)".}
## Computes the 1 dimensional FFT of a onesided Hermitian signal.
func hfft*(a: TorchTensor): TorchTensor {.importcpp: "torch::hfft(@)".}
## Computes the 1 dimensional FFT of a onesided Hermitian signal.
func ihfft*(a: TorchTensor, n: int, dim: int = -1, norm: CppString): TorchTensor {.importcpp: "torch::ihfft(@)".}
## Computes the inverse FFT of a real-valued Fourier domain signal.
func ihfft*(a: TorchTensor): TorchTensor {.importcpp: "torch::ihfft(@)".}
## Computes the inverse FFT of a real-valued Fourier domain signal.

{.pop.}
