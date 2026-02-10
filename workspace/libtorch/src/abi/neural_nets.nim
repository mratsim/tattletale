# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  # Internal
  workspace/libtorch/src/abi/torch_tensors,
  workspace/libtorch/src/abi/std_cpp,
  workspace/libtorch/src/abi/c10,
  workspace/libtorch/vendor/libtorch

# (Almost) raw bindings to PyTorch Neural Networks
# -----------------------------------------------------------------------
#
# This provides almost raw bindings to PyTorch tensors.
#
# "Nimification" (camelCase), ergonomic indexing and interoperability with Nim types is left to the "high-level" bindings.
# This should ease searching PyTorch and libtorch documentation,
# and make C++ tutorials easily applicable.

# C++ interop
# -----------------------------------------------------------------------

{.push cdecl, header: TorchHeader.}

# #######################################################################
#
#                       Autograd
#
# #######################################################################

type AutoGradMode* {.bycopy, pure, inheritable, importcpp: "torch::AutoGradMode".} = object

type NoGradGuard* {.bycopy, pure, inheritable, importcpp: "torch::NoGradGuard".} = object

func autogradMode(enabled: bool): AutoGradMode {.constructor, importcpp: "torch::AutoGradMode(#)".}

template with*(T: type AutoGradMode, enabled: bool, body: untyped): untyped =
  bind autogradMode
  block:
    let gradMode = autogradMode(enabled)
    body

template no_grad_mode*(body: untyped): untyped =
  ## Disable precomputations necessary for gradient propagation
  with(AutoGradMode, enabled = false):
    body

# #######################################################################
#
#                       LibTorch Functional API
#
# #######################################################################
#
# LibTorch Functional API is described here.
# https://pytorch.org/cppdocs/api/namespace_torch__nn__functional.html#namespace-torch-nn-functional
# libtorch/include/torch/csrc/api/include/torch/nn/functional
#
# It is stateless, meaning users need to track weight and bias themselves.
# It is suitable for layers with no learning parameters (for example reshaping),
# or when extra flexibility is required at a small price of ergonomics.
# The high-level Module API uses Functional internally.
#
# Note:
#   Function exists both in ATen TensorBody.h (namespace at:: or torch::)
#   and in torch::nn::functional.
#
#   We can have
#     func dropout*(input: Tensor, p = 0.5, training=true): Tensor {.importcpp: "torch::nn::functional::dropout(@)".}
#     func dropout_mut*(input: var Tensor, p = 0.5, training=true) {.importcpp: "torch::nn::functional::dropout(@, /*inplace=*/ true)".}
#
#     OR
#
#     func dropout*(input: Tensor, p = 0.5, training=true): Tensor {.importcpp: "torch::dropout(@)".}
#     func dropout_mut*(input: var Tensor, p = 0.5, training=true) {.importcpp: "torch::dropout_(@)".}
#
#   The functions in torch::nn::functional are thin inlined wrapper over TensorBody.h
#   so we directly use them.

# Linear Layers
# -------------------------------------------------------------------------

func linear*(input, weight: TorchTensor): TorchTensor {.importcpp: "torch::nn::functional::linear(@)".}
  ## Applies a linear transformation to the incoming data:
  ##   y = input * transpose(weight)
  ##
  ## Input: (N,∗,in_features)(N, *, in\_features)(N,∗,in_features)
  ##        N is the batch size, * means any number of additional dimensions
  ## Weight: (out_features,in_features)
  ## Output: (N,∗,out_features)

func linear*(input, weight, bias: TorchTensor): TorchTensor {.importcpp: "torch::nn::functional::linear(@)".}
  ## Applies a linear transformation to the incoming data:
  ##   y = input * transpose(weight) + bias
  ##
  ## Input: (N,∗,in_features)(N, *, in\_features)(N,∗,in_features)
  ##        N is the batch size, * means any number of additional dimensions
  ## Weight: (out_features,in_features)
  ## Bias: (out_features)
  ## Output: (N,∗,out_features)

# Pooling functions
# -------------------------------------------------------------------------

func max_pool2d*(input: TorchTensor): TorchTensor {.varargs, importcpp: "torch::max_pool2d(#, {@})".}
  ## MaxPool 2D function
  ## - `input`: a Tensor
  ## - `kernel_size`: the kernel shape

func max_pool2d*(input: TorchTensor, kernel_size: IntArrayRef): TorchTensor {.importcpp: "torch::max_pool2d(@)".}

# Activation functions
# -------------------------------------------------------------------------

func sigmoid*(input: TorchTensor): TorchTensor {.importcpp: "torch::sigmoid(@)".}
func sigmoid_mut*(input: var TorchTensor) {.importcpp: "torch::sigmoid_(@)".}

func relu*(input: TorchTensor): TorchTensor {.importcpp: "torch::relu(@)".}
func relu_mut*(input: var TorchTensor) {.importcpp: "torch::relu_(@)".}

func leakyRelu*(input: TorchTensor): TorchTensor {.importcpp: "torch::leaky_relu(@)".}
func leakyRelu_mut*(input: var TorchTensor) {.importcpp: "torch::leaky_relu_(@)".}

func gelu*(input: TorchTensor): TorchTensor {.importcpp: "torch::gelu(@)".}
func gelu_mut*(input: var TorchTensor) {.importcpp: "torch::gelu_(@)".}

func elu*(input: TorchTensor): TorchTensor {.importcpp: "torch::elu(@)".}
func elu_mut*(input: var TorchTensor) {.importcpp: "torch::elu_(@)".}

func pRelu*(input: TorchTensor): TorchTensor {.importcpp: "torch::prelu(@)".}
func pRelu_mut*(input: var TorchTensor) {.importcpp: "torch::prelu_(@)".}

func selu*(input: TorchTensor): TorchTensor {.importcpp: "torch::selu(@)".}
func selu_mut*(input: var TorchTensor) {.importcpp: "torch::selu_(@)".}

func silu*(self: TorchTensor): TorchTensor {.importcpp: "torch::silu(@)".}
  ## SiLU (Sigmoid Linear Unit) activation function: x / (1 + exp(-x))
  ## Also known as Swish.
func silu_mut*(self: TorchTensor): TorchTensor {.importcpp: "torch::silu_(@)".}

func tanh*(input: TorchTensor): TorchTensor {.importcpp: "torch::tanh(@)".}
func tanh_mut*(input: var TorchTensor) {.importcpp: "torch::tanh_(@)".}

func softmax*(input: TorchTensor, dim: int64): TorchTensor {.importcpp: "torch::softmax(@)".}
  ## Softmax activation function: softmax(x_i) = exp(x_i) / sum(exp(x_j))
  ## Converts logits to probabilities (output sums to 1 along dim).
  ## Critical for attention mechanisms in transformers and multi-class classification.
  ## dim: dimension along which to apply softmax (usually last dim for classification)
func softmax*(input: TorchTensor, dim: int64, dtype: ScalarKind): TorchTensor {.importcpp: "torch::softmax(@)".}
  ## Softmax with explicit output dtype (useful for mixed precision)

func log_softmax*(input: TorchTensor, axis: int64): TorchTensor {.importcpp: "torch::log_softmax(@)".}
func log_softmax*(input: TorchTensor, axis: int64, dtype: ScalarKind): TorchTensor {.importcpp: "torch::log_softmax(@)".}

# Dropout functions
# -------------------------------------------------------------------------

func dropout*(input: TorchTensor, p = 0.5, training = true): TorchTensor {.importcpp: "torch::dropout(@)".}
func dropout_mut*(input: var TorchTensor, p = 0.5, training = true) {.importcpp: "torch::dropout_(@)".}

# Normalization functions
# -------------------------------------------------------------------------
func rms_norm*(input: TorchTensor, normalized_shape: IntArrayRef) {.importcpp: "torch::rms_norm(@)".}
func rms_norm*(input: TorchTensor, normalized_shape: IntArrayRef, weight: TorchTensor) {.importcpp: "torch::rms_norm(@)".}

# Loss functions
# -------------------------------------------------------------------------

type Reduction* {.size: sizeof(cint), importcpp: "torch::Reduction::Reduction".} = enum
  None = 0 # Do not reduce
  Mean = 1 # (Possibly weighted) mean of losses
  Sum = 2 # Sum losses

func nll_loss*(input, target: TorchTensor): TorchTensor {.importcpp: "torch::nll_loss(@)".}
  ## Negative log likelihood loss. Target must be int64 (Long)!
  ## Uses mean reduction by default.

func nll_loss*(input, target: TorchTensor, weight: TorchTensor, red: Reduction): TorchTensor
  {.importcpp: "torch::nll_loss(@)".}
  ## Negative log likelihood loss with class weights and explicit reduction.
  ## Target must be int64 (Long)!
  ## Weight: optional 1D tensor of size C (num classes) for class weighting

func cross_entropy*(input, target: TorchTensor): TorchTensor {.importcpp: "torch::nn::functional::cross_entropy(@)".}
  ## Cross entropy loss: combines log_softmax and nll_loss in a single, numerically stable function.
  ## Standard loss for multi-class classification and language modeling.
  ##
  ## Input: (N, C) where N=batch size, C=number of classes (logits, NOT probabilities)
  ## Target: (N,) with class indices, where 0 <= target[i] < C (must be int64/Long)
  ## Output: scalar loss value (mean reduction by default)
  ##
  ## Example for LLM: Input shape (batch=4, vocab=50000), Target shape (batch=4,)
  ## Computes: -log(softmax(input)[i, target[i]]) for each i, then averages

func cross_entropy*(input, target, weight: TorchTensor): TorchTensor
  {.importcpp: "torch::nn::functional::cross_entropy(@)".}
  ## Cross entropy with class weights. Uses mean reduction.
  ## Weight: 1D tensor of size C (num classes) for per-class weighting

func cross_entropy*(input, target, weight: TorchTensor, reduction: Reduction): TorchTensor
  {.importcpp: "torch::nn::functional::cross_entropy(@)".}
  ## Cross entropy with class weights and explicit reduction mode (Mean, Sum, or None)

func binary_cross_entropy_with_logits*(
  input, target: TorchTensor
): TorchTensor {.importcpp: "torch::binary_cross_entropy_with_logits(@)".}
  ## Sigmoid + Log + Negative loglikelihood
  ## PyTorch naming
func sigmoid_cross_entropy*(
  input, target: TorchTensor
): TorchTensor {.importcpp: "torch::binary_cross_entropy_with_logits(@)".}
  ## Sigmoid + Log + Negative loglikelihood
  ## Arraymancer or Tensorflow naming

func mse_loss*(input, target: TorchTensor): TorchTensor {.importcpp: "torch::mse_loss(@)".} ## target must be int (Long)!

func l1_loss*(input, target: TorchTensor): TorchTensor {.importcpp: "torch::l1_loss(@)".} ## target must be int (Long)!

# #######################################################################
#
#                       LibTorch Module API
#
# #######################################################################
#
# LibTorch Module API is described here.
# https://pytorch.org/cppdocs/api/namespace_torch__nn.html#classes
# libtorch/include/torch/csrc/api/include/torch/nn/module.h
#
# It uses class derived from the base "Module" class.
# The modules keep track of weights and biases for the users.
# They also keep track of the training or evaluation mode,
# allow pretty-printing of a computation graph,
# serialization and deserialization.
#
# See Module ownership notes:
# - https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module_holder.html#classtorch_1_1nn_1_1_module_holder
# - https://pytorch.org/tutorials/advanced/cpp_frontend.html#module-ownership
# all modules are thin wrapper around shared_ptr + ModuleImpl
#
# Torch serialization expect the shared_ptr so we should respect their Module API.

type
  Module* {.bycopy, pure, inheritable, importcpp: "torch::nn::Module".} = object
    ## A LibTorch neural network module that can be inherited from
    # Impl detaim:
    #   Nim inheritable objects have runtime type information pointer
    #   as a hidden first field.
    #   {.pure, inheritable.} removes that to make the object C++ compatible.

  ModuleHolder* {.bycopy, pure, inheritable, importcpp: "torch::nn::ModuleHolder".} = object

  SharedModule*[T: Module] = CppSharedPtr[T]

proc register_module*[ParMod: ModuleHolder, ChildMod: ModuleHolder](
  parent: var ParMod, name: cstring, child: var ChildMod
) {.importcpp: "#.register_module(@)".} ## Register a submodule to a parent module.

proc register_module*[ParMod: ModuleHolder, ChildMod: ModuleHolder](
  parent: var ParMod, name: cstring, child: sink ChildMod
): ChildMod {.importcpp: "#.register_module(@)".} ## Register a submodule to a parent module.

proc register_module*[ParMod: Module, ChildMod: ModuleHolder](
  parent: var SharedModule[ParMod], name: cstring, child: var ChildMod
) {.importcpp: "#->register_module(@)".} ## Register a submodule to a parent module.

proc register_module*[ParMod: Module, ChildMod: ModuleHolder](
  parent: var SharedModule[ParMod], name: cstring, child: sink ChildMod
): ChildMod {.importcpp: "#->register_module(@)".} ## Register a submodule to a parent module.

proc register_module*[ParMod: Module, ChildMod: Module](
  parent: var SharedModule[ParMod], name: cstring, child: var SharedModule[ChildMod]
) {.importcpp: "#->register_module(@)".} ## Register a submodule to a parent module.

proc register_module*[ParMod: Module, ChildMod: Module](
  parent: var SharedModule[ParMod], name: cstring, child: sink SharedModule[ChildMod]
): SharedModule[ChildMod] {.importcpp: "#->register_module(@)".} ## Register a submodule to a parent module.

proc register_parameter*[ParMod: Module](
  parent: var SharedModule[ParMod], name: cstring, child: sink TorchTensor
): TorchTensor {.importcpp: "#->register_parameter(@)".} ## Register a submodule to a parent module.

func parameters*(module: Module, recurse = true): CppVector[TorchTensor] {.importcpp: "#.parameters(#)".}

func is_training*(module: Module): bool {.importcpp: "#.is_training()".}

proc to*(module: ModuleHolder or SharedModule, device: DeviceKind) {.importcpp: "#->to(#)".}
proc to*(module: ModuleHolder or SharedModule, device: Device) {.importcpp: "#->to(#)".}

func train*(module: var ModuleHolder or SharedModule, on = true) {.importcpp: "#->train(#)".} ## Enable training mode

func eval*(module: var ModuleHolder or SharedModule) {.importcpp: "#->eval()".} ## Enable evaluation mode

# Linear layer
# --------------------------------
# https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_linear_impl.html

type
  LinearOptions* {.bycopy, importcpp: "torch::nn::LinearOptions".} = object

  Linear* {.pure, bycopy, importcpp: "torch::nn::Linear".} = object of ModuleHolder
    # Linear is a shared_ptr underneath.
    # The ptr is bycopy which results in the actual data being byref.
    options* {.importc.}: LinearOptions
    weight* {.importc.}: TorchTensor
    bias* {.importc.}: TorchTensor

func init*(
  T: type LinearOptions, in_features, out_features: int64
): T {.constructor, importcpp: "torch::nn::LinearOptions(@)".}
func bias*(options: LinearOptions, bias: bool): LinearOptions {.importcpp: "#.bias(@)".}

func init*(T: type Linear, in_features, out_features: int64): T {.constructor, importcpp: "torch::nn::Linear(@)".}
func init*(T: type Linear, options: LinearOptions): T {.constructor, importcpp: "torch::nn::Linear(@)".}

# Non-generic wrappers to avoid Nim compiler type inference bug
proc newLinear*(in_features, out_features: int64): Linear {.importcpp: "torch::nn::Linear(@)", constructor.}

proc newLinear*(options: LinearOptions): Linear {.importcpp: "torch::nn::Linear(@)", constructor.}

func reset*(linear: Linear) {.importcpp: "#.reset()".}
  ## reset() must perform initialization of all members with reference semantics,
  ## most importantly parameters, buffers and submodules.

func reset_parameters*(linear: Linear) {.importcpp: "#.reset_parameters()".}

# pretty_print

func forward*(linear: Linear, input: TorchTensor): TorchTensor {.importcpp: "#->forward(#)".}
  ## Transforms the ``input`` tensor
  ## by multiplying with the ``weight``
  ## and optionally adding the ``bias``,
  ## if ``with_bias`` is true in the ``options``.

# Conv2D layer
# --------------------------------
# Link TODO

type
  Conv2dOptions* {.bycopy, importcpp: "torch::nn::Conv2dOptions".} = object

  Conv2d* {.pure, bycopy, importcpp: "torch::nn::Conv2d".} = object of ModuleHolder
    # Conv2d is a shared_ptr underneath.
    # The ptr is bycopy which results in the actual data being byref.
    options* {.importc.}: Conv2DOptions
    bias* {.importc.}: TorchTensor

func init*(
  T: type Conv2dOptions, in_channels, out_channels, kernel_size: int64 or array[2, int64]
): T {.constructor, importcpp: "torch::nn::Conv2dOptions(@)".}
func bias*(options: Conv2dOptions, bias: bool): Conv2dOptions {.importcpp: "#.bias(@)".}
func stride*(options: Conv2dOptions, stride: int64): Conv2dOptions {.importcpp: "#.stride(@)".}
func stride*(options: Conv2dOptions, stride: array[2, int64]): Conv2dOptions {.importcpp: "#.stride(@)".}
func padding*(options: Conv2dOptions, padding: int64): Conv2dOptions {.importcpp: "#.padding(@)".}
func dilation*(options: Conv2dOptions, dilation: int64 or array[2, int64]): Conv2dOptions {.importcpp: "#.dilation(@)".}
func groups*(options: Conv2dOptions, groups: int64): Conv2dOptions {.importcpp: "#.groups(@)".}

func stride*(options: Conv2dOptions): IntArrayRef {.importcpp: "at::ArrayRef<int64_t>(#.stride())".}

func init*(
  T: type Conv2d, in_channels, out_channels, kernel_size: int64
): T {.constructor, importcpp: "torch::nn::Conv2d(@)".}
func init*(
  T: type Conv2d, in_channels, out_channels, kernel_size: array[2, int64]
): T {.constructor, importcpp: "torch::nn::Conv2d(@)".}
func init*(T: type Conv2d, options: Conv2dOptions): T {.constructor, importcpp: "torch::nn::Conv2d(@)".}

# Non-generic wrappers to avoid Nim compiler type inference bug
proc newConv2d*(
  in_channels, out_channels, kernel_size: int64
): Conv2d {.importcpp: "torch::nn::Conv2d(@)", constructor.}

proc newConv2d*(options: Conv2dOptions): Conv2d {.importcpp: "torch::nn::Conv2d(@)", constructor.}

func `weight=`*(x: Conv2d, w: TorchTensor) {.importcpp: "#->weight = #".}

func reset*(conv2d: Conv2d) {.importcpp: "#.reset()".}
  ## reset() must perform initialization of all members with reference semantics,
  ## most importantly parameters, buffers and submodules.

func reset_parameters*(conv2d: Conv2d) {.importcpp: "#.reset_parameters()".}

# pretty_print

func forward*(conv2d: Conv2d, input: TorchTensor): TorchTensor {.importcpp: "#->forward(#)".}
  ## Transforms the ``input`` tensor
  ## by multiplying with the ``weight``
  ## and optionally adding the ``bias``,
  ## if ``with_bias`` is true in the ``options``.

# Dropout layers
# --------------------------------
# Link TODO

type
  DropoutOptions* {.bycopy, importcpp: "torch::nn::DropoutOptions".} = object

  Dropout* {.pure, bycopy, importcpp: "torch::nn::Dropout".} = object of ModuleHolder
    options* {.importc.}: DropoutOptions

  Dropout2d* {.pure, bycopy, importcpp: "torch::nn::Dropout2d".} = object of ModuleHolder
    options* {.importc.}: DropoutOptions

  Dropout3d* {.pure, bycopy, importcpp: "torch::nn::Dropout3d".} = object of ModuleHolder
    options* {.importc.}: DropoutOptions

  SomeDropout* = Dropout or Dropout2d or Dropout3d

func init*(T: type Dropout, proba = 0.5): T {.constructor, importcpp: "torch::nn::Dropout(@)".}
func init*(T: type Dropout2d, proba = 0.5): T {.constructor, importcpp: "torch::nn::Dropout2d(@)".}
func init*(T: type Dropout3d, proba = 0.5): T {.constructor, importcpp: "torch::nn::Dropout3d(@)".}

# Non-generic wrappers to avoid Nim compiler type inference bug
proc newDropout*(proba = 0.5): Dropout {.importcpp: "torch::nn::Dropout(@)", constructor.}

proc newDropout2d*(proba = 0.5): Dropout2d {.importcpp: "torch::nn::Dropout2d(@)", constructor.}

proc newDropout3d*(proba = 0.5): Dropout3d {.importcpp: "torch::nn::Dropout3d(@)", constructor.}

func forward*(dropout: SomeDropout, input: TorchTensor): TorchTensor {.importcpp: "#->forward(#)".}