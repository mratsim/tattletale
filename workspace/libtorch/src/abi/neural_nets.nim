# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
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

func softmax*(input: TorchTensor, dim: int): TorchTensor {.importcpp: "torch::softmax(@)".}
  ## Softmax activation function: softmax(x_i) = exp(x_i) / sum(exp(x_j))
  ## Converts logits to probabilities (output sums to 1 along dim).
  ## Critical for attention mechanisms in transformers and multi-class classification.
  ## dim: dimension along which to apply softmax (usually last dim for classification)
func softmax*(input: TorchTensor, dim: int, dtype: ScalarKind): TorchTensor {.importcpp: "torch::softmax(@)".}
  ## Softmax with explicit output dtype (useful for mixed precision)

func log_softmax*(input: TorchTensor, axis: int): TorchTensor {.importcpp: "torch::log_softmax(@)".}
func log_softmax*(input: TorchTensor, axis: int, dtype: ScalarKind): TorchTensor {.importcpp: "torch::log_softmax(@)".}

# Dropout functions
# -------------------------------------------------------------------------

func dropout*(input: TorchTensor, p = 0.5, training = true): TorchTensor {.importcpp: "torch::dropout(@)".}
func dropout_mut*(input: var TorchTensor, p = 0.5, training = true) {.importcpp: "torch::dropout_(@)".}

# Normalization functions
# -------------------------------------------------------------------------
func rms_norm*(input: TorchTensor, normalized_shape: IntArrayRef): TorchTensor {.importcpp: "torch::rms_norm(@)".}
func rms_norm*(input: TorchTensor, normalized_shape: IntArrayRef, weight: TorchTensor): TorchTensor {.importcpp: "torch::rms_norm(@)".}
func rms_norm*(input: TorchTensor, normalized_shape: IntArrayRef, weight: TorchTensor, eps: float64): TorchTensor {.importcpp: "torch::rms_norm(@)".}
  ## RMSNorm with optional eps parameter.
  ##
  ## C++ signature:
  ##   inline at::Tensor at::rms_norm(
  ##     const at::Tensor &input,
  ##     at::IntArrayRef normalized_shape,
  ##     const ::std::optional<at::Tensor> &weight = {},
  ##     ::std::optional<double> eps = ::std::nullopt
  ##   )
  ##
  ## Computes
  ##   y = x * weight / sqrt(mean(x^2) + eps)

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

# Scaled Dot Product Attention
# -------------------------------------------------------------------------
# https://pytorch.org/docs/stable/functional.html#torch.nn.functional.scaled_dot_product_attention
#
# Computes softmax(Q @ K^T / scale) @ V with efficient memory attention.
# Uses Paged KV cache for efficient generation and supports:
#   - Attention masking (padding masks, etc.)
#   - Causal/prefix attention (autoregressive decoding)
#   - Grouped-Query Attention (GQA) for efficient LLaMA-style models
#
# Input shapes:
#   - query: (B, H_q, L, d_k) or (B, L, H_q * d_k)
#   - key:   (B, H_kv, L, d_k) or (B, L, H_kv * d_k)
#   - value: (B, H_kv, L, d_v) or (B, L, H_kv * d_v)
#
# Output shape:
#   - (B, H_q, L, d_v) or (B, L, H_q * d_v)
#
# Parameters (forwarded to C++ std::optional):
#   - attn_mask: Mask to apply before softmax (broadcasts to batch).
#                Shape: (B, L, L) or (1, L, L) for broadcast.
#                Values: -inf or large negative for masked positions.
#   - dropout_p: Dropout probability. Default: 0.0 (no dropout).
#   - is_causal: Apply causal masking for autoregressive decoding.
#   - scale: Scale factor for Q @ K^T. Default: 1/sqrt(head_dim).
#   - enable_gqa: Enable grouped-query attention (H_kv must divide H_q).
#
# Backends
# -------------------------------------------------------------------------
# SDPA automatically selects the most efficient backend based on input constraints.
# Available backends (selected by priority on each device):
#
# | Backend              | CUDA                    | XPU (oneDNN)     | CPU  | MPS          |
# |---------------------|-------------------------|-----------------|------|--------------|
# | cuDNN Attention    | Hopper+ (SM 9/10)      | -               | -    | -           |
# | Flash Attention    | CUDA, XPU             | Flash           | Flash| -           |
# | Efficient Attention| CUDA (SM 70+)         | -               | -    | -           |
# | Overrideable       | -                     | oneDNN          | -    | -           |
# | Math (fallback)    | All devices           | All            | All  | Fast path   |
#
# Backend priority (CUDA):
#   1. cuDNN Attention (Hopper+ with cuDNN >9.15.0)
#   2. Flash Attention
#   3. Efficient Attention
#   4. Math (fallback)
#
# Backend priority (XPU):
#   1. Overrideable (oneDNN)
#   2. Flash Attention
#   3. Math
#   4. Efficient (logs warning, falls back to math)
#
# Backend constraints summary:
#   - Flash Attention: CUDA/XPU, dtype (FP16/BF16/FP32), head_dim % 8 == 0,
#                      no arbitrary mask (except causal), no nested tensors with training
#   - Efficient Attention: CUDA/ROCm (SM 70+), dtype (FP16/BF16/FP32),
#                          head_dim constraints, no nested tensors with training
#   - cuDNN Attention: Hopper/Blackwell GPUs, cuDNN >9.15.0
#   - Math: Fallback, supports all dtypes including FP64
#   - GQA: Supported only in Flash and Math backends on CUDA (experimental)
#
# Controlling backends:
#   - Context manager: torch.nn.attention.sdpa_kernel(backends=[SDPBackend.X])
#   - Global toggles:
#     - torch.backends.cuda.enable_flash_sdp
#     - torch.backends.cuda.enable_mem_efficient_sdp
#     - torch.backends.cuda.enable_math_sdp
#     - torch.backends.cuda.enable_cudnn_sdp
#
# Note: If no backend passes constraints, checks re-run with debug=True
#       and warnings print the rejection reasons.

func scaled_dot_product_attention*(
  query, key, value: TorchTensor,
  attn_mask: Optional[TorchTensor] = cpp_nullopt,
  dropout_p: cdouble = 0.0,
  is_causal: bool = false,
  scale: Optional[float64] = cpp_nullopt,
  enable_gqa: bool = false
): TorchTensor {.importcpp: "torch::scaled_dot_product_attention(@)".}
  ## SDPA - the core attention operation in Transformers.
  ##
  ## C++ signature:
  ##   inline at::Tensor at::scaled_dot_product_attention(
  ##     const at::Tensor &query,
  ##     const at::Tensor &key,
  ##     const at::Tensor &value,
  ##     const ::std::optional<at::Tensor> &attn_mask = {},
  ##     double dropout_p = 0.0,
  ##     bool is_causal = false,
  ##     ::std::optional<double> scale = ::std::nullopt,
  ##     bool enable_gqa = false
  ##   )
  ##
  ## Computes softmax(Q @ K^T / scale) @ V with efficient memory attention.
  ## Uses Paged KV cache for efficient generation and supports:
  ##   - Attention masking (padding masks, etc.)
  ##   - Causal/prefix attention (autoregressive decoding)
  ##   - Grouped-Query Attention (GQA) for efficient LLaMA-style models
  ##
  ## Input shapes:
  ##   - query: (B, H_q, L, d_k) or (B, L, H_q * d_k)
  ##   - key:   (B, H_kv, L, d_k) or (B, L, H_kv * d_k)
  ##   - value: (B, H_kv, L, d_v) or (B, L, H_kv * d_v)
  ##
  ## Output shape:
  ##   - (B, H_q, L, d_v) or (B, L, H_q * d_v)
  ##
  ## Parameters (forwarded to C++ std::optional):
  ##   - attn_mask: Mask to apply before softmax (broadcasts to batch).
  ##                Shape: (B, L, L) or (1, L, L) for broadcast.
  ##                Values: -inf or large negative for masked positions.
  ##   - dropout_p: Dropout probability. Default: 0.0 (no dropout).
  ##   - is_causal: Apply causal masking for autoregressive decoding.
  ##   - scale: Scale factor for Q @ K^T. Default: 1/sqrt(head_dim).
  ##   - enable_gqa: Enable grouped-query attention (H_kv must divide H_q).
  ##
  ## Backends: See module-level documentation for backend selection details.

func scaled_dot_product_attention*(
    query, key, value: TorchTensor,
    attn_mask = none(TorchTensor),
    dropout_p = 0.0,
    is_causal = false,
    scale = none(float64),
    enable_gqa = false): TorchTensor {.inline.} =

    scaled_dot_product_attention(
      query, key, value,
      attn_mask.toCppOptional(),
      dropout_p,
      is_causal,
      scale.toCppOptional(),
      enable_gqa
    )
