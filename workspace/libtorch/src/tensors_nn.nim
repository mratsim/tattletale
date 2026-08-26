# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  std/importutils,
  # Internal
  ./tensors {.all.},
  workspace/libtorch/src/raw_libtorch as F

#######################################################################
#
#                       Neural Network Functional API
#
#######################################################################
#
# Wraps the raw torch::nn::functional bindings from neural_nets.nim
# so users interact exclusively with Tensor types.
#
# Forward + mutation variants use the same pattern:
#   func foo(input: Tensor): Tensor
#   proc foo_mut(input: var Tensor)
#
#######################################################################

privateAccess(Tensor)

wrapLibtorch:
  # Activations
  # -------------------------------------------------------

  func silu*(input: Tensor): Tensor
    ## SiLU (Sigmoid Linear Unit) activation function: ``x / (1 + exp(-x))``
    ## Also known as Swish.

  proc silu_mut*(input: var Tensor)

  func sigmoid*(input: Tensor): Tensor
    ## Sigmoid activation function: ``1 / (1 + exp(-x))``

  proc sigmoid_mut*(input: var Tensor)

  # Normalized activations
  # -------------------------------------------------------

  func softmax*(input: Tensor, dim: int): Tensor
  func log_softmax*(input: Tensor, axis: int): Tensor

  # Normalization
  # -------------------------------------------------------

  func rms_norm*(input: Tensor, normalized_shape: varargs[int], weight: Tensor, eps: float64): Tensor

  # Linear
  # -------------------------------------------------------

  func linear*(input, weight: Tensor): Tensor
  func linear*(input, weight, bias: Tensor): Tensor

  # Loss functions
  # -------------------------------------------------------

  func mse_loss*(input, target: Tensor): Tensor

# Scaled Dot Product Attention
# -------------------------------------------------------
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
      query, key, value: Tensor,
      attn_mask: Option[Tensor] = none(Tensor),
      dropout_p: cdouble = 0.0,
      is_causal: bool = false,
      scale: Option[float64] = none(float64),
      enable_gqa: bool = false): Tensor {.inline, nodestroy.} =
  ## SDPA — Transformers' attention
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
  ##
  ## Equivalent to
  ##
  ## def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
  ##         is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
  ##     L, S = query.size(-2), key.size(-2)
  ##     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
  ##     attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
  ##     if is_causal:
  ##         assert attn_mask is None
  ##         temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
  ##         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
  ##
  ##     if attn_mask is not None:
  ##         if attn_mask.dtype == torch.bool:
  ##             attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
  ##         else:
  ##             attn_bias = attn_mask + attn_bias
  ##
  ##     if enable_gqa:
  ##         key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
  ##         value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
  ##
  ##     attn_weight = query @ key.transpose(-2, -1) * scale_factor
  ##     attn_weight += attn_bias
  ##     attn_weight = torch.softmax(attn_weight, dim=-1)
  ##     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
  ##     return attn_weight @ value


  # {.nodestroy.} is necessary or we have a segfault if compiled with GCC v15.2.0
  # but not GCC v15.2.1 or Clang.
  # as with `bug_test_embedded_tensors.nim`
  # and `wrapTorchTensorImpl` in `tensors.nim`
  # it looks like when moving in/out of an object (here Option[T])
  # Nim inserts destructors that interferes with the intrusive refcount

  convertLibTorchExceptions:
    wrapTorchTensor:

      if attn_mask.isSome():
        if scale.isSome():
          scaled_dot_product_attention(
            query.raw, key.raw, value.raw,
            attn_mask.unsafeGet().raw,
            dropout_p,
            is_causal,
            scale.unsafeGet(),
            enable_gqa
          )
        else:
          scaled_dot_product_attention(
            query.raw, key.raw, value.raw,
            attn_mask.unsafeGet().raw,
            dropout_p,
            is_causal,
            cpp_nullopt(),
            enable_gqa
          )
      else:
        if scale.isSome():
          scaled_dot_product_attention(
            query.raw, key.raw, value.raw,
            cpp_nullopt(),
            dropout_p,
            is_causal,
            scale.unsafeGet(),
            enable_gqa
          )
        else:
          scaled_dot_product_attention(
            query.raw, key.raw, value.raw,
            cpp_nullopt(),
            dropout_p,
            is_causal,
            cpp_nullopt(),
            enable_gqa
          )
