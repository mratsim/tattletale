# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/stateful/inference_context

{.experimental: "callOperator".}

type
  Lfm2ShortConv* = ref object
    ## Short-conv block of LFM2 hybrid layers (`layer_types[i] == "conv"`):
    ## a fused in_proj split into three hidden-size branches, a depthwise
    ## causal conv1d over the product of two branches, and an out_proj.
    ## Stateless layer. Causal-conv history lives in InferenceContext.convState,
    ## indexed by layer_idx.
    layer_idx*: int             # Layer index, indexes ctx.convState
    name*: string               # Safetensor key prefix (e.g. "model.layers.0.conv")
    in_proj: Linear             # [3 * hidden_size, hidden] fused projection, no bias
    conv_weight: Tensor         # [conv_dim, 1, K] bf16, depthwise causal conv, no bias
    out_proj: Linear            # [hidden, conv_dim]
    conv_kernel_size: int
    conv_dim: int

# =============================================================================
# Data flow through Lfm2ShortConv (vendored Lfm2ShortConv.forward)
# =============================================================================
#
#   x (batch, seq, hidden)
#   │
#   ├─→ in_proj → transpose(1, 2) → (b, 3*hidden_size, seq)
#   │        └─→ chunk 3 along dim -2 → B, C, x (each b, conv_dim, seq)
#   │        mixed = B * x     (input-dependent scaling before the conv)
#   │        cv = causal_conv1d(mixed, w)   (K=conv_L_cache, groups=conv_dim, no bias)
#   │              ├─ prefill: zero-left-pad K-1 → conv → take first T
#   │              └─ decode:  cat([state(K-1), mixed]) → conv → take last 1
#   │        y = C * cv         (input-dependent scaling after the conv)
#   │
#   └─→ transpose(1, 2).contiguous → (b, seq, conv_dim) → out_proj → (b, seq, hidden)
#
# Conv layers never touch ctx.pages. ctx.convState is their whole cache.
# =============================================================================

func init*(
    _: type Lfm2ShortConv,
    layer_idx: int,
    name: string,
    in_proj: Linear,
    conv_weight: Tensor,
    out_proj: Linear,
    conv_kernel_size, conv_dim: int): Lfm2ShortConv =
  ## Initialize an LFM2 short-conv layer.
  ##
  ## Args:
  ##   layer_idx: Layer index (0..num_layers-1), indexes the per-sequence
  ##     conv history in InferenceContext
  ##   name: Safetensor key prefix (e.g. "model.layers.0.conv")
  ##   in_proj: (3 * hidden_size, hidden) projection, no bias
  ##     (the vendored Lfm2ShortConv sizes the projection from hidden_size)
  ##   conv_weight: (conv_dim, 1, conv_kernel_size) bf16 depthwise conv weight
  ##   out_proj: (hidden, conv_dim) projection
  ##   conv_kernel_size: causal conv kernel width (conv_L_cache, 3)
  ##   conv_dim: width of each of the three branches (equals hidden_size)
  Lfm2ShortConv(
    layer_idx: layer_idx,
    name: name,
    in_proj: in_proj,
    conv_weight: conv_weight,
    out_proj: out_proj,
    conv_kernel_size: conv_kernel_size,
    conv_dim: conv_dim
  )

proc forward(
    self: Lfm2ShortConv,
    ctx: var InferenceContext,
    x: Tensor): Tensor =
  ## Forward pass for the short-conv block with per-sequence conv state.
  ##
  ## Args:
  ##   ctx: InferenceContext holding this layer's conv history
  ##     (ctx.convState[layer_idx], [conv_dim, K-1] bf16)
  ##   x: Input tensor of shape (batch, seq, hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (batch, seq, hidden_size)
  ##
  ## Decode (seq_len 1) reads the stored conv history and writes the updated
  ## history back. Prefill (seq_len > 1) starts from a zero left-pad,
  ## overwrites the stored history, so decode after prefill is bit-identical
  ## to a one-shot forward over the same tokens.
  let batch = x.size(0)
  if batch != 1:
    raise newException(ValueError,
      "[ttt] LFM2 short-conv currently supports batch_size == 1 only, got " & $batch)

  let seqLen = x.size(1)
  let device = x.deviceType()
  let convWidth = self.conv_kernel_size - 1
  ctx.ensureConvStates(
    self.layer_idx, self.conv_dim, convWidth, device)

  # ── Fused projection, split into three branches ──
  let bcx = self.in_proj.forward(x).transpose(1, 2)  # (b, 3*hidden_size, T)
  let split = F.chunk(bcx, 3, -2)                    # each (b, conv_dim, T)
  let bBranch = split[0]
  let cBranch = split[1]
  let xBranch = split[2]
  let mixed = bBranch * xBranch                      # (b, conv_dim, T)

  # ── Causal conv1d (K=conv_kernel_size, groups=conv_dim, no bias) ──
  var convOut: Tensor
  if seqLen == 1:
    # Decode step: prepend the stored conv history, valid conv, take last 1.
    let state = ctx.convState[self.layer_idx]  # (conv_dim, K-1) bf16
    let catInput = F.cat([state.unsqueeze(0), mixed], -1)  # (b, conv_dim, K)
    let conv = F.conv1d(
      catInput, self.conv_weight,
      padding = [0], groups = self.conv_dim)
    convOut = conv.narrow(2, conv.size(2) - seqLen, seqLen)
    # New state = last conv_kernel_size - 1 positions of the window
    # (the old history minus its head, plus the new token).
    ctx.convState[self.layer_idx] =
      catInput.narrow(2, catInput.size(2) - convWidth, convWidth)[0].contiguous()
  else:
    # Prefill: causal conv with (K-1)-position left padding, take first T.
    let conv = F.conv1d(
      mixed, self.conv_weight,
      padding = [convWidth], groups = self.conv_dim)
    convOut = conv.narrow(2, 0, seqLen)
    # New state = last K-1 positions of the zero-padded pre-conv input.
    # A prefill shorter than the kernel width also yields (K-1) columns.
    let padded = F.cat([
      F.zeros(batch, self.conv_dim, convWidth,
        F.tensorOptions(F.kBFloat16, device)),
      mixed], -1)
    ctx.convState[self.layer_idx] =
      padded.narrow(2, padded.size(2) - convWidth, convWidth)[0].contiguous()

  # ── Post-conv scaling, back to (batch, seq, hidden), out_proj ──
  let y = cBranch * convOut                       # (b, conv_dim, T)
  let reshaped = y.transpose(1, 2).contiguous()   # (b, T, conv_dim)
  result = self.out_proj.forward(reshaped)

template `()`*(layer: Lfm2ShortConv,
            ctx: var InferenceContext,
            x: Tensor): untyped =
  layer.forward(ctx, x)
