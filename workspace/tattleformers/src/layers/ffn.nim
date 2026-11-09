# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Transformer feed-forward layers:
## - GatedDenseFFN:
##   dense SwiGLU FFN with separate gate and up projections
##
## Out of scope, unported off the reference
## workspace/transformers/src/layers/ffn.nim:
## - GatedBlockSparseFFN: routed mixture-of-experts FFN with a shared
##   expert (router, per-expert decode/prefill bodies, scatter_add)
## - BlockSparseFFN: the ungated shared-expert form over the transformers
##   grouped_mm expert pairs

import
  workspace/tensorcery/src/tensor_datatypes,
  ../ops_2i1o/silu_mul,
  ./linear

{.experimental: "callOperator".}

type
  ActivationKind* {.size: sizeof(int8).} = enum
    ## Activation selector of the gated FFN layers. The reference enum
    ## lives in workspace/positron/src/kernels/portable/activations.nim,
    ## whose host ops bind libtorch.
    ## kSilu evaluates on the tattleformers face through the fused k_silu_mul kernel.
    kSilu = 0
    kGeluTanh
      ## Tanh-approximate GELU: the `gelu_pytorch_tanh` activation of the gemma checkpoints

type
  GatedDenseFFN* = ref object
    ## Dense SwiGLU feed-forward network (FFN) with separate gate and up projections.
    ##
    ## Forward:
    ##   gate = gate_proj(x)        # (..., intermediate_size)
    ##   up   = up_proj(x)          # (..., intermediate_size)
    ##   activation = silu(gate) * up
    ##   output = down_proj(activation)
    ##
    ## Input:
    ##   - An externally provided `x` of shape (..., hidden_size)
    ##
    ## Return:
    ##   - Output tensor of shape (..., hidden_size)
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    activation: ActivationKind

proc init*(
    _: type GatedDenseFFN,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: ActivationKind = kSilu
  ): GatedDenseFFN =
  ## Creates a GatedDenseFFN layer from pre-constructed Linear projections.
  GatedDenseFFN(
    gate_proj: gate_proj,
    up_proj: up_proj,
    down_proj: down_proj,
    activation: activation
  )

proc init*(
    _: type GatedDenseFFN,
    gate_weight, up_weight, down_weight: Tensor,
    activation: ActivationKind = kSilu
  ): GatedDenseFFN =
  ## Creates a GatedDenseFFN layer from separate gate and up weight tensors.
  ## The weights are still stored as separate projections (no fusion).
  GatedDenseFFN(
    gate_proj: Linear.init(gate_weight),
    up_proj: Linear.init(up_weight),
    down_proj: Linear.init(down_weight),
    activation: activation
  )

proc forward*(self: GatedDenseFFN, x: Tensor): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   x: Input tensor of shape (..., hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (..., hidden_size)
  ##
  ## Computes:
  ##   gate_out = self.gate_proj(x)     # (..., intermediate_size)
  ##   up_out   = self.up_proj(x)       # (..., intermediate_size)
  ##   act      = silu(gate_out) * up_out
  ##   return self.down_proj(act)
  let gate_out = self.gate_proj(x)
  let up_out = self.up_proj(x)
  let act_out =
    case self.activation
    of kSilu:
      # TODO silu_and_mul fusion
      # Port: the fused k_silu_mul kernel is this fusion. silu(`gate`) and the product stay f32, one rounding at the store.
      # Reference order: F.silu(gate_out) * up_out rounds silu(gate_out) to the input dtype ahead of the product.
      # ops_2i1o/silu_mul.nim records the rounding boundary against this order in its header.
      #
      # Flatten for the rank-2 silu·mul op: the activation of shape
      # (..., intermediate_size) flattens to (tokens, intermediate_size),
      # then the leading dimensions restore over the activation
      # (the Linear port's flatten).
      block:
        let s = gate_out.node.getImpl().shape()
        var tokens = 1
        for i in 0 ..< s.len - 1:
          tokens *= s[i]
        let inter = s[s.len - 1]
        let act_flat = Tensor(
          engine: gate_out.engine,
          node: tn_silu_mul(gate_out.engine,
            gate_out.reshape([tokens, inter]).node,
            up_out.reshape([tokens, inter]).node))
        var actShape: seq[int]
        for i in 0 ..< s.len:
          actShape.add(s[i])
        act_flat.reshape(actShape)
    of kGeluTanh:
      raise ValueError.newException(
        "GatedDenseFFN.forward: kGeluTanh unported, the reference " &
        "gelu_tanh(gate_out) * up_out host op binds libtorch " &
        "(gemma activation, out of scope)")
  result = self.down_proj(act_out)

template `()`*(layer: GatedDenseFFN, x: Tensor): untyped =
  forward(layer, x)
