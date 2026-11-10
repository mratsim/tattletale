# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/json,
  std/os,
  std/options,
  workspace/tensorcery/src/tensorcery,
  workspace/safetensors/src/collections,
  ../weights,
  ../layers/embedding,
  ../layers/linear,
  ../layers/norm,
  ../layers/rope,
  ../layers/attention,
  ../layers/ffn,
  ../layers/decoder_layers,
  ../layers/lmhead,
  ../stateful/page_pool,
  ../stateful/inference_context,
  ./all_interfaces

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════════════
# Qwen3 configuration
# ═══════════════════════════════════════════════════════════════════════

type
  Qwen3Config* = ref object
    architecture*: string
    model_type*: string
    num_hidden_layers*: int
    hidden_size*: int
    vocab_size*: int
    rms_norm_eps*: float
    torch_dtype*: string
    num_attention_heads*: int
    num_key_value_heads*: int
    head_dim*: int
    intermediate_size*: int
    hidden_act*: string
    max_position_embeddings*: int
    rope_theta*: float64
    use_qk_norm*: bool
    attention_bias*: bool
    tie_word_embeddings*: bool
    bos_token_id*: int
    eos_token_id*: int

proc parseQwen3Config*(json: JsonNode): Qwen3Config =
  ## Parse the config.json of a Qwen3 checkpoint.
  ##
  ## Args:
  ##   json: the parsed config.json
  ##
  ## The Metal port drops the fields it does not use:
  ## Qwen3-0.6B ships none of them:
  ## no sliding window, no quantization, no rope_scaling.
  result = new Qwen3Config

  result.architecture = json{"architectures"}[0].getStr()
  result.model_type = json{"model_type"}.getStr()
  result.vocab_size = json{"vocab_size"}.getInt().int
  result.hidden_size = json{"hidden_size"}.getInt().int
  result.num_hidden_layers = json{"num_hidden_layers"}.getInt().int
  result.rms_norm_eps = json{"rms_norm_eps"}.getFloat()
  result.torch_dtype = json{"torch_dtype"}.getStr("bfloat16")
  result.num_attention_heads = json{"num_attention_heads"}.getInt().int
  result.num_key_value_heads = json{"num_key_value_heads"}.getInt().int
  result.head_dim = json{"head_dim"}.getInt().int
  result.intermediate_size = json{"intermediate_size"}.getInt().int
  result.hidden_act = json{"hidden_act"}.getStr()
  result.max_position_embeddings = json{"max_position_embeddings"}.getInt().int
  result.rope_theta = json{"rope_theta"}.getFloat()
  result.use_qk_norm = json{"use_qk_norm"}.getBool(false)
  result.attention_bias = json{"attention_bias"}.getBool(false)
  result.tie_word_embeddings = json{"tie_word_embeddings"}.getBool(true)
  result.bos_token_id = json{"bos_token_id"}.getInt().int
  result.eos_token_id = json{"eos_token_id"}.getInt().int

proc loadQwen3Config*(path: string): Qwen3Config =
  parseQwen3Config(path.parseFile())

# ═══════════════════════════════════════════════════════════════════════
# Qwen3 model
# ═══════════════════════════════════════════════════════════════════════

type
  Qwen3DecoderLayer* = DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm]

  Qwen3Model* = ref object of RootRef
    embedTokens: Embedding
    layers: seq[Qwen3DecoderLayer]
    norm: RmsNorm
    lmHead: LMHead
    config*: Qwen3Config
    rotary*: RotaryPositionEmbedding

proc forwardHidden(self: Qwen3Model, ctx: var InferenceContext[uint32, Page], input_ids: openArray[int]): Tensor =
  ## Transformer stack over the long residual stream, without the LM head:
  ## final-normed hidden states of shape (1, seq_len, hidden_size).
  ##
  ## One eval per layer walks that layer's graph. The KV pool write
  ## hangs off the attention node.
  var x = self.embedTokens.forward(input_ids, @[1, input_ids.len])

  # Populate ctx.cos/sin from the model's RoPE cache
  ctx.setRopeForPositions(self.rotary)

  # Per-layer evals keep the walk linear: one eval over the whole stack
  # would re-walk the residual fan-out exponentially.
  # Each layer's hidden-residual pair shares only its own chain.
  var residual = none(Tensor)
  for layer in self.layers:
    let (x2, r2) = layer.forward(ctx, x, residual)
    eval(x2, r2)
    x = x2
    residual = some(r2)

  # Final norm over the full stream. The add folds x
  # with the accumulated residual, and the norm consumes the sum.
  let summed = x + residual.unsafeGet()
  self.norm.forward(summed)

proc forward*(self: Qwen3Model, ctx: var InferenceContext[uint32, Page], input_ids: openArray[int]): Tensor =
  ## Forward pass over the long residual stream, full-sequence logits.
  ##
  ## Args:
  ## - ctx: InferenceContext with page refs (populated by the orchestrator)
  ## - input_ids: token ids of shape (1, seq), passed flat with one sequence
  ##
  ## Returns:
  ##   Logits of shape (1, seq_len, vocab_size) in bfloat16.
  ##
  ## The generation path is forwardLast: it projects only the last
  ## position through the LM head. The full-sequence projection this proc
  ## performs is wasted work there, the greedy decision reads the last position.
  let normed = self.forwardHidden(ctx, input_ids)
  result = self.lmHead.forward(normed)
  eval(result)

proc forwardLast*(self: Qwen3Model, ctx: var InferenceContext[uint32, Page], input_ids: openArray[int]): Tensor =
  ## Forward pass over the long residual stream, last-position logits only.
  ##
  ## Args:
  ## - ctx: InferenceContext with page refs (populated by the orchestrator)
  ## - input_ids: token ids of shape (1, seq), passed flat with one sequence
  ##
  ## Returns:
  ##   Logits of shape (1, 1, vocab_size) in bfloat16, the last sequence
  ##   position's next-token distribution.
  ##
  ## The transformer stack runs over the whole prompt exactly like forward.
  ## Only the LM head projection narrows to the last position:
  ##
  ## - a full-sequence projection would build (seq_len, vocab_size) logits
  ##   whose entries beyond the last position are never read
  let normed = self.forwardHidden(ctx, input_ids)
  # Rank-2 narrow of the flattened stream stays dense contiguous
  # and reshapes. A rank-3 narrow would not reshape.
  let s = normed.shape()
  let flat = normed.reshape([s[1], s[2]])
  let lastRow = flat.narrow(0, s[1] - 1, 1)
  result = self.lmHead.forward(lastRow).reshape([1, 1, self.config.vocab_size])
  eval(result)

proc loadQwen3Model*(modelPath: string): Qwen3Model =
  ## Load a Qwen3 checkpoint from `modelPath` (config.json + model.safetensors).
  ##
  ## Args:
  ##   modelPath: checkpoint directory holding config.json, model.safetensors
  ##
  ## Every weight tensor takes one DMA through the zero-copy loader, bf16 only.
  let config = loadQwen3Config(modelPath / "config.json")
  let weights = SafetensorsCollection.open(modelPath / "model.safetensors")

  proc w(name: string): Tensor =
    weights.loadTensor(name)

  let embedTokens = Embedding.init(w("model.embed_tokens.weight"))

  let rotary = RotaryPositionEmbedding.new(
    config.head_dim, config.max_position_embeddings, config.rope_theta)

  var layers = newSeq[Qwen3DecoderLayer](config.num_hidden_layers)
  for i in 0 ..< config.num_hidden_layers:
    let lp = "model.layers." & $i & "."
    let attnNorm = RmsNorm.init(w(lp & "input_layernorm.weight"), config.rms_norm_eps)
    let mlpNorm = RmsNorm.init(w(lp & "post_attention_layernorm.weight"), config.rms_norm_eps)
    let attn = RopeGQAttention[RmsNorm].init(
      i, lp & "self_attn",
      Linear.init(w(lp & "self_attn.q_proj.weight")),
      Linear.init(w(lp & "self_attn.k_proj.weight")),
      Linear.init(w(lp & "self_attn.v_proj.weight")),
      Linear.init(w(lp & "self_attn.o_proj.weight")),
      config.num_attention_heads, config.num_key_value_heads, config.head_dim,
      rotary,
      RmsNorm.init(w(lp & "self_attn.q_norm.weight"), config.rms_norm_eps),
      RmsNorm.init(w(lp & "self_attn.k_norm.weight"), config.rms_norm_eps))
    let mlp = GatedDenseFFN.init(
      Linear.init(w(lp & "mlp.gate_proj.weight")),
      Linear.init(w(lp & "mlp.up_proj.weight")),
      Linear.init(w(lp & "mlp.down_proj.weight")))
    layers[i] = DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm].init(
      attnNorm, attn, mlpNorm, mlp)

  let norm = RmsNorm.init(w("model.norm.weight"), config.rms_norm_eps)
  # tie_word_embeddings: the LM head shares the embedding table
  # (Qwen3-0.6B ships tied). An untied checkpoint reads lm_head.weight instead.
  let lmHead =
    if config.tie_word_embeddings:
      LMHead.initTied(embedTokens)
    else:
      LMHead.init(w("lm_head.weight"))

  result = Qwen3Model(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    config: config,
    rotary: rotary
  )

static:
  # Register the Qwen3 model in the registry
  registerModel("Qwen3ForCausalLM", proc (modelPath: string): RootRef =
    cast[RootRef](loadQwen3Model(modelPath)))
