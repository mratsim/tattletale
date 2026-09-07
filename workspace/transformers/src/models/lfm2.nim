# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  std/os,
  std/memfiles,
  std/tables,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch,
  workspace/safetensors,
  workspace/positron,
  workspace/toktoktok,
  ../layers,
  ../layers/short_conv,
  ../deserialization,
  ../stateful/inference_context,
  ./all_interfaces

{.experimental: "callOperator".}

################################################################################
#                          LFM2 Configuration                                 #
################################################################################

type
  Lfm2Config* = ref object
    ## Config of the LFM2.5-230M hybrid stack (model_type `lfm2`, arch `Lfm2ForCausalLM`).
    ## Each layer is either a full softmax attention block or a short-conv block,
    ## named by `layer_types`.
    architecture*: string
    model_type*: string
    vocab_size*: int
    hidden_size*: int
    num_hidden_layers*: int
    num_attention_heads*: int
    num_key_value_heads*: int
    head_dim*: int
    intermediate_size*: int
    rms_norm_eps*: float      ## RMSNorm epsilon (config key `norm_eps`)
    max_position_embeddings*: int
    rope_theta*: float64
    conv_dim*: int            ## Branch width of the short-conv layers (defaults to hidden_size)
    conv_L_cache*: int        ## Causal conv kernel width (K), state width is K-1
    conv_bias*: bool
    layer_types*: seq[string] ## Per-layer kind: "full_attention" or "conv"
    full_attn_idxs*: seq[int]
    tie_word_embeddings*: bool
    dtype*: string            ## Activation dtype (config key `torch_dtype`, default bfloat16)
    bos_token_id*: Option[int]
    eos_token_id*: Option[int]
    pad_token_id*: Option[int]

proc parseLfm2Config(json: JsonNode): Lfm2Config =
  ## Parse an LFM2 config.json.
  ##
  ## Key mappings:
  ## - `rope_theta` lives in the nested `rope_parameters` object
  ## - the RMSNorm epsilon is spelled `norm_eps` (not `rms_norm_eps`)
  ## - `intermediate_size` applies the vendored Lfm2MLP auto-adjust:
  ##   with `block_auto_adjust_ff_dim`, halve it by 2/3, then apply
  ##   `block_ffn_dim_multiplier`, then round up to `block_multiple_of`
  ## - the activation dtype is spelled `dtype` in transformers 5.x configs,
  ##   `torch_dtype` in pre-5.x configs (default bfloat16)
  ## - `layer_types` names each of the `num_hidden_layers` layers
  ##   "full_attention" or "conv". When absent, the parser derives it
  ##   from `full_attn_idxs` (default: every layer full_attention)
  result = new Lfm2Config
  result.architecture = json{"architectures"}[0].getStr()
  result.model_type = json{"model_type"}.getStr()
  result.vocab_size = json{"vocab_size"}.getInt().int
  result.hidden_size = json{"hidden_size"}.getInt().int
  result.num_hidden_layers = json{"num_hidden_layers"}.getInt().int
  result.num_attention_heads = json{"num_attention_heads"}.getInt().int
  result.num_key_value_heads = json{"num_key_value_heads"}.getInt().int
  result.head_dim = if json{"head_dim"}.kind == JNull:
    result.hidden_size div result.num_attention_heads
  else:
    json{"head_dim"}.getInt().int
  result.intermediate_size = json{"intermediate_size"}.getInt().int
  if json{"block_auto_adjust_ff_dim"}.getBool(true):
    result.intermediate_size = int(2.0 * result.intermediate_size.float64 / 3.0)
    if json{"block_ffn_dim_multiplier"}.kind != JNull:
      let multiplier = json{"block_ffn_dim_multiplier"}.getFloat(1.0)
      result.intermediate_size = int(multiplier * result.intermediate_size.float64)
      let multipleOf = json{"block_multiple_of"}.getInt(256).int
      result.intermediate_size = multipleOf *
        ((result.intermediate_size + multipleOf - 1) div multipleOf)
  result.rms_norm_eps = json{"norm_eps"}.getFloat(json{"rms_norm_eps"}.getFloat(1e-5))
  result.max_position_embeddings = json{"max_position_embeddings"}.getInt().int
  result.rope_theta = json{"rope_parameters"}{"rope_theta"}.getFloat(1e6)
  result.conv_dim = json{"conv_dim"}.getInt(result.hidden_size).int
  result.conv_L_cache = json{"conv_L_cache"}.getInt(3).int
  result.conv_bias = json{"conv_bias"}.getBool(false)
  result.tie_word_embeddings = json{"tie_word_embeddings"}.getBool(true)
  result.dtype = json{"dtype"}.getStr(json{"torch_dtype"}.getStr("bfloat16"))
  result.bos_token_id = if json{"bos_token_id"}.kind == JNull:
    none(int)
  else:
    some(json{"bos_token_id"}.getInt().int)
  result.eos_token_id = if json{"eos_token_id"}.kind == JNull:
    none(int)
  else:
    some(json{"eos_token_id"}.getInt().int)
  result.pad_token_id = if json{"pad_token_id"}.kind == JNull:
    none(int)
  else:
    some(json{"pad_token_id"}.getInt().int)
  result.layer_types = newSeq[string]()
  if json{"layer_types"}.kind != JNull:
    for i in 0 ..< json{"layer_types"}.len:
      result.layer_types.add(json{"layer_types"}[i].getStr())
  result.full_attn_idxs = newSeq[int]()
  if json{"full_attn_idxs"}.kind != JNull:
    for i in 0 ..< json{"full_attn_idxs"}.len:
      result.full_attn_idxs.add(json{"full_attn_idxs"}[i].getInt().int)
  if result.layer_types.len == 0:
    if result.full_attn_idxs.len == 0:
      for i in 0 ..< result.num_hidden_layers:
        result.full_attn_idxs.add(i)
    for i in 0 ..< result.num_hidden_layers:
      result.layer_types.add(
        if result.full_attn_idxs.contains(i): "full_attention" else: "conv")

proc loadLfm2Config(path: string): Lfm2Config =
  ## LFM2 config parsed from a config.json file.
  ##
  ## `path` names the config.json itself, callers pass `modelDir / "config.json"`.
  let json = path.parseFile()
  result = parseLfm2Config(json)

################################################################################
#                          LFM2 Model                                         #
################################################################################

type
  Lfm2DecoderLayer* = ref object
    ## One decoder layer of the LFM2 hybrid stack. Exactly one of `self_attn`
    ## (RopeGQAttention) and `conv` (Lfm2ShortConv) is non-nil, per `layer_types[i]`.
    ## Residual pattern matches vendored Lfm2DecoderLayer.forward, local residuals
    ## with bf16 additions.
    layer_type*: string
    operator_norm: RmsNorm
    self_attn: RopeGQAttention  # nil on conv layers
    conv: Lfm2ShortConv         # nil on full_attention layers
    ffn_norm: RmsNorm
    mlp: GatedMLP

  Lfm2Model* = ref object
    embedTokens: Embedding
    layers: seq[Lfm2DecoderLayer]
    norm: RmsNorm
    lmHead: LMHead
    rotary: RotaryPositionEmbeddingRef
    config*: Lfm2Config
    tokenizer*: BPETokenizer
    device*: DeviceKind

func init*(
    _: type Lfm2DecoderLayer,
    layer_type: string,
    operator_norm, ffn_norm: RmsNorm,
    self_attn: RopeGQAttention,
    conv: Lfm2ShortConv,
    mlp: GatedMLP): Lfm2DecoderLayer =
  ## Assemble one hybrid decoder layer. Exactly one of `self_attn` / `conv`
  ## is non-nil, selected by `layer_type` ("full_attention" or "conv").
  Lfm2DecoderLayer(
    layer_type: layer_type,
    operator_norm: operator_norm,
    self_attn: self_attn,
    conv: conv,
    ffn_norm: ffn_norm,
    mlp: mlp
  )

proc forward*(self: Lfm2DecoderLayer, ctx: var InferenceContext, hidden: Tensor): Tensor =
  ## Run one hybrid decoder layer, vendored Lfm2DecoderLayer.forward
  ## with local residuals:
  ##
  ##   h = operator_norm(hidden)
  ##   h = hidden + attn_or_conv(ctx, h)
  ##   h = h + mlp(ffn_norm(h))
  ##
  ## Dispatch on `layer_type`. Conv layers run the short-conv block,
  ## conv history in ctx.convState. Full_attention layers run paged GQA attention,
  ## with KV pages in ctx.
  let residual = hidden
  let hNorm = self.operator_norm(hidden)
  let attnOut =
    if self.layer_type == "full_attention":
      self.self_attn(ctx, hNorm)
    else:
      self.conv(ctx, hNorm)
  let h1 = residual + attnOut
  let hNorm2 = self.ffn_norm(h1)
  result = h1 + self.mlp(hNorm2)

template `()`*(layer: Lfm2DecoderLayer,
            ctx: var InferenceContext,
            x: Tensor): untyped =
  layer.forward(ctx, x)

proc forward*(self: Lfm2Model, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Forward pass: embed → the `num_hidden_layers` hybrid layers →
  ## embedding_norm → tied lm_head.
  ##
  ## `ctx.setRopeForPositions` is called once per forward. Only
  ## full_attention layers read ctx.cos/sin (conv layers carry no rope).
  ## Conv per-sequence state lives in ctx.convState, full-attn state in ctx.pages.
  var h = self.embedTokens(input_ids)

  ctx.setRopeForPositions(self.rotary)

  for layer in self.layers:
    h = layer(ctx, h)

  let normed = self.norm(h)
  result = self.lmHead(normed)

proc getConfig(self: Lfm2Model): ModelConfigBase =
  ## Config subset `generate()` reads to size the Orchestrator and its InferenceContext.
  ModelConfigBase(
    architecture: self.config.architecture,
    model_type: self.config.model_type,
    num_hidden_layers: self.config.num_hidden_layers,
    hidden_size: self.config.hidden_size,
    vocab_size: self.config.vocab_size,
    rms_norm_eps: self.config.rms_norm_eps,
    torch_dtype: self.config.dtype,
    num_attention_heads: self.config.num_attention_heads,
    num_key_value_heads: self.config.num_key_value_heads,
    head_dim: self.config.head_dim,
    intermediate_size: self.config.intermediate_size,
    max_position_embeddings: self.config.max_position_embeddings,
    eosTokenId: self.config.eos_token_id.get(2)
  )

proc getTokenizer(self: Lfm2Model): BPETokenizer =
  self.tokenizer

proc getDeviceKind(self: Lfm2Model): DeviceKind =
  self.device

proc loadLfm2ModelRaw(modelPath: string, device = kCPU): Lfm2Model =
  ## Load the LFM2.5-230M stack from the single shard `model.safetensors`.
  ##
  ## Every layer loads its operator/ffn RMSNorms (epsilon from `config.rms_norm_eps`)
  ## and its SwiGLU MLP (`feed_forward.w1/w3/w2`).
  ## Operator block follows `config.layer_types[i]`:
  ## - full_attention layers load `self_attn.q_proj/k_proj/v_proj/out_proj`
  ##   plus the per-head `q_layernorm`/`k_layernorm` RMSNorms
  ## - conv layers load `conv.in_proj`, `conv.out_proj` and the depthwise
  ##   `conv.conv.weight` (shape (conv_dim, 1, K))
  ##
  ## Final norm is `model.embedding_norm`. The shard has no
  ## `lm_head.weight` tensor (`tie_word_embeddings`), so LMHead.load falls
  ## back to the tied embedding.
  let config = loadLfm2Config(modelPath / "config.json")
  let weightsPath = modelPath / "model.safetensors"
  var weightsMemFile = memFiles.open(weightsPath, mode = fmRead)
  defer: close(weightsMemFile)
  var weightsSt = safetensors.load(weightsMemFile)

  # Raw config JSON for deserialization (codecs inspect quantization_config)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = activationDtype(cfgJson)

  let embedWeight = Embedding.load(weightsSt, cfgJson, "model.embed_tokens", device)
  let embedTokens = Embedding.init(embedWeight)

  let rotary = RotaryPositionEmbeddingRef.new(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_theta,
    actDtype,
    device)

  var layers = newSeq[Lfm2DecoderLayer](config.num_hidden_layers)
  for i in 0 ..< config.num_hidden_layers:
    let lp = "model.layers." & $i & "."

    let operatorNorm = RmsNorm.load(
      weightsSt, cfgJson, lp & "operator_norm", device, eps = some(config.rms_norm_eps))
    let ffnNorm = RmsNorm.load(
      weightsSt, cfgJson, lp & "ffn_norm", device, eps = some(config.rms_norm_eps))

    let w1Proj = Linear.load(weightsSt, cfgJson, lp & "feed_forward.w1", device)
    let w3Proj = Linear.load(weightsSt, cfgJson, lp & "feed_forward.w3", device)
    let w2Proj = Linear.load(weightsSt, cfgJson, lp & "feed_forward.w2", device)
    let mlp = GatedMLP.init(w1Proj, w3Proj, w2Proj, kSilu)

    var selfAttn: RopeGQAttention = nil
    var conv: Lfm2ShortConv = nil
    if config.layer_types[i] == "full_attention":
      let qProj = Linear.load(weightsSt, cfgJson, lp & "self_attn.q_proj", device)
      let kProj = Linear.load(weightsSt, cfgJson, lp & "self_attn.k_proj", device)
      let vProj = Linear.load(weightsSt, cfgJson, lp & "self_attn.v_proj", device)
      let oProj = Linear.load(weightsSt, cfgJson, lp & "self_attn.out_proj", device)
      let qNorm = RmsNorm.load(
        weightsSt, cfgJson, lp & "self_attn.q_layernorm", device, eps = some(config.rms_norm_eps))
      let kNorm = RmsNorm.load(
        weightsSt, cfgJson, lp & "self_attn.k_layernorm", device, eps = some(config.rms_norm_eps))
      selfAttn = RopeGQAttention.init(
        i, lp & "self_attn",
        qProj, kProj, vProj, oProj,
        qNorm, kNorm,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        rotary)
    else:
      let inProj = Linear.load(weightsSt, cfgJson, lp & "conv.in_proj", device)
      let outProj = Linear.load(weightsSt, cfgJson, lp & "conv.out_proj", device)
      let convWeight = weightsSt.getTensorOwned(lp & "conv.conv.weight", device)
      conv = Lfm2ShortConv.init(
        i, lp & "conv",
        inProj, convWeight, outProj,
        config.conv_L_cache, config.conv_dim)

    layers[i] = Lfm2DecoderLayer.init(
      config.layer_types[i], operatorNorm, ffnNorm, selfAttn, conv, mlp)

  let norm = RmsNorm.load(
    weightsSt, cfgJson, "model.embedding_norm", device, eps = some(config.rms_norm_eps))

  let lmHead = LMHead.load(weightsSt, cfgJson, embedTokens, device)

  let tokenizerPath = modelPath / "tokenizer.json"
  let tokenizer = loadHFTokenizer(tokenizerPath)
  result = Lfm2Model(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    rotary: rotary,
    config: config,
    tokenizer: tokenizer,
    device: device,
  )

proc loadLfm2Model*(modelPath: string, device = kCPU): Model =
  let lfm2Model = loadLfm2ModelRaw(modelPath, device)
  # iface generates to[Model] converter automatically
  lfm2Model.to(Model)

static:
  ModelRegistry["Lfm2ForCausalLM"] = loadLfm2Model
