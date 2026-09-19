# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  std/options,
  std/os,
  std/tables,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/positron,
  workspace/toktoktok,
  ../layers,
  ../deserialization,
  ../models/loading/layer_kinds,
  workspace/safetensors/src/collections,
  ../stateful/inference_context,
  ./all_interfaces

{.experimental: "callOperator".}

# ─── Laguna configuration ──────────────────────────────────────────────────

type
  LagunaConfig* = ref object
    architecture*: string
    model_type*: string
    num_hidden_layers*: int
    hidden_size*: int
    vocab_size*: int
    head_dim*: int
    num_key_value_heads*: int
    rms_norm_eps*: float
    torch_dtype*: string
    intermediate_size*: int
      ## FFN width of the dense layers.
    moe_intermediate_size*: int
      ## Expert body width of the routed blocks.
    num_experts*: int
    num_experts_per_tok*: int
    norm_topk_prob*: bool
    moe_routed_scaling_factor*: float64
    mlp_only_layers*: seq[int]
      ## Layer indices that stay dense.
    decoder_sparse_step*: int
      ## Sparse-layer period outside `mlp_only_layers`.
    sliding_window*: int
      ## Visibility band of the sliding_attention layers.
    max_position_embeddings*: int
    numAttentionHeadsPerLayer*: seq[int]
      ## Query-head count per layer, the checkpoint's per-layer schedule.
    layerKinds*: seq[AttentionLayerKind]
      ## Per-layer attention kinds, parsed from the `layer_types` list.
    ropeFull*: LagunaRopeFull
      ## Yarn rope parameters of the full_attention layers.
    ropeSlidingTheta*: float64
      ## Rope base of the sliding_attention layers.
    bos_token_id*: int
    eos_token_ids*: seq[int]
      ## Stop set of the checkpoint config, an eos id list.

  LagunaRopeFull* = object
    ## Yarn rope parameters of the full_attention layers, the checkpoint's `rope_parameters.full_attention` block.
    theta*: float64
    factor*: float64
    betaFast*: float64
    betaSlow*: float64
    originalMaxPos*: int
    attentionFactor*: float64
    rotaryDim*: int
      ## Rotating width, head_dim * partial_rotary_factor.

func parseLagunaConfig(json: JsonNode): LagunaConfig =
  result = new LagunaConfig

  result.architecture = json{"architectures"}[0].getStr()
  result.model_type = json{"model_type"}.getStr()
  result.vocab_size = json{"vocab_size"}.getInt().int
  result.hidden_size = json{"hidden_size"}.getInt().int
  result.num_hidden_layers = json{"num_hidden_layers"}.getInt().int
  result.head_dim = json{"head_dim"}.getInt().int
  result.num_key_value_heads = json{"num_key_value_heads"}.getInt().int
  result.rms_norm_eps = json{"rms_norm_eps"}.getFloat()
  result.torch_dtype = json{"torch_dtype"}.getStr("bfloat16")
  result.intermediate_size = json{"intermediate_size"}.getInt().int
  result.moe_intermediate_size = json{"moe_intermediate_size"}.getInt().int
  result.num_experts = json{"num_experts"}.getInt().int
  result.num_experts_per_tok = json{"num_experts_per_tok"}.getInt().int
  result.norm_topk_prob = json{"norm_topk_prob"}.getBool(false)
  result.moe_routed_scaling_factor = json{"moe_routed_scaling_factor"}.getFloat(1.0)
  result.decoder_sparse_step = json{"decoder_sparse_step"}.getInt(1)
  for entry in json{"mlp_only_layers"}.items():
    result.mlp_only_layers.add entry.getInt().int
  result.sliding_window = json{"sliding_window"}.getInt().int
  result.max_position_embeddings = json{"max_position_embeddings"}.getInt().int
  for entry in json{"num_attention_heads_per_layer"}.items():
    result.numAttentionHeadsPerLayer.add entry.getInt().int
  checkValue(result.numAttentionHeadsPerLayer.len == result.num_hidden_layers,
    "[ttt] LagunaConfig: num_attention_heads_per_layer carries " &
    $result.numAttentionHeadsPerLayer.len & " entries for " &
    $result.num_hidden_layers & " layers")

  if json.hasKey("layer_types"):
    var layerIdx = 0
    for entry in json{"layer_types"}.items():
      result.layerKinds.add parseAttnFromHfTransformers(entry.getStr(),
        "layer_types[" & $layerIdx & "]")
      inc layerIdx
    checkValue(result.layerKinds.len == result.num_hidden_layers,
      "[ttt] LagunaConfig: layer_types carries " & $result.layerKinds.len &
      " entries for " & $result.num_hidden_layers & " layers")
  else:
    raise newException(ValueError,
      "[ttt] LagunaConfig: the checkpoint reads its layer kinds off the " &
      "layer_types list, the config carries none")

  # Per-layer-kind rope blocks. The full_attention rope block is yarn, the sliding_attention one the default theta.
  let ropeFull = json{"rope_parameters"}{"full_attention"}
  checkValue(ropeFull{"rope_type"}.getStr() == "yarn",
    "[ttt] LagunaConfig: full_attention rope_type is \"" &
    ropeFull{"rope_type"}.getStr() & "\", the port implements the yarn family only")
  result.ropeFull = LagunaRopeFull(
    theta: ropeFull{"rope_theta"}.getFloat(),
    factor: ropeFull{"factor"}.getFloat(),
    betaFast: ropeFull{"beta_fast"}.getFloat(),
    betaSlow: ropeFull{"beta_slow"}.getFloat(),
    originalMaxPos: ropeFull{"original_max_position_embeddings"}.getInt().int,
    attentionFactor: ropeFull{"attention_factor"}.getFloat(1.0),
    rotaryDim: int(result.head_dim.float64 *
      ropeFull{"partial_rotary_factor"}.getFloat(1.0)))
  result.ropeSlidingTheta = json{"rope_parameters"}{"sliding_attention"}{"rope_theta"}.getFloat()

  result.bos_token_id = json{"bos_token_id"}.getInt().int
  for e in json{"eos_token_id"}.items():
    result.eos_token_ids.add e.getInt().int

proc loadLagunaConfig(path: string): LagunaConfig =
  let json = path.parseFile()
  result = parseLagunaConfig(json)

# ─── Laguna model ──────────────────────────────────────────────────────────

type
  LagunaDenseLayer* = DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm]
    ## Plain local-residual decoder block, one norm per sublayer input.

  LagunaMoeLayer* = DecoderLayer[RopeGQAttention[RmsNorm], BlockSparseFFN, RmsNorm]
    ## Plain local-residual decoder block with the routed hidden mixer.

  LagunaModel* = ref object
    embedTokens: Embedding
    layers: seq[AnyDecoderLayer]
      ## Dense layer 0 and routed blocks on one sequence.
    norm: RmsNorm
    lmHead: LMHead
    config*: LagunaConfig
    rotaryFull*: RotaryPositionEmbedding
      ## Yarn rope table of the full_attention layers.
    rotarySliding*: RotaryPositionEmbedding
      ## Rope table of the sliding_attention layers, the default theta.
    tokenizerPath: string
      ## Deferred tokenizer binding, the Laguna tokenizer is loaded on first `getTokenizer` use, never at model load.
    tokenizer: BPETokenizer
      ## Nil until the first successful `getTokenizer` call.
    device*: DeviceKind

proc forward*(self: LagunaModel, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Full forward to logits over the input token ids.
  ##
  ## Expected input:
  ##
  ## - `input_ids`, token ids of shape (batch, seq_len)
  ##
  ## Output:
  ##
  ## - logits of shape (batch, seq_len, vocab_size)

  var x = self.embedTokens(input_ids)

  var residual: Option[Tensor]
  for i in 0 ..< self.layers.len:
    # Dual-theta rope. Each layer kind ropes with its own table, the rows
    # travel as per-forward request state on the context.
    ctx.setRopeForPositions(
      if self.config.layerKinds[i] == alkSlidingAttention: self.rotarySliding
      else: self.rotaryFull)
    let layerOut = self.layers[i].forward(ctx, x, residual)
    x = layerOut[0]
    residual = some(layerOut[1])

  let finalResidual = residual.get(x)
  let normed = self.norm(x + finalResidual)
  result = self.lmHead(normed)

func getConfig(self: LagunaModel): ModelConfigBase =
  ModelConfigBase(
    architecture: self.config.architecture,
    model_type: self.config.model_type,
    num_hidden_layers: self.config.num_hidden_layers,
    hidden_size: self.config.hidden_size,
    vocab_size: self.config.vocab_size,
    rms_norm_eps: self.config.rms_norm_eps.float,
    torch_dtype: self.config.torch_dtype,
    num_attention_heads: self.config.numAttentionHeadsPerLayer[0],
    num_key_value_heads: self.config.num_key_value_heads,
    head_dim: self.config.head_dim,
    intermediate_size: self.config.intermediate_size,
    max_position_embeddings: self.config.max_position_embeddings,
    eosTokenIds: self.config.eos_token_ids,
    layerKinds: self.config.layerKinds
  )

func getTokenizer(self: LagunaModel): BPETokenizer =
  ## Checkpoint tokenizer, loaded on first use.
  ##
  ## Raises:
  ##   - `ValueError` on every call, the Laguna tokenizer has no converter
  ##     yet and text tokenization is unavailable
  ##
  ## The suites replay recorded token ids and never encode.
  if self.tokenizer.isNil:
    raise newException(ValueError,
      "[ttt] LagunaModel: the Laguna tokenizer has no converter " &
      "in toktoktok yet, text tokenization is unavailable for " &
      self.tokenizerPath)
  self.tokenizer

func getDeviceKind(self: LagunaModel): DeviceKind =
  self.device

proc loadLagunaModelRaw(modelPath: string, device: DeviceKind): LagunaModel =
  let config = loadLagunaConfig(modelPath / "config.json")
  let weights = SafetensorsCollection.open(modelPath)

  # Raw config JSON for deserialization (codecs inspect quantization_config)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  # Dual-theta tables. The full_attention table is yarn over the partial
  # rotary width. The sliding_attention table runs the default theta over
  # the full head width.
  let rotaryFull = RotaryPositionEmbedding.new(
    config.head_dim,
    config.max_position_embeddings,
    config.ropeFull.theta,
    actDtype,
    device,
    rotary_dim = config.ropeFull.rotaryDim,
    yarnFactor = config.ropeFull.factor,
    yarnBetaFast = config.ropeFull.betaFast,
    yarnBetaSlow = config.ropeFull.betaSlow,
    yarnOriginalMaxPos = config.ropeFull.originalMaxPos,
    attentionFactor = config.ropeFull.attentionFactor)
  let rotarySliding = RotaryPositionEmbedding.new(
    config.head_dim,
    config.max_position_embeddings,
    config.ropeSlidingTheta,
    actDtype,
    device)

  var layers = newSeq[AnyDecoderLayer](config.num_hidden_layers)

  for i in 0 ..< config.num_hidden_layers:

    let lp = "model.layers." & $i & "."
    let sliding = config.layerKinds[i] == alkSlidingAttention

    let inputLN = RmsNorm.load(weights, cfgJson, lp & "input_layernorm", device)
    let postLN = RmsNorm.load(weights, cfgJson, lp & "post_attention_layernorm", device)

    let attn = RopeGQAttention[RmsNorm].load(
      weights, cfgJson, lp & "self_attn", i,
      config.numAttentionHeadsPerLayer[i], config.num_key_value_heads,
      config.head_dim,
      if sliding: rotarySliding else: rotaryFull, device,
      window = if sliding: config.sliding_window else: FullVisibilityWindow,
      perHeadGate = true)

    let routed = i notin config.mlp_only_layers and
      i mod config.decoder_sparse_step == 0
    if routed:
      # Sigmoid top-k router at its degenerate grouping, the reference
      # router runs no group limiting, n_group 1 keeps the group mask
      # all-ones inside the same op sequence.
      let mlpPrefix = lp & "mlp"
      let router = NoAuxTopCorr.init(
        weights.getTensorOwned(mlpPrefix & ".gate.weight", device),
        weights.getTensorOwned(mlpPrefix & ".experts.e_score_correction_bias", device),
        config.num_experts_per_tok, 1, 1,
        config.moe_routed_scaling_factor, config.norm_topk_prob)
      let mlp = BlockSparseFFN.load(weights, cfgJson, mlpPrefix, router, device)
      layers[i] = LagunaMoeLayer.init(
        input_layernorm = inputLN, sequence_mixer = attn,
        post_attention_layernorm = postLN,
        hidden_mixer = mlp).to(AnyDecoderLayer)
    else:
      let mlp = GatedDenseFFN.load(weights, cfgJson, lp & "mlp", device)
      layers[i] = LagunaDenseLayer.init(
        input_layernorm = inputLN, sequence_mixer = attn,
        post_attention_layernorm = postLN,
        hidden_mixer = mlp).to(AnyDecoderLayer)

  let embedTokens = Embedding.load(weights, cfgJson, "model.embed_tokens", device)
  let norm = RmsNorm.load(weights, cfgJson, "model.norm", device)
  let lmHead = LMHead.load(weights, cfgJson, embedTokens, device)
  result = LagunaModel(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    config: config,
    rotaryFull: rotaryFull,
    rotarySliding: rotarySliding,
    tokenizerPath: modelPath / "tokenizer.json",
    device: device
  )

proc loadLagunaModel*(modelPath: string, device: DeviceKind): AnyModel =
  ## Loads a Laguna checkpoint from `modelPath` onto `device`, wrapped
  ## in the AnyModel model interface.
  let lagunaModel = loadLagunaModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  lagunaModel.to(AnyModel)

static:
  # Register the Laguna model in the registry
  ModelRegistry["LagunaForCausalLM"] = loadLagunaModel
