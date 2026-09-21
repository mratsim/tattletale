# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
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

# ─── North-Mini configuration ──────────────────────────────────────────────

type
  NorthConfig* = ref object
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
      ## Expert body width of the routed blocks.
    prefixDenseIntermediateSize*: int
      ## FFN width of the dense prefix layers.
    firstKDenseReplace*: int
      ## Layer count of the dense prefix, every later layer routes.
    prefixDenseSlidingWindowPattern*: int
      ## Dense prefix forces rope when this is 1.
    hidden_act*: string
    max_position_embeddings*: int
    rope_theta*: float64
      ## Single rope base of the checkpoint.
    sliding_window*: int
      ## Visibility band of the sliding_attention layers.
    num_experts*: int
    num_experts_per_tok*: int
    expert_selection_fn*: string
    norm_topk_prob*: bool
    attention_bias*: bool
    use_qk_norm*: bool
    use_parallel_block*: bool
    logit_scale*: float
      ## Multiplier of the lm_head output.
    layerKinds*: seq[AttentionLayerKind]
      ## Per-layer attention kinds, parsed from the `layer_types` list.
    bos_token_id*: int
    eos_token_id*: int
      ## Stop token of the checkpoint config.

func parseNorthConfig(json: JsonNode): NorthConfig =
  result = new NorthConfig

  result.architecture = json{"architectures"}[0].getStr()
  result.model_type = json{"model_type"}.getStr()
  result.vocab_size = json{"vocab_size"}.getInt().int
  result.hidden_size = json{"hidden_size"}.getInt().int
  result.num_hidden_layers = json{"num_hidden_layers"}.getInt().int
  result.rms_norm_eps = json{"rms_norm_eps"}.getFloat()
  result.torch_dtype = json{"dtype"}.getStr(
    json{"torch_dtype"}.getStr("bfloat16"))
  result.num_attention_heads = json{"num_attention_heads"}.getInt().int
  result.num_key_value_heads = json{"num_key_value_heads"}.getInt().int
  result.head_dim = json{"head_dim"}.getInt().int
  result.intermediate_size = json{"intermediate_size"}.getInt().int
  result.prefixDenseIntermediateSize =
    json{"prefix_dense_intermediate_size"}.getInt().int
  result.firstKDenseReplace = json{"first_k_dense_replace"}.getInt(0)
  result.prefixDenseSlidingWindowPattern =
    json{"prefix_dense_sliding_window_pattern"}.getInt(0)
  result.hidden_act = json{"hidden_act"}.getStr()
  result.max_position_embeddings = json{"max_position_embeddings"}.getInt().int
  result.rope_theta = json{"rope_theta"}.getFloat(
    json{"rope_parameters"}{"rope_theta"}.getFloat())
  result.sliding_window = json{"sliding_window"}.getInt().int
  result.num_experts = json{"num_experts"}.getInt().int
  result.num_experts_per_tok = json{"num_experts_per_tok"}.getInt().int
  result.expert_selection_fn = json{"expert_selection_fn"}.getStr()
  result.norm_topk_prob = json{"norm_topk_prob"}.getBool(false)
  result.attention_bias = json{"attention_bias"}.getBool(false)
  result.use_qk_norm = json{"use_qk_norm"}.getBool(false)
  result.use_parallel_block = json{"use_parallel_block"}.getBool(false)
  result.logit_scale = json{"logit_scale"}.getFloat(1.0).float
  result.bos_token_id = json{"bos_token_id"}.getInt().int
  result.eos_token_id = json{"eos_token_id"}.getInt().int

  # The attention schedule ships as an explicit `layer_types` list
  # on this checkpoint shape.
  if json.hasKey("layer_types"):
    var layerIdx = 0
    for entry in json{"layer_types"}.items():
      result.layerKinds.add parseAttnFromHfTransformers(entry.getStr(),
        "layer_types[" & $layerIdx & "]")
      inc layerIdx
    checkValue(result.layerKinds.len == result.num_hidden_layers,
      "[ttt] NorthConfig: layer_types carries " & $result.layerKinds.len &
      " entries for " & $result.num_hidden_layers & " layers")
  else:
    raise newException(ValueError,
      "[ttt] NorthConfig: the f-first shape reads its layer kinds off the " &
      "layer_types list, the config carries none")

proc loadNorthConfig(path: string): NorthConfig =
  let json = path.parseFile()
  result = parseNorthConfig(json)

# ─── North-Mini model ──────────────────────────────────────────────────────

type
  NorthModel* = ref object
    embedTokens: Embedding
    layers: seq[AnyDecoderLayer]
      ## Dense prefix block and routed blocks on one sequence,
      ## both seated as FanoutDecoderLayer instantiations.
    norm: RmsNorm
    lmHead: LMHead
    config*: NorthConfig
    ropeApplied*: seq[bool]
      ## Per-layer rope decision.
      ##
      ## - Sliding layers and the forced-rope dense prefix rotate
      ## - Full routed layers do not
    rotary*: RotaryPositionEmbedding
      ## Single rope table of the rotating layers.
    tokenizerPath: string
      ## Deferred tokenizer binding, the North-Mini tokenizer is loaded
      ## on first `getTokenizer` use, never at model load.
    tokenizer: BPETokenizer
      ## Nil until the first successful `getTokenizer` call.
    device*: DeviceKind

proc loadRopePairedProjection(view: SafetensorsCollection, cfg: JsonNode,
    prefix: string, numHeads, headDim: int, device: DeviceKind): Linear =
  ## Q or K projection of a rope-carrying layer, the checkpoint rows
  ## reordered per head from the interleaved pair layout to the half-split
  ## layout the windowed spelling's rotation pairs.
  ##
  ## Channel pairing differs between the two rotations.
  ##
  ## - Checkpoint rows (cohere lineage) rotate the channel pairs
  ##   (2i, 2i + 1) by the row-i angle
  ## - Windowed spelling rotation pairs (i, i + head_dim/2)
  ##
  ## After the row reorder, spelling channels (i, i + head_dim/2)
  ## hold the checkpoint pair (2i, 2i + 1), so the rotation computes
  ## the reference pair values exactly.
  ##
  ## Rotated outputs stay in the reordered layout.
  ## The attention reads the same value set, one permutation
  ## of the reference channel order.
  ##
  ## Expected input:
  ##
  ## - the checkpoint `[num_heads * head_dim, hidden]` projection weight,
  ##   bias absent (`attention_bias` false on this checkpoint shape)
  ##
  ## Output:
  ##
  ## - a Linear whose output channels sit in the reordered layout
  let w = view.getTensorOwned(prefix & ".weight", device)
  checkValue(w.size(0) == numHeads * headDim,
    "[ttt] " & prefix & ".weight rows are " & $w.size(0) &
    ", expected num_heads * head_dim " & $(numHeads * headDim))
  let hidden = w.size(1)
  let paired = w.reshape([numHeads, headDim div 2, 2, hidden])
  let evens = paired.narrow(2, 0, 1).squeeze(2)
  let odds = paired.narrow(2, 1, 1).squeeze(2)
  Linear.init(F.cat([evens, odds], 1).reshape([numHeads * headDim, hidden]))

proc forward*(self: NorthModel, ctx: var InferenceContext, input_ids: Tensor): Tensor =
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
    # Rope rows are per-layer request state on the context.
    # - Sliding layers and the forced-rope dense prefix rotate
    # - Full routed layers run unrotated and consume no rope at all
    if self.ropeApplied[i]:
      ctx.setRopeForPositions(self.rotary)
      # The reference rotation computes in f32 over these bf16-grid rows,
      # one rounding back to the hidden dtype at the rotation output.
      ctx.cos = ctx.cos.to(F.kFloat32)
      ctx.sin = ctx.sin.to(F.kFloat32)
    else:
      # Zero-angle rows make the spelling's rotation an exact identity
      # (q * 1 + rotateHalf(q) * 0 = q), the data form of an unrotated
      # layer on a spelling that always rotates.
      let n = ctx.position_ids.numel()
      let opts = F.tensorOptions(F.kBFloat16, self.device)
      ctx.cos = F.ones(n, self.config.head_dim, opts)
      ctx.sin = F.zeros(n, self.config.head_dim, opts)
    let layerOut = self.layers[i].forward(ctx, x, residual)
    x = layerOut[0]
    residual = some(layerOut[1])

  let finalResidual = residual.get(x)
  let normed = self.norm(x + finalResidual)
  result = self.lmHead(normed) * Scalar(self.config.logit_scale)

func getConfig(self: NorthModel): ModelConfigBase =
  ModelConfigBase(
    architecture: self.config.architecture,
    model_type: self.config.model_type,
    num_hidden_layers: self.config.num_hidden_layers,
    hidden_size: self.config.hidden_size,
    vocab_size: self.config.vocab_size,
    rms_norm_eps: self.config.rms_norm_eps.float,
    torch_dtype: self.config.torch_dtype,
    num_attention_heads: self.config.num_attention_heads,
    num_key_value_heads: self.config.num_key_value_heads,
    head_dim: self.config.head_dim,
    intermediate_size: self.config.intermediate_size,
    max_position_embeddings: self.config.max_position_embeddings,
    eosTokenId: self.config.eos_token_id,
    layerKinds: self.config.layerKinds
  )

func getTokenizer(self: NorthModel): BPETokenizer =
  ## Checkpoint tokenizer, loaded on first use.
  ##
  ## Raises:
  ##   - `ValueError` on every call, the North-Mini tokenizer has no
  ##     converter yet and text tokenization is unavailable
  ##
  ## The North-Mini tokenizer pre-tokenizes through a Sequence of Split
  ## regex stages over a ByteLevel BPE model, a shape the HF-to-tiktoken
  ## conversion path rejects, hence the deferred binding.
  ##
  ## The suites replay recorded token ids and never encode.
  if self.tokenizer.isNil:
    raise newException(ValueError,
      "[ttt] NorthModel: the North-Mini tokenizer has no converter " &
      "in toktoktok yet, text tokenization is unavailable for " &
      self.tokenizerPath)
  self.tokenizer

func getDeviceKind(self: NorthModel): DeviceKind =
  self.device

proc loadNorthModelRaw(modelPath: string, device: DeviceKind): NorthModel =
  let config = loadNorthConfig(modelPath / "config.json")
  let weights = SafetensorsCollection.open(modelPath)

  # Raw config JSON for deserialization (codecs inspect quantization_config)
  let cfgJson = (modelPath / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)

  checkValue(config.hidden_act == "silu",
    "[ttt] NorthConfig: hidden_act is \"" & config.hidden_act &
    "\", the port implements the silu checkpoint family only")
  checkValue(not config.attention_bias,
    "[ttt] NorthConfig: attention_bias checkpoints carry biases the port " &
    "does not seat")
  checkValue(not config.use_qk_norm,
    "[ttt] NorthConfig: use_qk_norm checkpoints carry q/k norms the port " &
    "does not seat")
  checkValue(config.use_parallel_block,
    "[ttt] NorthConfig: the port seats the fanout residual block only")
  checkValue(config.expert_selection_fn == "sigmoid",
    "[ttt] NorthConfig: expert_selection_fn is \"" &
    config.expert_selection_fn &
    "\", the port routes through the sigmoid top-k router only")

  # Rope decisions of the f-first shape:
  # - sliding layers rotate
  # - dense prefix rotates when its sliding window pattern is 1
  # - full routed layers never do
  var ropeApplied = newSeq[bool](config.num_hidden_layers)
  for i in 0 ..< config.num_hidden_layers:
    let dense = i < config.firstKDenseReplace
    ropeApplied[i] = config.layerKinds[i] == alkSlidingAttention or
      (dense and config.prefixDenseSlidingWindowPattern == 1)

  let embedTokens = Embedding.load(weights, cfgJson, "model.embed_tokens", device)

  # Single-theta table over the full position range of the checkpoint.
  let rotary = RotaryPositionEmbedding.new(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_theta,
    actDtype,
    device
  )

  var layers = newSeq[AnyDecoderLayer](config.num_hidden_layers)

  for i in 0 ..< config.num_hidden_layers:

    let lp = "model.layers." & $i & "."
    let sliding = config.layerKinds[i] == alkSlidingAttention

    let inputLN = RmsNorm.load(weights, cfgJson, lp & "input_layernorm", device)

    # The rotating layers read their rope pairing off the reordered
    # projection rows, the unrotated layers load the checkpoint rows as is.
    let qProj =
      if ropeApplied[i]:
        loadRopePairedProjection(weights, cfgJson, lp & "self_attn.q_proj",
          config.num_attention_heads, config.head_dim, device)
      else:
        Linear.load(weights, cfgJson, lp & "self_attn.q_proj", device)
    let kProj =
      if ropeApplied[i]:
        loadRopePairedProjection(weights, cfgJson, lp & "self_attn.k_proj",
          config.num_key_value_heads, config.head_dim, device)
      else:
        Linear.load(weights, cfgJson, lp & "self_attn.k_proj", device)
    let vProj = Linear.load(weights, cfgJson, lp & "self_attn.v_proj", device)
    let oProj = Linear.load(weights, cfgJson, lp & "self_attn.o_proj", device)
    let attn = RopeGQAttention[void].init(
      i, lp & "self_attn", qProj, kProj, vProj, oProj,
      config.num_attention_heads, config.num_key_value_heads,
      config.head_dim, rotary,
      window = if sliding: config.sliding_window else: FullVisibilityWindow)

    # The fanout block feeds both mixers the one input-norm output.
    # Dense prefix block below the first_k_dense_replace boundary,
    # routed block everywhere else.
    layers[i] =
      if i < config.firstKDenseReplace:
        let mlp = GatedDenseFFN.load(weights, cfgJson, lp & "mlp", device)
        FanoutDecoderLayer[RopeGQAttention[void], GatedDenseFFN, RmsNorm].
          init(inputLN, attn, mlp).to(AnyDecoderLayer)
      else:
        # Sigmoid top-k router at its degenerate grouping.
        # - n_group 1 keeps the group mask at all-ones
        # - zero selection bias leaves each weight at the logits' sigmoid,
        #   no renorm, no scale
        # - the master NoAuxTopCorr corner unchanged
        let router = NoAuxTopCorr.init(
          weights.getTensorOwned(lp & "mlp.gate.weight", device),
          F.zeros(config.num_experts,
            F.tensorOptions(F.kFloat32, device)),
          config.num_experts_per_tok,
          1, 1,
          1.0'f64,
          config.norm_topk_prob)
        let mlp = BlockSparseFFN.load(weights, cfgJson, lp & "mlp", router, device)
        FanoutDecoderLayer[RopeGQAttention[void], BlockSparseFFN, RmsNorm].
          init(inputLN, attn, mlp).to(AnyDecoderLayer)

  let norm = RmsNorm.load(weights, cfgJson, "model.norm", device)
  let lmHead = LMHead.load(weights, cfgJson, embedTokens, device)
  result = NorthModel(
    embedTokens: embedTokens,
    layers: layers,
    norm: norm,
    lmHead: lmHead,
    config: config,
    ropeApplied: ropeApplied,
    rotary: rotary,
    tokenizerPath: modelPath / "tokenizer.json",
    device: device
  )

proc loadNorthModel*(modelPath: string, device: DeviceKind): AnyModel =
  ## Loads a North-Mini-Code checkpoint from `modelPath` onto `device`,
  ## wrapped in the AnyModel model interface.
  let northModel = loadNorthModelRaw(modelPath, device)
  # iface generates to[AnyModel] converter automatically
  northModel.to(AnyModel)

static:
  # Register North model in the registry
  ModelRegistry["Cohere2MoeForCausalLM"] = loadNorthModel
