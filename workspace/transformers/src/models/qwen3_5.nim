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
  std/strutils,
  std/tables,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch,
  workspace/safetensors,
  workspace/toktoktok,
  ../layers,
  ../deserialization,
  ../stateful/inference_context,
  ./all_interfaces

{.experimental: "callOperator".}

################################################################################
#                          Qwen3.5 Configuration                               #
################################################################################

type
  Qwen3_5Config* = ref object
    ## Config of the Qwen3.5-0.8B text stack, parsed from the nested
    ## `text_config` object of the wrapper `Qwen3_5ForConditionalGeneration`
    ## config.json. The top level only carries the wrapper architecture,
    ## the vision config and the image/video token ids.
    architecture*: string
    model_type*: string
    vocab_size*: int
    hidden_size*: int
    num_hidden_layers*: int
    num_attention_heads*: int
    num_key_value_heads*: int
    head_dim*: int
    intermediate_size*: int
    rms_norm_eps*: float
    hidden_act*: string
    max_position_embeddings*: int
    dtype*: string          ## Activation dtype (config uses `dtype`, there is no `torch_dtype` key)
    rope_theta*: float64
    partial_rotary_factor*: float64
    mrope_interleaved*: bool
    mrope_section*: seq[int]
    rope_type*: string
    attn_output_gate*: bool  ## Full-attention q_proj emits q and a raw per-head scaling vector
    full_attention_interval*: int
    layer_types*: seq[string]  ## Per-layer kind: "linear_attention" or "full_attention"
    linear_conv_kernel_dim*: int
    linear_key_head_dim*: int
    linear_num_key_heads*: int
    linear_num_value_heads*: int
    linear_value_head_dim*: int
    mamba_ssm_dtype*: string
    attention_bias*: bool
    attention_dropout*: float64
    use_cache*: bool
    tie_word_embeddings*: bool
    mlp_only_layers*: seq[string]
    mtp_num_hidden_layers*: int
    mtp_use_dedicated_embeddings*: bool
    bos_token_id*: Option[int]
    eos_token_id*: int
    image_token_id*: int
    video_token_id*: int
    transformers_version*: string

proc parseQwen3_5Config(json: JsonNode): Qwen3_5Config =
  ## Parse a Qwen3.5 wrapper config.json into the text-stack config.
  ##
  ## All model fields live in the nested `text_config` object:
  ## - `rope_parameters` holds rope_theta, partial_rotary_factor and the
  ##   mrope fields (text-only inference collapses mrope to plain partial
  ##   NeoX rotary, so only the scalar fields are consumed downstream)
  ## - `layer_types` names each of the 24 layers "linear_attention"
  ##   (Gated DeltaNet) or "full_attention"
  ## - `linear_*` fields size the Gated DeltaNet block
  ## - `bos_token_id` is absent in this model (Option[int] stays none)
  let textCfg = json{"text_config"}
  let ropeParams = textCfg{"rope_parameters"}

  result = new Qwen3_5Config
  result.architecture = json{"architectures"}[0].getStr()
  result.model_type = textCfg{"model_type"}.getStr()
  result.vocab_size = textCfg{"vocab_size"}.getInt().int
  result.hidden_size = textCfg{"hidden_size"}.getInt().int
  result.num_hidden_layers = textCfg{"num_hidden_layers"}.getInt().int
  result.num_attention_heads = textCfg{"num_attention_heads"}.getInt().int
  result.num_key_value_heads = textCfg{"num_key_value_heads"}.getInt().int
  result.head_dim = textCfg{"head_dim"}.getInt().int
  result.intermediate_size = textCfg{"intermediate_size"}.getInt().int
  result.rms_norm_eps = textCfg{"rms_norm_eps"}.getFloat()
  result.hidden_act = textCfg{"hidden_act"}.getStr()
  result.max_position_embeddings = textCfg{"max_position_embeddings"}.getInt().int
  result.dtype = textCfg{"dtype"}.getStr("bfloat16")
  result.rope_theta = ropeParams{"rope_theta"}.getFloat(1e6)
  result.partial_rotary_factor = ropeParams{"partial_rotary_factor"}.getFloat(1.0)
  result.mrope_interleaved = ropeParams{"mrope_interleaved"}.getBool(false)
  result.rope_type = ropeParams{"rope_type"}.getStr("default")
  result.mrope_section = newSeq[int]()
  if ropeParams{"mrope_section"}.kind != JNull:
    for i in 0 ..< ropeParams{"mrope_section"}.len:
      result.mrope_section.add(ropeParams{"mrope_section"}[i].getInt().int)
  result.attn_output_gate = textCfg{"attn_output_gate"}.getBool(false)
  result.full_attention_interval = textCfg{"full_attention_interval"}.getInt(4)
  result.layer_types = newSeq[string]()
  if textCfg{"layer_types"}.kind != JNull:
    for i in 0 ..< textCfg{"layer_types"}.len:
      result.layer_types.add(textCfg{"layer_types"}[i].getStr())
  result.linear_conv_kernel_dim = textCfg{"linear_conv_kernel_dim"}.getInt().int
  result.linear_key_head_dim = textCfg{"linear_key_head_dim"}.getInt().int
  result.linear_num_key_heads = textCfg{"linear_num_key_heads"}.getInt().int
  result.linear_num_value_heads = textCfg{"linear_num_value_heads"}.getInt().int
  result.linear_value_head_dim = textCfg{"linear_value_head_dim"}.getInt().int
  result.mamba_ssm_dtype = textCfg{"mamba_ssm_dtype"}.getStr("float32")
  result.attention_bias = textCfg{"attention_bias"}.getBool(false)
  result.attention_dropout = textCfg{"attention_dropout"}.getFloat(0.0)
  result.use_cache = textCfg{"use_cache"}.getBool(true)
  result.tie_word_embeddings = textCfg{"tie_word_embeddings"}.getBool(true)
  result.mlp_only_layers = newSeq[string]()
  if textCfg{"mlp_only_layers"}.kind != JNull:
    for i in 0 ..< textCfg{"mlp_only_layers"}.len:
      result.mlp_only_layers.add(textCfg{"mlp_only_layers"}[i].getStr())
  result.mtp_num_hidden_layers = textCfg{"mtp_num_hidden_layers"}.getInt().int
  result.mtp_use_dedicated_embeddings = textCfg{"mtp_use_dedicated_embeddings"}.getBool(false)
  result.bos_token_id = if textCfg{"bos_token_id"}.kind == JNull:
    none(int)
  else:
    some(textCfg{"bos_token_id"}.getInt().int)
  result.eos_token_id = textCfg{"eos_token_id"}.getInt().int
  result.image_token_id = json{"image_token_id"}.getInt().int
  result.video_token_id = json{"video_token_id"}.getInt().int
  result.transformers_version = json{"transformers_version"}.getStr("")

proc loadQwen3_5Config(path: string): Qwen3_5Config =
  ## Load a Qwen3.5 config.json from disk and parse it.
  let json = path.parseFile()
  result = parseQwen3_5Config(json)

proc numKvGroups(cfg: Qwen3_5Config): int =
  ## Number of query heads per KV head (GQA group size).
  cfg.num_attention_heads div cfg.num_key_value_heads

################################################################################
#                          Qwen3.5 Shard Layout                               #
################################################################################

type
  ShardTensorCounts* = object
    ## Tensor counts of a Qwen3.5 shard, grouped by prefix.
    total*: int
    languageModel*: int  ## `model.language_model.*` — the text stack this model loads
    visual*: int         ## `model.visual.*` — vision tower, never loaded
    mtp*: int            ## `mtp.*` — multi-token-prediction draft head, never loaded

proc countShardTensors*(st: Safetensor): ShardTensorCounts =
  ## Count the tensors of a Qwen3.5 shard by top-level prefix.
  ##
  ## The single shard holds three disjoint groups:
  ## - `model.language_model.*` — the text stack loaded by this model
  ## - `model.visual.*` — vision tower
  ## - `mtp.*` — MTP draft head
  ##
  ## The loader only requests `model.language_model.*` names, so the
  ## vision and MTP tensors are skipped without being read.
  result.total = st.tensors.len
  for name in st.tensors.keys():
    if name.startsWith("model.language_model."):
      inc result.languageModel
    elif name.startsWith("model.visual."):
      inc result.visual
    elif name.startsWith("mtp."):
      inc result.mtp

################################################################################
#                          Qwen3.5 Model                                      #
################################################################################

type
  Qwen3_5Model* = ref object
    embedTokens: Embedding
    norm: RmsNorm
    lmHead: LMHead
    config*: Qwen3_5Config
    tokenizer*: BPETokenizer
    device*: DeviceKind

proc forward*(self: Qwen3_5Model, ctx: var InferenceContext, input_ids: Tensor): Tensor =
  ## Text forward pass: embed → final RMSNorm → tied lm_head.
  ##
  ## The 24 hybrid layers (18 Gated DeltaNet, 6 full attention) are not
  ## wired yet, so the forward skips them entirely. This minimal path lets
  ## `generate()` run end to end. Layer dispatch is added in later work.
  let x = self.embedTokens(input_ids)
  let normed = self.norm(x)
  result = self.lmHead(normed)

proc getConfig(self: Qwen3_5Model): ModelConfigBase =
  ## Minimal config for InferenceContext creation in `generate()`.
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
    eosTokenId: self.config.eos_token_id
  )

proc getTokenizer(self: Qwen3_5Model): BPETokenizer =
  self.tokenizer

proc getDeviceKind(self: Qwen3_5Model): DeviceKind =
  self.device

proc loadQwen3_5ModelRaw(modelPath: string, device = kCPU): Qwen3_5Model =
  ## Load the Qwen3.5 text stack from the single shard
  ## `model.safetensors-00001-of-00001.safetensors`.
  ##
  ## Tensor requests are name-based under the `model.language_model.`
  ## prefix. The foreign `model.visual.*` (vision tower) and `mtp.*`
  ## (draft head) tensors living in the same shard are never requested,
  ## so the load skips them without error.
  ##
  ## The shard has no `lm_head` tensor (`tie_word_embeddings: true`), so
  ## LMHead.load falls back to the tied embedding.
  let config = loadQwen3_5Config(modelPath / "config.json")
  let weightsPath = modelPath / "model.safetensors-00001-of-00001.safetensors"
  var weightsMemFile = memFiles.open(weightsPath, mode = fmRead)
  defer: close(weightsMemFile)
  var weightsSt = safetensors.load(weightsMemFile)

  # Raw config JSON for deserialization (codecs inspect quantization_config)
  let cfgJson = (modelPath / "config.json").parseFile()

  let embedWeight = Embedding.load(weightsSt, cfgJson, "model.language_model.embed_tokens", device)
  let embedTokens = Embedding.init(embedWeight)

  let norm = RmsNorm.load(weightsSt, cfgJson, "model.language_model.norm", device)

  let lmHead = LMHead.load(weightsSt, cfgJson, embedTokens, device)

  let tokenizerPath = modelPath / "tokenizer.json"
  let tokenizer = loadHFTokenizer(tokenizerPath)
  result = Qwen3_5Model(
    embedTokens: embedTokens,
    norm: norm,
    lmHead: lmHead,
    config: config,
    tokenizer: tokenizer,
    device: device
  )

proc loadQwen3_5Model*(modelPath: string, device = kCPU): Model =
  let qwen3_5Model = loadQwen3_5ModelRaw(modelPath, device)
  # iface generates to[Model] converter automatically
  qwen3_5Model.to(Model)

static:
  # Register Qwen3.5 model in the registry
  ModelRegistry["Qwen3_5ForConditionalGeneration"] = loadQwen3_5Model
