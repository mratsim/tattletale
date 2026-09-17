# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Serialization dispatch — loads layer objects from safetensors.
##
## This is the ONLY file that bridges model loaders with quantization codecs.
## The quant format is detected from the model config.json, so callers
## stay quantization-blind.

import
  std/algorithm,
  std/sugar,
  std/tables,
  pkg/packedjson,
  workspace/safetensors,
  workspace/libtorch,
  ./layers,
  ./layers/attn_ssm/gated_delta_net,
  ./models/loading/config_json,
  ./quantizations/all_reexports,
  workspace/safetensors/src/collections,
  workspace/safetensors/src/safetensors_libtorch,
  ./instrumentation

const BottleneckNormEps* = 1e-6
  ## eps of the compressed-Q bottleneck norms (q_a_layernorm,
  ## kv_a_layernorm): the reference attention classes construct them
  ## WITHOUT an eps argument, so the checkpoint's rms_norm_eps never
  ## reaches them and the RMSNorm module default applies.

type
  ExpertKeyVocab* = enum
    ## Per-expert key naming of the routed block, the static discriminator
    ## of BlockSparseFFN.load. Both layouts fuse onto the same bodies: the
    ## gate rows at 0:I and the up rows at I:2I fill the [E, 2I, H] fused
    ## body, the [E, H, I] down body carries the output projection, all in
    ## the torch Linear [out, in] convention.
    ekvW1W3W2
      ## w1/w3/w2 triple, silu(w1) * w3 -> w2: the SwiGLU naming
      ## (Shazeer, GLU Variants Improve Transformer, 2020), used by
      ## Mistral and Mixtral
    ekvGateUpDown
      ## gate_proj/up_proj/down_proj: the HF LLaMA MLP convention,
      ## followed by Qwen and DeepSeek

# ─── Quant method detection ────────────────────────────────────────────

proc detectQuantization*(cfg: JsonNode): QuantFormatKind =
  if cfg.hasKey("quantization_config"):
    let qm = cfg["quantization_config"]["quant_method"].getStr("")
    checkValue(qm == "exl3",
      "[ttt] Unsupported quant_method: '" & qm & "' (expected 'exl3' or no quantization_config)")
    qExl3
  else:
    qBF16

# var ... {.compileTime.} are removed from the runtime.
# We force materializing them as `const` at runtime by shadowing them with a const ... = static(...)
const QuantLoaderRegistry = static(QuantLoaderRegistry)

# ─── Activations ────────────────────────────────────────────────────────

proc getDeployDtype*(cfg: JsonNode): ScalarKind =
  QuantLoaderRegistry[detectQuantization(cfg)].deployDtype

# ─── Linear ─────────────────────────────────────────────────────────────

proc load*(_: type Linear, view: SafetensorsCollection, cfg: JsonNode, prefix: string, device = kCPU): Linear =
  let quant = detectQuantization(cfg)
  let loader = QuantLoaderRegistry[quant].linear
  checkValue(loader != nil, "[ttt] No linear loader for " & $quant)
  loader(view, prefix, cfg, device)

# ─── RmsNorm ───────────────────────────────────────────────────────────

proc loadRmsWeight(view: SafetensorsCollection, cfg: JsonNode, prefix: string,
                   device: DeviceKind): tuple[quant: QuantFormatKind, weight: Tensor, eps: float64] =
  let quant = detectQuantization(cfg)
  let weight = view.getTensorOwned(prefix & ".weight", device)
    .to(QuantLoaderRegistry[quant].deployDtype)
  let textCfg = cfg{"text_config"}
  let eps = textCfg{"rms_norm_eps"}.getFloat(cfg{"rms_norm_eps"}.getFloat(1e-6))
  (quant: quant, weight: weight, eps: eps)

proc load*(_: type RmsNorm, view: SafetensorsCollection, cfg: JsonNode, prefix: string, device = kCPU): RmsNorm =
  let (quant, weight, eps) = loadRmsWeight(view, cfg, prefix, device)
  RmsNorm.init(weight, quant, eps)

proc load*(_: type RmsNormOne, view: SafetensorsCollection, cfg: JsonNode, prefix: string, device = kCPU): RmsNormOne =
  let (quant, weight, eps) = loadRmsWeight(view, cfg, prefix, device)
  RmsNormOne.init(weight, quant, eps)

proc load*(_: type RmsNormGated, view: SafetensorsCollection, cfg: JsonNode, prefix: string, device = kCPU): RmsNormGated =
  ## Checkpoints store this weight as F32 and it deploys as the format's dtype.
  let (_, weight, eps) = loadRmsWeight(view, cfg, prefix, device)
  RmsNormGated.init(weight, eps)

# ─── Embedding ──────────────────────────────────────────────────────────

proc load*(_: type Embedding, view: SafetensorsCollection, cfg: JsonNode, prefix: string, device = kCPU): Embedding =
  let quant = detectQuantization(cfg)
  let weight = view.getTensorOwned(prefix & ".weight", device)
    .to(QuantLoaderRegistry[quant].deployDtype)
  let textCfg = cfg{"text_config"}
  let vocab = textCfg{"vocab_size"}.getInt(cfg{"vocab_size"}.getInt())
  let hidden = textCfg{"hidden_size"}.getInt(cfg{"hidden_size"}.getInt())
  checkValue(
    weight.size(0) == vocab and weight.size(1) == hidden,
    "[ttt] " & prefix & ".weight is [" &
    $weight.size(0) & ", " & $weight.size(1) & "], expected [" &
    $vocab & ", " & $hidden & "]")
  Embedding.init(weight)

# ─── GatedDenseFFN ─────────────────────────────────────────────────────────

proc load*(_: type GatedDenseFFN, view: SafetensorsCollection, cfg: JsonNode,
           prefix: string, device = kCPU): GatedDenseFFN =
  let gate = Linear.load(view, cfg, prefix & ".gate_proj", device)
  let up = Linear.load(view, cfg, prefix & ".up_proj", device)
  let down = Linear.load(view, cfg, prefix & ".down_proj", device)
  GatedDenseFFN.init(gate, up, down)

# ─── GatedBlockSparseFFN ───────────────────────────────────────────────────

proc load*(_: type GatedBlockSparseFFN, view: SafetensorsCollection, cfg: JsonNode,
           prefix: string, numExpertsPerTok: int, device = kCPU): GatedBlockSparseFFN =
  let router = view.getTensorOwned(prefix & ".gate.weight", device)
  let gateUp = view.getTensorOwned(prefix & ".experts.gate_up_proj", device)
  let down = view.getTensorOwned(prefix & ".experts.down_proj", device)
  let shared = GatedDenseFFN.load(view, cfg, prefix & ".shared_expert", device)
  let sharedGate = view.getTensorOwned(prefix & ".shared_expert_gate.weight", device)
  GatedBlockSparseFFN.init(gateUp, down, router, shared, sharedGate, numExpertsPerTok)

# ─── BlockSparseFFN (ungated shared expert) ────────────────────────────────

proc load*(_: type BlockSparseFFN, view: SafetensorsCollection, cfg: JsonNode,
           prefix: string, router: NoauxTcRouter, device = kCPU,
           vocab: static ExpertKeyVocab = ekvGateUpDown): BlockSparseFFN =
  ## Loads the routed expert bodies and the ungated shared expert; the
  ## router arrives composed (NoauxTcRouter), nothing router-related
  ## loads here, bias-buffer key naming stays a model-load concern.
  ##
  ## Expert keys follow `vocab`: gate_proj/up_proj/down_proj or
  ## w1/w3/w2 (silu(w1) * w3 -> w2); both fuse to gate rows 0:I, up rows
  ## I:2I of the [E, 2I, H] fused body, down [E, H, I]. The first
  ## expert's views are shape-checked against the config widths, the
  ## rest numel-checked at assembly.
  ##
  ## Expert count reads n_routed_experts, falls back to num_experts;
  ## the load refuses when neither is a positive count.
  ##
  ## Each expert body copies once from its mmap view to its final device
  ## slot; a kCPU request assembles through stack/cat over the CPU views.
  let routedNode = cfg{"n_routed_experts"}
  let expertCount =
    if routedNode.kind == JInt:
      routedNode.getInt().int
    else:
      cfg{"num_experts"}.reqPosInt("num_experts")
  checkValue(expertCount > 0,
    "[ttt] BlockSparseFFN.load: config carries no positive n_routed_experts" &
    " or num_experts")
  # Config-vs-checkpoint cross-check widths: the first expert's views
  # shape-check against these two config values.
  let textCfg = cfg{"text_config"}
  let hiddenNode = textCfg{"hidden_size"}
  let hiddenSize =
    if hiddenNode.kind == JInt: hiddenNode.reqPosInt("hidden_size")
    else: cfg{"hidden_size"}.reqPosInt("hidden_size")
  let moeNode = textCfg{"moe_intermediate_size"}
  let moeIntermediate =
    if moeNode.kind == JInt: moeNode.reqPosInt("moe_intermediate_size")
    else: cfg{"moe_intermediate_size"}.reqPosInt("moe_intermediate_size")
  when vocab == ekvW1W3W2:
    let gateKey = ".w1.weight"
    let upKey = ".w3.weight"
    let downKey = ".w2.weight"
  else:
    let gateKey = ".gate_proj.weight"
    let upKey = ".up_proj.weight"
    let downKey = ".down_proj.weight"
  # One config-vs-view check on the first expert,
  # the remaining experts get the same check at assembly time.
  let first = prefix & ".experts.0"
  let gateW = view.getTensorView(first & gateKey)
  let upW = view.getTensorView(first & upKey)
  let downW = view.getTensorView(first & downKey)
  checkValue(gateW.size(0) == moeIntermediate and gateW.size(1) == hiddenSize,
    "[ttt] BlockSparseFFN.load: " & first & gateKey & " rows are [" &
    $gateW.size(0) & ", " & $gateW.size(1) & "], expected [" &
    $moeIntermediate & ", " & $hiddenSize & "]")
  checkValue(upW.size(0) == moeIntermediate and upW.size(1) == hiddenSize,
    "[ttt] BlockSparseFFN.load: " & first & upKey & " rows are [" &
    $upW.size(0) & ", " & $upW.size(1) & "], expected [" &
    $moeIntermediate & ", " & $hiddenSize & "]")
  checkValue(downW.size(0) == hiddenSize and downW.size(1) == moeIntermediate,
    "[ttt] BlockSparseFFN.load: " & first & downKey & " rows are [" &
    $downW.size(0) & ", " & $downW.size(1) & "], expected [" &
    $hiddenSize & ", " & $moeIntermediate & "]")
  if device == kCPU:
    # kCPU request: assemble through stack/cat over the CPU mmap views.
    let gateUp = stack(collect(newSeq, for e in 0 ..< expertCount:
      cat([view.getTensorView(prefix & ".experts." & $e & gateKey),
           view.getTensorView(prefix & ".experts." & $e & upKey)], 0))).
      to(device)
    let down = stack(collect(newSeq, for e in 0 ..< expertCount:
      view.getTensorView(prefix & ".experts." & $e & downKey))).to(device)
    let shared = GatedDenseFFN.load(view, cfg, prefix & ".shared_experts", device)
    return BlockSparseFFN.init(gateUp, down, shared, router)
  # Device request past kCPU: single-copy assembly. Both fused bodies
  # are allocated on the device at their final concatenated shapes
  # before the copies begin; each expert body then copies once from
  # its mmap view into its final device slot via copyFrom, no CPU
  # staging tensor. The per-expert numel and scalar-type checks refuse
  # every straggler row the earlier stack rule refused.
  let bodyOpts = tensorOptions().dtype(gateW.scalarType()).device(device)
  let gateUp = empty(expertCount, 2 * moeIntermediate, hiddenSize, bodyOpts)
  let down = empty(expertCount, hiddenSize, moeIntermediate, bodyOpts)
  let gateRowBytes = moeIntermediate * hiddenSize
  let downRowBytes = hiddenSize * moeIntermediate
  for e in 0 ..< expertCount:
    let rowPrefix = prefix & ".experts." & $e
    let gateRow = view.getTensorView(rowPrefix & gateKey)
    let upRow = view.getTensorView(rowPrefix & upKey)
    let downRow = view.getTensorView(rowPrefix & downKey)
    checkValue(gateRow.numel() == gateRowBytes and
      upRow.numel() == gateRowBytes and downRow.numel() == downRowBytes and
      gateRow.scalarType() == upRow.scalarType() and
      gateRow.scalarType() == downRow.scalarType() and
      gateRow.scalarType() == gateW.scalarType() and
      upRow.scalarType() == gateW.scalarType() and
      downRow.scalarType() == gateW.scalarType(),
      "[ttt] BlockSparseFFN.load: experts." & $e &
      " rows disagree with the fused shapes or the checkpoint scalar type")
    let gateUpSlice = gateUp.narrow(0, e, 1).squeeze(0)
    gateUpSlice.narrow(0, 0, moeIntermediate).copyFrom(gateRow)
    gateUpSlice.narrow(0, moeIntermediate, moeIntermediate).copyFrom(upRow)
    down.narrow(0, e, 1).squeeze(0).copyFrom(downRow)
  let shared = GatedDenseFFN.load(view, cfg, prefix & ".shared_experts", device)
  BlockSparseFFN.init(gateUp, down, shared, router)

# ─── GatedDeltaNet ─────────────────────────────────────────────────────────

func normalizeAGateLog*(key: string, aLog: Tensor, numHeads: int): Tensor =
  ## A_log normalized to (heads, 1) f32 from the checkpoint forms flat
  ## (heads), (heads, 1) or the (1, 1, heads, 1) rank-4 view. Raises
  ## ValueError naming `key` for any other form. Load-convention
  ## checkpoint fact: the mixer takes the normalized form.
  case aLog.dim()
  of 1:
    checkValue(aLog.numel() == numHeads,
      "[ttt] " & key & ": flat A_log holds " & $aLog.numel() &
      " entries, expected the head count " & $numHeads)
    aLog.reshape([numHeads, 1]).to(kFloat32)
  of 2:
    checkValue(aLog.size(0) == numHeads and aLog.size(1) == 1,
      "[ttt] " & key & ": A_log must be (heads, 1), got (" &
      $aLog.size(0) & ", " & $aLog.size(1) & ")")
    aLog.to(kFloat32)
  of 4:
    checkValue(aLog.size(0) == 1 and aLog.size(1) == 1 and
      aLog.size(2) == numHeads and aLog.size(3) == 1,
      "[ttt] " & key & ": rank-4 A_log must be (1, 1, heads, 1), got (" &
      $aLog.size(0) & ", " & $aLog.size(1) & ", " & $aLog.size(2) &
      ", " & $aLog.size(3) & ")")
    aLog.reshape([numHeads, 1]).to(kFloat32)
  else:
    raise newException(ValueError,
      "[ttt] " & key & ": A_log must be rank 1, 2, or 4, got rank " &
      $aLog.dim())

proc load*[Decay: static DecayAxis,
    GateIn: FullRankGateIn | LowRankGateIn, Form: static GateForm](
    _: type GatedDeltaNet[Decay, GateIn, Form],
    view: SafetensorsCollection, cfg: JsonNode,
    prefix: string, layerIdx: int,
    numKHeads, numVHeads, headKDim, headVDim, convKernelSize: int,
    device = kCPU, kdaLowerBound: float64 = 0.0): GatedDeltaNet[Decay, GateIn, Form] =
  ## One gated delta-rule mixer off the checkpoint. The static
  ## parameters select the gate-input and output-norm variants.
  ## kdaLowerBound is the config reader's parsed kda_lower_bound, read
  ## by the lowerBoundSigmoid gate form and ignored by the softplus
  ## forms, whose call sites omit it.
  when Decay == perHead:
    let qkvProj = Linear.load(view, cfg, prefix & ".in_proj_qkv", device)
    let zProj = Linear.load(view, cfg, prefix & ".in_proj_z", device)
    let aProj = Linear.load(view, cfg, prefix & ".in_proj_a", device)
    let bProj = Linear.load(view, cfg, prefix & ".in_proj_b", device)
    let convWeight = view.getTensorOwned(prefix & ".conv1d.weight", device)
    let aLog = view.getTensorOwned(prefix & ".A_log", device)
    let dtBias = view.getTensorOwned(prefix & ".dt_bias", device)
    let norm = RmsNormGated.load(view, cfg, prefix & ".norm", device)
    let outProj = Linear.load(view, cfg, prefix & ".out_proj", device)
    GatedDeltaNet[perHead, GateIn, Form].init(layerIdx, prefix,
      qkvProj, zProj, aProj, bProj,
      convWeight, aLog, dtBias, norm, outProj,
      numKHeads, numVHeads, headKDim, headVDim, convKernelSize)
  elif Form == GateForm.softplus:
    # The low-rank gate path: every weight loads through
    # getTensorOwned(device), the device parameter is a contract, the
    # checkpoint deploys bf16 only. A_log normalizes to (heads, 1) f32,
    # dt_bias to (heads, dk) f32.
    checkValue(detectQuantization(cfg) == qBF16,
      "[ttt] GatedDeltaNet.load: the low-rank gate path serves the" &
      " unquantized bf16 checkpoints only")
    let decayGate = LowRankGateIn.init(
      Linear.init(view.getTensorOwned(prefix & ".f_a_proj.weight", device)),
      Linear.init(view.getTensorOwned(prefix & ".f_b_proj.weight", device)))
    let normGate = LowRankGateIn.init(
      Linear.init(view.getTensorOwned(prefix & ".g_a_proj.weight", device)),
      Linear.init(view.getTensorOwned(prefix & ".g_b_proj.weight", device)))
    let aLog = normalizeAGateLog(prefix & ".A_log",
      view.getTensorOwned(prefix & ".A_log", device), numKHeads)
    let dtBias = view.getTensorOwned(prefix & ".dt_bias", device)
      .reshape([numKHeads, headKDim]).to(kFloat32)
    GatedDeltaNet[perChannel, GateIn, GateForm.softplus].init(layerIdx, prefix,
      decayGate, normGate,
      q_proj = Linear.init(view.getTensorOwned(prefix & ".q_proj.weight", device)),
      k_proj = Linear.init(view.getTensorOwned(prefix & ".k_proj.weight", device)),
      v_proj = Linear.init(view.getTensorOwned(prefix & ".v_proj.weight", device)),
      b_proj = Linear.init(view.getTensorOwned(prefix & ".b_proj.weight", device)),
      o_proj = Linear.init(view.getTensorOwned(prefix & ".o_proj.weight", device)),
      o_norm = RmsNormGatedSigmoid.init(
        view.getTensorOwned(prefix & ".o_norm.weight", device),
        cfg{"rms_norm_eps"}.reqPosFloat("rms_norm_eps")),
      conv_q = view.getTensorOwned(prefix & ".q_conv1d.weight", device),
      conv_k = view.getTensorOwned(prefix & ".k_conv1d.weight", device),
      conv_v = view.getTensorOwned(prefix & ".v_conv1d.weight", device),
      a_log = aLog, dt_bias = dtBias,
      num_heads = numKHeads, head_k_dim = headKDim,
      head_v_dim = headVDim, conv_kernel_size = convKernelSize)
  else:
    # The lower-bound-sigmoid registry path: projections load through
    # the quant loaders, and the config reader's kda_lower_bound rides
    # through to the mixer, the gate formula reading it at runtime.
    # A_log arrives flat (heads) on some checkpoints and normalizes to
    # (heads, 1) f32 on both paths.
    let decayGate = FullRankGateIn.init(
      Linear.load(view, cfg, prefix & ".f_proj", device))
    let normGate = FullRankGateIn.init(
      Linear.load(view, cfg, prefix & ".g_proj", device))
    let aLog = normalizeAGateLog(prefix & ".A_log",
      view.getTensorOwned(prefix & ".A_log", device), numKHeads)
    let dtBias = view.getTensorOwned(prefix & ".dt_bias", device)
      .reshape([numKHeads, headKDim]).to(kFloat32)
    GatedDeltaNet[perChannel, GateIn, lowerBoundSigmoid].init(layerIdx, prefix,
      decayGate, normGate,
      q_proj = Linear.load(view, cfg, prefix & ".q_proj", device),
      k_proj = Linear.load(view, cfg, prefix & ".k_proj", device),
      v_proj = Linear.load(view, cfg, prefix & ".v_proj", device),
      b_proj = Linear.load(view, cfg, prefix & ".b_proj", device),
      o_proj = Linear.load(view, cfg, prefix & ".o_proj", device),
      o_norm = FusedRmsNormGatedSigmoid.init(
        view.getTensorOwned(prefix & ".o_norm.weight", device),
        cfg{"rms_norm_eps"}.reqPosFloat("rms_norm_eps")),
      conv_q = view.getTensorOwned(prefix & ".q_conv1d.weight", device),
      conv_k = view.getTensorOwned(prefix & ".k_conv1d.weight", device),
      conv_v = view.getTensorOwned(prefix & ".v_conv1d.weight", device),
      a_log = aLog, dt_bias = dtBias,
      num_heads = numKHeads, head_k_dim = headKDim,
      head_v_dim = headVDim, conv_kernel_size = convKernelSize,
      kda_lower_bound = kdaLowerBound)

# ─── Grouped Query Attention ──────────────────────────────────────────────

proc load*[QKNorm](_: type RopeGQAttention[QKNorm], view: SafetensorsCollection,
                   cfg: JsonNode, prefix: string, layerIdx: int,
                   numQoHead, numKvHead, headDim: int,
                   rotary: RotaryPositionEmbedding,
                   device = kCPU): RopeGQAttention[QKNorm] =
  let qProj = Linear.load(view, cfg, prefix & ".q_proj", device)
  let kProj = Linear.load(view, cfg, prefix & ".k_proj", device)
  let vProj = Linear.load(view, cfg, prefix & ".v_proj", device)
  let oProj = Linear.load(view, cfg, prefix & ".o_proj", device)
  let qNorm = QKNorm.load(view, cfg, prefix & ".q_norm", device)
  let kNorm = QKNorm.load(view, cfg, prefix & ".k_norm", device)
  RopeGQAttention[QKNorm].init(layerIdx, prefix,
    qProj, kProj, vProj, oProj,
    numQoHead, numKvHead, headDim, rotary,
    q_norm = qNorm, k_norm = kNorm)

# ─── Gated Attention ───────────────────────────────────────────────────────

proc load*[QKNorm](_: type RopeElementWiseGatedAttention[QKNorm],
                   view: SafetensorsCollection, cfg: JsonNode, prefix: string,
                   layerIdx: int, numQoHead, numKvHead, headDim: int,
                   rotary: RotaryPositionEmbedding,
                   device = kCPU): RopeElementWiseGatedAttention[QKNorm] =
  let qProj = Linear.load(view, cfg, prefix & ".q_proj", device)
  let kProj = Linear.load(view, cfg, prefix & ".k_proj", device)
  let vProj = Linear.load(view, cfg, prefix & ".v_proj", device)
  let oProj = Linear.load(view, cfg, prefix & ".o_proj", device)
  let qNorm = QKNorm.load(view, cfg, prefix & ".q_norm", device)
  let kNorm = QKNorm.load(view, cfg, prefix & ".k_norm", device)
  RopeElementWiseGatedAttention[QKNorm].init(layerIdx, prefix,
    qProj, kProj, vProj, oProj,
    numQoHead, numKvHead, headDim, rotary,
    q_norm = qNorm, k_norm = kNorm)

# ─── LMHead ────────────────────────────────────────────────────────────────

proc load*(_: type LMHead, view: SafetensorsCollection, cfg: JsonNode, embedTokens: Embedding, device = kCPU): LMHead =
  let quant = detectQuantization(cfg)
  let loader = QuantLoaderRegistry[quant].lmHead
  checkValue(not loader.isNil(), "[ttt] No LMHead loader for " & $quant)
  let lmHead = loader(view, device)
  if not lmHead.isNil():
    return lmHead
  let textCfg = cfg{"text_config"}
  let tied = textCfg{"tie_word_embeddings"}.getBool(cfg{"tie_word_embeddings"}.getBool(true))
  if not tied:
    raise newException(IOError,
      "[ttt] LMHead.load: no lm_head tensors in the checkpoint and " &
      "tie_word_embeddings is false")
  LMHead.initTied(embedTokens)
