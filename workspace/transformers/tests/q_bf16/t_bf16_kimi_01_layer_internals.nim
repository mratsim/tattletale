# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-0 unit replay of the Kimi-Linear-48B-A3B-Instruct checkpoint:
## the KDA mixer forward from the bare hidden input, seq 5, head 0.
##
## - the fixture stores one bare bf16 driving tensor, kda.input, every
##   op-surface value living on the stats frame as fingerprints
## - the suite rebuilds the kernel-boundary surface through the layer
##   components over the bare hidden input, then asserts each recomputed
##   point against its fingerprint
## - the chunked-kernel record rows stay unread, the bitwise reference
##   for the Nim block stays the recurrent form
##
## Requires the local model at tests/hf_models/Kimi-Linear-48B-A3B-Instruct (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_kimi_01_layer_internals.nim

import
  std/os,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net {.all.},
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device

{.experimental: "callOperator".}
privateAccess(GatedDeltaNet[perChannel, LowRankGateIn, GateForm.softplus])

const
  ModelDir =
    currentSourcePath().parentDir() / ".." / "hf_models" / "Kimi-Linear-48B-A3B-Instruct"
  FixturePath =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" /
    "Kimi-Linear-48B-A3B-Instruct-layer-0" /
    "layer0-Kimi-Linear-48B-A3B-Instruct-00.safetensor"
  StatsPath = FixturePath & ".stats.json.zst"
  KdaPrefix = "model.layers.0.self_attn"

proc main() =
  ## Replay contract of the layer-0 KDA kernel boundary:
  ##
  ## - the fixture carries the bare hidden input, the mixer surface
  ##   projects it over the real layer-0 weights
  ## - every recomputed surface point meets the recorded stats, the f32
  ##   records under the elementwise band, the conv outputs and recurrence
  ##   records under the reduction bands
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
  let dev = testDevice()
  echo "    devices: ", deviceName(dev)
  let cfgJson = (ModelDir / "config.json").parseFile()
  let numHeads = cfgJson{"linear_attn_config"}{"num_heads"}.getInt()
  let headDim = cfgJson{"linear_attn_config"}{"head_dim"}.getInt()
  let convKernel = cfgJson{"linear_attn_config"}{"short_conv_kernel_size"}.getInt()
  let view = SafetensorsCollection.open(ModelDir)
  # The mixer loads exactly as the Kimi-Linear model file wires layer 0,
  # per-channel decay, low-rank projections, the softplus decay formula.
  let kda = GatedDeltaNet[perChannel, LowRankGateIn, GateForm.softplus].load(
    view, cfgJson, KdaPrefix, 0, numHeads, numHeads, headDim, headDim,
    convKernel, device = dev)
  var st = Safetensor.open(FixturePath)
  let x = st.getTensorOwned("kda.input", dev)   # (1, 5, 2304) bf16
  let seqLen = x.size(1)

  # Kernel-boundary surface recomputed through the layer components over
  # the bare hidden input, zero conv history, head-0 slices of the whole
  # full-head computation, feeding the recurrence exactly as the recording
  # fed its reference kernels.
  let qFlat = kda.q_proj.forward(x).transpose(1, 2)
  let kFlat = kda.k_proj.forward(x).transpose(1, 2)
  let vFlat = kda.v_proj.forward(x).transpose(1, 2)
  let (convQ, _) = shortConvSequence(qFlat, kda.conv_q, nil)
  let (convK, _) = shortConvSequence(kFlat, kda.conv_k, nil)
  let (convV, _) = shortConvSequence(vFlat, kda.conv_v, nil)
  let query = convQ.transpose(1, 2).reshape([1, seqLen, numHeads, headDim])
  let key = convK.transpose(1, 2).reshape([1, seqLen, numHeads, headDim])
  let value = convV.transpose(1, 2).reshape([1, seqLen, numHeads, headDim])
  let beta = F.sigmoid(kda.in_proj_b.forward(x).to(kFloat32))
  let g = computeG(perChannel, GateForm.softplus, kda.decay_gate,
    kda.a_log, kda.dt_bias, x)

  # Head slice of the full-head computation, the recorded op-surface form.
  let qHead = query.narrow(2, 0, 1).contiguous()
  let kHead = key.narrow(2, 0, 1).contiguous()
  let vHead = value.narrow(2, 0, 1).contiguous()
  let gHead = g.narrow(2, 0, 1).contiguous()
  let betaHead = beta.narrow(2, 0, 1).contiguous()
  assertStats(qHead, StatsPath, "kda.query", kReduction,
    msg = "kda query post-conv, head 0")
  assertStats(kHead, StatsPath, "kda.key", kReduction,
    msg = "kda key post-conv, head 0")
  assertStats(vHead, StatsPath, "kda.value", kReduction,
    msg = "kda value post-conv, head 0")
  assertStats(gHead, StatsPath, "kda.log_decay", kElementwise,
    msg = "kda f32 log decay, head 0")
  assertStats(betaHead, StatsPath, "kda.beta", kElementwise,
    msg = "kda f32 beta gate, head 0")

  # The recurrence l2-normalizes q/k in f32 after the staging cast,
  # the Kimi-linear native reference order, then runs the per-channel
  # decay delta rule to the final state.
  let (outRec, stateRec) = gatedDeltaRuleRecurrence(perChannel,
    qHead, kHead, vHead, gHead, betaHead, nil)
  assertStats(outRec, StatsPath, "kda.out_recurrent", kReduction,
    msg = "kda recurrent output")
  assertStats(stateRec, StatsPath, "kda.state_recurrent", kReduction,
    msg = "kda recurrent state")

when isMainModule:
  main()
