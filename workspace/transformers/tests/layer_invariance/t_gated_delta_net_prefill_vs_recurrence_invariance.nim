# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run through the test_tf_layer_invariance_gdn task in config.nims.
##
## Measured instrument drift of this suite, the committed budgets
##
## | instrument       | measured                                 | budget                         | source of the difference                |
## | ---------------- | ---------------------------------------- | ------------------------------ | --------------------------------------- |
## | conv context     | bit-equal (0.0)                          | must be exactly 0              | same 4-tap fp32 stencil both orders     |
## | block output     | 1.22e-4 (exactly 1 bf16 ulp at max 0.19) | 4 ulps (3.9e-3)                | batched-vs-per-token GEMM reassociation |
## | recurrence state | 2.98e-7 (~3 f32 ulps)                    | linear 70-step bound (1.67e-5) | f32 state carried 70 steps              |

import
  std/math,
  std/os,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net,
  workspace/transformers/tests/layer_utils

{.experimental: "callOperator".}

privateAccess(GatedDeltaNet)
privateAccess(InferenceContext)

proc main() =
  const
    ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
    Layer0Prefix = "model.language_model.layers.0"
  let
    prefillSeq = 70
    # Output budget factors, never measured headroom:
    # - the output budget sits on the norm-wise backward error
    #   analysis of Wilkinson and Higham, bounding a reassociated
    #   fp32 GEMM within one relative machine eps of the exact product
    # - the bf16 storage round adds one eps per rounded channel
    # - five rounded input channels q, k, v, z and decay plus one
    #   norm rounding stage give six relative eps at 2^-8
    # State budget factors:
    # - the state budget sits on the nonexpansive delta-rule propagator
    #   decay*(I - beta*k*k^T), contractive over nonexpansive iteration
    #   because decay <= 1 and the correction contracts the key direction
    # - one relative eps of per-step injection lands on the update
    #   of that step alone
    # - the triangle inequality telescopes the drift to eps times
    #   the accumulated update mass, one factor over the four state
    #   channels k, v, beta and decay
    outRelBudget = 6.0
    stateRelBudget = 4.0

  proc stimulus(t, hidden: int): Tensor =
    ## Deterministic (1, t, hidden) stimulus on the bf16 grid, all values
    ## sit inside the -1.0 .. 0.75 range at 0.25 steps, a rerun reproduces
    ## the exact tensor.
    var flat = newSeq[float32](t * hidden)
    for i in 0 ..< flat.len:
      flat[i] = (float32(i mod 8) * 0.25'f32) - 1.0'f32
    result = F.toTensor(flat).to(F.kBfloat16).reshape([1, t, hidden])

  proc maxAbsDiff(a, b: Tensor): float64 =
    (a.to(F.kFloat32) - b.to(F.kFloat32)).abs().max().item(float64)

  let cfgJson = (ModelDir / "config.json").parseFile()
  let tc = cfgJson{"text_config"}
  let hidden = tc{"hidden_size"}.getInt()
  let view = SafetensorsCollection.open(ModelDir)
  let gdn = cfgJson.setupGatedDeltaNet(
    GatedDeltaNet[perHead, FullRankGateIn, GateForm.softplus], view,
    "model.language_model.layers.", 0)
  let x = stimulus(prefillSeq, hidden)

  # One-shot prefill, the whole (1, 70, hidden) block enters one
  # forward call, the projections run as batched GEMMs, and the conv
  # runs the padded full-sequence stencil.
  var ctxOneShot = InferenceContext.init(
    num_layers = 1, batch_size = 1,
    kv_heads = tc{"linear_num_key_heads"}.getInt(), max_seq = 512,
    head_dim = tc{"linear_key_head_dim"}.getInt())
  let outOneShot = gdn(ctxOneShot, x)

  # Step decode, the same tokens enter as 70 narrow (1, 1, hidden) calls,
  # the projections run per-token GEMMs and the conv runs the cat-state
  # window stencil, while the recurrence is sequential in both orders.
  var ctxDecode = InferenceContext.init(
    num_layers = 1, batch_size = 1,
    kv_heads = tc{"linear_num_key_heads"}.getInt(), max_seq = 512,
    head_dim = tc{"linear_key_head_dim"}.getInt())
  var outDrift = 0.0'f64
  # The accumulated update mass of the decode run is the theory
  # scale of the state budget, one relative eps of projection
  # reassociation injects into each step update, the nonexpansive
  # propagator never amplifies, the drift telescopes to eps
  # times this mass.
  var updateMass = 0.0'f64
  var prevState = F.zeros(ctxOneShot.gdnSsmState[0].shape,
    F.tensorOptions(F.kFloat32, F.kCPU))
  for t in 0 ..< prefillSeq:
    let outT = gdn(ctxDecode, x.narrow(1, t.int64, 1))
    outDrift = max(outDrift, maxAbsDiff(outT, outOneShot.narrow(1, t.int64, 1)))
    let sNow = ctxDecode.gdnSsmState[0].to(F.kCPU)
    updateMass += maxAbsDiff(sNow, prevState)
    prevState = sNow

  # Causal conv stencil, the same 4-tap fp32 accumulation order applies
  # at every position in both orders, so the stored conv context
  # must land bit-equal.
  let convDiff = maxAbsDiff(ctxOneShot.gdnConvState[0], ctxDecode.gdnConvState[0])
  echo "  conv state bit difference: ", convDiff
  if convDiff != 0.0:
    echo "  FAIL: conv context differs across the evaluation orders"
    quit(1)

  # Block output, the batched-vs-per-token GEMM shapes reassociate
  # the fp32 accumulation, the norm-wise backward error bound plus
  # one storage rounding per channel give the six relative eps
  # behind the output budget, see the theory factors above.
  let outMax = outOneShot.to(F.kFloat32).abs().max().item(float64)
  let outBudget = outRelBudget * outMax * pow(2.0, -8.0)
  echo "  output drift: ", outDrift, ", budget: ", outBudget,
    " (output max ", outMax, ")"
  if outDrift > outBudget:
    echo "  FAIL: block output drift exceeds the bf16 ulp budget"
    quit(1)

  # Delta-rule recurrence state, fp32 carried across the 70
  # decode steps, the nonexpansive telescoping bounds the drift
  # by the relative eps over the accumulated update mass, see
  # the theory factors above.
  let ssmDiff = maxAbsDiff(ctxOneShot.gdnSsmState[0], ctxDecode.gdnSsmState[0])
  let stateBudget = stateRelBudget * pow(2.0, -8.0) * updateMass
  echo "  ssm state drift: ", ssmDiff, ", budget: ", stateBudget,
    " (update mass ", updateMass, ")"
  if ssmDiff > stateBudget:
    echo "  FAIL: recurrence state drift exceeds the theory budget"
    quit(1)

when isMainModule:
  main()
