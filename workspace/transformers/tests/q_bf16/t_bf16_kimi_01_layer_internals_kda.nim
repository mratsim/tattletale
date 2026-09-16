# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-0 Kimi-linear KDA recurrence of the Kimi-Linear checkpoint,
## replayed on cpu against the recorded layer-0 kernel-boundary payloads.
## Requires the layer-0 fixture directory (generated payloads, untracked).
##
## Run:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_kimi_01_layer_internals_kda.nim

import
  std/os,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Kimi-Linear-48B-A3B-Instruct-layer-0"

proc main() =
  # Kernel-boundary replay over the recorded post-conv q/k/v slabs,
  # the low-rank gate weights and the beta rates feed the recurrence
  # with no checkpoint weights loading.
  echo "    devices: ", deviceName(F.kCPU)
  for headNum in 0 ..< 2:
    for seqLen in [1, 64, 65]:
      let stem = "kda-Kimi-Linear-48B-A3B-Instruct-0" & $headNum & "t" & $seqLen
      var st = Safetensor.open(FixtureDir / (stem & ".safetensor"))
      let statsPath = FixtureDir / (stem & ".safetensor.stats.json.zst")
      echo "    devices: ", deviceName(F.kCPU), " (head ", headNum,
        " T ", seqLen, ")"

      # The recurrence l2-normalizes q/k in f32 after the staging cast,
      # the Kimi-linear native reference order, then runs the per-channel
      # decay delta rule to the final state.
      let (outRec, stateRec) = gatedDeltaRuleRecurrence(perChannel,
        st.getTensorOwned("query"), st.getTensorOwned("key"),
        st.getTensorOwned("value"), st.getTensorOwned("log_decay"),
        st.getTensorOwned("beta"), nil)
      assertStats(outRec, statsPath, "out_recurrent", kReduction,
        msg = stem & " recurrent output")
      assertStats(stateRec, statsPath, "state_recurrent", kReduction,
        msg = stem & " recurrent state")

  block:
    let runDev = testDevice()
    echo "    devices: ", deviceName(runDev)
    if runDev != F.kCPU:
      echo "    cross-device replay is out of scope for this suite, skipping on ", deviceName(runDev)

when isMainModule:
  main()
