# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-0 Kimi-linear KDA recurrence of the Kimi-Linear-48B-A3B-Instruct checkpoint,
## replayed on cpu against the recorded kernel-boundary fixture, seq 5, head 0.
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_kimi_01_layer_internals.nim

import
  std/os,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils

from workspace/transformers/tests/harness import zstdReadFixture

const
  FixturePath =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" /
    "Kimi-Linear-48B-A3B-Instruct-layer-0" /
    "layer0-Kimi-Linear-48B-A3B-Instruct-00.safetensor"
  StatsPath = FixturePath & ".stats.json.zst"
  MetaPath = FixturePath & ".metadata.json.zst"

proc main() =
  ## Replay contract of the layer-0 KDA kernel boundary:
  ##
  ## - the recorded fixture drives one recurrence on the suite device, an exact replay
  ## - the recurrent output and final state meet the recorded stats,
  ##   the reduction bands
  ## - the chunked-kernel record rows stay unread, the bitwise reference
  ##   for the Nim block stays the recurrent form
  ##
  ## Stage records without an assert, query, key, value, log_decay, beta:
  ##
  ## - the fixture carries the kernel-boundary inputs, the recurrence consumes them as recorded
  ## - no checkpoint weights load, the kernel-boundary contract feeds the recurrence directly
  ## - the recording is CPU-side, the generator locks device cpu
  echo "    devices: ", deviceName(F.kCPU)

  var st = Safetensor.open(FixturePath)
  let meta = zstdReadFixture(MetaPath).parseJson()

  # The recurrence l2-normalizes q/k in f32 after the staging cast,
  # the Kimi-linear native reference order, then runs the per-channel
  # decay delta rule to the final state.
  let (outRec, stateRec) = gatedDeltaRuleRecurrence(perChannel,
    st.blobSegment(meta, "kda", "query"),
    st.blobSegment(meta, "kda", "key"),
    st.blobSegment(meta, "kda", "value"),
    st.blobSegment(meta, "kda", "log_decay"),
    st.blobSegment(meta, "kda", "beta"), nil)
  assertStats(outRec, StatsPath, "kda.out_recurrent", kReduction,
    msg = "kda recurrent output")
  assertStats(stateRec, StatsPath, "kda.state_recurrent", kReduction,
    msg = "kda recurrent state")

when isMainModule:
  main()
