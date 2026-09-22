# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chain segment of the composition tier, an 8-step decode chain over carried
## state and ring, plus (in the `-d:RedSabotage` build) the failing-verdict run:
##   naive silu dropped → conv band catches the drop → failing verdict
##
## The three segments share the composition driver `ceramic_mega_gdn_composition.nim`.
##
## Run command, from the repo root:
## - nim test_positron_naive, the composition tier's untraced build
## - nim test_ceramic_red_sabotage, the RedSabotage build with the verdict inverted,
##   green means the conv band caught the silu drop
##   both commands derive from `suiteCmd` in config.nims

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/mega_kernels/decode_layers/qwen35_moe/qwen35_moe_decode_gdn_bf16
import ../naive/naive_rng
import ../naive/naive_tensors
import ../naive/naive_gdn
import ../naive/naive_grouped_mm
import ../naive/naive_layer_ops
from ../naive/naive_qwen35_layer import naiveQwen35GdnLayer, LayerOut, moeDecodeBody
import ceramic_pagebuf
# HwEngine is a concept. Symbols used inside its procs resolve at the call site,
# so each segment test imports the same module set the shared driver does,
# and the driver itself is imported last, carrying all the composition logic.
import ceramic_mega_gdn_composition
let engine = compositionInit()
let t0 = epochTime()
when RedSabotage:
  runRedSabotage(engine)
  echo &"[chain-red] wall clock {epochTime() - t0:.2f} s"
  echo "CERAMIC MEGA GDN COMPOSITION RED: conv band caught the silu drop"
else:
  runChain(engine)
  echo &"[chain-red] wall clock {epochTime() - t0:.2f} s"
  echo "CERAMIC MEGA GDN CHAIN VERDICT: per-stage bands, chain continuity"
