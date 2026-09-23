# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chain segment of the composition tier, an 8-step decode chain over carried
## state and ring:
##   stage bands hold at every step → the chain's carried state stays continuous
##
## The three segments share the composition driver `ceramic_mega_gdn_composition.nim`.
##
## Run command, from the repo root:
## - nim test_positron_naive, the composition tier's untraced build
##   (derives from `suiteCmd` in config.nims)

import std/[strformat, math, times]
import workspace/crucible
import workspace/ceramic
import ../../src/mega_kernels/decode_layers/gdn_moe_decode_megakernel
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
runChain(engine)
echo &"[chain] wall clock {epochTime() - t0:.2f} s"
echo "CERAMIC MEGA GDN CHAIN VERDICT: per-stage bands, chain continuity"
