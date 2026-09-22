# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Fused add+norm two-op check segment of the composition tier, validating
## the additive rounding model with reassociation budget before any multi-stage
## comparison is trusted. The shared driver lives in `ceramic_mega_gdn_composition.nim`.
##
## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests tests/ceramic/t_ceramic_mega_gdn_norm_probe.nim

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
runNormCheck(engine)
echo &"[norm-probe] wall clock {epochTime() - t0:.2f} s"
echo "CERAMIC MEGA GDN NORM PROBE VERDICT: fused add+norm two-op band"
