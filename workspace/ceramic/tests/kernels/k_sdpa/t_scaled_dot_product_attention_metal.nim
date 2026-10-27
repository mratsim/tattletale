# Tattletale
# Copyright (c) 2026 Mamy Andre-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SDPA fixture suite: every fixture replays in one binary.
## The kernel instantiates once per dtype, head dim and masking combo, all in one compile.
## A fixture must match its combo in metadata, dtype, Dk and mask flags, and checksums.
##
## Run from the repository root:
##   nim c -r --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/k_sdpa/t_scaled_dot_product_attention_metal.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_sdpa/kernel_sdpa_template

proc main() =
  replayFixture(float32, 16, false, false, ["tiny_f32"])
  replayFixture(float32, 32, false, false,
    ["gqa2_f32", "batched_gqa_f32", "cross_attn_f32", "scale_f32"])
  replayFixture(float32, 64, false, false, ["gqa4_f32", "gqa8_f32"])
  replayFixture(float32, 32, true, false, ["causal_f32"])
  replayFixture(float32, 64, true, false, ["gqa8_causal_f32"])
  replayFixture(float32, 128, true, false, ["d128_causal_f32"])
  replayFixture(float32, 32, false, true, ["windowed_f32"])
  replayFixture(float32, 64, true, true, ["causal_window_f32"])
  replayFixture(float16, 16, false, false, ["tiny_f16"])
  replayFixture(float16, 32, false, false, ["gqa4_f16"])
  replayFixture(float16, 64, true, false, ["decode_gqa8_f16"])
  replayFixture(float16, 32, true, false,
    ["causal_gqa2_f16", "gqa8_causal_f16"])
  replayFixture(float16, 32, false, true, ["windowed_f16"])
  replayFixture(float16, 64, true, true, ["windowed_gqa8_f16"])
  replayFixture(float16, 64, false, true, ["windowed_decode_f16"])
  replayFixture(bfloat16, 16, false, false, ["tiny_bf16"])
  replayFixture(bfloat16, 32, false, false, ["gqa2_bf16"])
  echo "all SDPA fixture replays pass (17 combos, 22 fixtures)"

main()
