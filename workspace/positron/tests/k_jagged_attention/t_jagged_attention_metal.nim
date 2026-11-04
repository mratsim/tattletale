# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Jagged attention fixture suite (Metal): every fixture replays in one binary.
##
## Test that each fixture replays inside its pre-stated band, that the output sentinels stay zero and that the relaunch is bit-identical.
##
## Run from the worktree root:
##   nim c -r -d:release --threads:off --hints:off --warnings:off \
##     -o:build/tests/t_jagged_attention_metal \
##     --nimcache:nimcache/t_jagged_attention_metal \
##     workspace/positron/tests/ceramic/k_jagged_attention/t_jagged_attention_metal.nim

import workspace/crucible
import kernel_jagged_attention_template

const denseCrossCheckFixture = "jagged_dense_degenerate_f16"
  ## Fixture the dense reference path cross-checks.

proc replayAllJaggedFixtures() =
  var engine = newEngine(runMetal)
  # One (fixture name, element dtype) line per fixture: the replay's dispatch table.
  template fixture(name: string, Elem: typedesc) =
    let bits = engine.jaggedAttnReplayCase(Elem, name)
    if name == denseCrossCheckFixture:
      engine.jaggedAttnDenseCrossCheck(Elem, name, bits)
  fixture("jagged_decode_gqa_f16", float16)
  fixture("jagged_prefill_gqa_f16", float16)
  fixture("jagged_prefix_share_f16", float16)
  fixture("jagged_dense_degenerate_f16", float16)
  fixture("jagged_prefill_f32", float32)
  echo "all jagged attention fixture replays pass"

replayAllJaggedFixtures()
