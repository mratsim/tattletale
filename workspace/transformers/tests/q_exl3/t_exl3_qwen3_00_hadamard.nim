# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 Hadamard rotation of the recorded block slices, every replayed
## output must carry the recorded uniform stats.
## Run through the test_tf_exl3_qwen3_00_hadamard task in config.nims.

import
  std/options,
  std/os,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/positron,
  workspace/transformers/tests/harness/harness

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-00-codec" / "Qwen3-0.6B-EXL3-5bpw"

proc main() =
  ## Replays hadamard_rotate_128 over the recorded block slices, each variant
  ## enforcing through assertStats against its recorded output tensor.
  ##
  ## kReduction carries the FWHT butterfly reordering band.
  const caseNames = ["single_block", "two_blocks", "eight_blocks",
                     "batch2_eight_blocks", "odd_blocks"]
  for name in caseNames:
    let path = FixtureDir / "hadamard_" & name & ".safetensor"
    if not fileExists(path):
      echo "case " & name & " has no recorded fixture, skipped"
      continue

    var st = Safetensor.open(path)
    let input = st.getTensorOwned("input")
    let suh = st.getTensorOwned("suh")
    let svh = st.getTensorOwned("svh")

    let yNone = hadamard_rotate_128(input,
      pre_scale = none(F.Tensor), post_scale = none(F.Tensor))
    assertStats(yNone, path & ".stats", "output_none", kReduction, msg = "Hadamard " & name & " no scale")

    # Pre-scale multiplies suh before the FWHT, post-scale multiplies
    # svh after the transform norm.
    let yPre = hadamard_rotate_128(input,
      pre_scale = some(suh), post_scale = none(F.Tensor))
    assertStats(yPre, path & ".stats", "output_pre", kReduction, msg = "Hadamard " & name & " pre scale")

    let yPost = hadamard_rotate_128(input,
      pre_scale = none(F.Tensor), post_scale = some(svh))
    assertStats(yPost, path & ".stats", "output_post", kReduction, msg = "Hadamard " & name & " post scale")

    echo "case " & name & " replayed under the recorded stats"

when isMainModule:
  main()
