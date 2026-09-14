# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run through the test_tf_layer_invariance_blocksparse task in config.nims.

import
  std/math,
  std/strformat,
  std/os,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/tests/layer_utils

{.experimental: "callOperator".}

const
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  Layer0Prefix = "model.language_model.layers.0"

proc main() =
  proc stimulusRows(t, hidden: int, device: F.DeviceKind): Tensor =
    ## Deterministic (T, hidden) stimulus on the bf16 grid, all values sit
    ## inside the -1.0 .. 0.75 range at 0.25 steps, so the property sees no
    ## input rounding noise and a rerun reproduces the exact tensor.
    var flat = newSeq[float32](t * hidden)
    for i in 0 ..< flat.len:
      flat[i] = (float32(i mod 8) * 0.25'f32) - 1.0'f32
    result = F.toTensor(flat).to(F.kBfloat16).to(device).reshape([t, hidden])

  let device =
    if paramCount() > 0 and paramStr(1) == "mps": F.kMPS
    else: F.kCPU
  echo &"Block-sparse FFN batch-vs-single property, device {device}"
  let cfgJson = (ModelDir / "config.json").parseFile()
  let tc = cfgJson{"text_config"}
  let numExpertsPerTok = tc{"num_experts_per_tok"}.getInt()
  let hidden = tc{"hidden_size"}.getInt()
  let view = SafetensorsCollection.open(ModelDir)
  let moe = GatedBlockSparseFFN.load(view, cfgJson, Layer0Prefix & ".mlp", numExpertsPerTok)

  var failures = 0
  for t in [2, 5, 9]:
    let hiddenRows = stimulusRows(t, hidden, device)
    let batchOut = moe.forward(hiddenRows)
    var worstDiff = 0.0'f32
    var worstMag = 0.0'f32
    for i in 0 ..< t:
      let row = hiddenRows.narrow(0, i.int64, 1)
      let singleOut = moe.forward(row)
      let diff = (batchOut.narrow(0, i.int64, 1) - singleOut).abs()
      worstDiff = max(worstDiff, diff.max().item(float32))
      worstMag = max(worstMag, singleOut.abs().max().item(float32))
    # Implementation-variant roundoff, the batched path accumulates experts
    # in routing order while the per-row path runs ascending expert ids, so
    # the budget holds two bf16 ulps at the output magnitude.
    let budget = 2.0'f32 * ulpBf16(worstMag)
    echo &"  T={t}: worst diff {worstDiff:.3e}, budget {budget:.3e} (max mag {worstMag:.1f})"
    if worstDiff > budget:
      failures += 1
      echo &"  FAIL at T={t}"
  if failures > 0:
    echo &"block-sparse batch property: {failures} case(s) FAIL"
    quit(1)

when isMainModule:
  main()
