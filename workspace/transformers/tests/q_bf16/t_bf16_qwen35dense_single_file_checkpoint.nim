# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35dense-single-file-checkpoint \
##   --nimcache:nimcache/tests/qwen35dense-single-file-checkpoint \
##   workspace/transformers/tests/q_bf16/t_bf16_qwen35dense_single_file_checkpoint.nim
# Requires: local model at tests/hf_models/Qwen3.5-0.8B (gitignored)

import
  std/tables,
  std/strutils,
  std/os,
  workspace/libtorch as F,
  workspace/safetensors,
  std/importutils,
  workspace/safetensors/src/safetensors {.all.},
  workspace/transformers/src/models,
  workspace/transformers/src/models/qwen35 {.all.},
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

privateAccess(SafetensorObj)

const ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"

proc main() =
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

  runCppTest "Qwen3.5-0.8B single safetensor file - language_model prefix, foreign tensors skipped":
    proc(): bool =
      let weightsPath = ModelPath / "model.safetensors-00001-of-00001.safetensors"
      var st = Safetensor.open(weightsPath)

      var total = 0
      var languageModel = 0
      var visual = 0
      var mtp = 0
      for name in st.tensors.keys():
        inc total
        if name.startsWith("model.language_model."):
          inc languageModel
        elif name.startsWith("model.visual."):
          inc visual
        elif name.startsWith("mtp."):
          inc mtp
      doAssert total == 488
      doAssert languageModel == 320
      doAssert visual == 153
      doAssert mtp == 15

      # Every language_model tensor must load by name, including all
      # `layers.*` tensors and the final norm. Foreign prefixes are never
      # requested, so the load skips them without error.
      var loaded = 0
      for name in st.tensors.keys():
        if name.startsWith("model.language_model."):
          discard st.getTensorOwned(name, kCPU)
          inc loaded
      doAssert loaded == languageModel

      true

  runCppTest "Qwen3.5-0.8B loadQwen35ModelRaw + generate plumbing":
    proc(): bool =
      let model = loadQwen35ModelRaw(ModelPath, kCPU)
      doAssert model.config.num_hidden_layers == 24
      doAssert model.config.vocab_size == 248320
      doAssert model.config.dtype == "bfloat16"

      # Loader footprint derived from a read-only second open of the file.
      # Requests are name-based: one per language_model key, all of them
      # consumed, plus one for the tied lm_head, as the checkpoint config
      # ties the embedding and the file carries no lm_head.weight key.
      # A regression that adds or drops a requested tensor flips the count.
      let footprintFilePath =
        ModelPath / "model.safetensors-00001-of-00001.safetensors"
      var footprintSt = Safetensor.open(footprintFilePath)
      var checkpointRequests = 0
      for name in footprintSt.tensors.keys():
        if name.startsWith("model.language_model."):
          inc checkpointRequests
      doAssert not footprintSt.tensors.hasKey("lm_head.weight")
      inc checkpointRequests
      doAssert checkpointRequests == 321

      let text = loadModel(ModelPath, kCPU).generate(
        "hi", temp = 1.0f, maxTokens = 3, maxContextLen = 512)
      doAssert text.len > 0

      true

  runCppTest "Qwen3.5-0.8B load plumbing, cross-device variant":
    proc(): bool =
      # No fixture family: the checkpoint loads by name and the counts
      # stay device-free bookkeeping. The computed side takes the run
      # device, the weights still come from the real checkpoint dir.
      let runDev = testDevice()
      echo "    device pair: no fixture family, load plumbing runs on ", deviceName(runDev)
      if runDev == F.kCPU:
        echo "    the run device matches the recorded device, the reference variant carries the replay"
        return true
      let model = loadQwen35ModelRaw(ModelPath, runDev)
      doAssert model.config.num_hidden_layers == 24
      doAssert model.config.vocab_size == 248320
      doAssert model.config.dtype == "bfloat16"
      let text = loadModel(ModelPath, runDev).generate(
        "hi", temp = 1.0f, maxTokens = 3, maxContextLen = 512)
      doAssert text.len > 0
      true

main()
