# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Real-shard load check for the Qwen3.5-0.8B text stack.
##
## Needs the local model copy at `tests/hf_models/Qwen3.5-0.8B` (gitignored).

import
  std/memfiles,
  std/tables,
  std/strutils,
  std/os,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/models,
  workspace/transformers/src/models/qwen3_5 {.all.},
  workspace/libtorch_testutils

const ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"

proc main() =
  runCppTest "Qwen3.5-0.8B single shard - language_model prefix, foreign tensors skipped":
    proc(): bool =
      let shardPath = ModelPath / "model.safetensors-00001-of-00001.safetensors"
      var memFile = memFiles.open(shardPath, mode = fmRead)
      defer: close(memFile)
      let st = safetensors.load(memFile)

      let counts = countShardTensors(st)
      doAssert counts.total == 488
      doAssert counts.languageModel == 320
      doAssert counts.visual == 153
      doAssert counts.mtp == 15

      # Every language_model tensor must load by name, including all
      # `layers.*` tensors and the final norm. Foreign prefixes are never
      # requested, so the load skips them without error.
      var loaded = 0
      for name in st.tensors.keys():
        if name.startsWith("model.language_model."):
          discard st.getTensorOwned(name, kCPU)
          inc loaded
      doAssert loaded == counts.languageModel

      true

  runCppTest "Qwen3.5-0.8B loadQwen3_5ModelRaw + generate plumbing":
    proc(): bool =
      let model = loadQwen3_5ModelRaw(ModelPath, kCPU)
      doAssert model.config.num_hidden_layers == 24
      doAssert model.config.vocab_size == 248320
      doAssert model.config.dtype == "bfloat16"

      # Loader footprint: the loader makes exactly 3 top-level name-based
      # tensor requests (embed, final norm, tied lm_head). A regression that
      # adds or drops a loader request changes this count.
      doAssert model.loadedTensorCount == 321

      let text = loadModel(ModelPath, kCPU).generate(
        "hi", temp = 1.0f, maxTokens = 3, maxContextLen = 512)
      doAssert text.len > 0

      true

main()
