# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Text-to-Text inference

import
  std/os,
  std/strformat,
  std/strutils,
  workspace/libtorch,
  workspace/transformers/src/models

const
  ModelPath = currentSourcePath().parentDir() / "hf_models" / "Qwen3-0.6B"

proc main*() =
  echo "Loading model..."
  let model = loadModel($ModelPath, kCPU)

  let prompt = "Hello, how are you?"
  echo &"Prompt: {prompt}"

  let output = model.generate(prompt, temp = 1.0f, maxTokens = 20)
  echo &"Output: {output}"

  assert output.len > prompt.len, "Output must be longer than prompt"
  assert output.startsWith(prompt), "Output must start with prompt"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✅ PASS | Qwen3-0.6B simple inference"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()
