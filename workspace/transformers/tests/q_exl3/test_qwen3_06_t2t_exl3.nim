# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Text-to-Text inference with EXL3-quantized model.
##
## Loads the EXL3 model via the generic loadModel interface,
## runs generation, and verifies the output is valid.

import
  std/os,
  std/strformat,
  std/strutils,
  workspace/libtorch,
  workspace/transformers/src/models

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"

proc main*() =
  echo "Loading EXL3-quantized model..."
  let model = loadModel($ModelPath, kCuda)

  let prompt = "Hello, how are you?"
  echo &"Prompt:\n----\n{prompt}\n"

  let output = model.generate(prompt, temp = 1.0f, maxTokens = 20)
  echo &"\n----\nOutput:\n----\n{output}\n"

  doAssert output.len > prompt.len, "Output must be longer than prompt"
  doAssert output.startsWith(prompt), "Output must start with prompt"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✅ PASS | Qwen3-0.6B-EXL3-5bpw simple inference"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()
