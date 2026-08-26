# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-t2t \
##   --nimcache:nimcache/tests/qwen35-t2t \
##   workspace/transformers/tests/q_bf16/test_qwen3_5_06_t2t_inference.nim

import
  std/json,
  std/os,
  std/strformat,
  std/strutils,
  workspace/libtorch,
  workspace/toktoktok,
  workspace/transformers/src/models,
  workspace/libtorch_testutils

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "greedy-decoding" / "Qwen3.5-0.8B"

# Decode entry for Qwen3.5-0.8B: the config has no bos_token_id and no
# generation_config.json, so generation starts from the prompt tokens
# directly. generate() tokenizes the raw prompt, prefills, then decodes;
# no special token is prepended. The stop condition is the config
# eos_token_id 248044, not the tokenizer's own eos (248046, im_end).
# The vendored fixture generator uses the same convention (its fixtures
# carry prompt_ids = tokenize(prompt) with nothing prepended), and the
# prompt_ids asserts below lock the match.

proc main() =
  runCppTest "Qwen3.5-0.8B t2t round-trip + decode entry (no bos, eos 248044)":
    proc(): bool =
      echo "Loading model..."
      let model = loadModel($(ModelPath), kCPU)

      # The generate() stop condition uses the config eos_token_id (248044).
      doAssert model.getConfig().eosTokenId == 248044,
        "generate() must stop at config eos 248044, not the tokenizer im_end 248046"

      # Decode entry: encode(prompt) must equal the vendored fixture
      # prompt_ids exactly, proving no bos token is prepended on either
      # side. The two NFC-clean fixture prompts lock the convention.
      for f in ["Hello_how_are_you.json", "What_is_the_capital_of_France.json"]:
        let data = parseJson(readFile(FixtureDir / f))
        let prompt = data["prompt"].getStr()
        var expectedIds: seq[int] = @[]
        for el in data["prompt_ids"]:
          expectedIds.add(el.getInt())
        doAssert model.getTokenizer().encode(prompt) == expectedIds,
          "decode entry diverges from the vendored convention in " & f

      # Combining marks: the pre-tokenizer regex includes \p{M}. The resume
      # fixture prompt is decomposed (e + U+0301). Tokenize then untokenize
      # must reproduce the prompt text byte for byte. (toktoktok does not
      # implement the tokenizer.json NFC normalizer, so the decomposed form
      # tokenizes to different ids than the vendored tokenizer; the contract
      # here is the tokenize/untokenize round-trip.)
      let resumePrompt = "The re\u0301sume\u0301 is ready"
      let markTokens = model.getTokenizer().encode(resumePrompt)
      doAssert model.getTokenizer().decodeToString(markTokens) == resumePrompt,
        "combining-mark prompt must round-trip through tokenize/untokenize"

      # End-to-end generate: prefill on the prompt, decode a short
      # continuation. maxContextLen is bounded so the page pool stays small
      # (the default maxContextLen = -1 would size it to
      # max_position_embeddings 262144).
      let output = model.generate(resumePrompt, temp = 1.0f, maxTokens = 16,
                                  maxContextLen = 512)
      echo "Output: " & output
      doAssert output.len > resumePrompt.len, "output must be longer than the prompt"
      doAssert output.startsWith(resumePrompt), "output must start with the prompt text"
      true

when isMainModule:
  main()
