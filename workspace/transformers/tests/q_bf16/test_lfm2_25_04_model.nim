# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/lfm2-model \
##   --nimcache:nimcache/tests/lfm2-model \
##   workspace/transformers/tests/q_bf16/test_lfm2_25_04_model.nim

import
  std/memfiles,
  std/os,
  std/strformat,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/lfm2 {.all.},
  workspace/libtorch_testutils

{.experimental: "callOperator".}

const
  FixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "ids-inference" / "LFM2.5-230M"
  # Weights load from the real checkpoint through the git-ignored
  # hf_models/LFM2.5-230M symlink, the layout the Qwen3.5 suites use.
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "LFM2.5-230M"
  NumLayers = 14
  NumKvHeads = 8
  HeadDim = 64
  # Last-position logits span ±16.75, one bf16 step at that magnitude = 0.125.
  # CPU libtorch accumulates the 14-layer bf16 stack in a different order
  # than the generator, and |Δ| reaches exactly one step, so the band takes
  # two. A flat 0.25 is two steps at 16.75 and over 60 steps at |logit| < 1.
  # Re-derive it if the fixture's logit range changes. Same value as LogitNoiseBand
  # in test_lfm2_25_05_greedy_decoding.nim.
  LogitsAbsTol = 0.25'f64

proc checkIdsToLogits(): bool =
  let model = loadLfm2ModelRaw(ModelDir, kCPU)
  doAssert model.config.num_hidden_layers == NumLayers
  doAssert model.config.num_key_value_heads == NumKvHeads
  doAssert model.config.head_dim == HeadDim

  var memFile = memFiles.open(FixtureDir / "ids-logits.safetensor", mode = fmRead)
  defer: close(memFile)
  let fixture = safetensors.load(memFile)

  # Token ids come from the fixture, which carries the reference tokenization.
  # No test compares Nim tokenization against the reference for LFM,
  # so nothing here re-tokenizes. Ids go straight into forward.
  let inputIds = fixture.getTensorOwned("input_ids")
  let seqLen = inputIds.size(1).int

  var ctx = InferenceContext.init(NumLayers, 1, NumKvHeads, seqLen, HeadDim)
  let pool = PagePool.init(
    64, num_layers = NumLayers, kv_heads = NumKvHeads, head_dim = HeadDim,
    dtype = F.kBFloat16, device = F.kCPU)
  for i in 0 ..< ceilDiv(seqLen, TokensPerPage):
    ctx.pages.add(pool.borrow())
  ctx.position_ids = F.arange(seqLen, F.tensorOptions(F.kInt64, F.kCPU))

  let logits = model.forward(ctx, inputIds)
  doAssert logits.size(1) == seqLen

  let last = logits.narrow(1, seqLen - 1, 1)
  let expected = fixture.getTensorOwned("logits_last")
  let maxDiff = (last.to(kFloat32) - expected.to(kFloat32)).abs().max().item(float)
  echo &"last-position logits: max |delta| = {maxDiff:.3e} over {logits.size(2)} vocab entries"
  echo &"next-token argmax: Nim {last.argmax().item(int)} vs fixture {expected.argmax().item(int)}"

  assertAllClose(last, expected,
    rtol = 0.0, abstol = LogitsAbsTol, msg = "full-model last-position logits mismatch")
  doAssert last.argmax().item(int) == expected.argmax().item(int),
    "full-model next-token argmax diverged from the fixture"
  result = true

when isMainModule:
  runCppTest("LFM2.5-230M ids -> last-position logits on the real checkpoint", checkIdsToLogits)
