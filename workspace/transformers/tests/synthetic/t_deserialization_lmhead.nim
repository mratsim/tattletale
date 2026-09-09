# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/synthetic --nimcache:nimcache/tests/synthetic \
##   workspace/transformers/tests/synthetic/t_deserialization_lmhead.nim

## When a checkpoint carries no lm_head tensors, the flag
## `tie_word_embeddings` decides: true builds the tied head, false
## breaks the untied promise and must raise IOError naming lm_head.
## Qwen3.5-0.8B ships head-less and serves both cases.

import
  std/os,
  std/strutils,
  std/unittest,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors/src/collections,
  workspace/libtorch_testutils,
  ../../src/deserialization,
  ../../src/layers/all_reexports

const ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"
const WeightsFile = ModelDir / "model.safetensors-00001-of-00001.safetensors"

proc headCfg(tied: JsonNode): JsonNode =
  %*{"text_config": {"tie_word_embeddings": tied}}

proc dummyEmbedding(): Embedding =
  Embedding.init(F.zeros(4, 8, options = F.tensorOptions(kBFloat16, kCPU)))

suite "LMHead tie decision":
  test "head-less checkpoint with tie_word_embeddings true builds a tied head":
    let view = SafetensorsCollection.open(WeightsFile)
    let head = LMHead.load(view, headCfg(%true), dummyEmbedding())
    check: not head.isNil

  test "head-less checkpoint with tie_word_embeddings false raises IOError":
    let view = SafetensorsCollection.open(WeightsFile)
    expect IOError:
      discard LMHead.load(view, headCfg(%false), dummyEmbedding())

  test "the IOError names the missing tensor and the config promise":
    let view = SafetensorsCollection.open(WeightsFile)
    try:
      discard LMHead.load(view, headCfg(%false), dummyEmbedding())
      check false # the raise below is the pass condition
    except IOError as e:
      check "lm_head" in e.msg
      check "tie_word_embeddings" in e.msg
