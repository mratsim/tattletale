# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Model-free analytic suite for the Kimi-Linear tokenizer, run over
## the checkpoint's own files (tiktoken.model + tokenizer_config.json),
## with no weight tensors touched.
##
## - MoonshotPatStrRegexp serves the kimi_linear facade, KimiK25Regexp is a distinct grammar, never the template
## - 258 special slots over 163584..163841, decoder entries direct-indexed, gaps as reserved fillers
## - spot encode rows vs the tiktoken 0.14.0 engine on the checkpoint pat_str and mergeable ranks, ASCII/CJK/contraction/digit/whitespace edges
## - special-token embedding at its id and decode roundtrips on every edge string
##
## Run:
##   nim cpp -d:release --stackTrace:on --debugger:native --passC:"-std=c++20" --verbosity:0 --hints:off \
##     --warnings:off --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_kimi_02_tokenizer.nim

import
  std/algorithm,
  std/json,
  std/os,
  std/strutils,
  std/tables,
  workspace/toktoktok,
  workspace/transformers/src/models/kimi_linear

const
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" /
    "Kimi-Linear-48B-A3B-Instruct"

template orRaise(cond: bool; msg: string) =
  ## Suite enforcement form, cond false raises ValueError carrying msg.
  if not cond:
    raise newException(ValueError, msg)

proc regexpIdentity() =
  ## MoonshotPatStrRegexp serves the kimi_linear facade, the checkpoint
  ## tokenization_kimi.py pat_str is identical to the Moonlight pat_str,
  ## the KimiK25Regexp sibling differs and is a distinct grammar.
  orRaise(KimiK25Regexp.regexp != MoonshotPatStrRegexp.regexp,
    "KimiK25Regexp equals MoonshotPatStrRegexp")

proc specialsSynthesis(tok: BPETokenizer, slots: int) =
  ## `slots` special slots past the mergeable ranks, every
  ## added_tokens_decoder entry direct-indexed at its id, every gap
  ## carrying the reserved filler literal.
  let decoder = parseFile(ModelDir / "tokenizer_config.json"){"added_tokens_decoder"}
  orRaise(decoder.kind == JObject,
    "no added_tokens_decoder object in tokenizer_config.json")
  var decoderIds = newSeq[int]()
  for idText, token in decoder:
    let id = parseInt(idText)
    let content = token{"content"}.getStr()
    orRaise(tok.specialTokensEncoder.getOrDefault(content, -1) == id,
      "special id " & $id & " for " & content)
    orRaise(tok.decodeToString(@[id]) == content,
      "decode of special id " & $id & " != its decoder content")
    decoderIds.add id
  decoderIds.sort()
  # Direct-index invariant this block guards:
  #   decoder ids distinct, a duplicate id would alias one special
  #   slot and corrupt the lo/hi window bounds plus the gap scan
  #   below. The check runs on the sorted ids where any duplicate
  #   must sit adjacent
  for i in 1 ..< decoderIds.len:
    orRaise(decoderIds[i] != decoderIds[i - 1],
      "duplicate added_tokens_decoder id " & $decoderIds[i])
  orRaise(tok.specialTokensEncoder.len == slots,
    "special token count " & $tok.specialTokensEncoder.len &
      " != " & $slots)
  var lo, hi = -1
  for _, v in tok.specialTokensEncoder:
    if lo == -1 or v < lo:
      lo = v
    if v > hi:
      hi = v
  orRaise(lo == decoderIds[0],
    "special id window lo " & $lo & " != first decoder id " & $decoderIds[0])
  orRaise(hi == decoderIds[0] + slots - 1,
    "special id window hi " & $hi &
      " != first decoder id + slots - 1 = " & $(decoderIds[0] + slots - 1))
  # Gap fillers:
  #   ids inside the window with no decoder entry spell the filler
  for id in lo .. hi:
    if id notin decoderIds:
      orRaise(tok.specialTokensEncoder.getOrDefault(
          "<|reserved_token_" & $id & "|>", -1) == id,
        "gap id " & $id & " misses the reserved filler literal")

proc spotEncodeRows(tok: BPETokenizer) =
  ## Spot ids measured against the tiktoken 0.14.0 reference engine,
  ## using the checkpoint pat_str + mergeable ranks as engine input.
  ##
  ## Split classes:
  ##   plain ASCII, CJK plus mixed-script, contractions, digit groups
  ##   (\p{N}{1,3} chunking), capitalized prose, whitespace runs.
  const rows = [
    ("Hello, how are you?", @[19180, 11, 1632, 554, 398, 30]),
    ("你好，世界！混合English文本。",
      @[33845, 378, 2243, 856, 13935, 44372, 26386, 292]),
    ("don't we're I'll it's can't", @[88709, 13810, 17916, 4643, 8971]),
    ("20260910 tokens 1234567 42",
      @[2975, 42335, 795, 18524, 220, 6694, 12972, 22, 220, 5512]),
    ("The capital of France is", @[1008, 10484, 318, 15383, 387]),
    ("  spaced\ttabs\nand newlines  ",
      @[220, 89198, 5604, 5609, 198, 516, 814, 11541, 256]),
  ]
  for text, expected in rows.items:
    orRaise(tok.encode(text) == expected, "spot row: " & text)
    orRaise(tok.encodeOrdinary(text) == expected, "ordinary spot row: " & text)

proc specialEmbeddingRow(tok: BPETokenizer) =
  ## A special literal inside text embeds at its id, the surrounding
  ## text splits ordinarily, measured against the reference engine
  ## under the allowed_special marker.
  orRaise(tok.encode("Hi<|im_end|>") == @[18699, 163586],
    "special literal embedding row: Hi<|im_end|>")
  orRaise(tok.decodeToString(@[18699, 163586]) == "Hi<|im_end|>",
    "special literal decode row: Hi<|im_end|>")

proc decodeRoundtrips(tok: BPETokenizer) =
  ## decodeToString(encode(t)) == t on every edge string including
  ## special tokens, no character lost or re-ordered across the codec.
  const texts = [
    "Hello, how are you?",
    "你好，世界！混合English文本。",
    "don't we're I'll it's can't",
    "20260910 tokens 1234567 42",
    "The capital of France is",
    "  spaced\ttabs\nand newlines  ",
    "[BOS][EOS][UNK][PAD]",
    "a<|im_end|>b",
  ]
  for text in texts.items:
    orRaise(tok.decodeToString(tok.encode(text)) == text, "roundtrip: " & text)

proc main() =
  echo "model dir: ", ModelDir
  let tok = loadKimiTokenizer(ModelDir)

  regexpIdentity()
  # Synthesis window:
  #   163584 mergeable ranks, then 258 special slots
  #   (num_reserved_special_tokens 256 plus 2), values measured
  #   with the checkpoint reference tokenizer and cross-checked
  #   on the loaded facade
  specialsSynthesis(tok, slots = 258)
  spotEncodeRows(tok)
  specialEmbeddingRow(tok)
  decodeRoundtrips(tok)

when isMainModule:
  main()
