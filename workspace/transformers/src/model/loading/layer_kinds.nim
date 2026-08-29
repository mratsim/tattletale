# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## The attention layer kind catalogue plus the parse bound where checkpoint
## vocabulary spellings die.

type
  AttentionLayerKind* = enum
    ## The attention variants of a hybrid decoder stack, named canonically:
    ## each checkpoint vocabulary maps into this catalogue at its parse
    ## bound (`parseAttnFromHfTransformers`) and every consumer downstream
    ## compares kinds.
    alkAttention     ## The softmax-attention slot of a hybrid stack: GQA
                     ## grouping, gating and rope policy are config-derived
                     ## properties of the wiring, not per-layer kinds. Spellings:
                     ## `full_attention`, `linear_attention` at the spelling wall
    alkGatedDeltaNet ## Linear attention: the Gated DeltaNet kind
    alkMla           ## Multi-head latent attention, a DeepSeek invention
                     ## (GLM-4.7 adopts it). No mapped spelling yet: the family
                     ## ships no per-layer attention alphabet, so the spelling
                     ## arrives with its port

proc parseAttnFromHfTransformers*(raw: string, source: string): AttentionLayerKind =
  ## Map one raw spelling from an HF transformers `config.json` `layer_types`
  ## entry to its catalogue kind, exact match. `source` names
  ## the entry's config path for the error message (`text_config.layer_types[3]`).
  ## Raises `ValueError`, with a `[ttt]` prefix, naming `raw` and `source`
  ## for every other text: vendor spellings die at this bound, callers
  ## downstream compare kinds.
  ##
  ## GGUF checkpoints read their kind tags through their own parser later,
  ## mapping into this same catalogue.
  ##
  ##   doAssert parseAttnFromHfTransformers("full_attention", "layer_types[0]") == alkAttention
  const spellings = [
    ("full_attention", alkAttention),
    ("linear_attention", alkGatedDeltaNet),
  ]
  for entry in spellings:
    if raw == entry[0]:
      return entry[1]
  var mapped = ""
  for i, entry in spellings:
    if i > 0:
      mapped.add ", "
    mapped.add "\"" & entry[0] & "\""
  raise newException(ValueError,
    "[ttt] parseAttnFromHfTransformers: unknown attention layer kind \"" & raw &
    "\" at " & source & ", mapped spellings are " & mapped)
