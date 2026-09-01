# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Generation-config reader: sampling and stop parameters of a checkpoint,
## parsed from `generation_config.json` next to `config.json`.

import
  std/options,
  pkg/packedjson,
  config_json

type
  GenerationConfig* = ref object
    ## Sampling and stop parameters of a checkpoint, parsed from the file
    ## `generation_config.json` next to `config.json`. For decoding, the values
    ## of this file are the ones a generator uses.
    ##
    ## Stop ids are a list in file order: `text_config` keeps one scalar eos,
    ## `generation_config.json` puts the conversation-end id first.
    eosTokenIds*: seq[int]
    bosTokenId*: Option[int] ## Same id as `text_config.bos_token_id`
    padTokenId*: Option[int] ## `null` becomes `none`
    doSample*: bool          ## The file's request, parsed verbatim. The decoding
                             ## rule sits at `loadGenerationConfig`

    temperature*: float64
    topK*: int
    topP*: float64

proc parseGenerationConfig*(json: JsonNode): GenerationConfig =
  ## Parse a generation_config.json body. Raises `ValueError` naming
  ## `eos_token_id` when no stop id could be read: a missing key, a `null`,
  ## a string and a float scalar all yield an empty seq.
  result = new GenerationConfig
  result.eosTokenIds = json{"eos_token_id"}.parseIntList("eos_token_id")
  if result.eosTokenIds.len == 0:
    raise newException(ValueError,
      "[ttt] GenerationConfig.parse: eos_token_id yielded no stop id, found " &
      $json{"eos_token_id"}.kind & ", and the stop set cannot be empty")
  result.bosTokenId = json{"bos_token_id"}.optInt("bos_token_id")
  result.padTokenId = json{"pad_token_id"}.optInt("pad_token_id")
  result.doSample = json{"do_sample"}.getBool(false)
  result.temperature = json{"temperature"}.getFloat(1.0)
  result.topK = json{"top_k"}.getInt()
  result.topP = json{"top_p"}.getFloat(1.0)

proc loadGenerationConfig*(path: string): GenerationConfig =
  ## Load a checkpoint's generation_config.json from disk and parse it.
  ##
  ## Decoding is greedy argmax over `eosTokenIds` with no implicit
  ## bos: instead of inheriting the file's flag, a generator passes
  ## `do_sample = false` explicitly, and `temperature`, `topK`, `topP` stay
  ## unused. Raises `ValueError` when the parsed stop set is empty.
  parseFile(path).parseGenerationConfig()
