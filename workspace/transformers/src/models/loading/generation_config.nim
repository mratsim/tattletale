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
    ## `generation_config.json` next to `config.json`.
    ##
    ## Stop ids are a list in file order: `text_config` keeps one scalar eos,
    ## `generation_config.json` puts the conversation-end id first.
    eosTokenIds*: seq[int]
    bosTokenId*: Option[int]
    padTokenId*: Option[int]

proc parseGenerationConfig*(json: JsonNode): GenerationConfig =
  ## Parse a generation_config.json body. Greedy decoding reads the stop set
  ## and the special ids, so only those fields are parsed. Sampling parameters
  ## in the file are ignored. Raises `ValueError` naming `eos_token_id`
  ## when no stop id could be read: an absent key, a `null` and an empty list
  ## all yield an empty seq. A wrong-typed value raises from the reader itself.
  result = new GenerationConfig
  result.eosTokenIds = json{"eos_token_id"}.parseIntList("eos_token_id")
  if result.eosTokenIds.len == 0:
    raise newException(ValueError,
      "[ttt] GenerationConfig.parse: eos_token_id yielded no stop id, found " &
      $json{"eos_token_id"}.kind & ", and the stop set cannot be empty")
  result.bosTokenId = json{"bos_token_id"}.optInt("bos_token_id")
  result.padTokenId = json{"pad_token_id"}.optInt("pad_token_id")

proc loadGenerationConfig*(path: string): GenerationConfig =
  ## Decoding is greedy argmax over `eosTokenIds` with no implicit bos.
  ## The generator reaches argmax by its own construction and inherits no
  ## sampling flags from the file.
  ## Raises `ValueError` when the parsed stop set is empty.
  parseFile(path).parseGenerationConfig()
