# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Toktoktok - BPE Tokenizer for Nim
#
# A BPE tokenizer that loads HuggingFace tokenizer.json files.

# Vibe-coded: This passes tests but need in-depth fixes, see VCR comments (Vibe-Code review)

import std/tables
import std/options
import std/os
import std/re
import std/json # TODO(VCR) switch to jsony and direct to object parsing

type
  ByteLevelConfig* = object
    add_prefix_space*: bool
    trim_offsets*: bool
    use_regex*: bool

  BPETokenizer* = object
    vocab*: seq[string]
    # TODO(VCR): I think we need a `decoder` cache.
    # TODO(VCR): Also ids_to_tokens and tokens_to_ids is clearer.
    encoder*: Table[string, int]
    merges*: seq[(int, int)]
    merge_ranks*: Table[string, int]
    special_tokens*: Table[int, string]
    special_tokens_encoder*: Table[string, int]
    max_token_id*: int
    byte_level_config*: ByteLevelConfig
    pattern*: Regex

  TokenizerError* = object of ValueError

  EncodingResult* = object
    ids*: seq[int]
    tokens*: seq[string]

proc new_byte_level_config*( # TODO(VCR) code style
  add_prefix_space = true,
  trim_offsets = true,
  use_regex = true
): ByteLevelConfig =
  ByteLevelConfig(
    add_prefix_space: add_prefix_space,
    trim_offsets: trim_offsets,
    use_regex: use_regex
  )

proc new_bpe_tokenizer*(): BPETokenizer =
  BPETokenizer(
    vocab: @[],
    encoder: init_table[string, int](), # TODO(VCR) code style
    merges: @[],
    merge_ranks: init_table[string, int](),
    special_tokens: init_table[int, string](),
    special_tokens_encoder: init_table[string, int](),
    max_token_id: 0,
    byte_level_config: new_byte_level_config(),
    pattern: re("")
  )

proc decode_token*(tokenizer: BPETokenizer, token_id: int): string =
  # TODO(VCR):
  #   I think all the range checks should be
  #   extracted in a separate proc to be done
  #   at the sequence level instead of here.
  #   While this makes 2 passes over the data, the first one
  #   is just a range-check that can use SIMD vectorization
  #   and will ready data in cache.
  #   It also avoids duplicate checks if the merge recursion below cannot be avoided.
  if token_id < 0:
    raise new_exception(
      TokenizerError,
      "Invalid token ID: " & $token_id & " (negative)"
    )
  if token_id < 256:
    return $chr(token_id)
  if token_id >= tokenizer.max_token_id:
    raise new_exception(
      TokenizerError,
      "Token ID " & $token_id & " exceeds vocabulary size " &
      "(max: " & $(tokenizer.max_token_id - 1) & ")"
    )
  if hasKey(tokenizer.special_tokens, token_id):
    return tokenizer.special_tokens[token_id]
  if token_id < tokenizer.vocab.len:
    return tokenizer.vocab[token_id]
  let merge_idx = token_id - tokenizer.vocab.len
  if merge_idx >= tokenizer.merges.len:
    raise new_exception(
      TokenizerError,
      "Merge index out of bounds: " & $merge_idx &
      " (merges count: " & $(tokenizer.merges.len) & ")"
    )
  # TODO(VCR): Can this be precomputed to avoid potentially costly recursion here?
  let (left_id, right_id) = tokenizer.merges[merge_idx]
  decode_token(tokenizer, left_id) & decode_token(tokenizer, right_id)

proc decode_to_string*(tokenizer: BPETokenizer, token_ids: seq[int]): string =
  result = "" # TODO(VCR): preallocate some sensible size, atleast token_ids.len
  for id in token_ids:
    # TODO(VCR): decode_token should likely work in-place of result
    result &= decode_token(tokenizer, id)

proc token_count*(tokenizer: BPETokenizer): int = tokenizer.max_token_id
proc get_vocab_size*(tokenizer: BPETokenizer): int = tokenizer.max_token_id
proc is_special_token*(tokenizer: BPETokenizer, token_id: int): bool =
  hasKey(tokenizer.special_tokens, token_id)

type
  MergePart* = tuple[idx: int, rank: int]

proc byte_pair_merge*(tokenizer: BPETokenizer, piece: string): seq[MergePart] =
  var parts: seq[MergePart] = new_seq[MergePart](piece.len + 1) # TODO(VCR): code style
  var min_rank = (high(int), -1)
  for i in 0..<piece.len - 1:
    let pair_str = piece[i..i+1]
    let rank = tokenizer.merge_ranks.getOrDefault(pair_str, high(int))
    parts[i] = (i, rank)
    if rank < min_rank[0]:
      min_rank = (rank, i)
  parts[piece.len - 1] = (piece.len - 1, high(int))
  parts[piece.len] = (piece.len, high(int))

  proc get_rank(parts: seq[MergePart], i: int, piece: string): int =
    if i + 2 < parts.len:
      let start_idx = parts[i].idx
      let end_idx = parts[i+2].idx
      if end_idx <= piece.len:
        let pair_str = piece[start_idx..<end_idx]
        result = tokenizer.merge_ranks.getOrDefault(pair_str, high(int))
      else:
        result = high(int)
    else:
      result = high(int)

  # TODO(VCR): That seems very complex or need a proper state machine to debug transitions
  var can_merge = false
  if min_rank[0] != high(int):
    can_merge = true
  while can_merge:
    let i = min_rank[1]
    let left_rank = if i > 0: parts[i-1][1] else: high(int)
    let right_rank = parts[i][1]
    if left_rank == high(int) or right_rank == high(int):
      let merged_start = parts[i].idx
      let merged_end = if i + 1 < parts.len: parts[i+1].idx else: merged_start
      let merged_token = piece[merged_start..<merged_end]
      if not hasKey(tokenizer.encoder, merged_token):
        break
    if i > 0:
      parts[i-1][1] = get_rank(parts, i-1, piece)
    parts[i][1] = get_rank(parts, i, piece)
    parts.delete(i + 1)
    can_merge = false
    min_rank = (high(int), -1)
    for j in 0..<parts.len - 1:
      let rank = parts[j][1]
      if rank != high(int) and rank < min_rank[0]:
        min_rank = (rank, j)
        can_merge = true
  parts

proc byte_pair_encode*(tokenizer: BPETokenizer, piece: string): seq[int] =
  if piece.len == 1:
    let char_str = piece
    # TODO(VCR): use `proc getOrDefault[A, B](t: OrderedTable[A, B]; key: A; def: B): B`
    if hasKey(tokenizer.encoder, char_str):
      return @[tokenizer.encoder[char_str]]
    else:
      return @[int(ord(char_str[0]))]
  let parts = byte_pair_merge(tokenizer, piece)
  var token_ids: seq[int] = @[]
  for i in 0..<parts.len - 1:
    let start_idx = parts[i].idx
    let end_idx = parts[i+1].idx
    let token_str = piece[start_idx..<end_idx]
    let token_id = tokenizer.encoder.getOrDefault(token_str, -1)
    if token_id >= 0:
      token_ids.add(token_id)
    elif hasKey(tokenizer.special_tokens_encoder, token_str):
      token_ids.add(tokenizer.special_tokens_encoder[token_str])
    else:
      let first_byte = int(ord(piece[start_idx]))
      if first_byte >= 128:
        token_ids.add(first_byte)
      else:
        for j in start_idx..<end_idx:
          token_ids.add(int(ord(piece[j])))
  token_ids

proc encode_ordinary*(tokenizer: BPETokenizer, text: string): seq[int] =
  if tokenizer.pattern == re(""):
    raise new_exception(TokenizerError, "Pattern not set for tokenizer")
  var ids: seq[int] = @[]
  var i = 0
  while i < text.len:
    let remaining = text[i..^1]
    let pieces = findAll(remaining, tokenizer.pattern)
    if pieces.len > 0:
      var piece_str = pieces[0]
      if piece_str.len == 0:
        i += 1
        continue
      if piece_str[0] == ' ' and pieces.len > 1:
        let next_piece = pieces[1]
        if next_piece.len > 0:
          let combined = "Ġ" & next_piece
          if hasKey(tokenizer.encoder, combined):
            ids.add(tokenizer.encoder[combined])
            i += piece_str.len + next_piece.len
            continue
      if hasKey(tokenizer.encoder, piece_str):
        ids.add(tokenizer.encoder[piece_str])
      else:
        for c in piece_str:
          ids.add(int(ord(c)))
      i += piece_str.len
    else:
      break
  ids

proc encode*(tokenizer: BPETokenizer, text: string, special: bool = false): EncodingResult =
  var result_ids: seq[int] = @[]
  var result_tokens: seq[string] = @[]
  var special_tokens: seq[string] = @[]
  for k in keys(tokenizer.special_tokens_encoder):
    special_tokens.add(k)
  var pos = 0

  while pos < text.len:
    var found_special = false
    for special_token in special_tokens:
      let token_pattern = re(escapeRe(special_token)) # TODO(VCR): Can this be cached?
      if match(text, token_pattern, pos):
        let special_token_str = special_token
        # TODO(VCR): why are `<` `>` special-cased? MiniMax-M2 has very unique tokens for example
        if special or (special_token_str.len > 1 and special_token_str[0] == '<' and special_token_str[^1] == '>'):
          let token_id = tokenizer.special_tokens_encoder[special_token_str]
          result_ids.add(token_id)
          result_tokens.add(special_token_str)
          pos += special_token_str.len
          found_special = true
          break
    if found_special:
      continue

    var matched = false
    let pieces = findAll(text[pos..^1], tokenizer.pattern)
    if pieces.len > 0:
      var piece_str = pieces[0]
      if piece_str.len == 0:
        pos += 1
        continue
      # TODO(VCR): That special casing seems absolutely buggy
      if piece_str[0] == ' ' and pieces.len > 1:
        let next_piece = pieces[1]
        if next_piece.len > 0:
          let combined = "Ġ" & next_piece
          if hasKey(tokenizer.encoder, combined):
            result_ids.add(tokenizer.encoder[combined])
            result_tokens.add(combined)
            pos += piece_str.len + next_piece.len
            matched = true
      if not matched:
        if hasKey(tokenizer.encoder, piece_str):
          result_ids.add(tokenizer.encoder[piece_str])
        else:
          let encoded = byte_pair_encode(tokenizer, piece_str)
          for id in encoded:
            result_ids.add(id)
        result_tokens.add(piece_str)
        pos += piece_str.len
        matched = true

    if not matched and pos < text.len:
      let c = text[pos]
      let char_str = $c
      if hasKey(tokenizer.encoder, char_str):
        result_ids.add(tokenizer.encoder[char_str])
        result_tokens.add(char_str)
      else:
        raise new_exception(TokenizerError, "Unknown character: " & $c)
      pos += 1

  EncodingResult(ids: result_ids, tokens: result_tokens)

type
  AddedTokenJson* = object
    id*: int
    content*: string
    special*: Option[bool]
    single_word*: Option[bool]
    lstrip*: Option[bool]
    rstrip*: Option[bool]
    normalized*: Option[bool]

proc load_tokenizer_json*(path: string): BPETokenizer = # TODO(VCR): code style in whole proc
  if not file_exists(path):
    raise new_exception(TokenizerError, "Tokenizer file not found: " & path)
  let content = read_file(path)
  if content.len == 0:
    raise new_exception(TokenizerError, "Tokenizer file is empty: " & path)

  let jsonNode = content.parseJson()
  var tokenizer = new_bpe_tokenizer()

  if jsonNode.hasKey("added_tokens"):
    var special_tokens = init_table[int, string]()
    var special_tokens_encoder = init_table[string, int]()
    let added_tokens = jsonNode["added_tokens"]
    for token in added_tokens:
      let id_val = token["id"]
      if id_val.kind == JInt:
        let id = int(id_val.getInt)
        let content_str = token["content"].getStr
        special_tokens[id] = content_str
        special_tokens_encoder[content_str] = id
    tokenizer.special_tokens = special_tokens
    tokenizer.special_tokens_encoder = special_tokens_encoder

  if not jsonNode.hasKey("model"):
    raise new_exception(TokenizerError, "Missing 'model' section")
  let model = jsonNode["model"]

  let model_type = model["type"].getStr
  if model_type != "BPE":
    raise new_exception(TokenizerError, "Unsupported model type: " & model_type)

  var vocab: seq[string]
  var encoder = init_table[string, int]()
  var token_to_id = init_table[string, int]()

  # I think all of that can be skipped with proper `jsony` parseJSON direct to object
  if model.hasKey("vocab"):
    let vocabNode = model["vocab"]
    if vocabNode.kind == JObject:
      var max_id = 0
      for key, value in vocabNode:
        if value.kind == JInt:
          let id = int(value.getInt)
          vocab.add(key)
          encoder[key] = id
          if id >= max_id:
            max_id = id + 1
      vocab = new_seq[string](max_id)
      for key, value in vocabNode:
        if value.kind == JInt:
          let id = int(value.getInt)
          vocab[id] = key
          encoder[key] = id
    elif vocabNode.kind == JArray:
      for item in vocabNode:
        if item.kind == JArray:
          let id = int(item[1].getInt)
          let token_str = item[0].getStr
          vocab[id] = token_str
          encoder[token_str] = id
  else:
    vocab = @[]

  for id in 0..<vocab.len:
    let token_str = vocab[id]
    if token_str.len > 0:
      token_to_id[token_str] = id

  tokenizer.vocab = vocab
  tokenizer.encoder = encoder

  var merges: seq[(int, int)]
  var merge_ranks = init_table[string, int]()

  if model.hasKey("merges"):
    let mergesNode = model["merges"]
    var merge_list: seq[string] = @[]
    if mergesNode.kind == JArray:
      for item in mergesNode:
        if item.kind == JString:
          merge_list.add(item.getStr)
        elif item.kind == JArray:
          merge_list.add(item[0].getStr & " " & item[1].getStr)
    merges = new_seq[(int, int)](merge_list.len)
    for i, merge_str in merge_list:
      let space_idx = merge_str.find(' ')
      let left_token = if space_idx >= 0: merge_str[0..<space_idx] else: merge_str
      let right_token = if space_idx >= 0: merge_str[space_idx+1..^1] else: ""
      let pair_key = left_token & right_token
      merge_ranks[pair_key] = i
      var left_id = token_to_id.getOrDefault(left_token, -1)
      var right_id = token_to_id.getOrDefault(right_token, -1)
      if left_id < 0:
        if left_token.len == 1:
          left_id = int(ord(left_token[0]))
        else:
          raise new_exception(TokenizerError, "Left token not found: " & left_token)
      if right_id < 0:
        if right_token.len == 1:
          right_id = int(ord(right_token[0]))
        else:
          raise new_exception(TokenizerError, "Right token not found: " & right_token)
      merges[i] = (left_id, right_id)
  else:
    merges = @[]

  tokenizer.merges = merges
  tokenizer.merge_ranks = merge_ranks
  tokenizer.max_token_id = vocab.len

  if jsonNode.hasKey("added_tokens"):
    let added_len = jsonNode["added_tokens"].len
    tokenizer.max_token_id = max(tokenizer.max_token_id, added_len)

  tokenizer.byte_level_config = new_byte_level_config()
  tokenizer.pattern = re("'s|'t|'re|'ve|'m|'ll|'d|[\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s+(?!\\S)|\\s+")

  tokenizer
