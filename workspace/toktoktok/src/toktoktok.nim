## Toktoktok - BPE Tokenizer for Nim
##
## A BPE tokenizer that loads HuggingFace tokenizer.json files.
## Uses jsony for fast JSON parsing. Supports both encoding and decoding.

import std/json
import std/tables
import std/options
import std/os
import std/json
import std/re

type
  ByteLevelConfig* = object
    add_prefix_space*: bool
    trim_offsets*: bool
    use_regex*: bool

  BPETokenizer* = object
    vocab*: seq[seq[byte]]
    encoder*: Table[string, int]
    merges*: seq[(int, int)]
    merge_ranks*: Table[string, int]
    special_tokens*: Table[int, seq[byte]]
    special_tokens_encoder*: Table[string, int]
    max_token_id*: int
    byte_level_config*: ByteLevelConfig
    pattern*: string

  TokenizerError* = object of ValueError
    message*: string

  EncodingResult* = object
    ids*: seq[int]
    tokens*: seq[string]

proc new_byte_level_config*(
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
    encoder: init_table[string, int](),
    merges: @[],
    merge_ranks: init_table[string, int](),
    special_tokens: init_table[int, seq[byte]](),
    special_tokens_encoder: init_table[string, int](),
    max_token_id: 0,
    byte_level_config: new_byte_level_config(),
    pattern: ""
  )

proc decode_token*(tokenizer: BPETokenizer, token_id: int): seq[byte] =
  if token_id < 0:
    raise new_exception(
      TokenizerError,
      "Invalid token ID: " & $token_id & " (negative)"
    )
  if token_id < 256:
    return @[byte(token_id)]
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
  let (left_id, right_id) = tokenizer.merges[merge_idx]
  let left_bytes = decode_token(tokenizer, left_id)
  let right_bytes = decode_token(tokenizer, right_id)
  left_bytes & right_bytes

proc decode_to_string*(tokenizer: BPETokenizer, token_ids: seq[int]): string =
  var decoded = new_string_of_cap(token_ids.len * 2)
  for id in token_ids:
    let bytes = decode_token(tokenizer, id)
    for b in bytes:
      decoded.add(chr(b))
  decoded

proc decode_to_bytes*(tokenizer: BPETokenizer, token_ids: seq[int]): seq[byte] =
  var decoded = new_seq[byte](token_ids.len * 2)
  for id in token_ids:
    let bytes = decode_token(tokenizer, id)
    decoded.add(bytes)
  decoded

proc token_count*(tokenizer: BPETokenizer): int = tokenizer.max_token_id
proc get_vocab_size*(tokenizer: BPETokenizer): int = tokenizer.max_token_id
proc is_special_token*(tokenizer: BPETokenizer, token_id: int): bool =
  hasKey(tokenizer.special_tokens, token_id)

proc to_utf8_bytes*(s: string): seq[byte] =
  var result = new_seq[byte](s.len)
  for i, c in s:
    result[i] = byte(ord(c))
  result

proc bytes_to_string*(bytes: seq[byte]): string =
  var result = new_string(bytes.len)
  for i, b in bytes:
    result[i] = chr(b)
  result

proc byte_pair_merge*(tokenizer: BPETokenizer, piece: seq[byte]): seq[(int, int)] =
  var parts: seq[(int, int)] = new_seq[(int, int)](piece.len + 1)
  var min_rank = (high(int), -1)
  for i in 0..<piece.len - 1:
    let pair_str = bytes_to_string(@[piece[i], piece[i+1]])
    let rank = tokenizer.merge_ranks.getOrDefault(pair_str, high(int))
    parts[i] = (i, rank)
    if rank < min_rank[0]:
      min_rank = (rank, i)
  parts[piece.len - 1] = (piece.len - 1, high(int))
  parts[piece.len] = (piece.len, high(int))

  proc get_rank(parts: seq[(int, int)], i: int, piece: seq[byte]): int =
    if i + 3 < parts.len:
      let start_idx = parts[i][0]
      let end_idx = parts[i+3][0]
      if end_idx <= piece.len:
        let pair_str = bytes_to_string(piece[start_idx..<end_idx])
        result = tokenizer.merge_ranks.getOrDefault(pair_str, high(int))
      else:
        result = high(int)
    else:
      result = high(int)

  while min_rank[0] != high(int):
    let i = min_rank[1]
    if i > 0:
      parts[i-1][1] = get_rank(parts, i-1, piece)
    parts[i][1] = get_rank(parts, i, piece)
    parts.delete(i + 1)
    min_rank = (high(int), -1)
    for j in 0..<parts.len - 1:
      if parts[j][1] < min_rank[0]:
        min_rank = (parts[j][1], j)
  parts

proc byte_pair_encode*(tokenizer: BPETokenizer, piece: seq[byte]): seq[int] =
  if piece.len == 1:
    let b = piece[0]
    if b < 256:
      let char_str = bytes_to_string(piece)
      if hasKey(tokenizer.encoder, char_str):
        return @[tokenizer.encoder[char_str]]
      else:
        return @[int(b)]
    else:
      raise new_exception(TokenizerError, "Invalid byte value: " & $b)
  let parts = byte_pair_merge(tokenizer, piece)
  var token_ids = new_seq[int](parts.len - 1)
  for i in 0..<parts.len - 1:
    let start_idx = parts[i][0]
    let end_idx = parts[i+1][0]
    let token_bytes = piece[start_idx..<end_idx]
    let s = bytes_to_string(token_bytes)
    token_ids[i] = tokenizer.encoder.getOrDefault(s, -1)
  token_ids

proc encode_ordinary*(tokenizer: BPETokenizer, text: string): seq[int] =
  if tokenizer.pattern == "":
    raise new_exception(TokenizerError, "Pattern not set for tokenizer")
  let regex = re(tokenizer.pattern)
  var result: seq[int] = @[]
  let pieces = findAll(text, regex)
  for piece_str in pieces:
    if piece_str.len == 0:
      continue
    let piece_bytes = to_utf8_bytes(piece_str)
    if hasKey(tokenizer.encoder, piece_str):
      result.add(tokenizer.encoder[piece_str])
    else:
      for pb in piece_bytes:
        let char_str = bytes_to_string(@[pb])
        if hasKey(tokenizer.encoder, char_str):
          result.add(tokenizer.encoder[char_str])
        else:
          result.add(int(pb))
  result

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
      let token_pattern = re(escapeRe(special_token))
      if match(text, token_pattern, pos):
        let special_token_str = special_token
        if special or (special_token_str.len > 1 and special_token_str[0] == '<' and special_token_str[^1] == '>'):
          let token_id = tokenizer.special_tokens_encoder[special_token_str]
          result_ids.add(token_id)
          result_tokens.add(special_token_str)
          pos += special_token_str.len
          found_special = true
          break
    if found_special:
      continue

    let regex = re(tokenizer.pattern)
    var matched = false
    let pieces = findAll(text[pos..^1], regex)
    if pieces.len > 0:
      let piece_str = pieces[0]
      if piece_str.len == 0:
        pos += 1
        continue
      let piece_bytes = to_utf8_bytes(piece_str)
      if hasKey(tokenizer.encoder, piece_str):
        result_ids.add(tokenizer.encoder[piece_str])
      else:
        let encoded = byte_pair_encode(tokenizer, piece_bytes)
        for id in encoded:
          result_ids.add(id)
      result_tokens.add(piece_str)
      pos += piece_str.len
      matched = true

    if not matched and pos < text.len:
      let c = text[pos]
      let piece_bytes = @[byte(ord(c))]
      let piece_str = bytes_to_string(piece_bytes)
      if hasKey(tokenizer.encoder, piece_str):
        result_ids.add(tokenizer.encoder[piece_str])
        result_tokens.add(piece_str)
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

  TokenizerJson* = object
    version*: Option[string]
    added_tokens*: Option[seq[AddedTokenJson]]
    model*: Option[ModelJson]

  ModelJson* = object
    modelType*: string
    vocab*: Option[JsonNode]
    merges*: Option[seq[seq[string]]]

proc load_tokenizer_json*(path: string): BPETokenizer =
  if not file_exists(path):
    raise new_exception(TokenizerError, "Tokenizer file not found: " & path)
  let content = read_file(path)
  if content.len == 0:
    raise new_exception(TokenizerError, "Tokenizer file is empty: " & path)
  let jsonNode = content.parseJson()
  var tokenizer = new_bpe_tokenizer()

  if jsonNode.hasKey("added_tokens"):
    var special_tokens = init_table[int, seq[byte]]()
    var special_tokens_encoder = init_table[string, int]()
    let added_tokens = jsonNode["added_tokens"]
    for token in added_tokens:
      let id = int(token["id"].getInt)
      let content_str = token["content"].getStr
      special_tokens[id] = content_str.to_utf8_bytes
      special_tokens_encoder[content_str] = id
    tokenizer.special_tokens = special_tokens
    tokenizer.special_tokens_encoder = special_tokens_encoder

  if not jsonNode.hasKey("model"):
    raise new_exception(TokenizerError, "Missing 'model' section")
  let model = jsonNode["model"]
  if model["type"].getStr != "BPE":
    raise new_exception(TokenizerError, "Unsupported model type: " & model["type"].getStr)

  var vocab: seq[seq[byte]]
  var encoder = init_table[string, int]()
  let vocabNode = model["vocab"]
  if vocabNode.kind == JObject:
    vocab = new_seq[seq[byte]]()
    for key, value in vocabNode:
      let id = int(value.getInt)
      if id >= vocab.len:
        let old_len = vocab.len
        vocab.set_len(id + 1)
        for i in old_len ..< id:
          vocab[i] = @[]
      vocab[id] = key.to_utf8_bytes
      encoder[key] = id
  else:
    vocab = new_seq[seq[byte]](vocabNode.len)
    for item in vocabNode:
      let id = int(item["id"].getInt)
      let token_str = item["token"].getStr
      vocab[id] = token_str.to_utf8_bytes
      encoder[token_str] = id
  tokenizer.vocab = vocab
  tokenizer.encoder = encoder

  var merges: seq[(int, int)]
  var merge_ranks = init_table[string, int]()
  var token_to_id = init_table[string, int]()
  for id in 0 ..< vocab.len:
    let bytes = vocab[id]
    if bytes.len > 0:
      var token_str = new_string(bytes.len)
      for i, b in bytes:
        token_str[i] = chr(b)
      token_to_id[token_str] = id

  let mergesNode = model["merges"]
  merges = new_seq[(int, int)](mergesNode.len)
  for i in 0 ..< mergesNode.len:
    let mergeItem = mergesNode[i]
    var left_token, right_token: string
    if mergeItem.kind == JArray:
      left_token = mergeItem[0].getStr
      right_token = mergeItem[1].getStr
    else:
      let merge_str = mergeItem.getStr
      let space_idx = merge_str.find(' ')
      left_token = if space_idx >= 0: merge_str[0..<space_idx] else: merge_str
      right_token = if space_idx >= 0: merge_str[space_idx+1..^1] else: ""
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
  tokenizer.merges = merges
  tokenizer.merge_ranks = merge_ranks
  tokenizer.max_token_id = vocab.len

  if jsonNode.hasKey("post_processor"):
    let post_processor = jsonNode["post_processor"]
    if post_processor.hasKey("type") and post_processor["type"].getStr == "ByteLevel":
      tokenizer.byte_level_config = new_byte_level_config()
  if jsonNode.hasKey("pre_tokenizer"):
    let pre_tokenizer = jsonNode["pre_tokenizer"]
    if pre_tokenizer.hasKey("type") and pre_tokenizer["type"].getStr == "ByteLevel":
      tokenizer.byte_level_config = new_byte_level_config()

  tokenizer.pattern = "'s|'t|'re|'ve|'m|'ll|'d|[\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s+(?!\\S)|\\s+"

  tokenizer
