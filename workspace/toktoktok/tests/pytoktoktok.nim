# Toktoktok Python Extension Module
#
# This module exports toktoktok functionality to Python using nimpy.
# Compile with: nim c --app:lib --out:toktoktok.so pytoktoktok.nim

import nimpy
import std/os
import std/tables

import workspace/toktoktok

type
  TokenizerRef* = ref object of PyNimObjectExperimental
    tokenizer*: BPETokenizer

proc load_tokenizer*(path: string): TokenizerRef {.exportpy.} =
  result = TokenizerRef()
  result.tokenizer = load_tokenizer_json(path)

proc encode*(self: TokenizerRef, text: string): seq[int] {.exportpy.} =
  result = self.tokenizer.encode(text).ids

proc decode*(self: TokenizerRef, ids: seq[int]): string {.exportpy.} =
  result = decode_to_string(self.tokenizer, ids)

proc vocab_size*(self: TokenizerRef): int {.exportpy.} =
  result = self.tokenizer.max_token_id

proc `$`*(self: TokenizerRef): string {.exportpy.} =
  result = "Tokenizer(vocab_size=" & $self.tokenizer.max_token_id & ")"

setModuleDocString("Toktoktok BPE Tokenizer - A fast tokenizer written in Nim")
setDocStringForType(TokenizerRef, "Opaque wrapper around a loaded BPE tokenizer")
