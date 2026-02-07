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
  result.tokenizer = loadTokenizerJson(path)

proc encode*(self: TokenizerRef, text: string): seq[int] {.exportpy.} =
  result = self.tokenizer.encodeWithSpecial(text)

proc encode_ordinary*(self: TokenizerRef, text: string): seq[int] {.exportpy.} =
  result = self.tokenizer.encodeOrdinary(text)

proc decode*(self: TokenizerRef, ids: seq[int]): string {.exportpy.} =
  result = decodeToString(self.tokenizer, ids)

proc decode_bytes*(self: TokenizerRef, ids: seq[int]): seq[byte] {.exportpy.} =
  result = self.tokenizer.decodeToBytes(ids)

proc vocab_size*(self: TokenizerRef): int {.exportpy.} =
  result = self.tokenizer.tokenCount

proc special_tokens*(self: TokenizerRef): seq[string] {.exportpy.} =
  result = self.tokenizer.getSpecialTokens()

proc `$`*(self: TokenizerRef): string {.exportpy.} =
  result = "Tokenizer(vocabSize=" & $self.tokenizer.tokenCount & ")"

setModuleDocString("Toktoktok tokenizer - A fast byte-level BPE tokenizer written in Nim")
setDocStringForType(TokenizerRef, "Opaque wrapper around a loaded Toktoktok tokenizer")
