# Toktoktok Python Extension Module
#
# This module exports toktoktok functionality to Python using nimpy.
# Compile with: nim c --app:lib --out:pytoktoktok.so pytoktoktok.nim

import nimpy
import std/tables

import workspace/toktoktok

type
  TokenizerRef* = ref object of PyNimObjectExperimental
    tokenizer*: BPETokenizer

proc load_tokenizer_hf*(path: string): TokenizerRef {.exportpy.} =
  result = TokenizerRef()
  result.tokenizer = loadHFTokenizer(path)

proc load_tokenizer_tiktoken*(path: string): TokenizerRef {.exportpy.} =
  result = TokenizerRef()
  result.tokenizer = loadTiktokenizer(path)

proc encode*(self: TokenizerRef, text: string): seq[int] {.exportpy.} =
  self.tokenizer.encode(text)

proc encode_ordinary*(self: TokenizerRef, text: string): seq[int] {.exportpy.} =
  self.tokenizer.encodeOrdinary(text)

proc decode*(self: TokenizerRef, ids: seq[int]): string {.exportpy.} =
  decodeToString(self.tokenizer, ids)

proc vocab_size*(self: TokenizerRef): int {.exportpy.} =
  self.tokenizer.tokenCount

proc `$`*(self: TokenizerRef): string {.exportpy.} =
  "Tokenizer(vocabSize=" & $self.tokenizer.tokenCount & ")"

setModuleDocString("Toktoktok tokenizer - A fast byte-level BPE tokenizer written in Nim")
setDocStringForType(TokenizerRef, "Opaque wrapper around a loaded Toktoktok tokenizer")
