# Toktoktok Python Extension Module
#
# This module exports toktoktok functionality to Python using nimpy.
# Compile with: nim c --app:lib --out:pytoktoktok.so pytoktoktok.nim

import nimpy
import std/tables

import workspace/toktoktok

type
  RegexPattern* {.size: 4, pure.} = enum
    r50k = 1,
    p50k = 2,
    cl100k = 3,
    o200k = 4,
    kimik25 = 5

  TokenizerRef* = ref object of PyNimObjectExperimental
    tokenizer*: BPETokenizer

proc load_tokenizer_hf*(path: string): TokenizerRef {.exportpy.} =
  result = TokenizerRef()
  result.tokenizer = loadHFTokenizer(path)

proc load_tokenizer_tiktoken*(path: string, pattern: string): TokenizerRef {.exportpy.} =
  result = TokenizerRef()
  let regexp = case pattern
    of "r50k": R50kRegexp
    of "p50k": P50kRegexp
    of "cl100k": Cl100kRegexp
    of "o200k": O200kRegexp
    of "kimik2.5": KimiK25Regexp
    else:
      raise newException(ValueError, "Unknown pattern: " & pattern)
  result.tokenizer = loadTiktokenizer(path, regexp)

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
