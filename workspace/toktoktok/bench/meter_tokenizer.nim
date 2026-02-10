# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Compile with
# nim c -r --hints:off --warnings:off --verbosity:0 -d:danger -d:TTL_METER --outdir:build workspace/toktoktok/bench/meter_tokenizer.nim

import workspace/probes # probes MUST be imported before measured import

import std/os
import std/strutils

import workspace/toktoktok

const
  FixturesDir = currentSourcePath().parentDir() / ".." / "tests" / "fixtures" / "large"
  TokenizersDir = currentSourcePath().parentDir() / ".." / "tests" / "tokenizers"
  VernePath = FixturesDir / "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt"
  KimiK25Path = TokenizersDir / "kimik2.5.tiktoken"

proc readVerneText(maxChars: int): string =
  if not fileExists(VernePath):
    raise newException(ValueError, "Verne fixture not found: " & VernePath)
  let content = readFile(VernePath)
  result = content[0..<min(maxChars, content.len)]

proc measureEncode(tokenizer: BPETokenizer, text: string) =
  echo ""
  echo "=".repeat(60)
  echo "Metering tokenizer.encode on Verne text (", text.len, " chars)"
  echo "=".repeat(60)

  resetMetering()
  let tokens = tokenizer.encode(text)
  let numTokens = tokens.len

  reportMetering()

  echo ""
  echo "Result: ", numTokens, " tokens encoded"

when isMainModule:
  echo ""
  echo "=".repeat(70)
  echo "PERFORMANCE METERING: BPETokenizer.encode"
  echo "=".repeat(70)

  if not fileExists(KimiK25Path):
    echo "ERROR: KimiK2.5 tokenizer not found at: ", KimiK25Path
    quit(1)

  resetMetering() # loadTiktokenizer, calls one of the traced function

  echo "[INFO] Loading KimiK2.5 tokenizer..."
  let tokenizer = loadTiktokenizer(KimiK25Path, KimiK25Regexp)
  echo "[OK] Tokenizer loaded with ", tokenizer.tokenCount(), " tokens"

  echo "[INFO] Reading Verne text (limited to 10000 chars)..."
  let verneText = readVerneText(10000)
  echo "[OK] Read ", verneText.len, " chars"

  measureEncode(tokenizer, verneText)
