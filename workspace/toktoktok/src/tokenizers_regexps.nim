# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# https://github.com/openai/tiktoken/blob/0.12.0/tiktoken_ext/openai_public.py

## Served family patterns as flat alternation strings,
## with the split-triple shape notes per pattern
## (see the per-const docs).

import strutils

## Flat pattern-string wrapper, the checkpoint's split pattern as a plain string
## (the family chain resolution lives in scan.nim).
type TokRegexp* = object
  regexp*: string

# Equivalent but slower than R50k
# const Gpt2Regexp* = TokRegexp(pattern: r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s0-9a-zA-Z]+|\r?\n|\s+(?!\S)|\s+")

const
  ## Verbatim tiktoken r50k_base pattern, alternatives in source order:
  ## the trailing whitespace alternative stays `\s`, not the `\s+` the rust-gems
  ## documentation shows (the pattern-split pat3 carries it verbatim, scan.nim pattern-split section).
  R50kRegexp* = TokRegexp(regexp: r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")
  Gpt2Regexp* = R50kRegexp
  P50kRegexp* = R50kRegexp
  ## Verbatim tiktoken cl100k_base pattern, alternatives in source order, the trailing
  ## whitespace alternative `\s` verbatim (split pat3 carries it verbatim, scan.nim pattern-split section).
  Cl100kRegexp* = TokRegexp(regexp: r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""")
  ## Verbatim tiktoken o200k_base pattern, alternatives in source order:
  ## - the lowercase-led letter alternative stays before the uppercase-led one,
  ##   the upstream tiktoken order, priority order is segmentation-visible
  ##   (the uppercase-led alternative matches farther, `_文D`).
  ## - the trailing whitespace alternative is `\s+` as documented (split pat3 carries it verbatim, scan.nim pattern-split section).
  O200kRegexp* = TokRegexp(regexp:
      # This works too
      # "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
      [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
      ].join("|")
    )
  ## Verbatim Kimi-K2.5 pat_str, alternatives in source order, the trailing whitespace
  ## alternative `\s+` as documented (split pat3 carries it verbatim, scan.nim pattern-split section).
  KimiK25Regexp* = TokRegexp(regexp:
      # From https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/tokenization_kimi.py
      # Note:
      #   Using \p{Script=Han} instead of \p{Han} to exclude punctuation like 。 (U+3002)
      # that have Han in ScriptExtensions but are not actually Han script characters.
      # Tiktoken uses Rust regex which seems to mismatch PCRE2
      # TODO:
      #   pending a checkpoint that ships a [\p{Han}]+ pattern,
      #   add a preprocessing phase translating it to [\p{Script=Han}]
      [
            r"""[\p{Script=Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
      ].join("|")
    )
  ## Verbatim Moonlight pat_str under the Rust-regex translation above,
  ## alternatives in source order, the trailing whitespace alternative
  ## `\s+` as documented (split pat3 carries it verbatim).
  MoonlightRegexp* = TokRegexp(regexp:
      # From the Moonlight checkpoint tokenization_moonshot.py pat_str
      # (huggingface.co/moonshotai/Moonlight-16B-A3B). Two Rust-regex constructs have no
      # PCRE2 spelling, translated here and verified token-identical against the tiktoken
      # Rust engine on ASCII and CJK corpora, letters, contractions, digits, Han
      # punctuation adjacency, mixed-script runs:
      # - the class intersections
      #   [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]] and [\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]
      #   become lookahead-guarded alternations. Han script characters carry general
      #   category Lo, so guarding \p{Lo} with a negative lookahead and keeping the other
      #   categories in plain classes reproduces the intersection.
      # - \p{Han} reads as \p{Script=Han}:
      #   PCRE2 resolves the plain
      #   name through Script_Extensions and would also take the Han
      #   punctuation U+3002 that the Rust engine refuses
      #   (KimiK25 precedent, same file).
      [
            r"""[\p{Script=Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?(?:(?!\p{Script=Han})\p{Lo}|[\p{Lt}\p{Lu}\p{Lm}\p{M}])*(?:(?!\p{Script=Han})\p{Lo}|[\p{Ll}\p{Lm}\p{M}])+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?(?:(?!\p{Script=Han})\p{Lo}|[\p{Lt}\p{Lu}\p{Lm}\p{M}])+(?:(?!\p{Script=Han})\p{Lo}|[\p{Ll}\p{Lm}\p{M}])*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
      ].join("|")
    )
