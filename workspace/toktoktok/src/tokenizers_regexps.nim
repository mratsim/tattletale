# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# https://github.com/openai/tiktoken/blob/0.12.0/tiktoken_ext/openai_public.py

import strutils

type TokRegexp* = object
  regexp*: string

# Equivalent but slower than R50k
# const Gpt2Regexp* = TokRegexp(pattern: r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s0-9a-zA-Z]+|\r?\n|\s+(?!\S)|\s+")

const
  R50kRegexp* = TokRegexp(regexp: r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""")
  Gpt2Regexp* = R50kRegexp
  P50kRegexp* = R50kRegexp
  Cl100kRegexp* = TokRegexp(regexp: r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""")
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
  KimiK25Regexp* = TokRegexp(regexp:
      # From https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/tokenization_kimi.py
      # Note: Using \p{Script=Han} instead of \p{Han} to exclude punctuation like 。 (U+3002)
      # that have Han in ScriptExtensions but are not actually Han script characters.
      # Tiktoken uses Rust regex which seems to mismatch PCRE2
      # TODO:
      #   Given that in many cases the regex is supplied by the model
      #   we might want to add a preprocessing phase
      #   that would translate [\p{Han}]+ into [\p{Script=Han}]
      # see commit bc8d9df32db81a9d08c4458bce5df2b1098a8f68
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
  MoonshotPatStrRegexp* = TokRegexp(regexp:
      # From the Moonlight checkpoint tokenization_moonshot.py pat_str
      # (huggingface.co/moonshotai/Moonlight-16B-A3B). The Kimi-Linear
      # checkpoint pat_str is byte-identical, so one spelling serves
      # the whole Moonshot pat_str family. Two Rust-regex
      # constructs have no PCRE2 spelling, translated here and verified
      # token-identical against the tiktoken Rust engine on ASCII and CJK
      # corpora: letters, contractions, digits, Han punctuation adjacency,
      # mixed-script runs:
      # - the class intersections
      #   [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]] and [\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]
      #   become
      #   lookahead-guarded alternations. Han script characters carry
      #   general category Lo, so guarding \p{Lo} with a negative
      #   lookahead and keeping the other categories in plain classes
      #   reproduces the intersection.
      # - \p{Han} reads as \p{Script=Han}: PCRE2 resolves the plain
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
