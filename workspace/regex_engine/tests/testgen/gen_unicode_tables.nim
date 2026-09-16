# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Generator for the regex engine's Unicode property tables.
##
## Queries the bundled PCRE2 engine (workspace/pcre2) over the whole
## Unicode range, emitting sorted inclusive codepoint ranges covering
## the engine's pattern classes plus one caseless-fold partner set
## for each contraction letter.
## - querying the reference engine keeps the tables at the exact classes PCRE2 resolves, immune to UCD snapshot drift
## - output overwrites workspace/regex_engine/src/regex_unicode_tables.nim in place
## - byte determinism applies, run the generator twice and commit only when the sha256 hashes of both outputs match run-to-run
##
## Run:
##   nim cpp -r --outdir:build/tests/gen_unicode_tables \
##     --nimcache:nimcache/tests/gen_unicode_tables workspace/regex_engine/tests/testgen/gen_unicode_tables.nim  # repo root

import std/[os, strutils, monotimes, times]

import workspace/pcre2

const
  OutPath = currentSourcePath().parentDir().parentDir().parentDir() /
    "src" / "regex_unicode_tables.nim"
  MaxCp = 0x10FFFF

  ClassNames = [
    "RangesL", r"[\p{L}]",
    "RangesLu", r"[\p{Lu}]",
    "RangesLt", r"[\p{Lt}]",
    "RangesLm", r"[\p{Lm}]",
    "RangesLo", r"[\p{Lo}]",
    "RangesLl", r"[\p{Ll}]",
    "RangesM", r"[\p{M}]",
    "RangesN", r"[\p{N}]",
    "RangesP", r"[\p{P}]",
    "RangesS", r"[\p{S}]",
    "RangesSpace", r"[\s]",
    "RangesHan", r"[\p{Script=Han}]",
  ]

  # Each (?i:...) contraction group in the served patterns carries
  # exactly these letters, one capture per fold class.
  # Two distinct ASCII letters never share a simple-fold class,
  # so no branch of the alternation shadows another.
  FoldLetters = "sdmtlvre"

proc utf8Encode(cp: uint32, buf: var array[4, char]): int =
  if cp < 0x80:
    buf[0] = char(cp)
    return 1
  elif cp < 0x800:
    buf[0] = char(0xC0 or (cp shr 6))
    buf[1] = char(0x80 or (cp and 0x3F))
    return 2
  elif cp < 0x10000:
    buf[0] = char(0xE0 or (cp shr 12))
    buf[1] = char(0x80 or ((cp shr 6) and 0x3F))
    buf[2] = char(0x80 or (cp and 0x3F))
    return 3
  else:
    buf[0] = char(0xF0 or (cp shr 18))
    buf[1] = char(0x80 or ((cp shr 12) and 0x3F))
    buf[2] = char(0x80 or ((cp shr 6) and 0x3F))
    buf[3] = char(0x80 or (cp and 0x3F))
    return 4

type
  CompiledProbe = object
    code: ptr Code
    matchData: ptr MatchData
    ovector: ptr UncheckedArray[int]

proc init(_: type CompiledProbe, pattern: string): CompiledProbe =
  var err: CompileError
  var errOffset: csize_t
  result.code = compile(pattern, flag(UTF, UCP), err, errOffset)
  if result.code == nil:
    raise newException(ValueError, "probe pattern failed to compile: " & pattern)
  result.matchData = match_data_create_from_pattern(result.code, nil)
  if result.matchData == nil:
    raise newException(ValueError, "probe match data allocation failed")
  result.ovector = get_ovector_pointer(result.matchData)

proc matches(p: CompiledProbe, subj: openArray[char]): bool =
  let rc = match(p.code, subj, 0, flag(MatchOption.NO_UTF_CHECK), p.matchData, nil)
  result = rc >= 0

proc closeProbe(p: var CompiledProbe) =
  code_free(p.code)
  match_data_free(p.matchData)
  p.code = nil
  p.matchData = nil

proc emitRanges(outp: var string, name: string, hits: seq[bool]) =
  var pairs: seq[tuple[lo, hi: uint32]] = @[]
  var i = 0
  while i <= MaxCp:
    if hits[i]:
      var j = i
      while j + 1 <= MaxCp and hits[j + 1]:
        inc j
      pairs.add (uint32(i), uint32(j))
      i = j + 1
    else:
      inc i
  outp.add "const " & name & "* = ["
  for k, pair in pairs:
    if k mod 4 == 0:
      outp.add "\n  "
    outp.add "(0x" & toHex(int(pair.lo), 5).toLowerAscii & "'u32, 0x" &
      toHex(int(pair.hi), 5).toLowerAscii & "'u32)"
    if k != pairs.len - 1:
      outp.add ", "
  outp.add "\n]\n\n"

proc main() =
  let t0 = getMonoTime()

  var probes: array[ClassNames.len div 2, CompiledProbe]
  for k in 0 ..< probes.len:
    probes[k] = CompiledProbe.init(ClassNames[2 * k + 1])

  var foldProbes: array[FoldLetters.len, CompiledProbe]
  for k in 0 ..< FoldLetters.len:
    let c = FoldLetters[k]
    foldProbes[k] = CompiledProbe.init("(?i:([" & c & "]))")

  var classHits: array[ClassNames.len div 2, seq[bool]]
  for k in 0 ..< classHits.len:
    classHits[k] = newSeq[bool](MaxCp + 1)
  var foldHits: array[FoldLetters.len, seq[bool]]
  for k in 0 ..< FoldLetters.len:
    foldHits[k] = newSeq[bool](MaxCp + 1)

  var buf: array[4, char]
  var nProbed = 0
  for cp in 0 .. MaxCp:
    if cp >= 0xD800 and cp <= 0xDFFF:
      continue
    let n = utf8Encode(uint32(cp), buf)
    for k in 0 ..< probes.len:
      classHits[k][cp] = probes[k].matches(buf.toOpenArray(0, n - 1))
    for k in 0 ..< foldProbes.len:
      foldHits[k][cp] = foldProbes[k].matches(buf.toOpenArray(0, n - 1))
    inc nProbed

  for k in 0 ..< probes.len:
    probes[k].closeProbe()
  for k in 0 ..< foldProbes.len:
    foldProbes[k].closeProbe()

  let probeMillis = (getMonoTime() - t0).inMilliseconds

  var outp =
    "## Generated Unicode property tables for the regex engine.\n" &
    "## Generated by workspace/regex_engine/tests/testgen/gen_unicode_tables.nim,\n" &
    "## which queries the bundled PCRE2 engine for every codepoint, so the ranges\n" &
    "## match the exact UCD PCRE2 resolves.\n" &
    "## - do not hand-edit\n" &
    "## - do not commit by hand without rerunning the generator, run twice, both outputs byte-identical\n" &
    "## - query scope covers codepoints 0x0000..0x10FFFF minus UTF-16 surrogates, " & $nProbed & " codepoints per pattern\n\n"

  for k in 0 ..< classHits.len:
    emitRanges(outp, ClassNames[2 * k], classHits[k])

  for k in 0 ..< FoldLetters.len:
    let c = FoldLetters[k]
    outp.add "const Fold" & toUpperAscii($c) & "* = ["
    var first = true
    for cp in 0 .. MaxCp:
      if foldHits[k][cp]:
        if not first:
          outp.add(", ")
        first = false
        outp.add("0x" & toHex(cp, 4).toLowerAscii & "'u32")
    outp.add "]\n"

  writeFile(OutPath, outp)
  echo "wrote ", OutPath, " (", outp.len, " bytes, ", nProbed,
    " codepoints probed, ", probeMillis, " ms probe wall time)"

when isMainModule:
  main()
