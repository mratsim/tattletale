# Toktoktok Python Extension Module
#
# Exports the toktoktok pipeline machine to Python using nimpy.
# Compile command:
#   nim c --app:lib --out:pytoktoktok.so pytoktoktok.nim
#
# Consumption protocol:
#   a pipeline machine is built from a checkpoint, the caller resets
# the text (or drives a chunked stream with an explicit finish), then
# pulls token ids in bounded batches until the stream drains
# (bounded pulls carry the machine state across calls). The encode /
# encode_ordinary one-shot conveniences are thin loops over exactly
# that protocol (reset + bounded pulls), for callers that want the whole
# id stream in one call.
#
# Loading and decoding use the toktoktok deserializer surface
# (src/deserializers.nim, rank files and HF json checkpoints build the byte-level codec, the id decode included). The composed encoder
# machine feeds special-scan decisions into region-restricted
# pre-tokenization which streams BPE ids.

import nimpy
import std/tables
import std/strutils

import workspace/toktoktok/src/scan
import workspace/regex_engine
import workspace/toktoktok/src/pipeline
import workspace/toktoktok/src/serialization
import workspace/toktoktok/src/deserializers
import pull_chunks

type
  PipelineRef* = ref object of PyNimObjectExperimental
    ## Opaque wrapper around one toktoktok pipeline machine, the composed
    ## encoder plus the codec it loaded from (decode surface and special-token table source):
    ## - `pipe`, the machine carrying the load flavor's special-token setting.
    ## - `plain`, the specials-free machine built lazily for encode_ordinary.
    pipe*: TokPipeline
    plain*: TokPipeline
    fam*: scan.Family
    codec*: TiktokenCodec
    cache*: PreTokRegexCache

proc normRegexEsc(s: string): string =
  ## Normalizes literal control bytes to their escaped spelling:
  ## - the staged checkpoint jsons carry real `\r` `\n` `\t` bytes inside regex
  ##   character classes where the Nim pattern constants carry the backslash-escaped
  ##   two-character sequences, regex-equivalent, so matching runs on the normalized form.
  result = s
  for pair in [("\r", "\\r"), ("\n", "\\n"), ("\t", "\\t")]:
    result = result.replace(pair[0], pair[1])

proc familyFromRegexp(regexp: string): scan.Family =
  ## Resolves the pre-tokenization family from the flat alternation
  ## the HF-to-tiktoken conversion derives from a checkpoint json pre_tokenizer.
  ##
  ## Matching:
  ## - single-regex families match their chain pattern (normalized)
  ## - the step-3.5-flash chain matches its three steps joined in chain order
  ## - the ling checkpoint json is the one alias row
  ##
  ## That alias row spells the same case-insensitive suffix alternation,
  ## `(?i:[sdmt]|ll|ve|re)` on the json side and `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
  ## on the nim side.
  ## Loud failure on unknown shapes.
  const Ling3JsonSpelling =
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}|""" &
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
  let arg = normRegexEsc(regexp)
  let candidates: seq[tuple[pat: string, fam: Family]] = @[
    (R50kPat, famR50k), (Cl100kPat, famCl100k), (O200kPat, famO200k),
    (KimiK25Pat, famKimiK25), (QwenPat, famQwen), (Qwen35Pat, famQwen35),
    (Glm47Pat, famGlm47), (Ling3Pat, famLing3), (Ling3JsonSpelling, famLing3),
    (ExaoneStepPat, famExaone), (MoonlightPat, famMoonlight),
    (Step35DigitsPat & "|" & Step35CjkPat & "|" & Step35MainPat, famStep35Flash)]
  for c in candidates:
    if arg == c.pat:
      return c.fam
  raise newException(ValueError, "unresolved pre-tokenization family")

proc mkPipeline(codec: TiktokenCodec, fam: scan.Family,
    withSpecials: bool, cache: var PreTokRegexCache): TokPipeline =
  new result
  if withSpecials:
    # Special dictionary in tie-priority order extracted from the live codec
    # table (the same-start tie rule keeps the first pattern the table iteration yields), never assumed.
    var pats: seq[string]
    var ids: seq[int]
    for token, id in codec.specials:
      pats.add token
      ids.add id
    result = TokPipeline.init(cache, codec.ranks, pats, ids, fam)
  else:
    result = TokPipeline.init(cache, codec.ranks, @[], @[], fam)

proc hfFamily(path: string): scan.Family =
  let hf = deserializeHfTokenizer(readFile(path))
  familyFromRegexp(convertHfToTiktoken(hf).pattern.regexp)

proc load_tokenizer_hf*(path: string): PipelineRef {.exportpy.} =
  ## HF tokenizer json, special tokens active (encode splits every added token out like the converted tiktoken special table does).
  let codec = loadHfCodec(path)
  new result
  result.codec = codec
  result.fam = hfFamily(path)
  result.pipe = mkPipeline(codec, result.fam, true, result.cache)

proc load_tokenizer_hf_ordinary*(path: string): PipelineRef {.exportpy.} =
  ## HF tokenizer json, no special tokens (the ordinary encode semantics).
  let codec = loadHfCodec(path)
  new result
  result.codec = codec
  result.fam = hfFamily(path)
  result.pipe = mkPipeline(codec, result.fam, false, result.cache)

proc tiktokenFamily(pattern: string): scan.Family =
  ## Rank-file pattern name -> family
  ## (the pattern string names the checkpoint's split pattern, the chain lives in scan.nim).
  case pattern
  of "r50k": famR50k
  of "p50k": famP50k
  of "cl100k": famCl100k
  of "o200k": famO200k
  of "kimik2.5": famKimiK25
  else:
    raise newException(ValueError, "Unknown pattern: " & pattern)

proc load_tokenizer_tiktoken*(path: string, pattern: string): PipelineRef {.exportpy.} =
  ## Tiktoken base64 rank file, special tokens active, loader contract:
  ## - the specialTokens table is empty for these files, the special
  ##   strings of a checkpoint live beside the rank file.
  ## - the pipeline degenerates to ordinary encoding, the predecessor codec behavior.
  let codec = loadTiktokenCodec(path)
  new result
  result.codec = codec
  result.fam = tiktokenFamily(pattern)
  result.pipe = mkPipeline(codec, result.fam, true, result.cache)

proc load_tokenizer_tiktoken_ordinary*(path: string,
    pattern: string): PipelineRef {.exportpy.} =
  ## tiktoken base64 rank file, no special tokens
  ## (the ordinary encode semantics).
  let codec = loadTiktokenCodec(path)
  new result
  result.codec = codec
  result.fam = tiktokenFamily(pattern)
  result.pipe = mkPipeline(codec, result.fam, false, result.cache)

proc ordinaryPipeline(self: PipelineRef): TokPipeline =
  ## Returns the specials-free machine for encode_ordinary, built lazily:
  ## the rank-table engine build dominates load time, so an ordinary-only
  ## consumer must not pay for it at load.
  if self.plain.isNil:
    self.plain = TokPipeline.init(self.cache, self.codec.ranks, @[], @[], self.fam)
  self.plain

proc drainAll(p: TokPipeline): seq[int] =
  ## Whole-stream drain as a bounded windowed pull, one-shot
  ## conveniences follow the same pull discipline as the streaming
  ## surface (in-flight decision carry and drain state, never bypassed).
  var pc = p.pullChunks(PullCap)
  for window in pc:
    result.add window

proc reset_text*(self: PipelineRef, text: string) {.exportpy.} =
  ## Whole-input mode, the text moves into the pipeline (zero copy)
  ## and the stream is complete from the start.
  self.pipe.resetText(text)

proc begin_stream*(self: PipelineRef) {.exportpy.} =
  ## Chunked mode start:
  ##   feed chunks, then finish_stream.
  self.pipe.beginStream()

proc feed*(self: PipelineRef, chunk: string) {.exportpy.} =
  ## Chunked mode, appends one chunk. Every id pulled so far must be
  ## drained before the next feed (no decision in flight).
  self.pipe.feed(chunk)

proc finish_stream*(self: PipelineRef) {.exportpy.} =
  ## Chunked mode:
  ##   marks the stream complete (idempotent).
  self.pipe.finishStream()

proc pull*(self: PipelineRef, cap: int): seq[int] {.exportpy.} =
  ## One bounded consumption step on the load flavor's pipeline.
  pullBatch(self.pipe, cap)

proc drained*(self: PipelineRef): bool {.exportpy.} =
  ## True once the whole stream is encoded and every id was pulled.
  self.pipe.drained()

proc encode*(self: PipelineRef, text: string): seq[int] {.exportpy.} =
  ## One-shot encode over the consumption protocol (special tokens per the load flavor):
  ## reset + bounded pulls, the streaming surface in a loop.
  self.pipe.resetText(text)
  drainAll(self.pipe)

proc encode_ordinary*(self: PipelineRef, text: string): seq[int] {.exportpy.} =
  ## One-shot encode with no special tokens, over the same protocol
  ## on the specials-free machine.
  let p = self.ordinaryPipeline()
  p.resetText(text)
  drainAll(p)

proc decode*(self: PipelineRef, ids: seq[int]): string {.exportpy.} =
  ## Byte-level decode from the loader codec (the vocabularies are byte rank tables, no remap stage participates).
  decodeToString(self.codec, ids)

proc vocab_size*(self: PipelineRef): int {.exportpy.} =
  ## Vocabulary size (mergeable ranks plus special tokens).
  self.codec.tokenCount

proc `$`*(self: PipelineRef): string {.exportpy.} =
  "Pipeline(vocabSize=" & $self.codec.tokenCount & ")"

setModuleDocString("Toktoktok tokenizer - machine-consumption pipeline over byte-level BPE (nimpy binding)")
setDocStringForType(PipelineRef, "Opaque wrapper around a loaded toktoktok pipeline machine")
