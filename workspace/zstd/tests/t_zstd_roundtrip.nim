# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Round-trip check for the raw zstd binding: a known json payload must survive
## compress plus decompress byte-exactly through the one-shot path and the streaming path.
##
## Run from the project root:
##   nim cpp -d:release --stackTrace:on --lineTrace:on --lineDir:on \
##     --debugger:native --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests/rel --nimcache:nimcache/tests/rel \
##     workspace/zstd/tests/t_zstd_roundtrip.nim
##
## The system path is the binding default on macOS and Linux
## (TTT_USE_SYSTEM_ZSTD=true), the vendored path is the default on
## Windows and the opt-in elsewhere (-d:TTT_USE_SYSTEM_ZSTD=false,
## the submodule materialized). Both stay exercised:
##   nim cpp -d:TTT_USE_SYSTEM_ZSTD=false ... (all other flags unchanged)

import std/strutils
import workspace/zstd

const KnownJson* = """{
  "model": "Qwen3.5-0.8B",
  "layer": "model.language_model.layers.3.self_attn.",
  "case": "prefill_batch2_seq8",
  "note": "磨刀不误砍柴工, the payload carries multibyte utf-8",
  "positions": [0, 1, 2, 3, 4, 5, 6, 7],
  "theta": 10000000,
  "values": [0.123456789, -3.141592653589793, 1e-30, 6.02214076e23],
  "nested": {"a": [true, false, null], "b": {"c": "d"}}
}
"""

proc main() =
  echo "zstd version: ", $versionString(),
       " (versionNumber ", versionNumber(), ")"

  # one-shot compress through a context, level 19
  let cctx = createCCtx()
  doAssert not cctx.isNil
  # ZSTD_CCtx_setParameter returns the applied value on success
  # (zstd_compress.c ZSTD_CCtxParams_setParameter), never 0 for a
  # nonzero value: success is isError-free, an error is negated
  doAssert isError(setParameter(cctx, CompressionParameter.compressionLevel, 19)) == 0
  doAssert isError(setParameter(cctx, contentSizeFlag, 1)) == 0
  doAssert isError(setParameter(cctx, checksumFlag, 1)) == 0
  let bound = int compressBound(csize_t KnownJson.len)
  var frame = newSeq[char](bound)
  let csize = compress2(
    cctx,
    toOpenArrayByte(frame, 0, bound - 1),
    toOpenArrayByte(KnownJson, 0, KnownJson.len - 1))
  doAssert isError(csize) == 0, "compress2 failed"
  frame.setLen(int csize)
  doAssert freeCCtx(cctx) == 0

  echo "payload ", KnownJson.len, " B -> frame ", frame.len, " B at level 19"
  doAssert frame.len > 0 and frame.len < KnownJson.len,
    "compression must shrink the payload"

  # content size recorded in the frame header
  let contentSize = getFrameContentSize(
    toOpenArrayByte(frame, 0, frame.len - 1))
  doAssert contentSize == culonglong KnownJson.len,
    "frame must carry the exact content size, got " & $contentSize

  # one-shot decompress round-trip, byte-exact
  let dstBound = int contentSize
  var back = newSeq[char](dstBound)
  let dsize = decompress(
    toOpenArrayByte(back, 0, dstBound - 1),
    toOpenArrayByte(frame, 0, frame.len - 1))
  doAssert isError(dsize) == 0, "decompress failed"
  doAssert (int dsize) == KnownJson.len,
    "one-shot length mismatch: " & $(int dsize) & " vs " & $KnownJson.len
  doAssert back == KnownJson

  # streaming round-trip, byte-exact, through ZSTD_decompressStream
  let dctx = createDCtx()
  doAssert not dctx.isNil
  var input = InBuffer(
    src: cast[ptr byte](frame[0].unsafeAddr),
    size: csize_t frame.len,
    pos: csize_t 0)
  var streamed = newSeq[char](0)
  while true:
    let offset = streamed.len
    streamed.setLen(offset + 65536)
    var output = OutBuffer(
      dst: streamed[offset].addr,
      size: csize_t 65536,
      pos: csize_t 0)
    let code = decompressStream(dctx, output.addr, input.addr)
    doAssert isError(code) == 0, "decompressStream failed"
    streamed.setLen(offset + int output.pos)
    if code == 0:
      break
  doAssert freeDCtx(dctx) == 0
  doAssert streamed == KnownJson, "streaming round-trip mismatch"

  # error codes surface through isError, the 1:1 port keeps no
  # exception path: a corrupt frame returns an error code
  var bodyCorrupted = frame
  bodyCorrupted[frame.len div 2] =
    char(uint8(bodyCorrupted[frame.len div 2]) xor 0x5A)
  let bad = decompress(
    toOpenArrayByte(back, 0, dstBound - 1),
    toOpenArrayByte(bodyCorrupted, 0, bodyCorrupted.len - 1))
  doAssert isError(bad) != 0, "a corrupt frame must return an error code"
  echo "corrupt frame returned error code: ", $getErrorName(bad)

  echo "zstd round-trip check passed, one-shot and streaming byte-exact"

when isMainModule:
  main()
