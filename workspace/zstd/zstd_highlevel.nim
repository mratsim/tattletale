# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## The fixture-frame contract over the raw zstd C API.
##
## Contract (version 1):
## - A frame carries its content size and a checksum in its header,
##   the reader asserts the content size instead of guessing buffers.
## - The compression level is 19, matching the python producers.
## - A corrupt frame raises ZstdError.

import
  std/streams,
  workspace/zstd

type ZstdError* = object of ValueError
  ## Raised on a corrupt frame, on a missing or unknown content size
  ## and on a failed zstd call, the message carries the error name

const zstdFrameLevel: cint = 19
  ## The recorded fixture compression level, matching the python
  ## producers (compression.zstd level 19, content size + checksum)

proc raiseZstdError(code: csize_t, context: string) {.noreturn, noinline.} =
  ## Every error from a zstd call raises through here
  raise newException(
    ZstdError, "zstd: " & context & " (" & $getErrorName(code) & ")")

proc raiseZstdError(msg: string) {.noreturn, noinline.} =
  ## Raises for contract violations caught before a zstd call,
  ## for example a frame with no content size
  raise newException(ZstdError, "zstd: " & msg)

proc raiseZstdIO(msg: string) {.noreturn, noinline.} =
  ## The stream handlers convert the contract errors to IOError,
  ## the std/streams handler fields are typed to IO errors
  raise newException(IOError, "zstd: " & msg)

proc raiseZstdIO(code: csize_t, context: string) {.noreturn, noinline.} =
  raise newException(
    IOError, "zstd: " & context & " (" & $getErrorName(code) & ")")

proc assertContentSize[S: char|byte](src: openArray[S]): int =
  ## Check that the frame carries its content size: the fixture contract
  ## requires it, one check shared by the one-shot decompress flavors
  if src.len == 0:
    raiseZstdError("decompress: empty input is not a zstd frame")
  let contentSize = getFrameContentSize(src)
  if contentSize == ContentSizeUnknown:
    raiseZstdError(
      "decompress: frame carries no content size, " &
      "the fixture contract requires content-size frames")
  if contentSize == ContentSizeError:
    raiseZstdError("decompress: malformed frame content size field")
  if contentSize > uint64 int.high:
    raiseZstdError(
      "decompress: content size " & $contentSize &
      " beyond the addressable range")
  result = int contentSize

func zstdCompress*[S: char|byte](
    src: openArray[S],
    Dst: typedesc[string|seq[byte]] = seq[byte]): Dst =
  ## Compresses src into a fixture frame at the recorded level,
  ## with content size and checksum parameters set
  if src.len == 0:
    raiseZstdError("compress: empty input is not a zstd frame producer use case")
  let cctx = createCCtx()
  if cctx.isNil:
    raiseZstdError("compress: context allocation failed")
  var opFailure: ref ZstdError = nil
  try:
    var code = setParameter(
      cctx, CompressionParameter.compressionLevel, zstdFrameLevel)
    if isError(code) != 0:
      raiseZstdError(code, "setParameter compressionLevel failed")
    code = setParameter(cctx, contentSizeFlag, 1)
    if isError(code) != 0:
      raiseZstdError(code, "setParameter contentSizeFlag failed")
    code = setParameter(cctx, checksumFlag, 1)
    if isError(code) != 0:
      raiseZstdError(code, "setParameter checksumFlag failed")
    let bound = int compressBound(csize_t src.len)
    when Dst is string:
      result = newString(bound)
    else:
      result = newSeq[byte](bound)
    code = compress2(
      cctx,
      result,
      src)
    if isError(code) != 0:
      raiseZstdError(code, "compress failed")
    result.setLen(int code)
  except ZstdError as e:
    opFailure = e
  let freeCode = freeCCtx(cctx)
  # the real failure re-raises first, a free failure raises otherwise
  if not opFailure.isNil:
    raise opFailure
  if isError(freeCode) != 0:
    raiseZstdError(freeCode, "freeCCtx failed")

func zstdDecompress*[S: char|byte](
    src: openArray[S],
    Dst: typedesc[string|seq[byte]] = seq[byte]): Dst {.discardable.} =
  ## Decompresses a complete fixture frame in one shot, the frame
  ## must carry its content size
  let contentSize = assertContentSize(src)
  when Dst is string:
    result = newString(contentSize)
  else:
    result = newSeq[byte](contentSize)
  let code = result.decompress(src)
  if isError(code) != 0:
    raiseZstdError(code, "decompress failed")

type ZstdFrameStream = ref object of StreamObj
  ## The decode pipeline behind a fixture-frame stream: compressed
  ## bytes are pulled from the source stream in chunks and decoded
  ## through the dctx, neither side holds the whole payload
  dctx: ptr DCtx
  source: Stream
  chunk: seq[byte]
  input: InBuffer
  primed: bool
  frameDone: bool
  freed: bool

const ChunkSize = 65536

proc fsFree(fs: ZstdFrameStream) =
  if not fs.freed:
    fs.freed = true
    let freeCode = freeDCtx(fs.dctx)
    fs.dctx = nil
    if isError(freeCode) != 0:
      raiseZstdIO(freeCode, "streaming freeDCtx failed")

proc fsReadData(s: Stream, buffer: pointer, bufLen: int): int =
  ## Pulls compressed bytes from the source stream and decodes
  ## them into the caller's buffer, the frame contract rules
  ## apply on every read. A short return means the frame is
  ## fully decoded. A truncated frame raises IOError instead of
  ## a short read.
  let fs = ZstdFrameStream(s)
  if bufLen <= 0 or fs.frameDone:
    return 0
  if not fs.primed:
    # the first chunk carries the frame header: validate the
    # content size contract before any payload flows
    fs.chunk = newSeq[byte](ChunkSize)
    let got = fs.source.readData(fs.chunk[0].addr, ChunkSize)
    if got == 0:
      raiseZstdIO("streaming decompress: the source holds no bytes")
    fs.chunk.setLen(got)
    let contentSize = getFrameContentSize(fs.chunk)
    if contentSize == ContentSizeUnknown:
      raiseZstdIO(
        "decompress: frame carries no content size, " &
        "the fixture contract requires content-size frames")
    if contentSize == ContentSizeError:
      raiseZstdIO("decompress: malformed frame content size field")
    if contentSize > uint64 int.high:
      raiseZstdIO(
        "decompress: content size " & $contentSize &
        " beyond the addressable range")
    fs.input = InBuffer(
      src: fs.chunk[0].addr,
      size: csize_t got,
      pos: csize_t 0)
    fs.primed = true
  var written = 0
  while true:
    if fs.input.pos < fs.input.size:
      var output = OutBuffer(
        dst: cast[pointer](cast[ByteAddress](buffer) + written),
        size: csize_t(bufLen - written),
        pos: csize_t 0)
      let code = decompressStream(fs.dctx, output.addr, fs.input.addr)
      if isError(code) != 0:
        fs.fsFree()
        raiseZstdIO(code, "streaming decompress failed")
      written += int output.pos
      if code == 0:
        # the frame is fully decoded and flushed
        fs.frameDone = true
        fs.fsFree()
        return written
      if written == bufLen:
        return written
      if fs.input.pos < fs.input.size:
        continue
    let got = fs.source.readData(fs.chunk[0].addr, ChunkSize)
    if got == 0:
      # the source is exhausted with the frame still incomplete:
      # a truncated frame is a corrupt frame here
      fs.fsFree()
      raiseZstdIO(
        "streaming decompress: truncated frame, " &
        $fs.input.pos & " of " & $fs.input.size & " bytes consumed")
    fs.input = InBuffer(
      src: fs.chunk[0].addr,
      size: csize_t got,
      pos: csize_t 0)

proc fsAtEnd(s: Stream): bool =
  ZstdFrameStream(s).frameDone

proc fsClose(s: Stream) =
  fsFree(ZstdFrameStream(s))

proc zstdDecompressStream*(source: Stream): Stream =
  ## A std/streams decode pipeline over one fixture frame:
  ##
  ##   source stream -> chunk reads -> dctx decode -> caller pull buffer
  ##
  ## Large frames stay bounded by the consumer's pull size on both sides.
  ## The frame contract applies: the content size must be present and
  ## valid, a corrupt or truncated frame raises IOError carrying the zstd
  ## message on the read that hits it. The source stream stays owned by
  ## the caller, close the frame stream after reading it to the end.
  let dctx = createDCtx()
  if dctx.isNil:
    raiseZstdError("decompress: context allocation failed")
  result = ZstdFrameStream(
    dctx: dctx,
    source: source,
    chunk: newSeq[byte](0))
  result.readDataImpl = fsReadData
  result.atEndImpl = fsAtEnd
  result.closeImpl = fsClose

