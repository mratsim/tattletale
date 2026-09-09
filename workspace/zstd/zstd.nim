# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# This file links the system zstd or statically compiles
# the vendored copy, and exposes its API in Nim. The system
# build is the default on macOS and Linux, the vendored
# submodule build is the default on Windows.
# We don't modify or patch the original library.
#
# Version updates only need:
# - moving the submodule pointer in workspace/zstd/vendor/zstd
# - re-running the round-trip test,
#   workspace/zstd/tests/t_zstd_roundtrip.nim
#
# The ZSTD_VERSION_* macros in the submodule zstd.h
# lock the version for every compiled consumer.
#
# The {.compile:} lines read plain upstream sources
# inside the submodule lib/ subtree. The v1.5.7 set
# carries no basename collisions, so the object names
# cannot collide in nimcache.

import workspace/zstd/c_abi

when defined(windows):
  const TTT_USE_SYSTEM_ZSTD_DEFAULT = false
else:
  const TTT_USE_SYSTEM_ZSTD_DEFAULT = true
const TTT_USE_SYSTEM_ZSTD* {.booldefine.} = TTT_USE_SYSTEM_ZSTD_DEFAULT

when not TTT_USE_SYSTEM_ZSTD:
  # Vendored static build, opt-in:
  # the submodule must be materialized (see the header, Windows intent)
  {.compile:"vendor/zstd/lib/common/debug.c".}
  {.compile:"vendor/zstd/lib/common/entropy_common.c".}
  {.compile:"vendor/zstd/lib/common/error_private.c".}
  {.compile:"vendor/zstd/lib/common/fse_decompress.c".}
  {.compile:"vendor/zstd/lib/common/pool.c".}
  {.compile:"vendor/zstd/lib/common/threading.c".}
  {.compile:"vendor/zstd/lib/common/xxhash.c".}
  {.compile:"vendor/zstd/lib/common/zstd_common.c".}
  {.compile:"vendor/zstd/lib/compress/fse_compress.c".}
  {.compile:"vendor/zstd/lib/compress/hist.c".}
  {.compile:"vendor/zstd/lib/compress/huf_compress.c".}
  {.compile:"vendor/zstd/lib/compress/zstd_compress.c".}
  {.compile:"vendor/zstd/lib/compress/zstd_compress_literals.c".}
  {.compile:"vendor/zstd/lib/compress/zstd_compress_sequences.c".}
  {.compile:"vendor/zstd/lib/compress/zstd_compress_superblock.c".}
  {.compile:"vendor/zstd/lib/compress/zstd_double_fast.c".}
  {.compile:"vendor/zstd/lib/compress/zstd_fast.c".}
  {.compile:"vendor/zstd/lib/compress/zstd_lazy.c".}
  {.compile:"vendor/zstd/lib/compress/zstd_ldm.c".}
  {.compile:"vendor/zstd/lib/compress/zstd_opt.c".}
  {.compile:"vendor/zstd/lib/compress/zstd_preSplit.c".}
  {.compile:"vendor/zstd/lib/compress/zstdmt_compress.c".}
  {.compile:"vendor/zstd/lib/decompress/huf_decompress.c".}
  {.compile:"vendor/zstd/lib/decompress/zstd_ddict.c".}
  {.compile:"vendor/zstd/lib/decompress/zstd_decompress.c".}
  {.compile:"vendor/zstd/lib/decompress/zstd_decompress_block.c".}
  {.compile:"vendor/zstd/lib/dictBuilder/cover.c".}
  {.compile:"vendor/zstd/lib/dictBuilder/divsufsort.c".}
  {.compile:"vendor/zstd/lib/dictBuilder/fastcover.c".}
  {.compile:"vendor/zstd/lib/dictBuilder/zdict.c".}

# ################################################## #
#                 zstd API                           #
# ################################################## #

# Overview of general API
# ══════════════════════════════════════════════════
# the public headers live inside the submodule:
#   workspace/zstd/vendor/zstd/lib/zstd.h
#   workspace/zstd/vendor/zstd/lib/zstd_errors.h
#
# This is a low-level binding, it stays close to the upstream API.
#
# Namely:
# - #define constants become enums and consts
# - buffers are openArray[byte] with the length explicit, no strlen
# - calls return size_t codes, errors are the negated form of the
#   error enum, tested with isError; the raw surface raises nothing
#
when not TTT_USE_SYSTEM_ZSTD:
  {.pragma: zstd, importc: "ZSTD_$1", cdecl.}
  {.pragma: zstdCctx, importc: "ZSTD_CCtx_$1", cdecl.}
else:
  import workspace/zstd/zstd_config
  {.passL: ZstdLinkFlags.}
  # qualified dynlib ref: generic procs expand pragmas at the
  # instantiation site, the bare symbol would not resolve there
  {.pragma: zstd, importc: "ZSTD_$1", cdecl,
    dynlib: zstd_config.ZstdDynlib.}
  {.pragma: zstdCctx, importc: "ZSTD_CCtx_$1", cdecl,
    dynlib: zstd_config.ZstdDynlib.}

# Constants
# ══════════════════════════════════════════════════

const ContentSizeUnknown* = not 0'u64
  ## ZSTD_CONTENTSIZE_UNKNOWN, getFrameContentSize result for a frame
  ## that does not carry its content size
const ContentSizeError* = not 1'u64
  ## ZSTD_CONTENTSIZE_ERROR, getFrameContentSize result for a frame
  ## with a malformed content size field

const DefaultCompressionLevel* = 3.cint
  ## ZSTD_CLEVEL_DEFAULT, the upstream default compression level

# Parameter ids of the advanced API, workspace/zstd/vendor/zstd/lib/zstd.h
# ZSTD_cParameter section. Only the parameters the container contract
# depends on are enumerated, pass the raw ids for the rest
type CompressionParameter* {.size: sizeof(cint).} = enum
  compressionLevel = 100
  windowLog        = 101
  strategy         = 107
  contentSizeFlag  = 200
  checksumFlag     = 201

# Error codes, workspace/zstd/vendor/zstd/lib/zstd_errors.h
# ══════════════════════════════════════════════════

type ErrorCode* {.size: sizeof(cint).} = enum
  noError                                 = 0
  GENERIC                                 = 1
  prefixUnknown                           = 10
  versionUnsupported                      = 12
  frameParameterUnsupported               = 14
  frameParameterWindowTooLarge            = 16
  corruptionDetected                      = 20
  checksumWrong                           = 22
  literalsHeaderWrong                     = 24
  dictionaryCorrupted                     = 30
  dictionaryWrong                         = 32
  dictionaryCreationFailed                = 34
  parameterUnsupported                    = 40
  parameterCombinationUnsupported          = 41
  parameterOutOfBound                     = 42
  tableLogTooLarge                        = 44
  maxSymbolValueTooLarge                  = 46
  maxSymbolValueTooSmall                  = 48
  cannotProduceUncompressedBlock          = 49
  stabilityConditionNotRespected          = 50
  stageWrong                              = 60
  initMissing                             = 62
  memoryAllocation                        = 64
  workSpaceTooSmall                       = 66
  dstSizeTooSmall                         = 70
  srcSizeWrong                            = 72
  dstBufferNull                           = 74
  noForwardProgressDestFull               = 80
  noForwardProgressInputEmpty             = 82
  # the following codes are NOT STABLE upstream,
  # they can be removed or changed in future versions
  frameIndexTooLarge                      = 100
  seekableIO                              = 102
  dstBufferWrong                          = 104
  srcBufferWrong                          = 105
  sequenceProducerFailed                  = 106
  externalSequencesInvalid                = 107

# Streaming buffer types
# ══════════════════════════════════════════════════
# man-style reference: workspace/zstd/vendor/zstd/lib/zstd.h, streaming section
# a decompression call returns 0 when a frame is fully decoded
# and flushed, a positive value when it needs more input or output

# Plain Nim objects mirroring the public ZSTD_inBuffer and ZSTD_outBuffer
# structs field for field, layout per workspace/zstd/vendor/zstd/lib/zstd.h.
# The binding only ever passes pointers to them, so the generated
# code needs no zstd.h include and no -I path.
type InBuffer* = object
  src*: pointer
  size*: csize_t
  pos*: csize_t

type OutBuffer* = object
  dst*: pointer
  size*: csize_t
  pos*: csize_t

type
  DCtx* = object
  CCtx* = object

# Version
# ══════════════════════════════════════════════════

proc versionNumber*(): cuint {.zstd.}
  ## Returns the library version as MAJOR*100*100 + MINOR*100 + RELEASE

proc versionString*(): cstring {.zstd.}
  ## Returns the library version as a string, for example "1.5.7"

# Errors
# ══════════════════════════════════════════════════

proc isError*(code: csize_t): cuint {.zstd.}
  ## 1 when the code from a zstd call is an error, 0 otherwise

proc getErrorName*(code: csize_t): cstring {.zstd.}
  ## Symbolic error name of a zstd error code, for example
  ## "DstSize_corruption"

proc getErrorCode*(code: csize_t): ErrorCode {.zstd.}
  ## Typed error code of a zstd error code

# One-shot compression and decompression
# ══════════════════════════════════════════════════

proc createCCtx*(): ptr CCtx {.zstd.}
  ## Creates a compression context, returns nil on allocation failure

proc freeCCtx*(cctx: ptr CCtx): csize_t {.zstd.}
  ## Frees a compression context, error code on failure

proc setParameter*(cctx: ptr CCtx,
                  param: CompressionParameter,
                  value: cint): csize_t {.zstdCctx.}
  ## Sets one advanced compression parameter
  ## man-style reference: workspace/zstd/vendor/zstd/lib/zstd.h, ZSTD_cParameter

proc compress2*[D: char|byte, S: char|byte](cctx: ptr CCtx,
               dst: openArray[D],
               src: openArray[S]): csize_t {.
    wrapOpenArrayLenType: csize_t, zstd.}
  ## Compresses src into dst in one shot using the cctx parameters
  ## man-style reference: workspace/zstd/vendor/zstd/lib/zstd.h, ZSTD_compress2

proc compressBound*(srcSize: csize_t): csize_t {.zstd.}
  ## Maximum compressed size in worst case single-call scenario
  ## man-style reference: workspace/zstd/vendor/zstd/lib/zstd.h#L232

proc maxCLevel*(): cint {.zstd.}
  ## Maximum compression level currently supported, 22 at v1.5.7

proc minCLevel*(): cint {.zstd.}
  ## Negative compression level lower bound, -131072 at v1.5.7

proc defaultCLevel*(): cint {.zstd.}
  ## Default compression level of the library, 3 at v1.5.7

proc compress*[D: char|byte, S: char|byte](dst: openArray[D],
              src: openArray[S],
              compressionLevel: cint): csize_t {.
    wrapOpenArrayLenType: csize_t, zstd.}
  ## Doc:
  ##    workspace/zstd/vendor/zstd/lib/zstd.h#L160
  ##
  ## Compresses src into dst in one shot, no streaming state.
  ## dstCapacity must be at least compressBound(srcSize)

proc decompress*[D: char|byte, S: char|byte](dst: openArray[D],
                src: openArray[S]): csize_t {.
    wrapOpenArrayLenType: csize_t, zstd.}
  ## Doc:
  ##    workspace/zstd/vendor/zstd/lib/zstd.h#L173
  ##
  ## Decompresses a complete frame into dst in one shot.
  ## dstCapacity must be an upper bound of the content size,
  ## the recorded fixture contract guarantees the exact size
  ## through the frame header

proc getFrameContentSize*[S: char|byte](src: openArray[S]): culonglong {.
    wrapOpenArrayLenType: csize_t, zstd.}
  ## Doc:
  ##    workspace/zstd/vendor/zstd/lib/zstd.h#L205
  ##
  ## Content size written into the frame header at compression time.
  ## ContentSizeUnknown when the producer skipped it,
  ## ContentSizeError on a malformed field

proc createDCtx*(): ptr DCtx {.zstd.}
  ## Creates a decompression context, returns nil on allocation failure
  ## man-style reference: workspace/zstd/vendor/zstd/lib/zstd.h, ZSTD_createDCtx

proc freeDCtx*(dctx: ptr DCtx): csize_t {.zstd.}
  ## Frees a decompression context, error code on failure

proc decompressStream*(dctx: ptr DCtx,
                      output: ptr OutBuffer,
                      input: ptr InBuffer): csize_t {.zstd.}
  ## Doc:
  ##    workspace/zstd/vendor/zstd/lib/zstd.h, ZSTD_decompressStream
  ##
  ## Streaming decompression: consumes input.pos up to input.size,
  ## writes output.pos up to output.size, both pos fields advance.
  ## Returns 0 when a frame is completely decoded and flushed,
  ## a positive value when more input or output room is needed,
