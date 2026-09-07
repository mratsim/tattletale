# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# This file implements a safetensors reader.
# `open(path)` parses the header of one safetensor file, memory-mapping
# the file read-only. That mapping belongs to the returned Safetensor
# until the last reference dies.
# While the mapping is open, zero-copy views and direct loads to GPU
# are available without materializing tensors in RAM.
#
# Assuming NVMe drives (and especially no HDD) actual loading
# might benefit from parallelism or multiple Cuda streams.
#
# ## Error model
#
# Given that:
# - we compile to C++ and exceptions there are "zero-cost"
# - that we build an application not a library hence we control all use-cases
# - that failing to load a model is unrecoverable
# we use exceptions.
#
# This might change once safetensors is deemed ready for public consumption

import
  std/memfiles,
  std/options,
  std/strformat,
  std/sugar,
  std/tables,
  pkg/jsony,
  pkg/stew/endians2

const MAX_HEADER_SIZE = 100_000_000 # From hf/safetensors. Avoids attack vector via large memory request

type
  ST_dtype* = enum
    ## Available dtypes. They MUST be in increasing alignment order
    ## Boolean type
    BOOL
    ## MXF4 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F4
    ## MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F6_E2M3
    ## MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F6_E3M2
    ## Unsigned byte
    U8
    ## Signed byte
    I8
    ## FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    F8_E5M2
    ## FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    F8_E4M3
    ## F8_E8M0 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F8_E8M0
    ## Signed integer (16-bit)
    I16
    ## Unsigned integer (16-bit)
    U16
    ## Half-precision floating point
    F16
    ## Brain floating point
    BF16
    ## Signed integer (32-bit)
    I32
    ## Unsigned integer (32-bit)
    U32
    ## Floating point (32-bit)
    F32
    ## Complex (32-bit parts)
    C64
    ## Floating point (64-bit)
    F64
    ## Signed integer (64-bit)
    I64
    ## Unsigned integer (64-bit)
    U64

  TensorInfo* = ref object
    ## A single tensor information.
    ## Endianness is assumed to be little endian
    ## Ordering is assumed to be 'C' (i.e. row-major, as opposed to Fortran col-major)
    ## The dataOffsets are relative to the start of the `data` section.
    ## They ignore the initial 8 bytes for headerSize + the actual header.
    dtype*: ST_dtype
    shape*: seq[int]
      # The reference impl uses usize, but AFAIK Cuda doesn't support 32-bit. This makes conversion to IntArrayRef easier.
    dataOffsets*: tuple[start, stopEx: int]
      # stop is exclusive

  SafetensorObj = object
    metadata: Option[OrderedTable[string, string]]
    tensors: OrderedTable[string, TensorInfo]
    dataSectionOffset: int ## Offset of the data section in the file. Set after parsing.
    memFile: MemFile ## The mapping owned by this reader.

  Safetensor* = ref SafetensorObj
    ## A safetensor file loaded into memory.
    ## Stores for each tensor
    ##   * tensor names
    ##   * the type of the data
    ##   * the shape of the data
    ##   * start and (exclusive) stop offset of the tensor data relative to the data offset

proc `=destroy`(st: var SafetensorObj) =
  ## Release the memory mapping this reader acquired in `open`.
  ## A nil reader, or one whose mapping was never acquired, releases
  ## nothing.
  if st.memFile.mem != nil:
    close(st.memFile)

proc skipHook(T: typedesc[Safetensor], key: string): bool =
  key == "dataSectionOffset" or key == "memFile"

const DtypeSize: array[ST_dtype, int] = [
  ## Size in bytes.
  ## Unsure why the reference library bothers with bits
  ## when packing is done at another level
  ## and all what safetensor stores is a large type.
  BOOL: 1, F4: 1, F6_E2M3: 1, F6_E3M2: 1,
  U8: 1, I8: 1, F8_E5M2: 1, F8_E4M3: 1, F8_E8M0: 1,
  I16: 2, U16: 2, F16: 2, BF16: 2,
  I32: 4, U32: 4, F32: 4,
  C64: 8, F64: 8, I64: 8, U64: 8
]

proc parseHook(src: string, pos: var int, value: var Safetensor) =
  # Who got the bright idea to put heterogenous data at the same level?
  var safetensor = Safetensor()

  eatChar(src, pos, '{')
  while pos < src.len:
    eatSpace(src, pos)
    if pos < src.len and src[pos] == '}':
      value = safetensor
      break

    var key: string
    parseHook(src, pos, key)
    eatChar(src, pos, ':')
    if key == "__metadata__":
      parseHook(src, pos, safetensor.metadata)
    else:
      var tensorInfo = new TensorInfo
      parseHook(src, pos, tensorInfo)
      safetensor.tensors[key] = tensorInfo

    if pos < src.len and src[pos] == ',':
      inc pos
    else:
      value = safetensor
      break
  eatChar(src, pos, '}')

template `+%`(p: pointer, offset: SomeInteger): pointer =
  ## Pointer arithmetic | increment
  cast[pointer](cast[uint](p) + uint(offset))

func product(a: openArray[SomeInteger]): SomeInteger {.inline.} =
  if unlikely(a.len == 0):
    return 0
  result = 1
  for value in items(a):
    result *= value

func validate_offsets(st: Safetensor, dataSectionSize: int) =
  ## Sanity checks for data offsets
  ## Assumes the tensors are sorted by ascending offsets
  ## Checks:
  ## - Soundness
  ## - Contiguity (not specified but enforced by reference impl)
  ## - No overlap
  ## - No incomplete reads or read past the file
  var cur = 0
  for (name, info) in st.tensors.pairs():
    let (start, stopEx) = info.dataOffsets
    if start != cur or start >= stopEx:
      raise newException(RangeDefect, &"safetensors: Tensor '{name}' has invalid offsets")

    let numel = info.shape.product()
    let size = numel * DtypeSize[info.dtype]
    if stopEx - start != size:
      raise newException(RangeDefect, &"safetensors: Tensor '{name}' has invalid offsets or shape")

    cur = stopEx

  if cur != dataSectionSize:
    raise newException(RangeDefect, &"safetensors: Tensor offsets and data section size mismatch")

proc parseMapped(memFile: MemFile): Safetensor =
  ## Parse and validate the header of a memory-mapped safetensors file.
  ## Raises RangeDefect on any header or data-offset defect.
  let parsedHeaderSize = uint64.fromBytesLE(toOpenArray(cast[ptr UncheckedArray[byte]](memFile.mem), 0, sizeof(uint64)-1))
  let headerSize = int(parsedHeaderSize)

  if headerSize > MAX_HEADER_SIZE:
    raise newException(RangeDefect, "safetensors: Safetensor header too large")
  if sizeof(uint64) + headerSize > memFile.size:
    raise newException(RangeDefect, "safetensors: Safetensor header has an invalid length")

  # Jsony requires copying the header, no zero-copy :/
  # https://github.com/treeform/jsony/issues/102
  var rawHeader = newString(headerSize)
  copyMem(rawHeader[0].addr, memFile.mem +% sizeof(uint64), headerSize)
  result = fromJson(rawHeader, Safetensor)

  # Sort tensors by offsets
  result.tensors.sort((lhs, rhs) => system.cmp(lhs[1].dataOffsets.start, rhs[1].dataOffsets.start))

  result.dataSectionOffset = sizeof(uint64) + headerSize

  # Validate that offsets are within the file with no gap or overlap
  result.validate_offsets(memFile.size - result.dataSectionOffset)

proc open*(_: typedesc[Safetensor], path: string): Safetensor =
  ## Read the safetensor file at `path`, memory-mapping it read-only.
  ##
  ## The header is parsed and validated: tensor names, dtypes, shapes
  ## and data offsets (contiguous, non-overlapping, within the file).
  ## Tensor bytes stay in the mapping until read through `getMmapView`
  ## or the libtorch bridge.
  ##
  ## Failures propagate as the exception raised: `OSError` from the memory-map
  ## open for an absent or unreadable path, a parser defect for a corrupt header,
  ## per the error model at the top of this file.
  var memFile = memFiles.open(path, mode = fmRead)
  result = memFile.parseMapped()
  result.memFile = memFile


# Individual tensor API (WIP)
# ---------------------------------------------------------
#
# The API here might change with the following consideration
# - How to allow fast loading (async Streams, parallel workers, direct to GPU, ...)
#
# Views returned here borrow from the mapping owned by `st`.
# The borrow is a documented contract, not a compiler-checked one.
# `MemFile` predates `lent` and view openarrays, so no borrow-checking
# exists.
# https://nim-lang.org/docs/manual.html#var-return-type-future-directions

proc getMmapView*(st: Safetensor, tensorName: string): MemSlice {.inline.} =
  ## Returns a zero-copy `MemSlice` view of the tensor data of `tensorName`.
  ##
  ## Preconditions: `st` is a reader returned by `open` and still alive,
  ## `tensorName` is a key of `st.tensors`.
  ## Postconditions: the view is valid while the reader is alive,
  ## dangling after its destructor ran.
  let info = st.tensors[tensorName]
  let (start, stopEx) = info.dataOffsets
  MemSlice(
    data: st.memFile.mem +% st.dataSectionOffset +% start,
    size: stopEx - start
  )
