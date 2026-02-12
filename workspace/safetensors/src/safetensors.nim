# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# This file implements a safetensor loader.
# It returns an iterator tuple[key: string, loc: MemSlice] with loc being memory-mapped (read-only)
# to the original file.
# This allows zero-copy access to the data if needed and it allows direct loading to GPU
# without materializing the tensors in RAM.
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
  Dtype* = enum
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

  TensorInfo* = object
    ## A single tensor information.
    ## Endianness is assumed to be little endian
    ## Ordering is assumed to be 'C' (i.e. row-major, as opposed to Fortran col-major)
    ## The dataOffsets are relative to the start of the `data` section.
    ## They ignore the initial 8 bytes for headerSize + the actual header.
    dtype*: Dtype
    shape*: seq[int] # The reference impl uses usize, but AFAIK Cuda doesn't support 32-bit. This makes conversion to IntArrayRef easier.
    dataOffsets*: tuple[start, stopEx: int] # stop is exclusive

  Safetensor* = object
    metadata*: Option[OrderedTable[string, string]]
    tensors*: OrderedTable[string, TensorInfo]
    dataSectionOffset*: int ## Offset of the data section in the file. Set after parsing.

proc skipHook*(T: typedesc[Safetensor], key: string): bool =
  key == "dataSectionOffset"

const DtypeSize: array[Dtype, int] = [
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
  var safetensor = default(Safetensor)

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
      var tensorInfo: TensorInfo
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

proc load*(memFile: MemFile): Safetensor =
  ## Load a safetensor file and return
  ## - for each tensor
  ##   * tensor names
  ##   * the type of the data
  ##   * the shape of the data
  ##   * start and (exclusive) stop offset of the tensor data relative to the data offset
  ## - the dataSection offset

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
  var header = fromJson(rawHeader, Safetensor)

  # Sort tensors by offsets
  header.tensors.sort((lhs, rhs) => system.cmp(lhs[1].dataOffsets.start, rhs[1].dataOffsets.start))

  header.dataSectionOffset = sizeof(uint64) + headerSize

  # Validate that offsets are within the file with no gap or overlap
  header.validate_offsets(memFile.size - header.dataSectionOffset)

  return header

# Individual tensor API (WIP)
# ---------------------------------------------------------
#
# The API here will likely change with the following considerations
# - How to allow fast loading (async Streams, parallel workers, direct to GPU, ...)
# - How to associate lifetimes of `MemFile` and `MemSlice`
# - Don't leak implementation details like `dataSectionOffset`
#
# For now the goal is to get something working
# and keep it independent from the backend.

proc getMmapView*(st: Safetensor, memFile: MemFile, tensorName: string): MemSlice {.inline.} =
  ## Get a memory view to the tensor data.
  ## This allows zero-copy access to the tensor data.
  ## Lifetime:
  ##   Unfortunately MemFile predates `lent` and `openarray` as values view `{.experimental: "views".}`
  ##   so we don't get compiler-enforced borrow-checking.
  ##   https://github.com/nim-lang/nimony/issues/1517#issuecomment-3859350630
  ##
  ##   And this is not available yet
  ##   https://nim-lang.org/docs/manual.html#var-return-type-future-directions
  ##   `proc foo(other: Y; container: var X): var T from container`
  let info = st.tensors[tensorName]
  let (start, stopEx) = info.dataOffsets
  MemSlice(
    data: memFile.mem +% st.dataSectionOffset +% start,
    size: stopEx - start
  )
