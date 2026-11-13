## Grouped GEMV fixture loader: python-generated raw binary fixtures as host tensors.
##
## One directory per fixture under `fixtures/`: meta.json plus raw arrays. The arrays are little-endian, contiguous, last axis fastest.
## Element-dtype arrays load as bit patterns (uint16 for float16, uint32 for float32) and suites widen them through `ceramic_dtype`.

import std/[json, os]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import ../positron_testutils


export int_tuples, layout_algebra, tensors

export positron_testutils.HostTensor, positron_testutils.makeHostTensor
export positron_testutils.len
export positron_testutils.loadFixtureTensor, positron_testutils.xorHash

const GroupedGemvFixturesDir* = currentSourcePath().parentDir() / "fixtures" / "gemv"

type GroupedGemvFixtureMetadata* = object
  ## One fixture's geometry and checksums, parsed from meta.json.
  name*: string
  dtype*: string          # "float16" | "float32"
  t*, topk*, e*, k*, n*: int
  xorhashChecksums*: seq[(string, uint32)]  # per-file bit-pattern xors

proc parseGroupedGemvMetadata*(fixtureDir: string): GroupedGemvFixtureMetadata =
  ## Returns the parsed meta.json of one fixture directory.
  let j = parseFile(fixtureDir / "meta.json")
  result.name = j["name"].getStr()
  result.dtype = j["dtype"].getStr()
  result.t = j["T"].getInt()
  result.topk = j["topk"].getInt()
  result.e = j["E"].getInt()
  result.k = j["K"].getInt()
  result.n = j["N"].getInt()
  for fname, chk in j["xorhash_checksums"].pairs():
    result.xorhashChecksums.add (fname, uint32(chk.getInt()))

func xorhashChecksum*(meta: GroupedGemvFixtureMetadata, file: string): uint32 =
  ## Looks up one file's recorded checksum.
  for (fname, chk) in meta.xorhashChecksums:
    if fname == file:
      return chk
  doAssert false, "no recorded checksum for " & file & " in " & meta.name

proc loadIds2DFixtureTensor*(fixtureDir: string, file: string, t, topk: int): auto =
  ## Two-dimensional int32 fixture array (the per-token expert ids).
  ##
  ## Returns:
  ##   A `HostTensor[int32, ...]` over (t, topk) freshly allocated elements,
  ##   filled from `file`'s raw bytes.
  result = makeHostTensor(int32, (int32(t), int32(topk)), (int32(topk), int32(1)))
  let bytes = readFile(fixtureDir / file)
  doAssert bytes.len == t * topk * sizeof(int32),
    "loadIds2DFixtureTensor: " & file & " is " & $bytes.len & " bytes, expected " &
    $(t * topk * sizeof(int32))
  copyMem(result.data, unsafeAddr bytes[0], bytes.len)

proc loadWeights2DFixtureTensor*(fixtureDir: string, file: string, t, topk: int): auto =
  ## Two-dimensional float32 fixture array (the per-token routing weights).
  ##
  ## Returns:
  ##   A `HostTensor[float32, ...]` over (t, topk) freshly allocated elements,
  ##   filled from `file`'s raw bytes.
  result = makeHostTensor(float32, (int32(t), int32(topk)), (int32(topk), int32(1)))
  let bytes = readFile(fixtureDir / file)
  doAssert bytes.len == t * topk * sizeof(float32),
    "loadWeights2DFixtureTensor: " & file & " is " & $bytes.len & " bytes, expected " &
    $(t * topk * sizeof(float32))
  copyMem(result.data, unsafeAddr bytes[0], bytes.len)
