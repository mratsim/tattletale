## Grouped GEMM fixture loader: python-generated raw binary fixtures as host tensors.
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

const GroupedGemmFixturesDir* = currentSourcePath().parentDir() / "fixtures" / "gemm"

type GroupedGemmFixtureMetadata* = object
  ## One fixture's geometry and checksums, parsed from meta.json.
  name*: string
  dtype*: string          # "float16" | "float32"
  e*, k*, n*, totalM*, mTile*: int
  counts*, pads*: seq[int]
  xorhashChecksums*: seq[(string, uint32)]  # per-file bit-pattern xors

proc parseGroupedGemmMetadata*(fixtureDir: string): GroupedGemmFixtureMetadata =
  ## Returns the parsed meta.json of one fixture directory.
  let j = parseFile(fixtureDir / "meta.json")
  result.name = j["name"].getStr()
  result.dtype = j["dtype"].getStr()
  result.e = j["E"].getInt()
  result.k = j["K"].getInt()
  result.n = j["N"].getInt()
  result.totalM = j["total_m"].getInt()
  result.mTile = j["m_tile"].getInt()
  for v in j["counts"]:
    result.counts.add v.getInt()
  for v in j["pads"]:
    result.pads.add v.getInt()
  for fname, chk in j["xorhash_checksums"].pairs():
    result.xorhashChecksums.add (fname, uint32(chk.getInt()))

func xorhashChecksum*(meta: GroupedGemmFixtureMetadata, file: string): uint32 =
  ## Looks up one file's recorded checksum.
  for (fname, chk) in meta.xorhashChecksums:
    if fname == file:
      return chk
  doAssert false, "no recorded checksum for " & file & " in " & meta.name

proc loadIdsFixtureTensor*(fixtureDir: string, file: string, n: int): auto =
  ## One-dimensional int32 fixture array (the per-line expert ids).
  ##
  ## Returns:
  ##   A `HostTensor[int32, ...]` over `n` freshly allocated elements,
  ##   filled from `file`'s raw bytes.
  result = makeHostTensor(int32, (n,), (int32(1),))
  let bytes = readFile(fixtureDir / file)
  doAssert bytes.len == n * sizeof(int32),
    "loadIdsFixtureTensor: " & file & " is " & $bytes.len & " bytes, expected " &
    $(n * sizeof(int32))
  copyMem(result.data, unsafeAddr bytes[0], bytes.len)
