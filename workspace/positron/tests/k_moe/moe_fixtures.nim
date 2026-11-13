## MoE frontend fixture loader: python-generated raw binary fixtures as host tensors.
##
## One directory per fixture under `fixtures/`: meta.json plus raw arrays.
##
## Arrays are little-endian, contiguous, last axis fastest.
##
## Element-dtype arrays load as bit patterns: uint16 for float16 and uint32
## for float32. Suites widen them through the staging casts.
##
## Fixture families:
##
## - router: scores, bias, ids, weights
## - dispatch: ids, x, packedIds, packedOffsets, sortedPairs, slotOfPair, packedX
## - e2e: x, b, scores, bias, d

import std/[json, os]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import ../positron_testutils


export int_tuples, layout_algebra, tensors

export positron_testutils.HostTensor, positron_testutils.makeHostTensor
export positron_testutils.len
export positron_testutils.loadFixtureTensor, positron_testutils.xorHash

const MoEFixturesDir* = currentSourcePath().parentDir() / "fixtures"

type MoEFixtureMetadata* = object
  ## One fixture's geometry, family and checksums, parsed from meta.json.
  name*: string
  family*: string         # "router" | "dispatch" | "e2e"
  dtype*: string          # "float16" | "float32", e2e fixtures only
  t*, topk*, e*, k*, n*: int
  variant*: int           # moeSoftmax | moeSigmoid
  scaling*: float64
  xorhashChecksums*: seq[(string, uint32)]  # per-file bit-pattern xors

proc parseMoEFixtureMetadata*(fixtureDir: string): MoEFixtureMetadata =
  ## Returns the parsed meta.json of one fixture directory.
  let j = parseFile(fixtureDir / "meta.json")
  result.name = j["name"].getStr()
  result.family = j["family"].getStr()
  if j.hasKey("dtype"):
    result.dtype = j["dtype"].getStr()
  result.t = j["T"].getInt()
  result.topk = j["topk"].getInt()
  result.e = j["E"].getInt()
  if j.hasKey("K"):
    result.k = j["K"].getInt()
  if j.hasKey("N"):
    result.n = j["N"].getInt()
  if j.hasKey("variant"):
    result.variant = j["variant"].getInt()
  if j.hasKey("scaling"):
    result.scaling = j["scaling"].getFloat()
  for fname, chk in j["xorhash_checksums"].pairs():
    result.xorhashChecksums.add (fname, uint32(chk.getInt()))

func xorhashChecksum*(meta: MoEFixtureMetadata, file: string): uint32 =
  ## Looks up one file's recorded checksum.
  for (fname, chk) in meta.xorhashChecksums:
    if fname == file:
      return chk
  doAssert false, "no recorded checksum for " & file & " in " & meta.name
