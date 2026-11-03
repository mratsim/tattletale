# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Paged SDPA fixture loader: python-generated raw binary fixtures into host tensors.
##
## One directory per fixture under `fixtures/`: meta.json plus the raw arrays.
## The arrays are little-endian, contiguous, last axis fastest.

import std/[json, os]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_sdpa/sdpa_fixtures

export int_tuples, tensors

export ceramic_testutils.HostTensor, ceramic_testutils.makeHostTensor
export sdpa_fixtures.loadFixtureTensor, sdpa_fixtures.xorHash

const SdpaPagedFixturesDir* = currentSourcePath().parentDir() / "fixtures"

type SdpaPagedFixtureMetadata* = object
  name*: string
  dtype*: string      # "float16"
  numSeqs*, Hq*, Hkv*, Dk*, Dv*: int
  pageSize*, maxPages*, numLayers*, numPages*, layer*: int
  scale*: float32
  cacheSeqlens*: seq[int]
  checksum*: uint32
  xorhashChecksums*: seq[(string, uint32)]  # per-file bit-pattern xors

proc parseSdpaPagedMetadata*(fixtureDir: string): SdpaPagedFixtureMetadata =
  let j = parseFile(fixtureDir / "meta.json")
  result.name = j["name"].getStr()
  result.dtype = j["dtype"].getStr()
  result.numSeqs = j["num_seqs"].getInt()
  result.Hq = j["Hq"].getInt()
  result.Hkv = j["Hkv"].getInt()
  result.Dk = j["Dk"].getInt()
  result.Dv = j["Dv"].getInt()
  result.pageSize = j["page_size"].getInt()
  result.maxPages = j["max_pages"].getInt()
  result.numLayers = j["num_layers"].getInt()
  result.numPages = j["num_pages"].getInt()
  result.layer = j["layer"].getInt()
  result.scale = j["scale"].getFloat().float32
  for v in j["cache_seqlens"]:
    result.cacheSeqlens.add v.getInt()
  result.checksum = uint32(j["checksum"].getInt())
  for name, chk in j["xorhash_checksums"].pairs():
    result.xorhashChecksums.add (name, uint32(chk.getInt()))

proc xorhashChecksum*(meta: SdpaPagedFixtureMetadata, file: string): uint32 =
  for (name, chk) in meta.xorhashChecksums:
    if name == file:
      return chk
  doAssert false, "no recorded checksum for " & file & " in " & meta.name

proc loadFlatFixtureTensor*(fixtureDir: string, file: string, n: int): auto =
  ## One-dimensional int32 fixture array (block table, cache seqlens).
  ##
  ## Returns:
  ##   A `HostTensor[int32, ...]` over `n` freshly allocated elements,
  ##   filled from `file`'s raw bytes.
  ##
  ## Why not `loadFixtureTensor`: `makeHostTensor` rejects rank-1 shapes,
  ## `originOffset` dies with a type mismatch on the `(n,)` layout
  ## (ceramic_testutils.nim stride loop), so the layout is built here.
  let layout = make_layout((n,))
  result = HostTensor[int32, typeof(layout.shape), typeof(layout.stride)](
    data: cast[ptr UncheckedArray[int32]](allocShared(n * sizeof(int32))),
    offset: 0,
    layout: layout)
  let bytes = readFile(fixtureDir / file)
  doAssert bytes.len == n * sizeof(int32),
    "loadFlatFixtureTensor: " & file & " is " & $bytes.len & " bytes, expected " & $(n * sizeof(int32))
  copyMem(result.data, unsafeAddr bytes[0], bytes.len)

proc expectedShape*(meta: SdpaPagedFixtureMetadata): tuple =
  (meta.numSeqs, meta.Hq, 1, meta.Dv)

