# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Jagged attention fixture loader: python-generated raw binary fixtures as host tensors.
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
export positron_testutils.len, positron_testutils.bridge
export positron_testutils.loadFixtureTensor, positron_testutils.xorHash

const JaggedAttnFixturesDir* = currentSourcePath().parentDir() / "fixtures"

type JaggedAttnFixtureMetadata* = object
  ## One fixture's geometry and checksums, parsed from meta.json.
  name*: string
  dtype*: string          # "float16" | "float32"
  numSeqs*, H*, Nkv*, D*: int
  pageSize*, numLayers*, numPages*, layer*: int
  windowSize*: int        # 0 = no window
  numQoTokens*, numIndices*: int
  scale*: float32
  qOffsets*, kvOffsets*, kvSeqlens*: seq[int]
  xorhashChecksums*: seq[(string, uint32)]  # per-file bit-pattern xors

proc parseJaggedAttnMetadata*(fixtureDir: string): JaggedAttnFixtureMetadata =
  ## Returns the parsed meta.json of one fixture directory.
  let j = parseFile(fixtureDir / "meta.json")
  result.name = j["name"].getStr()
  result.dtype = j["dtype"].getStr()
  result.numSeqs = j["num_seqs"].getInt()
  result.H = j["H"].getInt()
  result.Nkv = j["Nkv"].getInt()
  result.D = j["D"].getInt()
  result.pageSize = j["page_size"].getInt()
  result.numLayers = j["num_layers"].getInt()
  result.numPages = j["num_pages"].getInt()
  result.layer = j["layer"].getInt()
  result.windowSize = j["window_size"].getInt()
  result.numQoTokens = j["num_qo_tokens"].getInt()
  result.numIndices = j["num_indices"].getInt()
  result.scale = j["scale"].getFloat().float32
  for v in j["q_offsets"]:
    result.qOffsets.add v.getInt()
  for v in j["kv_offsets"]:
    result.kvOffsets.add v.getInt()
  for v in j["kv_seqlens"]:
    result.kvSeqlens.add v.getInt()
  for fname, chk in j["xorhash_checksums"].pairs():
    result.xorhashChecksums.add (fname, uint32(chk.getInt()))

func xorhashChecksum*(meta: JaggedAttnFixtureMetadata, file: string): uint32 =
  ## Looks up one file's recorded checksum.
  for (fname, chk) in meta.xorhashChecksums:
    if fname == file:
      return chk
  doAssert false, "no recorded checksum for " & file & " in " & meta.name

proc loadRank1FixtureTensor*(fixtureDir: string, file: string, n: int): auto =
  ## One-dimensional int32 fixture array (offsets, indices, seqlens).
  ##
  ## Returns:
  ##   A `HostTensor[int32, ...]` over `n` freshly allocated elements,
  ##   filled from `file`'s raw bytes.
  ##
  ## Why not `loadFixtureTensor`: `makeHostTensor` rejects rank-1 shapes,
  ## `originOffset` dies with a type mismatch on the `(n,)` layout
  ## (ceramic_testutils.nim stride loop), so the layout is built here.
  let layout = make_layout((n,))
  result = HostTensor[int32, typeof(layout)](
    data: cast[ptr UncheckedArray[int32]](allocShared(n * sizeof(int32))),
    offset: 0,
    layout: layout)
  let bytes = readFile(fixtureDir / file)
  doAssert bytes.len == n * sizeof(int32),
    "loadRank1FixtureTensor: " & file & " is " & $bytes.len & " bytes, expected " &
    $(n * sizeof(int32))
  copyMem(result.data, unsafeAddr bytes[0], bytes.len)
