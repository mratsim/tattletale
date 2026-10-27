# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SDPA fixture loader: python-generated raw binary fixtures into host tensors.
##
## One directory per fixture under `fixtures/`: meta.json plus the raw arrays.
## The arrays are little-endian, contiguous, row-major.

import std/[json, os, math]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils

export int_tuples, tensors

export ceramic_testutils.HostTensor, ceramic_testutils.makeHostTensor

const SdpaFixturesDir* = currentSourcePath().parentDir() / "fixtures"

type SdpaFixtureMetadata* = object
  name*: string
  dtype*: string      # "float32" | "float16" | "bfloat16"
  B*, Hq*, Hkv*, L*, S*, Dk*, Dv*: int
  scale*: float32
  causal*: bool
  windowSize*: int    # 0 = no window
  checksum*: uint32
  xorhashChecksums*: seq[(string, uint32)]  # per-file bit-pattern xors

proc parseSdpaMetadata*(fixtureDir: string): SdpaFixtureMetadata =
  let j = parseFile(fixtureDir / "meta.json")
  result.name = j["name"].getStr()
  result.dtype = j["dtype"].getStr()
  result.B = j["B"].getInt()
  result.Hq = j["Hq"].getInt()
  result.Hkv = j["Hkv"].getInt()
  result.L = j["L"].getInt()
  result.S = j["S"].getInt()
  result.Dk = j["Dk"].getInt()
  result.Dv = j["Dv"].getInt()
  result.scale = j["scale"].getFloat().float32
  result.causal = j["causal"].getBool()
  result.windowSize = j["window_size"].getInt(0)
  result.checksum = uint32(j["checksum"].getInt())
  for name, chk in j["xorhash_checksums"].pairs():
    result.xorhashChecksums.add (name, uint32(chk.getInt()))

proc xorhashChecksum*(meta: SdpaFixtureMetadata, file: string): uint32 =
  for (name, chk) in meta.xorhashChecksums:
    if name == file:
      return chk
  doAssert false, "no recorded checksum for " & file & " in " & meta.name

func xorHash*[T, Sh, St](ht: HostTensor[T, Sh, St]): uint32 =
  ## Order-independent checksum over the raw bit patterns of a host tensor
  for i in 0 ..< ht.len():
    when sizeof(T) == 4:
      result = result xor cast[uint32](ht.data[i])
    elif sizeof(T) == 2:
      result = result xor uint32(cast[uint16](ht.data[i]))
    else:
      result = result xor uint32(ht.data[i])

proc loadFixtureTensor*(fixtureDir: string, file: string, shape: tuple, elemType: typedesc): auto =
  result = makeHostTensor(elemType, shape, LayoutRight)
  let bytes = readFile(fixtureDir / file)
  doAssert bytes.len == result.len() * sizeof(elemType),
    "loadFixtureTensor: " & file & " is " & $bytes.len & " bytes, expected " &
    $(result.len() * sizeof(elemType))
  copyMem(result.data, unsafeAddr bytes[0], bytes.len)

proc expectedShape*(meta: SdpaFixtureMetadata): tuple =
  (meta.B, meta.Hq, meta.L, meta.Dv)
