# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## SDPA fixture loader: python-generated raw binary fixtures into host tensors.
##
## One directory per fixture under `fixtures/`:
##
## - meta.json
## - q.bin, k.bin, v.bin, o.bin, all raw little-endian, contiguous row-major
## - mask.bin, only for the additive-mask fixtures
##
## Meta checksums: `checksum` sums the expected output in float64.
## `input_checksums` sums each of q/k/v/mask bytes the same way.
## Loader callers verify the byte replay against them.

import std/[json, os, math]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils

export int_tuples, tensors

export ceramic_testutils.HostTensor, ceramic_testutils.makeHostTensor

const SdpaFixturesDir* = currentSourcePath().parentDir() / "fixtures"

type SdpaFixtureMeta* = object
  name*: string
  dtype*: string      # "float32" | "float16" | "bfloat16"
  B*, Hq*, Hkv*, L*, S*, Dk*, Dv*: int
  scale*: float32
  causal*: bool
  mask*: string       # "none" | "additive"
  checksum*: float64
  inputChecksums*: seq[(string, float64)]  # per-file byte checksums
  maskNegInfCount*: int    # additive masks: blocked entries, excluded from the mask checksum

proc readSdpaMeta*(fixtureDir: string): SdpaFixtureMeta =
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
  result.mask = j["mask"].getStr()
  result.checksum = j["checksum"].getFloat()
  if "input_checksums" in j:
    for name, chk in j["input_checksums"].pairs():
      result.inputChecksums.add (name, chk.getFloat())
  if "mask_neg_inf_count" in j:
    result.maskNegInfCount = j["mask_neg_inf_count"].getInt()

proc inputChecksum*(meta: SdpaFixtureMeta, file: string): float64 =
  ## Recorded byte checksum of one fixture file, q/k/v/mask/o
  for (name, chk) in meta.inputChecksums:
    if name == file:
      return chk
  doAssert false, "no recorded checksum for " & file & " in " & meta.name

proc rowMajorStrides4(shape: (int, int, int, int)): (int, int, int, int) =
  # contiguous (B, H, S, D): stride of the first axis covers the tail
  let (d0, d1, d2, d3) = shape
  (d1 * d2 * d3, d2 * d3, d3, 1)

proc loadFixtureTensor*[T](fixtureDir: string, file: string, shape: tuple): auto =
  ## Reads one raw binary array, the element type the caller picks
  ## (bfloat16 fixtures load as uint16 bits at the call site).
  let strides = rowMajorStrides4(shape)
  result = makeHostTensor(T, shape, strides)
  let bytes = readFile(fixtureDir / file)
  doAssert bytes.len == result.len() * sizeof(T),
    "loadFixtureTensor: " & file & " is " & $bytes.len & " bytes, expected " &
    $(result.len() * sizeof(T))
  copyMem(result.data, unsafeAddr bytes[0], bytes.len)

proc expectedShape*(meta: SdpaFixtureMeta): tuple =
  (meta.B, meta.Hq, meta.L, meta.Dv)
