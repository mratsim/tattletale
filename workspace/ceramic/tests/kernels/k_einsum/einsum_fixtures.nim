# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Einsum fixture loader: python-generated raw binaries into host tensors.
##
## One directory per fixture under `fixtures/`: meta.json plus the raw arrays.
## Arrays are little-endian float32, stored contiguously in one of two axis
## orders over the logical shape, meta.json's strides name the order:
## - C order, last axis fastest
## - Fortran order, first axis fastest
##
## Rank-1 tensors load through the shared `makeHostTensor` too.
## Every stride here is positive, so the origin offset is 0.

import std/[json, os, tables]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_sdpa/sdpa_fixtures
  # xorHash: the order-independent bit-pattern checksum the generator records

export int_tuples, layout_algebra, tensors

const EinsumFixturesDir* = currentSourcePath().parentDir() / "fixtures"

type EinsumMeta* = object
  name*: string
  recipe*: string
  operandShapes*: seq[seq[int]]
  operandStrides*: seq[seq[int]]
  outShape*: seq[int]
  outStrides*: seq[int]
  xorhash*: Table[string, uint32]

proc parseEinsumMeta*(fixtureDir: string): EinsumMeta =
  let j = parseFile(fixtureDir / "meta.json")
  result.name = j["name"].getStr()
  result.recipe = j["recipe"].getStr()
  proc readInts(node: JsonNode): seq[int] =
    for v in node:
      result.add int(v.getInt())
  for op in j["operands"]:
    result.operandShapes.add readInts(op["shape"])
    result.operandStrides.add readInts(op["strides"])
  result.outShape = readInts(j["out"]["shape"])
  result.outStrides = readInts(j["out"]["strides"])
  for file, chk in j["xorhash_checksums"].pairs():
    result.xorhash[file] = uint32(chk.getInt())

proc checkXorhash*[T, Sh, St](meta: EinsumMeta, file: string, ht: HostTensor[T, Sh, St]) =
  doAssert meta.xorhash[file] == xorHash(ht),
    meta.name & ": " & file & " xorhash mismatch"

proc loadFixtureBin[T](fixtureDir: string, file: string, n: int): seq[T] =
  let bytes = readFile(fixtureDir / file)
  doAssert bytes.len == n * sizeof(T),
    "loadFixtureBin: " & file & " is " & $bytes.len & " bytes, expected " & $(n * sizeof(T))
  result = newSeq[T](n)
  copyMem(addr result[0], unsafeAddr bytes[0], bytes.len)

proc loadFixtureTensor*[T](fixtureDir: string, file: string, shape, strides: tuple, elemType: typedesc[T]): HostTensor[T, typeof(shape), typeof(strides)] =
  ## Host tensor over a fixture bin, laid out as the caller names
  result = makeHostTensor(elemType, shape, strides)
  let raw = loadFixtureBin[T](fixtureDir, file, toInt(result.layout.size()))
  copyMem(result.data, addr raw[0], raw.len * sizeof(T))
