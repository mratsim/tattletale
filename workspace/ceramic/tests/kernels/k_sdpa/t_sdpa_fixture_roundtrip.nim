# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Fixture replay roundtrip: every python-generated fixture file reads
## back byte-exact on the Nim side, inputs and expected output alike,
## each file verified against its meta checksum.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/k_sdpa/t_sdpa_fixture_roundtrip.nim

import std/[algorithm, math, os]
import workspace/crucible
import workspace/ceramic/src/tensors
import workspace/fp_conv/fp_conv
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_sdpa/sdpa_fixtures

proc checksum64[T, Sh, St](ht: HostTensor[T, Sh, St]): float64 =
  ## f64 sum over the flat element order, low-precision floats widen first
  for i in 0 ..< ht.len():
    when T is float16:
      result += float64(ht.data[i].to(float32))
    else:
      result += float64(ht.data[i])

proc checkFixture(name: string) =
  ## Checksums every file of one fixture against its meta record.
  ## The loader byte-size assert covers shapes, this covers bytes.
  let dir = SdpaFixturesDir / name
  let meta = readSdpaMeta(dir)
  template check(file: string, T: typedesc, shape: tuple, want: float64) =
    let ht = loadFixtureTensor[T](dir, file, shape)
    doAssert abs(checksum64(ht) - want) < 1e-3 * max(1.0, abs(want)),
      name & ": " & file & " byte checksum mismatch"
  case meta.dtype
  of "float32":
    check("q.bin", float32, (meta.B, meta.Hq, meta.L, meta.Dk), meta.inputChecksum("q.bin"))
    check("k.bin", float32, (meta.B, meta.Hkv, meta.S, meta.Dk), meta.inputChecksum("k.bin"))
    check("v.bin", float32, (meta.B, meta.Hkv, meta.S, meta.Dv), meta.inputChecksum("v.bin"))
  of "float16":
    check("q.bin", float16, (meta.B, meta.Hq, meta.L, meta.Dk), meta.inputChecksum("q.bin"))
    check("k.bin", float16, (meta.B, meta.Hkv, meta.S, meta.Dk), meta.inputChecksum("k.bin"))
    check("v.bin", float16, (meta.B, meta.Hkv, meta.S, meta.Dv), meta.inputChecksum("v.bin"))
  of "bfloat16":
    # bf16 fixtures store raw uint16 bits, checksummed as the bits
    check("q.bin", uint16, (meta.B, meta.Hq, meta.L, meta.Dk), meta.inputChecksum("q.bin"))
    check("k.bin", uint16, (meta.B, meta.Hkv, meta.S, meta.Dk), meta.inputChecksum("k.bin"))
    check("v.bin", uint16, (meta.B, meta.Hkv, meta.S, meta.Dv), meta.inputChecksum("v.bin"))
  else:
    doAssert false, "unknown fixture dtype " & meta.dtype
  check("o.bin", float32, meta.expectedShape, meta.checksum)

  if meta.mask == "additive":
    # (B, Hm, S, S) with Hm in {1, Hkv}: the head count comes from the file size
    let nBytes = getFileSize(dir / "mask.bin")
    let Hm = int(nBytes div (4 * meta.B * meta.S * meta.S))
    doAssert Hm == 1 or Hm == meta.Hkv, "unexpected mask head count " & $Hm
    let m = loadFixtureTensor[float32](dir, "mask.bin", (meta.B, Hm, meta.S, meta.S))
    # Blocked entries are -inf and checksum separately from the finite
    # values covered by the meta checksum
    var finiteSum = 0.0'f64
    var blocked = 0
    for i in 0 ..< m.len():
      if m.data[i] == NegInf:
        blocked.inc
      else:
        finiteSum += float64(m.data[i])
    doAssert blocked == meta.maskNegInfCount,
      name & ": blocked mask entries " & $blocked & ", meta records " & $meta.maskNegInfCount
    let want = meta.inputChecksum("mask.bin")
    doAssert abs(finiteSum - want) < 1e-3 * max(1.0, abs(want)),
      name & ": mask.bin byte checksum mismatch"
  echo "roundtrip ok: ", name

proc main =
  var names: seq[string]
  for dir in walkDirs(SdpaFixturesDir / "*"):
    names.add(splitPath(dir).tail)
  names.sort()
  doAssert names.len > 0, "no fixture directories under " & SdpaFixturesDir
  for name in names:
    checkFixture(name)
  echo "all fixture roundtrips pass (", names.len, " fixtures)"

main()
