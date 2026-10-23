# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Fixture replay roundtrip: python-generated raw binary fixtures read
## back byte-exact on the Nim side, verified against the meta checksum.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/k_sdpa/t_sdpa_fixture_roundtrip.nim

import std/[math, os]
import workspace/crucible
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_sdpa/sdpa_fixtures

proc checksum64[T, Sh, St](ht: HostTensor[T, Sh, St]): float64 =
  ## f64 sum over the flat element order
  for i in 0 ..< ht.len():
    result += float64(ht.data[i])

proc checkFixture(name: string) =
  let dir = SdpaFixturesDir / name
  let meta = readSdpaMeta(dir)
  let inShape = (meta.B, meta.Hq, meta.Hkv, meta.L, meta.S, meta.Dk)
  doAssert inShape[0] > 0

  case meta.dtype
  of "float32":
    let q = loadFixtureTensor[float32](dir, "q.bin",
      (meta.B, meta.Hq, meta.L, meta.Dk))
    let o = loadFixtureTensor[float32](dir, "o.bin", meta.expectedShape)
    doAssert q.len() == meta.B * meta.Hq * meta.L * meta.Dk
    doAssert abs(checksum64(o) - meta.checksum) < 1e-3 * max(1.0, abs(meta.checksum)),
      name & ": f32 expected-output checksum mismatch"
  of "float16":
    let q = loadFixtureTensor[float16](dir, "q.bin",
      (meta.B, meta.Hq, meta.L, meta.Dk))
    let o = loadFixtureTensor[float32](dir, "o.bin", meta.expectedShape)
    doAssert q.len() == meta.B * meta.Hq * meta.L * meta.Dk
    doAssert abs(checksum64(o) - meta.checksum) < 1e-3 * max(1.0, abs(meta.checksum)),
      name & ": f16 fixture f32 expected-output checksum mismatch"
  of "bfloat16":
    # bf16 fixtures store raw uint16 bits
    let qBits = loadFixtureTensor[uint16](dir, "q.bin",
      (meta.B, meta.Hq, meta.L, meta.Dk))
    let o = loadFixtureTensor[float32](dir, "o.bin", meta.expectedShape)
    doAssert qBits.len() == meta.B * meta.Hq * meta.L * meta.Dk
    doAssert abs(checksum64(o) - meta.checksum) < 1e-3 * max(1.0, abs(meta.checksum)),
      name & ": bf16 fixture f32 expected-output checksum mismatch"
  else:
    doAssert false, "unknown fixture dtype " & meta.dtype

  if meta.mask == "additive":
    # (B, Hm, S, S) with Hm in {1, Hkv}: the head count comes from the file size
    let nBytes = getFileSize(dir / "mask.bin")
    let Hm = int(nBytes div (4 * meta.B * meta.S * meta.S))
    doAssert Hm == 1 or Hm == meta.Hkv, "unexpected mask head count " & $Hm
    let m = loadFixtureTensor[float32](dir, "mask.bin",
      (meta.B, Hm, meta.S, meta.S))
    doAssert m.len() == meta.B * Hm * meta.S * meta.S
  echo "roundtrip ok: ", name

proc main =
  for name in ["tiny_f32", "gqa2_f32", "tiny_f16", "tiny_bf16",
               "mask_f32", "mask_broadcast_f32", "causal_f32",
               "decode_gqa8_f16"]:
    checkFixture(name)
  echo "all fixture roundtrips pass"

main()
