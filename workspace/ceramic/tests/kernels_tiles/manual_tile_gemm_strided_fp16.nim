## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##

import workspace/crucible
import ../tile_test_utils
import ../libtest_epilogues
import ../../src/kernels/k_tile_gemm

const stridedMsl = metal:
  proc fusedGemm(D: ptr UncheckedArray[float32],
                 A, B: ptr UncheckedArray[float16],
                 M, N, K: int32, rsa, csa, rsb, csb: int32) {.global.} =
    gemm_with_epilogue(D, N, 1, A, rsa, csa, B, rsb, csb, M, K, N, EpiIdentity())

proc runStrided(engine: var auto; M, N, K: int;
                rsa, csa, rsb, csb, aBase, bBase: int) =
  let aSize = aBase + max(0, (M - 1) * rsa) + (K - 1) * csa + 1
  let bSize = bBase + max(0, (K - 1) * rsb) + (N - 1) * csb + 1
  var Ah = newSeq[uint16](aSize)
  var Bh = newSeq[uint16](bSize)
  for m in 0 ..< M:
    for k in 0 ..< K:
      Ah[aBase + m * rsa + k * csa] = fp32ToFp16(float32(1 + 2 * m + 7 * k))
  for k in 0 ..< K:
    for n in 0 ..< N:
      Bh[bBase + k * rsb + n * csb] = fp32ToFp16(float32(1 + 3 * k + 11 * n))
  let pA = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](Ah[0].unsafeAddr),
                          len: aSize, off: aBase)
  let pB = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](Bh[0].unsafeAddr),
                          len: bSize, off: bBase)
  var D = newSeq[float32](M * N)
  engine.run << (grid: (N div 32, M div 32), blk: (32, 1)) >> (
    "fusedGemm", D,
    (pA, pB, int32(M), int32(N), int32(K),
     int32(rsa), int32(csa), int32(rsb), int32(csb)))
  assertAllClose(D, gemmRefStrided(M, N, K, rsa, csa, rsb, csb, aBase, bBase, Ah, Bh))
  echo "STRIDED ", M, "x", N, "x", K,
       " (rsa=", rsa, ", csa=", csa, ", rsb=", rsb, ", csb=", csb, "): PASS"

proc runTest() =
  const M = 64
  const N = 64
  const K = 32
  var engine = bkMetal.init()
  engine.ingest(stridedMsl)
  runStrided(engine, M, N, K, K, 1, N, 1, 0, 0)      # row-major
  runStrided(engine, M, N, K, 1, M, 1, K, 0, 0)      # col-major
  runStrided(engine, M, N, K, 2 * K, 1, 2 * N, 1, 0, 0)  # 1 out of 2
  # Negative stride: the base-shifted view needs the full buffer
  # addressable below the base.
  runStrided(engine, M, N, K, -K, 1, -N, 1,
             (M - 1) * K, (K - 1) * N)

when isMainModule:
  runTest()
