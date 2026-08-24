## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## On-device fused linear+relu (manual, Metal): `linear_relu`
## vs the triple-loop GEMM reference with bias and relu applied by hand.
##
## The 8×8×8 mma's cross-lane reduction needs subgroup shuffles, which
## Apple's OpenCL-to-Metal translation rejects, so this gate runs on
## Metal on this machine; on OpenCL 2.0+ platforms it runs on OpenCL.
##
## Run: nim c -r --hints:off --warnings:off \
##   --outdir:build/tests/manual_tile_linear_relu_fp16 \
##   --nimcache:nimcache/tests/manual_tile_linear_relu_fp16 \
##   workspace/ceramic/tests/kernels_tiles/manual_tile_linear_relu_fp16.nim

import workspace/crucible
import ../libtest_epilogues
import ../../src/kernels/k_tile_gemm_epilogues

{.experimental: "callOperator".}

const msl = metal:
  proc fusedBiasRelu(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                     Bias: ptr UncheckedArray[float32], N, K, M: int32) {.global.} =
    linear_relu(D, A, B, Bias, N, K, M)

proc runTest() =
  let M = 64
  let N = 32
  let K = 32
  let (Ah, Bh) = fillAB(M, N, K, M, N, K)
  var Bias = newSeq[float32](N)
  for n in 0 ..< N:
    Bias[n] = (if n mod 3 == 0: -float32(1 + 2 * n) else: float32(1 + 2 * n))
  var Bb = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      Bb[m * N + n] = Bias[n]
  var D = newSeq[float32](M * N)
  var engine = bkMetal.init()
  engine.ingest(msl)
  engine.run << (grid: (N div 32, M div 32), blk: (32, 1)) >> (
    "fusedBiasRelu", D, (Ah, Bh, Bias, int32(M), int32(K), int32(N)))
  assertAllClose(D, relu(sum(gemmRef(M, N, K, M, N, K, Ah, Bh), Bb)))
  echo "linear_relu ", M, "x", N, "x", K, " (bias+relu fused): PASS on Metal"

when isMainModule:
  runTest()
