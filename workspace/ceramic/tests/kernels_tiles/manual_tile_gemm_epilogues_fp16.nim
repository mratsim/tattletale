## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/manual_tile_gemm_epilogues_fp16 \
##   --nimcache:nimcache/tests/manual_tile_gemm_epilogues_fp16 \
##   workspace/ceramic/tests/kernels_tiles/manual_tile_gemm_epilogues_fp16.nim

import workspace/crucible
import ../libtest_epilogues
import ../tile_test_utils
import ../../src/kernels/k_tile_gemm

{.experimental: "callOperator".}

const epiloguesMsl = metal:
  proc fusedIdentity(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                     N, K, M: int32) {.global.} =
    matmul(D, A, B, N, K, M)

  proc fusedRelu(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                 N, K, M: int32) {.global.} =
    gemm_relu(D, A, B, N, K, M)

  proc fusedBias(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                 Bias: ptr UncheckedArray[float32], N, K, M: int32) {.global.} =
    linear(D, A, B, Bias, N, K, M)

  proc fusedAxpby(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                  C: ptr UncheckedArray[float32], Alpha, Beta: float32,
                  N, K, M: int32) {.global.} =
    gemm(D, K, 1, N, K, M, Alpha, A, K, 1, B, M, 1, Beta, C, M, 1)

  proc fusedAxpbyS(D: ptr UncheckedArray[float32], A, B: ptr UncheckedArray[float16],
                   C: ptr UncheckedArray[float32], Alpha, Beta: float32,
                   N, K, M: int32, rsc, csc: int32) {.global.} =
    gemm(D, K, 1, N, K, M, Alpha, A, K, 1, B, M, 1, Beta, C, rsc, csc)

# The dtype-parametrized tile applies live in a separate program.
# Fusing them with the kernels above trips a legalization-pass limit.
const dtypeMsl = metal:
  proc dtypeFp16(D: ptr UncheckedArray[float32], A: ptr UncheckedArray[float16]) {.global.} =
    var t_rtl: rt_l(float16, 16, 16)
    t_rtl.frags[0][0].frag[0] = A[0]
    apply(EpiIdentity(), t_rtl, t_rtl)
    D[0] = float32(A[0])

  proc dtypeFp32(D: ptr UncheckedArray[float32]) {.global.} =
    var t_rtl: rt_l(float32, 16, 16)
    t_rtl.frags[0][0].frag[0] = 5.0'f32
    apply(EpiIdentity(), t_rtl, t_rtl)
    D[0] = 1.0'f32

proc runTest() =
  let M = 64
  let N = 32
  let K = 32
  let (Ah, Bh) = fillAB(M, N, K, M, N, K)
  var D = newSeq[float32](M * N)
  var engine = bkMetal.init()
  engine.ingest(epiloguesMsl)
  let grid = (N div 32, M div 32)

  engine.run << (grid: grid, blk: (32, 1)) >> (
    "fusedIdentity", D, (Ah, Bh, int32(M), int32(K), int32(N)))
  assertAllClose(D, gemmRef(M, N, K, M, N, K, Ah, Bh))

  engine.run << (grid: grid, blk: (32, 1)) >> (
    "fusedRelu", D, (Ah, Bh, int32(M), int32(K), int32(N)))
  assertAllClose(D, relu(gemmRef(M, N, K, M, N, K, Ah, Bh)))

  var Bias = newSeq[float32](N)
  for n in 0 ..< N:
    Bias[n] = (if n mod 3 == 0: -float32(1 + 2 * n) else: float32(1 + 2 * n))
  var Bb = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      Bb[m * N + n] = Bias[n]
  engine.run << (grid: grid, blk: (32, 1)) >> (
    "fusedBias", D, (Ah, Bh, Bias, int32(M), int32(K), int32(N)))
  assertAllClose(D, sum(gemmRef(M, N, K, M, N, K, Ah, Bh), Bb))

  var Cmat = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      Cmat[m * N + n] = float32(1 + 3 * m + 5 * n)
  engine.run << (grid: grid, blk: (32, 1)) >> (
    "fusedAxpby", D, (Ah, Bh, Cmat, 2.0'f32, 4.0'f32,
                       int32(M), int32(K), int32(N)))
  assertAllClose(D, sum(scale(gemmRef(M, N, K, M, N, K, Ah, Bh), 2.0'f32),
                        scale(Cmat, 4.0'f32)))

  # Strided C: the β-operand at (rsc, csc) = (2·N, 1), every other padded
  # row — the runtime strides must be honored by the axpby epilogue.
  var Cp = newSeq[float32](M * 2 * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      Cp[m * 2 * N + n] = float32(1 + 3 * m + 5 * n)
  var Cref = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      Cref[m * N + n] = Cp[m * 2 * N + n]
  engine.run << (grid: grid, blk: (32, 1)) >> (
    "fusedAxpbyS", D, (Ah, Bh, Cp, 2.0'f32, 4.0'f32,
                        int32(M), int32(K), int32(N), int32(2 * N), int32(1)))
  assertAllClose(D, sum(scale(gemmRef(M, N, K, M, N, K, Ah, Bh), 2.0'f32),
                        scale(Cref, 4.0'f32)))

  # The dtype variants resolve and run.
  var dres: array[1, float32]
  var a16 = newSeq[uint16](1)
  a16[0] = fp32ToFp16(5.0'f32)
  engine.ingest(dtypeMsl)
  engine.run << (grid: (1, 1), blk: (32, 1)) >> ("dtypeFp16", dres, (a16,))
  engine.run << (grid: (1, 1), blk: (32, 1)) >> ("dtypeFp32", dres, ())
  echo "gemm_epilogues ", M, "x", N, "x", K,
       " (fusedIdentity/Relu/Bias/Axpby + dtypeFp16/Fp32): PASS on Metal"

when isMainModule:
  runTest()
