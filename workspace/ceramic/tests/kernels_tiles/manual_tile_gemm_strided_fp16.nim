## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/tests/manual_tile_gemm_strided_fp16 \
##   --nimcache:nimcache/tests/manual_tile_gemm_strided_fp16 \
##   workspace/ceramic/tests/kernels_tiles/manual_tile_gemm_strided_fp16.nim

import std/[strformat]
import workspace/crucible
import ../libtest_epilogues
import gemm_test_utils
import ../../src/kernels/k_tile_gemm

# ═════════════════════════════════════════════════════════════════════════
#  Device-level launchers: the strided gemm over fp16 views.
#  rsd is the D row stride (the padded pitch), separate from the raw N.
#  C loads bounded into a register tile at the original strides,
#  out-of-range lanes zero-filled in-register on ragged edges.
# ═════════════════════════════════════════════════════════════════════════

const stridedMsl = metal:
  proc fusedGemm(D: ptr UncheckedArray[float32],
                 A, B: ptr UncheckedArray[float16],
                 M, N, K: int32, rsd, rsa, csa, rsb, csb: int32) {.global.} =
    gemm_with_epilogue(D, rsd, 1, A, rsa, csa, B, rsb, csb, M, K, N,
                       EpiIdentity())

  proc fusedAxpby(D: ptr UncheckedArray[float32],
                  A, B: ptr UncheckedArray[float16],
                  C: ptr UncheckedArray[float32],
                  Alpha, Beta: float32,
                  M, N, K: int32, rsd, rsa, csa, rsb, csb, rsc, csc: int32) {.global.} =
    gemm(D, rsd, 1, M, N, K, Alpha, A, rsa, csa, B, rsb, csb, Beta, C, rsc, csc)

# ═════════════════════════════════════════════════════════════════════════
#  Device-level checks
# ═════════════════════════════════════════════════════════════════════════

proc runStrided(engine: var auto; M, N, K: int; rsa, csa, rsb, csb: int) =
  ## Device-level strided gemm (fp16, EpiIdentity) over RAW dims.
  ## The buffers span the padded tile extent (canary-filled).
  ## D is padded with the canary so the masked store is verified to leave it untouched.
  ## Negative strides rebase the view so the whole padded extent stays addressable.
  let Mp = ((M + 31) div 32) * 32
  let Np = ((N + 31) div 32) * 32
  let Kp = max(((K + 15) div 16) * 16, 16)
  let ab = paddedAB(M, N, K, Mp, Np, Kp, rsa, csa, rsb, csb)
  let pA = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](ab.Ah[0].unsafeAddr),
                          len: ab.Ah.len, off: ab.aBase)
  let pB = PtrArg[uint16](buf: cast[ptr UncheckedArray[uint16]](ab.Bh[0].unsafeAddr),
                          len: ab.Bh.len, off: ab.bBase)
  var D = canaryD(Mp, Np)
  engine.run << (grid: (Np div 32, Mp div 32), blk: (32, 1)) >> (
    "fusedGemm", D,
    (pA, pB, int32(M), int32(N), int32(K), int32(Np),
     int32(rsa), int32(csa), int32(rsb), int32(csb)))
  let refC = gemmRefStrided(M, N, K, rsa, csa, rsb, csb,
                            ab.aBase, ab.bBase, ab.Ah, ab.Bh)
  checkStrided(D, Np, M, N, refC,
    &"STRIDED {M}×{N}×{K} (rsa={rsa}, csa={csa}, rsb={rsb}, csb={csb})")

proc runAxpbyRagged(engine: var auto; M, N, K: int; alpha, beta: float32) =
  ## Ragged strided-C device case: D = α·A·B + β·C over RAW dims.
  ## C and D are padded Mp×Np and canary-filled.
  ## C loads bounded into a register tile (out-of-range lanes never touch C memory).
  ## The masked store leaves the D padding untouched.
  let Mp = ((M + 31) div 32) * 32
  let Np = ((N + 31) div 32) * 32
  let Kp = max(((K + 15) div 16) * 16, 16)
  let ab = paddedAB(M, N, K, Mp, Np, Kp, Kp, 1, Np, 1)
  var C = paddedC(M, N, Mp, Np)
  var D = canaryD(Mp, Np)
  let gridX = (N + 31) div 32
  let gridY = (M + 31) div 32
  engine.run << (grid: (gridX, gridY), blk: (32, 1)) >> (
    "fusedAxpby", D,
    (ab.Ah, ab.Bh, C, alpha, beta, int32(M), int32(N), int32(K),
     int32(Np), int32(Kp), int32(1), int32(Np), int32(1),
     int32(Np), int32(1)))
  let refG = gemmRefStrided(M, N, K, Kp, 1, Np, 1, 0, 0, ab.Ah, ab.Bh)
  var refC = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      refC[m * N + n] = alpha * refG[m * N + n] + beta * C[m * Np + n]
  checkStrided(D, Np, M, N, refC,
    &"AXPBY-STRIDED {M}×{N}×{K} α={alpha} β={beta}")

# ═════════════════════════════════════════════════════════════════════════
#  Host BLIS gemm_strided exercise
# ═════════════════════════════════════════════════════════════════════════

proc checkHostGemmStrided(M, N, K: int; alpha, beta: float32;
                          rsa, csa, rsb, csb, rsc, csc: int) =
  ## Runs the public host launcher on a freshly built fp32 operand set.
  ## A/B/C are allocated exactly at their accessed spans.
  ## C is the exact-span in-place output.
  let ops = hostOperands(M, N, K, rsa, csa, rsb, csb, rsc, csc)
  let refC = gemmRefFp32(M, N, K, alpha, beta,
                         ops.A, ops.aBase, rsa, csa,
                         ops.B, ops.bBase, rsb, csb,
                         ops.C, ops.cBase, rsc, csc)
  gemm_strided(M, N, K, alpha, ops.A[0].addr, rsa, csa,
               ops.B[0].addr, rsb, csb, beta, ops.C[0].addr, rsc, csc)
  checkHostStrided(
    &"HOST gemm_strided {M}×{N}×{K} α={alpha} β={beta} " &
    &"(rsa={rsa}, csa={csa}, rsb={rsb}, csb={csb}, rsc={rsc}, csc={csc})",
    ops.C, ops.cBase, rsc, csc, M, N, refC)

# ═════════════════════════════════════════════════════════════════════════
#  Tests
# ═════════════════════════════════════════════════════════════════════════

proc runTest() =
  var engine = bkMetal.init()
  engine.ingest(stridedMsl)

  const M = 64
  const N = 64
  const K = 32
  runStrided(engine, M, N, K, K, 1, N, 1)          # row-major
  runStrided(engine, M, N, K, 1, M, 1, K)          # col-major
  runStrided(engine, M, N, K, 2 * K, 1, 2 * N, 1)  # 1 out of 2
  runStrided(engine, M, N, K, -K, 1, -N, 1)        # negative A/B strides
  runStrided(engine, 37, 55, 50, 50, 1, 55, 1)     # ragged M/N/K

  # Ragged strided-C: C loads bounded, so the C and D paddings stay canary.
  runAxpbyRagged(engine, 37, 55, 50, 2.0'f32, 4.0'f32)

  # Host BLIS gemm_strided: α/β combos (incl. β = 0 and β ≠ 0), strided C,
  # ragged and small shapes, negative A/B strides, K = 0 (D = β·C).
  checkHostGemmStrided(37, 55, 50, 1.0'f32, 1.0'f32, 50, 1, 55, 1, 55, 1)
  checkHostGemmStrided(65, 89, 71, 2.0'f32, 0.5'f32, 71, 1, 89, 1, 89, 1)
  checkHostGemmStrided(17, 33, 16, 1.0'f32, 0.0'f32, 16, 1, 33, 1, 33, 1)
  checkHostGemmStrided(7, 9, 5, 0.5'f32, -1.0'f32, 5, 1, 9, 1, 9, 1)
  checkHostGemmStrided(32, 64, 32, 1.0'f32, 1.0'f32, 32, 1, 64, 1, 64, 1)   # tiled control
  checkHostGemmStrided(37, 55, 0, 1.0'f32, 2.0'f32, 55, 1, 55, 1, 55, 1)    # K = 0: D = 2·C
  checkHostGemmStrided(37, 55, 50, 1.0'f32, 1.0'f32, 50, 1, 55, 1, 110, 1)  # strided C (1 out of 2)
  checkHostGemmStrided(37, 55, 50, 1.0'f32, 1.0'f32, -50, 1, -55, 1, 55, 1) # negative A/B

when isMainModule:
  runTest()
