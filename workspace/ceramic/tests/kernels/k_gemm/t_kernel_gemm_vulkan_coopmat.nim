## KHR cooperative-matrix tile GEMM on Vulkan:
## D = alpha·A·B^T + beta·C through the opaque-fragment family.
##
## - one squadron per workgroup, `blk = (32, 1)` on NVIDIA, the tile grid
##   iterated by workgroup ID
## - shapes are multiples of the (32, 32, 16) tile
## - the KHR load's inner-dimension stride is implicitly 1 and its SPIR-V
##   layout operand must be a constant instruction, so every operand
##   carries a statically-typed stride
##   (see the C14D06 brief and `gemm_via_khr_coopmat`)
##
## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_gemm/t_kernel_gemm_vulkan_coopmat.nim

import std/[random, times]
import workspace/crucible
import workspace/fp_conv/fp_conv
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_gemm/kernel_gemm_template

# ═══════════════════════════════════════════════════════════════════════
#  Test cases: tile-multiple shapes, contiguous storage, static strides
# ═══════════════════════════════════════════════════════════════════════

# The KHR load/store layout operand must be a constant instruction.
# The orientation is decided from the static stride types.
# make_layout with literal strides lifts them to compile-time constants.

const KhrGemmHighlights = (
  # row-major storage on every operand, the alpha/beta legs
  (alpha: 2.0'f32, A: make_layout((32, 32), (32, 1)), B: make_layout((32, 32), (32, 1)), beta: 1.0'f32, C: make_layout((32, 32), (32, 1))),
  (alpha: 1.0'f32, A: make_layout((32, 32), (32, 1)), B: make_layout((32, 32), (32, 1)), beta: 0.0'f32, C: make_layout((32, 32), (32, 1))),
  (alpha: 0.5'f32, A: make_layout((32, 32), (32, 1)), B: make_layout((32, 32), (32, 1)), beta: 2.0'f32, C: make_layout((32, 32), (32, 1))),
  # column-major storage on every operand, the transposed load path
  (alpha: 1.0'f32, A: make_layout((32, 32), (1, 32)), B: make_layout((32, 32), (1, 32)), beta: 1.0'f32, C: make_layout((32, 32), (1, 32))),
  # mixed orientations
  (alpha: 1.0'f32, A: make_layout((32, 32), (32, 1)), B: make_layout((32, 32), (1, 32)), beta: 1.0'f32, C: make_layout((32, 32), (1, 32))),
  (alpha: 2.0'f32, A: make_layout((32, 32), (1, 32)), B: make_layout((32, 32), (32, 1)), beta: 0.0'f32, C: make_layout((32, 32), (32, 1))),
  # multi-tile shapes, the tile grid needs grid-stride loops
  (alpha: 2.0'f32, A: make_layout((64, 32), (32, 1)), B: make_layout((64, 32), (32, 1)), beta: 1.0'f32, C: make_layout((64, 64), (64, 1))),
  (alpha: 1.0'f32, A: make_layout((96, 64), (64, 1)), B: make_layout((96, 64), (1, 96)), beta: 0.5'f32, C: make_layout((96, 96), (1, 96))),
  (alpha: 1.0'f32, A: make_layout((64, 96), (1, 64)), B: make_layout((32, 96), (96, 1)), beta: 2.0'f32, C: make_layout((64, 32), (32, 1))),
  # K deeper than one k-tile
  (alpha: 1.0'f32, A: make_layout((32, 64), (64, 1)), B: make_layout((32, 64), (64, 1)), beta: 1.0'f32, C: make_layout((32, 32), (32, 1))),
  (alpha: 2.0'f32, A: make_layout((32, 96), (1, 32)), B: make_layout((32, 96), (96, 1)), beta: 1.0'f32, C: make_layout((32, 32), (1, 32))))

template khrGemmFuzzStridePerms(m, n, k: static int): untyped =
  ## Every row-major/column-major combination on A (m, k), B (n, k),
  ## C (m, n). alpha = 1, beta = 0, the geometry is the subject
  (
    (alpha: 1.0'f32, A: make_layout((m, k), (k, 1)), B: make_layout((n, k), (k, 1)), beta: 0.0'f32, C: make_layout((m, n), (n, 1))),
    (alpha: 1.0'f32, A: make_layout((m, k), (k, 1)), B: make_layout((n, k), (k, 1)), beta: 0.0'f32, C: make_layout((m, n), (1, m))),
    (alpha: 1.0'f32, A: make_layout((m, k), (k, 1)), B: make_layout((n, k), (1, n)), beta: 0.0'f32, C: make_layout((m, n), (n, 1))),
    (alpha: 1.0'f32, A: make_layout((m, k), (k, 1)), B: make_layout((n, k), (1, n)), beta: 0.0'f32, C: make_layout((m, n), (1, m))),
    (alpha: 1.0'f32, A: make_layout((m, k), (1, m)), B: make_layout((n, k), (k, 1)), beta: 0.0'f32, C: make_layout((m, n), (n, 1))),
    (alpha: 1.0'f32, A: make_layout((m, k), (1, m)), B: make_layout((n, k), (k, 1)), beta: 0.0'f32, C: make_layout((m, n), (1, m))),
    (alpha: 1.0'f32, A: make_layout((m, k), (1, m)), B: make_layout((n, k), (1, n)), beta: 0.0'f32, C: make_layout((m, n), (n, 1))),
    (alpha: 1.0'f32, A: make_layout((m, k), (1, m)), B: make_layout((n, k), (1, n)), beta: 0.0'f32, C: make_layout((m, n), (1, m))))

const KhrGemmFuzzCases = (
  khrGemmFuzzStridePerms(32, 32, 32),
  khrGemmFuzzStridePerms(64, 64, 32),
  khrGemmFuzzStridePerms(32, 32, 64))

# ═══════════════════════════════════════════════════════════════════════
#  The suite, one leg per operand type the KHR combos cover
# ═══════════════════════════════════════════════════════════════════════

proc main =
  var engine = newEngine(runVulkan)
  let devName = engine.deviceName()
  var rng: Rand
  let seedVal = uint32(getTime().toUnix() and (1'i64 shl 32 - 1)) # unixTime mod 2^32
  rng.seed(int64(seedVal))
  echo "test_k_tile_gemm_vulkan_coopmat xoroshiro128+ seed: ", seedVal
  # KHR 16x16x16 f16/bf16 with f32 accumulation, the portable classic combo.
  # Tile (32, 32, 16) = 2x2 accumulator fragments per squadron.
  # Mesa's lavapipe, the other KHR device on this box, fails numerically:
  # its coopMatStore writes one element per lane and the 8x8x8 muladd
  # half-accumulates. A Mesa limitation, see the C14D06 brief.
  discard devName
  block:
    const mmaF16 = tiledMma(float32, float16, float16,
        KHR_16x16x16_F32F16F16F32, 32, 32, 16)
    rng.runGemmChecksAtomKhr(engine, mmaF16, KhrGemmHighlights, Iters = 2)
    rng.runGemmFuzzAtomKhr(engine, mmaF16, KhrGemmFuzzCases)
  block:
    const mmaBf16 = tiledMma(float32, bfloat16, bfloat16,
        KHR_16x16x16_F32BF16BF16F32, 32, 32, 16)
    rng.runGemmChecksAtomKhr(engine, mmaBf16, KhrGemmHighlights, Iters = 2)
    rng.runGemmFuzzAtomKhr(engine, mmaBf16, KhrGemmFuzzCases)

when isMainModule:
  main()
