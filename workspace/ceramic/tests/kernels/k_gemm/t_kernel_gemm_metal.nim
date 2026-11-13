## Test that tile-algebra gemm computes D = alpha·A·B^T + beta·C on Metal
## for the operand types the backend's atoms cover, over explicit cases
## and generated stride permutations.
##
## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_gemm/t_kernel_gemm_metal.nim

import std/[random, times]
import workspace/crucible
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_gemm/kernel_gemm_template

# ═══════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════

const GemmHighlights = [
  # stride permutations at the atom tile size, (32, 32, 32)
  (alpha: 2.0'f32, A: ((32, 32), (32, 1)), B: ((32, 32), (32, 1)), beta: 1.0'f32, C: ((32, 32), (32, 1))),
  (alpha: 2.0'f32, A: ((32, 32), (1, 32)), B: ((32, 32), (1, 32)), beta: 1.0'f32, C: ((32, 32), (1, 32))),
  (alpha: 1.0'f32, A: ((32, 32), (1, 32)), B: ((32, 32), (32, 1)), beta: 0.0'f32, C: ((32, 32), (1, 32))),
  (alpha: 1.0'f32, A: ((32, 32), (32, 1)), B: ((32, 32), (1, 32)), beta: 2.0'f32, C: ((32, 32), (32, 1))),
  # non-contiguous skips, rows strided apart and inner skips
  (alpha: 1.0'f32, A: ((48, 40), (121, 3)), B: ((24, 40), (1, 80)), beta: 1.0'f32, C: ((48, 24), (24, 1))),
  (alpha: 2.0'f32, A: ((64, 32), (65, 1)), B: ((48, 32), (2, 96)), beta: 1.0'f32, C: ((64, 48), (48, 1))),
  (alpha: 1.0'f32, A: ((50, 30), (1, 60)), B: ((40, 30), (31, 1)), beta: 1.0'f32, C: ((50, 40), (1, 50))),
  (alpha: 1.0'f32, A: ((33, 33), (2, 67)), B: ((33, 33), (67, 2)), beta: 1.0'f32, C: ((33, 33), (1, 33))),
  (alpha: 1.0'f32, A: ((128, 64), (64, 1)), B: ((128, 64), (1, 128)), beta: 0.0'f32, C: ((128, 128), (128, 1))),
  # ragged odd shapes
  (alpha: 1.0'f32, A: ((17, 23), (23, 1)), B: ((19, 23), (1, 19)), beta: 1.0'f32, C: ((17, 19), (19, 1))),
  (alpha: 1.0'f32, A: ((65, 62), (62, 1)), B: ((63, 62), (63, 1)), beta: 2.0'f32, C: ((65, 63), (63, 1))),
  (alpha: 2.0'f32, A: ((24, 96), (96, 1)), B: ((24, 96), (1, 24)), beta: 1.0'f32, C: ((24, 24), (24, 1))),
  (alpha: 1.0'f32, A: ((40, 96), (1, 96)), B: ((96, 96), (96, 1)), beta: 1.0'f32, C: ((40, 96), (96, 1))),
  # sub-tile shapes, tensors smaller than one mma tile
  (alpha: 1.0'f32, A: ((8, 8), (8, 1)), B: ((8, 8), (8, 1)), beta: 0.0'f32, C: ((8, 8), (8, 1))),
  (alpha: 1.0'f32, A: ((12, 28), (28, 1)), B: ((20, 28), (1, 20)), beta: 1.0'f32, C: ((12, 20), (20, 1))),
  (alpha: 2.0'f32, A: ((2, 16), (16, 1)), B: ((130, 16), (1, 130)), beta: 1.0'f32, C: ((2, 130), (130, 1))),
  # multi-tile shapes, the tile grid needs grid-stride loops
  (alpha: 2.0'f32, A: ((70, 100), (100, 1)), B: ((50, 100), (1, 50)), beta: 1.0'f32, C: ((70, 50), (50, 1))),
  (alpha: 1.0'f32, A: ((96, 96), (1, 96)), B: ((96, 96), (96, 1)), beta: 0.5'f32, C: ((96, 96), (1, 96))),
  # alpha scaling and beta accumulation
  (alpha: 0.5'f32, A: ((48, 40), (40, 1)), B: ((24, 40), (1, 24)), beta: 2.0'f32, C: ((48, 24), (24, 1))),
  (alpha: 1.0'f32, A: ((17, 23), (23, 1)), B: ((19, 23), (19, 1)), beta: 1.0'f32, C: ((17, 19), (19, 1)))]

const GemmFuzzCases = genGemmCases((32, 32, 32), (40, 45, 41))

# ═══════════════════════════════════════════════════════════════
#  The suite, one leg per operand type the backend's atoms cover
# ═══════════════════════════════════════════════════════════════

proc main =
  var engine = newEngine(runMetal)
  var rng: Rand
  let seedVal = uint32(getTime().toUnix() and (1'i64 shl 32 - 1)) # unixTime mod 2^32
  rng.seed(int64(seedVal))
  echo "test_k_tile_gemm_metal xoroshiro128+ seed: ", seedVal
  # Metal simdgroup atoms: f16 and f32 operands, both accumulate in f32
  rng.runGemmChecks(engine, float32, float16, 32, 32, 16, GemmHighlights, Iters = 2)
  rng.runGemmChecks(engine, float32, float32, 32, 32, 16, GemmHighlights, Iters = 2)
  rng.runGemmFuzz(engine, float32, float16, 32, 32, 16, GemmFuzzCases)
  rng.runGemmFuzz(engine, float32, float32, 32, 32, 16, GemmFuzzCases)

when isMainModule:
  main()
