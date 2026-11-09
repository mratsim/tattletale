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
