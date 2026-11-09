## Test that tile-algebra gemm computes D = alpha·A·B^T + beta·C on Vulkan
## (the GLSL ingestion path) with the fp32 generic-tile atom: the shared
## GemmHighlights at Iters = 2, one pass over the generated GemmFuzzCases
## for (32, 32, 32) and (40, 45, 41), ragged edges and sub-tile shapes included.
##
## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_gemm/t_kernel_gemm_vulkan.nim

import std/[random, times]
import workspace/crucible
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_gemm/kernel_gemm_template

static:
  # Fragment layout tables of VULKAN_32x32x2_F32, hand-computed:
  # A register v: cell (m, k) = (v mod 4, v div 4), offset m + 32·k,
  # B register v: (n, k) = (v mod 8, v div 8), offset n + 32·k,
  # C register i: (m, n) = (i mod 4, i div 4), offset m + 32·n.
  # Thread 0 coordinates: rowBlock = 0, colBlock = 0.
  doAssert VULKAN_32x32x2_F32.valuesPerThread(opA).toInt() == 8
  doAssert VULKAN_32x32x2_F32.valuesPerThread(opB).toInt() == 16
  doAssert VULKAN_32x32x2_F32.valuesPerThread(opC).toInt() == 32
  for v in 0 ..< 8:
    doAssert VULKAN_32x32x2_F32_aLayout(((0, 0), (v mod 4, v div 4))).toInt() ==
      (v mod 4) + 32 * (v div 4)
  for v in 0 ..< 16:
    doAssert VULKAN_32x32x2_F32_bLayout(((0, 0), (v mod 8, v div 8))).toInt() ==
      (v mod 8) + 32 * (v div 8)
  for i in 0 ..< 32:
    doAssert VULKAN_32x32x2_F32_cLayout(((0, 0), (i mod 4, i div 4))).toInt() ==
      (i mod 4) + 32 * (i div 4)

# ═══════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════

const GemmFuzzCases = genGemmCases((32, 32, 32), (40, 45, 41))

# ═══════════════════════════════════════════════════════════════════════
#  The suite, the fp32 leg the Vulkan generic-tile atom covers
# ═══════════════════════════════════════════════════════════════════════

proc main =
  var engine = newEngine(runVulkan)
  var rng: Rand
  let seedVal = uint32(getTime().toUnix() and (1'i64 shl 32 - 1)) # unixTime mod 2^32
  rng.seed(int64(seedVal))
  echo "t_kernel_gemm_vulkan xoroshiro128+ seed: ", seedVal
  rng.runGemmChecks(engine, float32, float32, 32, 32, 2, GemmHighlights, Iters = 2, lang = "glsl")
  # GLSL route only: the SPIR-V printer cannot walk the var Tile params of the tile load/store
  rng.runGemmFuzz(engine, float32, float32, 32, 32, 2, GemmFuzzCases, lang = "glsl")

when isMainModule:
  main()
