# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Benchmark:
##   k_tile_gemm throughput on Metal
##
## Run:
##   nim c -r -d:release --outdir:build/bench \
##     workspace/ceramic/benchmark/bench_tile_gemm_metal.nim (from tattletale)

import std/[os, strutils]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/layout_algebra/layouts
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/benchmark/bench_utils
import workspace/ceramic/benchmark/bench_tile_gemm_template
import workspace/fp_conv/fp_conv

{.experimental: "callOperator".}

const
  ProblemSizes = [1024, 2048, 4096]
  WarmupSeconds = 1.0
  Rounds = 3

proc main =
  var engine = bkMetal.init()
  # CLI size filter: bench 4096 runs only that size
  var wanted: seq[int]
  for p in commandLineParams():
    wanted.add parseInt(p)
  for mnk in ProblemSizes:
    if wanted.len == 0 or mnk in wanted:
      engine.tileGemmBench(mnk, mnk, mnk, WarmupSeconds, Rounds)

when isMainModule:
  main()
