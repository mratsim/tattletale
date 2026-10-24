# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Run:
#   nim c -r -d:release --outdir:build/bench workspace/ceramic/bench/gemm_gpu/bench_gemm_cuda.nim
#
# The MLX baseline runs out-of-process, python mlx with the CUDA backend:
#   python3 workspace/ceramic/bench/gemm_gpu/bench_gemm_mlx.py
#
# Correctness is owned by the kernel test suite
# (tests/kernels/k_gemm/t_kernel_gemm_cuda.nim), the suite launches
# and verifies the CUDA gemm end to end.
#
# Unlike the Metal bench, no cross-check here:
# - the Metal bench cross-checks implementations on shared inputs
# - this bench has one implementation only

import std/[os, strutils]
import workspace/crucible
import workspace/ceramic/bench/gemm_gpu/bench_gemm_template

const ProblemSizes = [256, 512, 1024, 2048, 4096]

proc main() =
  var engine = newEngine(runCuda)
  var wanted: seq[int]
  for p in commandLineParams():
    wanted.add parseInt(p)
  separator()
  for mnk in ProblemSizes:
    if wanted.len == 0 or mnk in wanted:
      let iters = if mnk >= 4096: 20 elif mnk >= 2048: 100 else: 1000
      discard benchCeramic(engine, mnk, mnk, mnk, iters)
      separator()
  separator()

when isMainModule:
  main()
