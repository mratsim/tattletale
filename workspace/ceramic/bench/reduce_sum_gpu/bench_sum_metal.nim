# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: full-sum reduction bandwidth on Metal, GB/s.
# - atomic: the shipped `sum(dst, src)`, grid-stride strips banked atomically
# - 1 group: one threadgroup folds the whole tensor
# Run:
#   nim c -r -d:release --outdir:build/bench \
#     workspace/ceramic/bench/reduce_sum_gpu/bench_sum_metal.nim (from tattletale)

import std/[os, strformat, strutils]
import workspace/crucible
import workspace/ceramic/bench/reduce_sum_gpu/bench_sum_template

const ProblemSizes = [1_000, 16_000, 100_000, 200_000, 1_000_000, 4_000_000, 16_000_000]

proc main() =
  var engine = newEngine(runMetal)
  echo "full sum, Metal, blk = (32, 1)"
  var wanted: seq[int]
  for p in commandLineParams():
    wanted.add parseInt(p)
  separator()
  for np in ProblemSizes:
    if wanted.len == 0 or np in wanted:
      let iters = if np <= 200_000: 200
                  elif np <= 1_000_000: 50
                  else: 20
      let dRep = benchSumCeramic(engine, np, iters)
      let dOne = benchOneThreadgroup(engine, np, iters)
      # the implementations must agree with the host reference
      let expect = sumExpect(np)
      # the sumLoop rows run `iters` sums per launch, the accumulated
      # total divides back to the per-sum value
      for (name, d, reps) in [("atomic", dRep, iters), ("1 group", dOne, iters)]:
        doAssert abs(d.data[0].float64 / reps.float - expect) < 1e-2 * max(1.0, abs(expect)),
          &"{name} sum wrong at {np}: got {d.data[0]}, expect {expect}"
      separator()
  separator()

when isMainModule:
  main()
