# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Baseline benchmark: full-sum reduction bandwidth on Metal, GB/s.
# - replicate: the shipped `sum(dst, src)` today
# - 1 group: one threadgroup folds the whole tensor
# - 2-pass ceiling: raw Metal kernels, one partial per threadgroup +
#   one-threadgroup finalize, both dispatches in one command buffer
# Run:
#   nim c -r -d:release --outdir:build/bench \
#     workspace/ceramic/bench/reduce_sum_gpu/bench_sum_metal.nim (from tattletale)

import std/[os, strformat, strutils]
import workspace/crucible
import workspace/crucible/src/runtime/engines/metal
import workspace/ceramic/bench/reduce_sum_gpu/bench_sum_template
import workspace/ceramic/bench/reduce_sum_gpu/sum_twopasses_metal{.all.}

const ProblemSizes = [1_000, 16_000, 100_000, 200_000, 1_000_000, 4_000_000, 16_000_000]

proc benchTwoPasses*(engine: MetalEngine, np, G, iters: int): auto =
  ## Two-pass ceiling, both dispatches of a sum in one command buffer
  ##
  ## - one partial per threadgroup, one threadgroup finalizes
  ## - each timed iteration is one end-to-end sum with commit+wait
  let one = 1  # runtime int for the layouts to stay dynamic
  var
    srcHost = makeHostTensor(float32, (np, one), (one, np))
    partialsHost = makeHostTensor(float32, (G, one), (one, G))
    dstHost = makeHostTensor(float32, (1, 1), (1, 1))
  fillSumInputs(srcHost.view(), np)
  if engine.getArtifact() != sumTwoPassSrc:
    engine.ingest(sumTwoPassSrc)
  let
    srcDev = engine.bridge(srcHost)
    partialsDev = engine.bridge(partialsHost)
    dstDev = engine.bridge(dstHost)
    srcBuf = srcDev.data.mem
    partialsBuf = partialsDev.data.mem
    dstBuf = dstDev.data.mem
  # warmup, one end-to-end sum
  launchSumTwoPasses(engine, partialsBuf, srcBuf, dstBuf, uint64(np), uint32(G))
  engine.copyToHostFromDevice(addr dstHost.data[0], dstDev.data, 1)
  # timed, iters end-to-end sums back-to-back
  benchSum("2-pass", G, np, iters):
    for _ in 0 ..< iters:
      launchSumTwoPasses(engine, partialsBuf, srcBuf, dstBuf, uint64(np), uint32(G))
  engine.copyToHostFromDevice(addr dstHost.data[0], dstDev.data, 1)
  dstHost

proc main() =
  var engine = bkMetal.newEngine()
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
      let dRep = benchReplicate(engine, np, iters)
      let dOne = benchOneThreadgroup(engine, np, iters)
      # a G wider than the input sums empty threadgroups, skip it
      let dTwo256 = benchTwoPasses(engine, np, min(256, np), iters)
      let dTwo1024 = benchTwoPasses(engine, np, min(1024, np), iters)
      # the implementations must agree with the host reference
      let expect = sumExpect(np)
      for (name, d) in [("replicate", dRep), ("1 group", dOne),
                        ("2-pass G=256", dTwo256), ("2-pass G=1024", dTwo1024)]:
        doAssert abs(d.data[0].float64 - expect) < 1e-2 * max(1.0, abs(expect)),
          &"{name} sum wrong at {np}: got {d.data[0]}, expect {expect}"
      separator()
  separator()

when isMainModule:
  main()
