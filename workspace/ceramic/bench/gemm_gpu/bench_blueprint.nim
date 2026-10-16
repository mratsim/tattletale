# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#             Benchmark blueprint
#
# ############################################################

import std/[cpuinfo, monotimes, strformat, strutils, times]

# ═══════════════════════════════════════════════════════════════
#  Warmup
# ═══════════════════════════════════════════════════════════════

proc warmup*() =
  # Warmup - make sure cpu is on max perf
  let start = cpuTime()
  var foo = 123
  for i in 0 ..< 300_000_000:
    foo += i*i mod 456
    foo = foo mod 789

  # Compiler shouldn't optimize away the results as cpuTime rely on sideeffects
  let stop = cpuTime()
  echo &"Warmup: {stop - start:>4.4f} s, result {foo} (displayed to avoid compiler optimizing warmup away)\n"

# ═══════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════

proc separator*(length = 118) =
  echo "-".repeat(length)

proc report*(op, config: string, startTime, stopTime: MonoTime, iters: int, flops: float64 = 0): int64 =
  let ns = inNanoseconds(stopTime - startTime) div iters
  let throughput = 1e9 / ns.float
  if flops > 0:
    echo &"{op:<40} {config:<16} {throughput:>15.3f} ops/s     {ns:>9} ns/op     {flops / (ns.float / 1e9) / 1e12:>8.2f} TFlops"
  else:
    echo &"{op:<40} {config:<16} {throughput:>15.3f} ops/s     {ns:>9} ns/op"
  ns

template measure*(iters: int, startTime, stopTime: untyped, body: untyped): untyped =
  let startTime = getMonotime()
  for _ in 0 ..< iters:
    body
  let stopTime = getMonotime()

template bench*(op, config: string, iters: int, flops: float64 = 0, body: untyped): int64 =
  block:
    let benchStart = getMonotime()
    for _ in 0 ..< iters:
      body
    let benchStop = getMonotime()
    report(op, config, benchStart, benchStop, iters, flops)
