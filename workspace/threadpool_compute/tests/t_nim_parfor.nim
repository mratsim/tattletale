# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## threadpool_compute parallelFor over the Nim wrapper
##
## Test that parallelFor computes a value-checked parallel sum reduction
## across pool sizes, chunks splitting lazily to idle workers
##
## Run:
##   nim c --path:../.. -r tests/t_nim_parfor.nim

import std/cpuinfo
import std/syncio
import std/atomics

import threadpool_compute

const numItems = 1_000_000

type SumEnv = object
  data: ptr UncheckedArray[int64]
  total: ptr Atomic[int64]

proc sumBody(idx: int32, env: ptr SumEnv) {.cdecl.} =
  discard env.total[].fetchAdd(env.data[][idx])

proc runSum(numThreads: int32) =
  let tp = threadpool_new(numThreads)
  var data = newSeq[int64](numItems)
  for i in 0 ..< numItems:
    data[i] = int64(i)
  var total: Atomic[int64]
  var env = SumEnv(data: cast[ptr UncheckedArray[int64]](data[0].addr),
                   total: total.addr)
  syncScope:
    parallelFor(0, int32 numItems, env, sumBody)
  let expected = int64(numItems) * (int64(numItems) - 1) div 2
  doAssert total.load == expected,
    "sum mismatch with " & $numThreads & " workers: " & $total.load &
    " != " & $expected
  stdout.writeLine "t_nim_parfor: sum over " & $numItems & " items, " &
    $numThreads & " workers: OK"
  threadpool_shutdown(tp)

proc main() =
  runSum(2)
  runSum(8)
  runSum(int32 countProcessors())

main()
