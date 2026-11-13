# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/random
from workspace/crucible import to
import ../tensorcery_testutils

export tensorcery_testutils


proc refAdd(dst: var openArray[bfloat16], a, b: openArray[bfloat16]) =
  for i in 0 ..< dst.len:
    dst[i] = (a[i].to(float32) + b[i].to(float32)).to(bfloat16)

proc runAddTest*(run: static Runtime, name: string) =
  var rng = initRand(0xFACADE)
  for shape in [ @[2, 4], @[5, 13], @[2, 3, 7] ]:
    var size = 1
    for dim in shape:
      size *= dim

    var a = rng.randomSeq(bfloat16, size)
    var b = rng.randomSeq(bfloat16, size)
    var dst = newSeq[bfloat16](size)
    dst.refAdd(a, b)

    let x = a.toTensor(run, shape)
    let y = b.toTensor(run, shape)

    doAssert (x + y).allClose(dst.toTensor(run, shape)), name & ": add " & $shape

  block: # allClose evaluates its tested argument once: the argument of a quote is re-substituted at every use
    let x = @[1.0'f32, 2, 3, 4].toTensor(run, @[4])
    var pending = @[x]
    doAssert pending.pop().allClose(x), name & ": allClose binds its tested argument once"

  block: # eval evaluates its argument once: a template re-substitutes each argument at every use
    let x = @[1.0'f32, 2, 3, 4].toTensor(run, @[4])
    var pending = @[x]
    eval(pending.pop())
    doAssert (x + x).item(float32) == 2.0'f32, name & ": eval binds its argument once"

  when defined(TTT_tensorgraphMetrics):
    block: # Dedup: a second eval of the same graph skips the already-allocated nodes
      let x = @[1.0'f32, 2, 3, 4].toTensor(run, @[4])
      let y = x + x
      doAssert y.item(float32) == 2.0'f32
      let afterFirst = graphCounters()
      doAssert y.item(float32) == 2.0'f32
      let afterSecond = graphCounters()
      doAssert afterSecond.evals - afterFirst.evals == 1
      doAssert afterSecond.nodesDispatched - afterFirst.nodesDispatched == 0
      doAssert afterSecond.kernelsEnqueued - afterFirst.kernelsEnqueued == 0
      doAssert afterSecond.nodesSkippedAllocated - afterFirst.nodesSkippedAllocated == 1

  echo "tensorcery add (", name, "): all checks passed"
