# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensorcery CPU backend: deferred ops over the flow-events drain
##
## Test that CPU add and scalar-multiply recorded on the context compute correct
## values through one synchronous eval, with the completion counter asserting
## every recorded op ran before eval returns
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --nimcache:nimcache/t_cpu_backend workspace/tensorcery/tests/t_cpu_backend.nim

import std/atomics
import std/importutils
import pkg/iface
import workspace/data_structures/src/small_seqs
import workspace/tensorcery
import workspace/tensorcery/src/backends/cpu

privateAccess(Tensor)

proc main() =
  let ctx = Ctx()

  block evalSynchronous:
    let a = toTensor(ctx, @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
    let b = toTensor(ctx, @[10.0'f32, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3)
    let c = a + b
    let d = c * 2.0

    doAssert ctx.getPendingCount() == 2
    doAssert completionCount.load == 0, "no op may run before eval"

    resetCompletionCount()
    eval(ctx)
    doAssert completionCount.load == 2, "eval must complete every recorded op"
    doAssert ctx.getPendingCount() == 0

    doAssert toSeq[float32](c) == @[11.0'f32, 22.0, 33.0, 44.0, 55.0, 66.0]
    doAssert toSeq[float32](d) == @[22.0'f32, 44.0, 66.0, 88.0, 110.0, 132.0]

  block readMaterializes:
    let a = toTensor(ctx, @[1.0'f32, 2.0, 3.0], 3)
    let b = a * 3.0
    doAssert ctx.getPendingCount() == 1

    resetCompletionCount()
    doAssert toSeq[float32](b) == @[3.0'f32, 6.0, 9.0]
    doAssert completionCount.load == 1, "reading a pending tensor evals the context"
    doAssert ctx.getPendingCount() == 0

  block erasedFaceRoundTrip:
    let a = toTensor(ctx, @[1.0'f32, 2.0], 2)
    let c = a + a
    discard toSeq[float32](c)
    let impl = c.impl.to(TensorImpl[float32, bkCpu])
    doAssert impl != nil
    doAssert impl.shape == [2'i32]
    doAssert impl.storage == @[2.0'f32, 4.0]
    doAssert impl.seqNo != 0, "a computed tensor carries its producing op id"

  block rank4:
    let a = toTensor(ctx, @[1.0'f32, 2, 3, 4, 5, 6, 7, 8], 1, 2, 2, 2)
    let b = toTensor(ctx, @[10.0'f32, 20, 30, 40, 50, 60, 70, 80], 1, 2, 2, 2)
    let c = a + b
    doAssert toSeq[float32](c) == @[11.0'f32, 22, 33, 44, 55, 66, 77, 88]

  stdout.writeLine "t_cpu_backend: OK"

main()
