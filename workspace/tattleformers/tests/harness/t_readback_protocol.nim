# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Readback protocol smoke test (Metal), a 2-op graph.
##
## Test that the harness readback protocol passes on the smallest graph
## that has an intermediate:
##
## - eval the graph root, one batch walks and enqueues the whole chain
## - readback the intermediate op node on the evaluated path, no second
##   eval and no re-enqueue
##
## `copyToHostFromDevice` never evaluates:
## - a readback of a node whose kernel was never enqueued returns
##   the buffer's uninitialized content
## - an assert must always read a node on an evaluated root's path
## The harness module header states the protocol, this test proves
## the intermediate readback half of the contract.
##
## Run from the worktree root:
##   nim c -r -d:release --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wipM09readback --outdir:build/wipM09readback \
##     workspace/tattleformers/tests/harness/t_readback_protocol.nim

import
  workspace/crucible,
  workspace/ceramic/src/tensors,
  workspace/ceramic/src/layout_algebra,
  workspace/tensorcery/src/device_view,
  workspace/tensorcery/src/tensor_datatypes,
  workspace/tattleformers/src/ops_2i1o/add

proc k_primer*(z, x: distinct DeviceView, shift: static int) =
  ## z = x + shift, one element.
  ##
  ## Why this kernel exists:
  ## - the compilation must hold one enqueue before the first erased-Tensor call
  ## - Tensor views dispatch through the erased interface, whose vtable
  ##   instantiation compiles eval and its batch submit
  ## - submit's finalize macro is a compile-time check erroring on an empty
  ##   codegen frontend
  let
    zv = make_view(z.data +% z.offset, z.layout)
    xv = make_view(x.data +% x.offset, x.layout)
  let s = xv[0].to(float32) + float32(shift)
  zv[0] = s.to(bfloat16)

proc main[E: HwEngine](engine: E) =
  # Primer block first: the primer enqueue must textually precede
  # every erased-Tensor call below. Erased vtable instantiation
  # compiles the submit finalize on that first call. No erased call
  # may sit above the primer in instantiation order.
  block:
    let
      zDev = engine.allocDevice(bfloat16, 1)
      xDev = engine.allocDevice(bfloat16, 1)
      zb = TensorBridge[typeof(zDev), (int,), (int,)](
        data: zDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
      xb = TensorBridge[typeof(xDev), (int,), (int,)](
        data: xDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
    var batch = engine.recordBatch()
    batch.enqueue<<(1, 1)>> k_primer(zb, xb, 3)
    batch.submit()


  const N = 4
  var xHost, yHost: array[N, bfloat16]
  for i in 0 ..< N:
    xHost[i] = (1.5'f32 + 2.0'f32 * float32(i)).to(bfloat16)
    yHost[i] = (0.5'f32 * float32(i + 1)).to(bfloat16)

  let x = newTensor(engine, @[N], kBf16)
  x.copyToDeviceFromHost(xHost)
  let y = newTensor(engine, @[N], kBf16)
  y.copyToDeviceFromHost(yHost)

  # 2-op graph: op1 = x + y, op2 = op1 + y. The intermediate op1
  # is the readback protocol's asserted node.
  let op1 = x + y
  let op2 = op1 + y

  # Step 1: eval the root, one batch, both add kernels enqueue
  # in walk order.
  var roots = @[op2.node]
  op2.engine.eval(roots.toSpan())

  # Step 2: readback the intermediate node on the evaluated path.
  # copyToHostFromDevice does not eval, the op1 kernel already ran
  # inside the single root-eval batch.
  var got1: array[N, bfloat16]
  copyToHostFromDevice(got1, op1)
  var got2: array[N, bfloat16]
  copyToHostFromDevice(got2, op2)
  for i in 0 ..< N:
    let want1 = xHost[i].to(float32) + yHost[i].to(float32)
    let want2 = want1 + yHost[i].to(float32)
    doAssert got1[i].to(float32) == want1.to(bfloat16).to(float32),
      "op1 readback " & $got1[i].to(float32) & " != " & $want1
    doAssert got2[i].to(float32) == want2.to(bfloat16).to(float32),
      "op2 readback " & $got2[i].to(float32) & " != " & $want2

  stderr.writeLine "readback protocol smoke (2-op graph, Metal): all checks passed"

when isMainModule:
  main(gMetalEngine())
