# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Batch enqueue shapes beyond k_add's three-view form.
##
## Test that one batch enqueues kernels taking a static param, a DeviceMem-leaf struct arg, and a pointer-formal metadata buffer.
## Each kernel is judged against a host reference.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wipM02 --outdir:build/wip \
##     workspace/tattleformers/tests/t_batch_enqueue_poc.nim

import std/math
import workspace/crucible
import workspace/ceramic
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/ptr_arithmetic
import workspace/tensorcery/src/device_view
import workspace/tensorcery/src/tensor_datatypes

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════════════
#  Probe kernels
# ═══════════════════════════════════════════════════════════════════════

type PocBias* = object
  ## Device-side descriptor of the bias table: the kernel reads the buffer
  ## through the pointer leaf, `count` bounds the walk.
  bias*: ptr UncheckedArray[float32]
  count*: int

proc k_poc_static*(z, x: distinct DeviceView, shift: static int) =
  ## z = x + shift, one element per thread, grid-stride over the flat range.
  ## bfloat16 sums promote to float32 and round back to bfloat16 inline.
  let
    zv = make_view(z.data +% z.offset, z.layout)
    xv = make_view(x.data +% x.offset, x.layout)
  let
    numel = size(zv)
    tid = thread_position_in_grid.x
    stride = threadgroups_per_grid.x * threads_per_threadgroup.x
  for i in countup(tid, numel - 1, stride):
    let s = xv[i].to(float32) + float32(shift)
    when zv.T is bfloat16:
      zv[i] = s.to(bfloat16)
    else:
      zv[i] = s

proc k_poc_struct*(z, x: distinct DeviceView, desc: PocBias) =
  ## z = x + bias[i mod count]:
  ##
  ## - one element per thread, grid-stride
  ## - bfloat16 sums promote to float32 and round back to bfloat16 inline
  let
    zv = make_view(z.data +% z.offset, z.layout)
    xv = make_view(x.data +% x.offset, x.layout)
  let
    numel = size(zv)
    tid = thread_position_in_grid.x
    stride = threadgroups_per_grid.x * threads_per_threadgroup.x
  for i in countup(tid, numel - 1, stride):
    let s = xv[i].to(float32) + desc.bias[i mod desc.count]
    when zv.T is bfloat16:
      zv[i] = s.to(bfloat16)
    else:
      zv[i] = s

proc k_poc_ids*(z, x: distinct DeviceView, ids: ptr UncheckedArray[int32]) =
  ## Mini gather: z[i] = x[ids[i]].
  ##
  ## - one element per thread, grid-stride
  ## - ids binds a DeviceMem buffer to the pointer formal: the launch
  ##   path for typed metadata buffers such as token ids
  ## - the dynamic index exercises the gather's indexing path
  let
    zv = make_view(z.data +% z.offset, z.layout)
    xv = make_view(x.data +% x.offset, x.layout)
  let
    numel = size(zv)
    tid = thread_position_in_grid.x
    stride = threadgroups_per_grid.x * threads_per_threadgroup.x
  for i in countup(tid, numel - 1, stride):
    zv[i] = xv[int(ids[i])]

# ═══════════════════════════════════════════════════════════════════════
#  Enqueue shapes under test
# ═══════════════════════════════════════════════════════════════════════

proc runProbe[E: HwEngine](engine: E) =
  const N = 64
  const Blk = 32
  const Grid = ceil_div(N, Blk)

  # seeded inputs, bf16 rounding applied identically on both sides
  var xHost = newSeq[bfloat16](N)
  for i in 0 ..< N:
    xHost[i] = (float32(i) * 0.25 - 8.0).to(bfloat16)

  let xDev = engine.allocDevice(bfloat16, N)
  engine.copyToDeviceFromHost(xDev, xHost)
  let xb = TensorBridge[typeof(xDev), (int,), (int,)](
    data: xDev, offset: 0, layout: make_layout((int(N),), (int(1),)))

  # 1. static param: z = x + 3
  let zDevS = engine.allocDevice(bfloat16, N)
  let zbS = TensorBridge[typeof(zDevS), (int,), (int,)](
    data: zDevS, offset: 0, layout: make_layout((int(N),), (int(1),)))

  # 2. DeviceMem-leaf struct arg: z = x + bias[i mod 4]. The launch struct
  # lists the device descriptor's fields in the same order.
  # DeviceMem leaves explode to buffer bindings. `count` packs as bytes.
  const biasHost = [0.5'f32, -1.25'f32, 2.0'f32, 7.5'f32]
  let biasDev = engine.allocDevice(float32, biasHost.len)
  engine.copyToDeviceFromHost(biasDev, biasHost)
  type BiasBridge = object
    bias: typeof(biasDev)
    count: int
  let zDevB = engine.allocDevice(bfloat16, N)
  let zbB = TensorBridge[typeof(zDevB), (int,), (int,)](
    data: zDevB, offset: 0, layout: make_layout((int(N),), (int(1),)))

  # 3. DeviceMem buffer binding to a pointer formal: z[i] = x[ids[i]].
  # 7 is coprime with 64 so the ids form a permutation of 0 ..< N
  var idsHost = newSeq[int32](N)
  for i in 0 ..< N:
    idsHost[i] = int32((i * 7) mod N)
  let idsDev = engine.allocDevice(int32, N)
  engine.copyToDeviceFromHost(idsDev, idsHost)
  let zDevG = engine.allocDevice(bfloat16, N)
  let zbG = TensorBridge[typeof(zDevG), (int,), (int,)](
    data: zDevG, offset: 0, layout: make_layout((int(N),), (int(1),)))

  # one recordBatch per test, one submit: the artifact compiles once
  var batch = engine.recordBatch()
  batch.enqueue<<(Grid, Blk)>> k_poc_static(zbS, xb, 3)
  batch.enqueue<<(Grid, Blk)>> k_poc_struct(
    zbB, xb, BiasBridge(bias: biasDev, count: biasHost.len))
  batch.enqueue<<(Grid, Blk)>> k_poc_ids(zbG, xb, idsDev)
  batch.submit()

  var gotS = newSeq[bfloat16](N)
  engine.copyToHostFromDevice(gotS, zDevS)
  for i in 0 ..< N:
    let expected = (xHost[i].to(float32) + 3.0'f32).to(bfloat16)
    doAssert gotS[i].to(float32) == expected.to(float32),
      "static param " & $i & ": got " & $gotS[i].to(float32) &
      ", expected " & $expected.to(float32)

  var gotB = newSeq[bfloat16](N)
  engine.copyToHostFromDevice(gotB, zDevB)
  for i in 0 ..< N:
    let expected = (xHost[i].to(float32) + biasHost[i mod biasHost.len]).to(bfloat16)
    doAssert gotB[i].to(float32) == expected.to(float32),
      "struct arg " & $i & ": got " & $gotB[i].to(float32) &
      ", expected " & $expected.to(float32)

  var gotG = newSeq[bfloat16](N)
  engine.copyToHostFromDevice(gotG, zDevG)
  for i in 0 ..< N:
    doAssert gotG[i].to(float32) == xHost[idsHost[i]].to(float32),
      "bare DeviceMem " & $i & ": got " & $gotG[i].to(float32) &
      ", expected " & $xHost[idsHost[i]].to(float32)

when isMainModule:
  # enqueues live in a generic proc. The kernel symbol resolves at instantiation, as the op enqueue handlers do.
  let engine = gMetalEngine()
  runProbe(engine)
  echo "batch enqueue POC (Metal): all checks passed"
