# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: jagged attention on Metal, effective TFLOP/s.
#
# Effective flops count only the KV blocks a query block visits
# within its segment's causal band (k_jagged_attention), qBase = qBlock*MT:
# - a query block at qBase attends KV tokens up to kvLen - qLen + qBase + MT
# - the visited KV blocks are ceil(attended/NT) of the segment's blocks
# - per visited block: QK costs 2*MT*NT*D, PV 2*MT*NT*D, over all H*batchSize heads
#
# Run:
#   nim c -r -d:release \
#     -d:jagDtype=f32|f16 \
#     --outdir:build/bench workspace/positron/bench/jagged_attention/bench_jagged_attention_metal.nim

import std/[algorithm, math, monotimes, strformat, strutils, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/positron/src/kernels/k_jagged_attention
import workspace/positron/tests/positron_testutils

const jagDtype {.strdefine.} = "f32"

when jagDtype == "f16":
  type Elem = float16
else:
  type Elem = float32

const
  ## Tile shape of the bench kernel, page_size = the mma tile
  benchTile = 32
  benchD = 64

proc jaggedBenchLoop(o, q, kPool, vPool: distinct TensorView,
    qOffsets: distinct TensorView, kv: JaggedKv, scale: float32,
    layer, H, Nkv: int, iters: int) =
  ## `iters` attention passes in one launch, idempotent stores
  for _ in 0 ..< iters:
    jagged_attn_fwd(o, q, kPool, vPool, qOffsets, kv,
      scale, layer, H, Nkv, benchTile, benchD,
      tiledMma(float32, q.T, kPool.T, benchTile, benchTile, benchD),
      tiledMma(float32, o.T, vPool.T, benchTile, benchTile, benchD))

proc makeOffsetsTensor(n: int): auto =
  ## One-dimensional int32 host tensor over `n` zeroed elements
  let layout = make_layout((n,))
  result = HostTensor[int32, typeof(layout.shape), typeof(layout.stride)](
    data: cast[ptr UncheckedArray[int32]](allocShared(n * sizeof(int32))),
    offset: 0,
    layout: layout)
  zeroMem(cast[pointer](result.data), n * sizeof(int32))

proc fillPattern[T, Sh, St](tv: var TensorView[T, Sh, St]) =
  ## Logical position mod 7, times 0.25 plus 0.5
  for i in 0 ..< tv.size():
    tv(i) = to(float32(i mod 7) * 0.25'f32 + 0.5'f32, T)

proc visitedKvBlocks(qLen, kvLen, qBase: int, NT: static int): int =
  ## KV blocks the query block at qBase visits in its segment
  let attended = min(kvLen, kvLen - qLen + qBase + NT)
  if attended <= 0:
    return 0
  result = (attended + NT - 1) div NT

func computeJaggedFlops(seqlens: seq[int], H: int, MT, NT, D: static int): float64 =
  ## Effective flops over the batch: per query block and head, the visited
  ## KV blocks cost 2*MT*NT*D (QK) plus 2*MT*NT*D (PV)
  for kvLen in seqlens:
    let qLen = kvLen
    let qBlocks = (qLen + MT - 1) div MT
    for qBlock in 0 ..< qBlocks:
      result += float64(visitedKvBlocks(qLen, kvLen, qBlock * MT, NT)) *
        float64(H) * 2.0 * float64(MT * NT) * 2.0 * float64(D)

proc runCase(engine: MetalEngine, kvLen: int, numSeqs: int, iters: int) =
  ## One case: numSeqs equal-length segments, warmup, 3 timed launches, median
  const
    H = 32
    Nkv = 8
    MT = benchTile
    NT = benchTile
  let numQo = kvLen * numSeqs
  let numPages = (kvLen * numSeqs + benchTile - 1) div benchTile
  var
    qHost = makeHostTensor(Elem, (numQo, H, benchD), LayoutRight)
    kHost = makeHostTensor(Elem, (numPages, 2, benchTile, Nkv, benchD), LayoutRight)
    vHost = makeHostTensor(Elem, (numPages, 2, benchTile, Nkv, benchD), LayoutRight)
    oHost = makeHostTensor(Elem, (numQo, H, benchD), LayoutRight)
    qOffHost = makeOffsetsTensor(numSeqs + 1)
  block:
    var qv = qHost.view()
    var kv = kHost.view()
    var vv = vHost.view()
    fillPattern(qv)
    fillPattern(kv)
    fillPattern(vv)
  var kvHost = new(JaggedKvHost, capacity = numPages)
  for s in 0 ..< numSeqs:
    var pageIds: seq[int32]
    let firstPage = s * (kvLen div benchTile)
    for p in 0 ..< (kvLen div benchTile):
      pageIds.add(int32(firstPage + p))
    kvHost.addSequence(tokens = kvLen, pageIds = pageIds)
  for s in 0 .. numSeqs:
    qOffHost.data[s] = int32(s * kvLen)

  let
    scaleVal = 1.0'f32 / sqrt(float32(benchD))
    qBlocks = (kvLen + MT - 1) div MT
    launchGrid = (qBlocks, H, numSeqs)

  # warmup
  block:
    let
      qDev = engine.bridge(qHost)
      kDev = engine.bridge(kHost)
      vDev = engine.bridge(vHost)
      oDev = engine.bridge(oHost)
      qOffDev = engine.bridge(qOffHost)
      kvB = engine.bridge(kvHost)
    engine.launchDeviceFn jaggedBenchLoop(
        oDev, qDev, kDev, vDev, qOffDev, kvB, scaleVal, 0, H, Nkv, 2) do:
      grid = launchGrid
      blk = (32, 1)

  # timed, fresh bridges per rep and the copy drains the launch
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    block:
      let
        qDev = engine.bridge(qHost)
        kDev = engine.bridge(kHost)
        vDev = engine.bridge(vHost)
        oDev = engine.bridge(oHost)
        qOffDev = engine.bridge(qOffHost)
        kvB = engine.bridge(kvHost)
      let benchStart = getMonoTime()
      engine.launchDeviceFn jaggedBenchLoop(
          oDev, qDev, kDev, vDev, qOffDev, kvB, scaleVal, 0, H, Nkv, iters) do:
        grid = launchGrid
        blk = (32, 1)
      engine.copyToHostFromDevice(addr oHost.data[0], oDev.data, oHost.len())
      timings[rep] = inNanoseconds(getMonoTime() - benchStart) div iters
  sort(timings)
  let
    med = timings[1].float64 / 1e9
    flops = computeJaggedFlops(seqlens = @[kvLen], H = H, MT, NT, D = benchD) * float64(numSeqs)
    tflops = flops / med / 1e12
  writeLine(stdout, &"kvLen={kvLen:>5} seqs={numSeqs:>3}  median {med * 1e6:>10.1f} us  effective {tflops:>6.2f} TFLOP/s")

proc main =
  writeLine(stdout, "Jagged attention Metal bench, H=32 Nkv=8 D=64 page=32 " & jagDtype & " (M4 Max)")
  var engine = newEngine(runMetal)
  runCase(engine, 512, 1, iters = 10)
  runCase(engine, 1024, 1, iters = 5)
  runCase(engine, 2048, 1, iters = 2)
  runCase(engine, 2048, 8, iters = 2)
  runCase(engine, 32, 1, iters = 20)
  runCase(engine, 32, 64, iters = 10)

main()
