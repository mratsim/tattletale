# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: scaled dot product attention on Metal, effective TFLOP/s.
#
# Effective flops count only the key tiles a query tile visits:
# - the causal band visits min(ceil((qTile+1)*MT, NT), kvTiles) tiles
# - the full sweep visits kvTiles
# - per visited tile: QK costs 2*MT*NT*Dk, PV 2*MT*NT*Dv
#
# Run:
#   nim c -r -d:release --maxLoopIterationsVM:100000000 \
#     -d:sdpaDtype=f32|bf16 \
#     --outdir:build/bench workspace/ceramic/bench/sdpa_gpu/bench_scaled_dot_product_attention_metal.nim

import std/[algorithm, math, monotimes, strformat, strutils, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_tile_sdpa
import workspace/ceramic/tests/ceramic_testutils

const sdpaDtype {.strdefine.} = "f32"

when sdpaDtype == "bf16":
  type Elem = bfloat16
else:
  type Elem = float32

const
  ## Tile shape of the bench kernel, measured per dtype on M4 Max:
  ## f32 peaks at (32, 16), bf16 at (16, 32). The flops formula reads
  ## these consts too, so the kernel and the formula cannot drift apart
  benchMT = when Elem is float32: 32 else: 16
  benchNT = when Elem is float32: 16 else: 32

proc sdpaBenchLoop(o, q, k, v: distinct TensorView, scale: float32, Dk, Dv: static int, Causal: static bool, iters: int) =
  ## `iters` attention passes in one launch, idempotent stores
  const
    MT = benchMT
    NT = benchNT
  for _ in 0 ..< iters:
    scaled_dot_product_attention(
      tiledMma(float32, Elem, Elem, MT, NT, 32),
      tiledMma(float32, Elem, Elem, MT, NT, 32),
      o, q, k, v, scale, Dk, Dv, Causal)

proc fillPattern[T, Sh, St](tv: var TensorView[T, Sh, St]) =
  ## Logical position mod 7, times 0.25 plus 0.5
  for i in 0 ..< tv.size():
    tv(i) = to(float32(i mod 7) * 0.25'f32 + 0.5'f32, T)

proc visitedTiles(qTiles, seqLen: int, Causal: static bool, MT, NT: static int): int =
  ## Total key tiles visited across the query-tile grid
  let kvTiles = seqLen.ceil_div(NT)
  for qTile in 0 ..< qTiles:
    when Causal:
      let diagTile = (qTile * MT + MT - 1) div NT
      result += min(diagTile + 1, kvTiles)
    else:
      result += kvTiles

func computeAttnFlops(qTiles, seqLen, Hq, B: int, Causal: static bool, MT, NT, Dk, Dv: static int): float64 =
  ## Effective flops: only the key tiles a query tile visits.
  ## Per visited tile QK costs 2*MT*NT*Dk and PV 2*MT*NT*Dv, over all Hq*B heads
  float64(visitedTiles(qTiles, seqLen, Causal, MT, NT)) *
    float64(Hq * B) * 2.0 * MT * NT * (Dk + Dv)

proc runCase(engine: MetalEngine, L, S: int, Causal: static bool, iters: int) =
  ## One (L, S, causal) case: warmup, 3 timed launches, median TFLOP/s
  const
    B = 1
    Hq = 32
    Hkv = 8
    Dk = 128
    Dv = 128
    MT = benchMT
    NT = benchNT
  let shapeQ = (B, Hq, L, Dk)
  let shapeKV = (B, Hkv, S, Dk)
  let shapeO = (B, Hq, L, Dv)
  var
    qHost = makeHostTensor(Elem, shapeQ, LayoutRight)
    kHost = makeHostTensor(Elem, shapeKV, LayoutRight)
    vHost = makeHostTensor(Elem, shapeKV, LayoutRight)
    oHost = makeHostTensor(Elem, shapeO, LayoutRight)
  block:
    var qv = qHost.view()
    var kv = kHost.view()
    var vv = vHost.view()
    fillPattern(qv)
    fillPattern(kv)
    fillPattern(vv)
  let
    scaleVal = 1.0'f32 / sqrt(float32(Dk))
    qTiles = L.ceil_div(MT)
    launchGrid = (qTiles, Hq, B)

  # warmup
  block:
    let
      qDev = engine.bridge(qHost)
      kDev = engine.bridge(kHost)
      vDev = engine.bridge(vHost)
      oDev = engine.bridge(oHost)
    engine.launchDeviceFn sdpaBenchLoop(oDev, qDev, kDev, vDev, scaleVal, Dk, Dv, Causal, 2) do:
      grid = launchGrid
      blk = (32, 1)

  # timed, fresh bridges per rep. The launch overhead is measured.
  # Drain: wait for the iterations before the next rep.
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    block:
      let
        qDev = engine.bridge(qHost)
        kDev = engine.bridge(kHost)
        vDev = engine.bridge(vHost)
        oDev = engine.bridge(oHost)
      let benchStart = getMonoTime()
      engine.launchDeviceFn sdpaBenchLoop(oDev, qDev, kDev, vDev, scaleVal, Dk, Dv, Causal, iters) do:
        grid = launchGrid
        blk = (32, 1)
      engine.copyToHostFromDevice(addr oHost.data[0], oDev.data, oHost.len())
      timings[rep] = inNanoseconds(getMonoTime() - benchStart) div iters
  sort(timings)
  let
    med = timings[1].float64 / 1e9
    flops = computeAttnFlops(qTiles, S, Hq, B, Causal, MT, NT, Dk, Dv)
    tflops = flops / med / 1e12
  writeLine(stdout, &"L={L:>5} S={S:>5} causal={Causal:>5}  median {med * 1e6:>10.1f} us  effective {tflops:>6.2f} TFLOP/s")

proc main =
  writeLine(stdout, "SDPA Metal bench, B=1 Hq=32 Hkv=8 Dk=Dv=128 " & sdpaDtype & " (M4 Max)")
  var engine = newEngine(runMetal)
  runCase(engine, 512, 512, Causal = true, iters = 10)
  runCase(engine, 1024, 1024, Causal = true, iters = 5)
  runCase(engine, 2048, 2048, Causal = true, iters = 2)
  runCase(engine, 2048, 2048, Causal = false, iters = 2)
  runCase(engine, 32, 2048, Causal = true, iters = 20)
  runCase(engine, 32, 8192, Causal = true, iters = 10)

main()
