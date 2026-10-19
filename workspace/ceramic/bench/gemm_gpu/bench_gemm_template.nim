# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Run:
#   nim c -r -d:release --outdir:build/bench \
#     workspace/ceramic/bench/gemm_gpu/bench_gemm_metal.nim (from tattletale)

import std/[monotimes, strformat, strutils, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/bench/gemm_gpu/bench_blueprint
import workspace/ceramic/tests/kernels/k_gemm/kernel_gemm_template
import workspace/fp_conv/fp_conv

export bench_blueprint
export int_tuples
export layout_algebra
export tensors
export tile_algebra
export ceramic_testutils
export fp_conv

proc fillGemmInputs*[TVa, TVb](aTV: TVa, bTV: TVb, M, N, K: int) =
  for m in 0 ..< M:
    for k in 0 ..< K:
      aTV[m, k] = F16.from(float32(1 + (2 * m + 7 * k) mod 13))
  for n in 0 ..< N:
    for k in 0 ..< K:
      bTV[n, k] = F16.from(float32(1 + (3 * k + 11 * n) mod 17))

proc gemmLoop*(mma: static TiledMma, C, A, B: distinct TensorView, iters: int) =
  for _ in 0 ..< iters:
    matmul(mma, C, A, B)

template benchGemm*(op: string, M, N, K, iters: int, body: untyped) =
  let
    config = $M & "x" & $N & "x" & $K
    flops = 2.0 * M.float * N.float * K.float
    benchStart = getMonoTime()
  body
  let benchStop = getMonoTime()
  discard report(op & ": gemm", config, benchStart, benchStop, iters, flops)

proc benchCeramic*[E: HwEngine](engine: E, M, N, K, iters: int): auto =
  let one = 1  # runtime int, the layouts stay dynamic
  var
    aHost = makeHostTensor(F16, (M, K), (K, one))
    bHost = makeHostTensor(F16, (N, K), (one, N))
    dHost = makeHostTensor(float32, (M, N), (N, one))
  fillGemmInputs(aHost.view(), bHost.view(), M, N, K)
  # warmup, the kernel loops 5 times inside one launch
  block:
    let
      aDev = engine.bridge(aHost)
      bDev = engine.bridge(bHost)
      dDev = engine.bridge(dHost)
    engine.launchDeviceFn gemmLoop(tiledMma(float32, float16, float16, 32, 32, 16), dDev, aDev, bDev, 5) do:
      grid = (ceil_div(M, 32), ceil_div(N, 32))
      blk = (32, 1)
  # timed, the launch overhead leaves the measurement
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  benchGemm("Tattletale Ceramic", M, N, K, iters):
    engine.launchDeviceFn gemmLoop(tiledMma(float32, float16, float16, 32, 32, 16), dDev, aDev, bDev, iters) do:
      grid = (ceil_div(M, 32), ceil_div(N, 32))
      blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, M * N)
  dHost
