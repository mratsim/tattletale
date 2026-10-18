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
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/layout_algebra/layouts
import workspace/ceramic/src/tensors
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_tile_gemm
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/bench/gemm_gpu/bench_blueprint
import workspace/ceramic/tests/kernels/k_gemm/t_kernel_gemm_template
import workspace/fp_conv/fp_conv

export bench_blueprint
export int_tuples, int_tuples_streams
export layout_algebra, layout_compiletime, layouts
export tensors
export h_mma_configgen, h_mma_registry, h_mma_properties, h_mma_dispatch
export k_tile_gemm
export tile_algebra
export static_for
export ceramic_testutils
export t_kernel_gemm_template
export fp_conv

{.experimental: "callOperator".}

type DevTensorView*[DevMem, Sh, St] = object
  ## Host-side TensorView for DeviceMem
  data: DevMem
  layout: Layout[Sh, St]

func devView2D*[DevMem](data: DevMem, shape, strides: (int, int)): DevTensorView[DevMem, (int, int), (int, int)] =
  DevTensorView[DevMem, (int, int), (int, int)](data: data, layout: make_layout(shape, strides))

type LaunchFn[E, H] = proc (engine: E,
    dDev: DeviceMem[H, float32],
    aDev, bDev: DeviceMem[H, F16],
    M, N, K: int, iters: int) {.nimcall.}

proc fillGemmInputs[TVa, TVb](aTV: TVa, bTV: TVb, M, N, K: int) =
  for m in 0 ..< M:
    for k in 0 ..< K:
      aTV[m, k] = F16.from(float32(1 + (2 * m + 7 * k) mod 13))
  for n in 0 ..< N:
    for k in 0 ..< K:
      bTV[n, k] = F16.from(float32(1 + (3 * k + 11 * n) mod 17))

proc checkGemmRef[E: HwEngine, H](engine: E, M, N, K: int, launch: LaunchFn[E, H], msg: string) =
  let one = 1  # runtime int, the layouts stay dynamic
  var
    aHost = newSeq[F16](M * K)
    bHost = newSeq[F16](K * N)
  let
    aTV = make_view(addr aHost[0], make_layout((M, K), (K, one)))
    bTV = make_view(addr bHost[0], make_layout((N, K), (one, N)))
  fillGemmInputs(aTV, bTV, M, N, K)
  let
    aDev = engine.allocDevice(F16, M * K)
    bDev = engine.allocDevice(F16, K * N)
    dDev = engine.allocDevice(float32, M * N)
  engine.copyToDeviceFromHost(aDev, addr aHost[0], M * K)
  engine.copyToDeviceFromHost(bDev, addr bHost[0], K * N)
  stderr.writeLine "DBG checkGemmRef: launch done " & msg
  when not defined(skipLaunch):
    launch(engine, dDev, aDev, bDev, M, N, K, 1)
  stderr.writeLine "DBG checkGemmRef: kernel done " & msg
  var gotHost = newSeq[float32](M * N)
  var refHost = newSeq[float32](M * N)
  engine.copyToHostFromDevice(addr gotHost[0], dDev, M * N)
  let gotTV = make_view(cast[ptr UncheckedArray[float32]](addr gotHost[0]),
    make_layout((M, N), (N, one)))
  let refTV = make_view(cast[ptr UncheckedArray[float32]](addr refHost[0]),
    make_layout((M, N), (N, one)))
  gemm_ref(refTV, 1.0'f32, aTV, bTV, 0.0'f32, gotTV)
  assertAllClose(gotTV, refTV, rtol = 1e-5, abstol = 1e-3, msg = msg)
  engine.freeDevice(aDev)
  engine.freeDevice(bDev)
  engine.freeDevice(dDev)

proc gemmBench*[E: HwEngine, H](engine: E, launch: LaunchFn[E, H], M, N, K: int, iters: int, text: string) =
  checkGemmRef(engine, 128, 128, 512, launch, text & " vs gemm_ref") # Correctness + warmup
  checkGemmRef(engine, 1024, 64, 512, launch, text & " vs gemm_ref")

  let one = 1  # runtime int, the layouts stay dynamic
  var
    aHost = newSeq[F16](M * K)
    bHost = newSeq[F16](K * N)
  let
    aTV = make_view(addr aHost[0], make_layout((M, K), (K, one)))
    bTV = make_view(addr bHost[0], make_layout((N, K), (one, N)))
  fillGemmInputs(aTV, bTV, M, N, K)
  let
    aDev = engine.allocDevice(F16, M * K)
    bDev = engine.allocDevice(F16, K * N)
    dDev = engine.allocDevice(float32, M * N)
  engine.copyToDeviceFromHost(aDev, addr aHost[0], M * K)
  engine.copyToDeviceFromHost(bDev, addr bHost[0], K * N)

  let
    config = $M & "x" & $N & "x" & $K
    flops = 2.0 * M.float * N.float * K.float
  # the GPU loops the device function `iters` times inside one launch,
  # the launch overhead leaves the measurement
  let benchStart = getMonoTime()
  stderr.writeLine "DBG timing launch start " & text
  launch(engine, dDev, aDev, bDev, M, N, K, iters)
  stderr.writeLine "DBG timing launch done " & text
  let benchStop = getMonoTime()
  discard report(text & ": gemm", config, benchStart, benchStop, iters, flops)

  engine.freeDevice(aDev)
  engine.freeDevice(bDev)
  engine.freeDevice(dDev)
