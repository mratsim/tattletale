# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Two-batch accumulation
##
## Run:
##   nim c -r -d:release --hints:off --warnings:off \
##     --nimcache:nimcache/t_batch_two_wave_metal --outdir:build/tests/t_batch_two_wave_metal \
##     workspace/crucible/tests/batching/t_batch_two_wave_metal.nim

import std/importutils
import std/strutils
import workspace/crucible

privateAccess(BatchCommands)

const N = 16

proc fillKernel*(outP: ptr UncheckedArray[uint32], v: uint32) =
  let i = int(thread_position_in_grid.x)
  if i < N:
    outP[i] = v

proc scaleKernel*(outP, x: ptr UncheckedArray[float32], s: float32) =
  let i = int(thread_position_in_grid.x)
  outP[i] = s * x[i]

proc countKernels(artifact: string): int =
  for line in artifact.splitLines():
    if line.startsWith("kernel void "):
      result.inc

proc main =
  let engine = newEngine(runMetal)
  var fills: array[N, uint32]
  var xs: array[N, float32]
  for i in 0 ..< N:
    fills[i] = 0
    xs[i] = float32(i)
  var fillD = engine.allocDevice(uint32, N)
  var xD = engine.allocDevice(float32, N)
  var outD = engine.allocDevice(float32, N)
  engine.copyToDeviceFromHost(fillD, fills.toOpenArray(0, N - 1))
  engine.copyToDeviceFromHost(xD, xs.toOpenArray(0, N - 1))

  # First batch: accumulate, materialize.
  var batch1 = engine.recordBatch()
  batch1.enqueue<<(1, 16)>> scaleKernel(outD, xD, 3.0'f32)
  batch1.submit()
  doAssert countKernels(batch1.artifact) == 1, "first artifact holds " &
    $countKernels(batch1.artifact) & " kernels, expected 1 (first-batch prefix)"

  # Second batch: accumulate more, materialize again.
  var batch2 = engine.recordBatch()
  batch2.enqueue<<(1, 16)>> fillKernel(fillD, 7'u32)
  batch2.submit()
  doAssert countKernels(batch2.artifact) == 2, "second artifact holds " &
    $countKernels(batch2.artifact) & " kernels, expected 2 (complete stream)"

  # Both batches hold their own artifact and launch correctly.
  var hostFills: array[N, uint32]
  var hostOut: array[N, float32]
  engine.copyToHostFromDevice(hostFills.toOpenArray(0, N - 1), fillD)
  engine.copyToHostFromDevice(hostOut.toOpenArray(0, N - 1), outD)
  for i in 0 ..< N:
    doAssert hostFills[i] == 7'u32
    doAssert hostOut[i] == 3.0'f32 * float32(i)
  echo "crucible two-batch staleness test (Metal): passed"

when isMainModule:
  main()
