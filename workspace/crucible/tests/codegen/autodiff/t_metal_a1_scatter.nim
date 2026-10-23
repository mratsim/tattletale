# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## A1 scatter test: plain-buffer scatter on Metal (bwd-diff assumption A1)
##
## Test that a transposed loop whose backward scatters cotangent values
## into a plain `ptr UncheckedArray[float32]` shadow buffer writes every
## value on Metal, element-for-element against a CPU closed form:
##
## - 8x8 transpose scatter: thread (x, y) reads d_y[y*8 + x], writes d_a[x*8 + y]
## - consecutive threads write stride-8 apart, a scatter pattern, not a map, every slot written exactly once (bijective transpose)
## - `d_a[j] = alpha * d_y[i]` is the statement shape the transposer emits for a buffer-slot cotangent
##
## Run:
##   nim c -r --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/autodiff/t_metal_a1_scatter.nim

import workspace/crucible

const N = 8

const scatterBwd = metal:
  proc scatterBwd(d_a: ptr UncheckedArray[float32], d_y: ptr UncheckedArray[float32], alpha: float32) {.global, workgroup: (4, 4).} =
    let gx = thread_position_in_grid.x
    let gy = thread_position_in_grid.y
    let src = gy * N + gx
    let dst = gx * N + gy
    d_a[dst] = alpha * d_y[src]

proc runTest() =   # private: engines are destroyed at return
  var engine = newEngine(runMetal)
  engine.ingest(scatterBwd)
  let dADev = engine.allocDevice(float32, N * N)
  let dYDev = engine.allocDevice(float32, N * N)
  var dYHost: array[N * N, float32]
  for i in 0 ..< N * N:
    dYHost[i] = float32(i) + 0.5'f32
  engine.copyToDeviceFromHost(dYDev, addr dYHost[0], N * N)

  let alpha = 2.5'f32
  engine.launchKernel<<(grid: (2, 2), blk: (4, 4))>>("scatterBwd", dADev, (dYDev, alpha))

  var got: array[N * N, float32]
  engine.copyToHostFromDevice(addr got[0], dADev, N * N)
  for gy in 0 ..< N:
    for gx in 0 ..< N:
      let src = gy * N + gx
      let dst = gx * N + gy
      let expected = alpha * dYHost[src]
      doAssert got[dst] == expected,
        "scatter landed wrong: dst slot " & $dst & " got " & $got[dst] &
        " want " & $expected & " (src slot " & $src & ")"
  echo "  OK — plain-buffer scatter: all 64 transposed cotangent writes landed"

when isMainModule:
  runTest()
