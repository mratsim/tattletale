# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Batch launch instantiated outside crucible's scope.
##
## Test that a batch launch wrapper compiles and runs when the instantiating
## module sees only a downstream wrapper module, never crucible.
## The generated wrapper's pragmas bind at the launch machinery itself.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --nimcache:nimcache/slices --outdir:build/slices \
##     workspace/crucible/tests/runtime/t_launch_wrapper_scope.nim

import ./launch_wrapper_scope_helper

proc runTest() =
  var engine = newTestEngine()
  const n = 16
  var xs: array[n, float32]
  var ys: array[n, float32]
  for i in 0 ..< n:
    xs[i] = float32(i)
    ys[i] = 1.0'f32

  var xD = engine.allocDevice(float32, n)
  var yD = engine.allocDevice(float32, n)
  var outD = engine.allocDevice(float32, n)
  engine.copyToDeviceFromHost(xD, xs.toOpenArray(0, n - 1))
  engine.copyToDeviceFromHost(yD, ys.toOpenArray(0, n - 1))

  engine.enqueueAxpy(outD, xD, yD, n, 2.5'f32)

  var got: array[n, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, n - 1), outD)
  for i in 0 ..< n:
    doAssert got[i] == 2.5'f32 * (xs[i] + ys[i]),
      "axpy at " & $i & ": got " & $got[i] & " want " & $(2.5'f32 * (xs[i] + ys[i]))
  echo "[OK] batch launch wrapper instantiated outside crucible's scope"

runTest()
