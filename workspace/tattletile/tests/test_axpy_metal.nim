# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## axpy end to end on Metal: Tile IR fixture compiled by tattletile, launched through crucible's Metal engine
##
## Test that the compiled axpy kernel, launched over 2 blocks x 4 threads,
## produces results bitwise-identical (int32 view) against the python float64-then-narrow reference.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_axpy_metal.nim

import workspace/crucible
import workspace/tattletile/tattletile

const N = 16
  ## Element count, a multiple of the 8-element tile. The axpy tracer emits
  ## no out-of-bounds predication (recorded for bullet 5), so the launch
  ## covers the view exactly.

const AxpyMsl = compileTileIrToMsl(
  parseTileIrFile("../fixtures/vec_add_axpy.tileir"), "axpy", threads = 4)

const Alpha = static:
  ## Axpy multiplier, extracted from the fixture constant like the python runner does.
  let m = parseTileIrFile("../fixtures/vec_add_axpy.tileir")
  var alpha = 0.0
  for op in m.entries[0].ops:
    if op.kind == okConstant and op.constElem == eF32:
      alpha = op.constValue.scalar.floatVal
  alpha

const AHost = static:
  ## Test input of the python runner: a[i] = i mod 7, integer-valued f32.
  var a: array[N, float32]
  for i in 0 ..< N:
    a[i] = float32(i mod 7)
  a

const BHost = static:
  ## Test input of the python runner: b[i] = i mod 5, integer-valued f32.
  ## Integer values keep every intermediate exactly representable.
  var b: array[N, float32]
  for i in 0 ..< N:
    b[i] = float32(i mod 5)
  b

const Expected = static:
  ## Reference bits: c = f32(f64(alpha) * f64(a) + f64(b)), the python axpy_ref order.
  var e: array[N, uint32]
  for i in 0 ..< N:
    e[i] = cast[uint32](float32(Alpha * float64(AHost[i]) + float64(BHost[i])))
  e

proc f32Bits(v: float32): uint32 =
  cast[uint32](v)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runMetal)
  engine.ingest(AxpyMsl)

  let aDev = engine.allocDevice(float32, N)
  let bDev = engine.allocDevice(float32, N)
  let cDev = engine.allocDevice(float32, N)
  engine.copyToDeviceFromHost(aDev, addr AHost[0], N)
  engine.copyToDeviceFromHost(bDev, addr BHost[0], N)

  # Launch order = kernel parameter order. Slot 0 binds the output buffer,
  # so the store target goes first, then the remaining entry args.
  # args tuple is a named var, an inline tuple would copy DeviceMem handles.
  # readback below reads the output buffer again.
  var args = (int32(N), aDev, int32(N), bDev, int32(N))
  engine.launchKernel("axpy", cDev, args) do:
    grid = (N div 8, 1)
    blk = (4, 1)

  var got: array[N, float32]
  engine.copyToHostFromDevice(addr got[0], cDev, N)
  for i in 0 ..< N:
    doAssert f32Bits(got[i]) == Expected[i],
      "axpy element " & $i & ": got bits " & $f32Bits(got[i]) &
      " want bits " & $Expected[i]
  echo "[OK] axpy on Metal: bitwise-exact (int32 view) over ", N, " elements"

when isMainModule:
  runTest()
