# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GEMM end to end on Metal: Tile IR fixture compiled by tattletile, launched through crucible's Metal engine
##
## Test that the compiled gemm kernel, launched over 1 block x 4 threads,
## produces results bitwise-identical (int32 view) against the python
## float64-then-narrow matmul reference.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_gemm_metal.nim

import workspace/crucible
import workspace/tattletile/tattletile

const M = 16
  ## View size of the 16x16 fixture views: M elements per side, tile == view.
  ## The gemm tracer emits no out-of-bounds predication (recorded in bullet 3),
  ## so the launch covers the views exactly: one block.

const K = 16
  ## Reduction size of the fixture's mmaf, one 16-term dot product per
  ## owned accumulator coordinate.

const GemmMsl = compileTileIrToMsl(
  parseTileIrFile("../fixtures/gemm.tileir"), "gemm", threads = 4)

const AHost = static:
  ## Test input of the python runner: a[i] = i mod 5, integer-valued f32.
  var a: array[M * K, float32]
  for i in 0 ..< M * K:
    a[i] = float32(i mod 5)
  a

const BHost = static:
  ## Test input of the python runner: b[i] = i mod 4, integer-valued f32.
  ## Integer values keep every intermediate exactly representable.
  var b: array[M * K, float32]
  for i in 0 ..< M * K:
    b[i] = float32(i mod 4)
  b

const Expected = static:
  ## Reference bits: C = f32(f64(A) @ f64(B)), the python gemm_ref order.
  var e: array[M * M, uint32]
  for i in 0 ..< M:
    for j in 0 ..< M:
      var s = 0.0
      for kk in 0 ..< K:
        s += float64(AHost[i * K + kk]) * float64(BHost[kk * M + j])
      e[i * M + j] = cast[uint32](float32(s))
  e

proc f32Bits(v: float32): uint32 =
  cast[uint32](v)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runMetal)
  engine.ingest(GemmMsl)

  let aDev = engine.allocDevice(float32, M * K)
  let bDev = engine.allocDevice(float32, M * K)
  let cDev = engine.allocDevice(float32, M * M)
  engine.copyToDeviceFromHost(aDev, addr AHost[0], M * K)
  engine.copyToDeviceFromHost(bDev, addr BHost[0], M * K)

  # Launch order = kernel parameter order. Slot 0 binds the output buffer,
  # so the store target goes first, then the remaining entry args.
  # grid is one block: the 16x16 views equal their 16x16 partition tiles.
  # args tuple is a named var, an inline tuple would copy DeviceMem handles.
  var args = (aDev, bDev)
  engine.launchKernel("gemm", cDev, args) do:
    grid = (M div 16, M div 16)
    blk = (4, 1)

  var got: array[M * M, float32]
  engine.copyToHostFromDevice(addr got[0], cDev, M * M)
  for i in 0 ..< M * M:
    doAssert f32Bits(got[i]) == Expected[i],
      "gemm element " & $i & ": got bits " & $f32Bits(got[i]) &
      " want bits " & $Expected[i]
  echo "[OK] gemm on Metal: bitwise-exact (int32 view) over ", M * M, " outputs"

when isMainModule:
  runTest()
