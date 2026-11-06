# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Demo host: one binary, two Metal kernels compiled from Tile IR text at build time
##
## Test that one binary embeds both fixtures as `const`, runs them on Metal bitwise-exact
## (int32 view), and ships no parser machinery.
##
## Coverage:
##
## - the pipeline runs in the Nim VM at build time: the bridge builds a CIRv2 module
##   through crucible's `cirv2_build`, only the emitted MSL ships as `const`
## - axpy runs on Metal, c = alpha * a + b over 16 elements, 2 blocks x 4 threads
## - gemm runs on Metal, C = A @ B over 16x16, one block x 4 threads
## - the binary introspects itself: it ships the MSL text and no parser machinery
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/manual_embed_axpy_gemm.nim

import std/os
import std/strutils

import workspace/crucible
import workspace/tattletile/tattletile

const Axpy = embedKernel("../fixtures/vec_add_axpy.tileir", "axpy", threads = 4)
  ## Axpy kernel, compiled from Tile IR text in the Nim VM.

const Gemm = embedKernel("../fixtures/gemm.tileir", "gemm", threads = 4)
  ## Gemm kernel, compiled from Tile IR text in the Nim VM.

const N = 16
  ## Axpy element count, a multiple of the 8-element tile. The axpy tracer emits
  ## no out-of-bounds predication, so the launch covers the view exactly.

const M = 16
  ## Gemm view size, M elements per side, tile == view, one block covers them.

const K = 16
  ## Reduction size of the gemm mmaf, one 16-term dot product per accumulator.

const Alpha = static:
  ## Axpy multiplier, extracted from the fixture constant.
  let m = parseTileIrFile("../fixtures/vec_add_axpy.tileir")
  var alpha = 0.0
  for op in m.entries[0].ops:
    if op.kind == okConstant and op.constElem == eF32:
      alpha = op.constValue.scalar.floatVal
  alpha

const AHost = static:
  ## Axpy input, a[i] = i mod 7, integer-valued f32.
  var a: array[N, float32]
  for i in 0 ..< N:
    a[i] = float32(i mod 7)
  a

const BHost = static:
  ## Axpy input, b[i] = i mod 5, integer-valued f32.
  var b: array[N, float32]
  for i in 0 ..< N:
    b[i] = float32(i mod 5)
  b

const AxpyExpected = static:
  ## Reference bits: c = f32(f64(alpha) * f64(a) + f64(b)), the python axpy_ref order.
  var e: array[N, uint32]
  for i in 0 ..< N:
    e[i] = cast[uint32](float32(Alpha * float64(AHost[i]) + float64(BHost[i])))
  e

const AHostM = static:
  ## Gemm input, a[i] = i mod 5, integer-valued f32.
  var a: array[M * K, float32]
  for i in 0 ..< M * K:
    a[i] = float32(i mod 5)
  a

const BHostM = static:
  ## Gemm input, b[i] = i mod 4, integer-valued f32.
  var b: array[M * K, float32]
  for i in 0 ..< M * K:
    b[i] = float32(i mod 4)
  b

const GemmExpected = static:
  ## Reference bits: C = f32(f64(A) @ f64(B)), the python gemm_ref order.
  var e: array[M * M, uint32]
  for i in 0 ..< M:
    for j in 0 ..< M:
      var s = 0.0
      for kk in 0 ..< K:
        s += float64(AHostM[i * K + kk]) * float64(BHostM[kk * M + j])
      e[i * M + j] = cast[uint32](float32(s))
  e

proc checkShippedBinary() =
  ## Compile-time embedding, verified against the running executable:
  ## MSL text in the binary, parser and bridge machinery absent.
  let binary = readFile(getAppFilename())
  doAssert "kernel void axpy(" in binary, "embedded axpy MSL not in the binary"
  doAssert "kernel void gemm(" in binary, "embedded gemm MSL not in the binary"
  # Marker halves concatenate at runtime, the check strings are not themselves
  # literals in the binary.
  let p1 = "make_tens"
  let p2 = "or_view"
  let p3 = "cuda_tile"
  let p4 = ".module"
  let r1 = "redefines a kernel"
  let r2 = "-level name"
  doAssert (p1 & p2) notin binary, "parser machinery leaked into the binary"
  doAssert (p3 & p4) notin binary, "parser machinery leaked into the binary"
  doAssert (r1 & r2) notin binary, "bridge machinery leaked into the binary"
  echo "[OK] the binary ships the MSL text and no parser machinery"

proc runAxpy() =
  var engine = newEngine(runMetal)
  engine.ingest(Axpy.msl)

  let aDev = engine.allocDevice(float32, N)
  let bDev = engine.allocDevice(float32, N)
  let cDev = engine.allocDevice(float32, N)
  engine.copyToDeviceFromHost(aDev, addr AHost[0], N)
  engine.copyToDeviceFromHost(bDev, addr BHost[0], N)

  # Launch order = kernel parameter order, the store target binds slot 0.
  var args = (int32(N), aDev, int32(N), bDev, int32(N))
  engine.launchKernel(Axpy.name, cDev, args) do:
    grid = (N div 8, 1)
    blk = (4, 1)

  var got: array[N, float32]
  engine.copyToHostFromDevice(addr got[0], cDev, N)
  for i in 0 ..< N:
    doAssert cast[uint32](got[i]) == AxpyExpected[i],
      "axpy element " & $i & ": got bits " & $cast[uint32](got[i]) &
      " want bits " & $AxpyExpected[i]
  echo "[OK] embedded axpy on Metal: bitwise-exact (int32 view) over ", N, " elements"

proc runGemm() =
  var engine = newEngine(runMetal)
  engine.ingest(Gemm.msl)

  let aDev = engine.allocDevice(float32, M * K)
  let bDev = engine.allocDevice(float32, M * K)
  let cDev = engine.allocDevice(float32, M * M)
  engine.copyToDeviceFromHost(aDev, addr AHostM[0], M * K)
  engine.copyToDeviceFromHost(bDev, addr BHostM[0], M * K)

  # One block: the 16x16 views equal their 16x16 partition tiles.
  var args = (aDev, bDev)
  engine.launchKernel(Gemm.name, cDev, args) do:
    grid = (M div 16, M div 16)
    blk = (4, 1)

  var got: array[M * M, float32]
  engine.copyToHostFromDevice(addr got[0], cDev, M * M)
  for i in 0 ..< M * M:
    doAssert cast[uint32](got[i]) == GemmExpected[i],
      "gemm element " & $i & ": got bits " & $cast[uint32](got[i]) &
      " want bits " & $GemmExpected[i]
  echo "[OK] embedded gemm on Metal: bitwise-exact (int32 view) over ", M * M, " outputs"

when isMainModule:
  checkShippedBinary()
  runAxpy()
  runGemm()
  echo "manual_embed_axpy_gemm: PASS"
