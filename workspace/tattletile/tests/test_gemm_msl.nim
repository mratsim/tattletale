# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## GEMM Tile IR to MSL through the tattletile bridge and crucible's shipped pipeline
##
## Test that the gemm fixture compiles to the expected Metal Shading
## Language, stable across compiles:
## - the kernel signature with buffer slots and coordinate builtins
## - the fused per-thread value loop over the 2-d tile coordinates
## - the abstract mmaf lowered to a per-thread FMA loop over the k dimension
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_gemm_msl.nim

import std/strutils

import workspace/tattletile/tattletile

const GemmMsl = compileTileIrToMsl(
  parseTileIrFile("../fixtures/gemm.tileir"), "gemm", threads = 4)

const GemmMslAgain = compileTileIrToMsl(
  parseTileIrFile("../fixtures/gemm.tileir"), "gemm", threads = 4)

proc expect(msl, fragment: string) =
  doAssert fragment in msl, "emitted MSL misses:\n  " & fragment &
    "\nemitted:\n" & msl

proc checkSignature() =
  expect(GemmMsl, "kernel void gemm(")
  # pointers take explicit buffer slots in launch order, the store target
  # first (the launch binds the output to slot 0), then the entry args
  expect(GemmMsl, "device float* ptr_c [[buffer(0)]]")
  expect(GemmMsl, "device float* ptr_a [[buffer(1)]]")
  expect(GemmMsl, "device float* ptr_b [[buffer(2)]]")
  # coordinate builtins are emitted as uint3 attribute parameters
  expect(GemmMsl, "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]")
  expect(GemmMsl, "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]")

proc checkBody() =
  # block-level index math through the coordinate builtins, the y component
  # indexes the second view dimension
  expect(GemmMsl, "thread_position_in_threadgroup.x")
  expect(GemmMsl, "threadgroup_position_in_grid.y")
  # fused value loop over the thread-value slots, 4 threads x 64 values,
  # while-form loop idiom of the CIRv2 lowering
  expect(GemmMsl, "int tt_v;")
  expect(GemmMsl, "while ((tt_v < 64)) {")
  # flat owned coordinate decomposes into the 2-d tile coordinates
  expect(GemmMsl, "int tt_c0 = (tt_coord / 16);")
  expect(GemmMsl, "int tt_c1 = (tt_coord % 16);")
  # loads through the view layout, the tile dataflow in program order
  expect(GemmMsl, "tile_a = ptr_a[tt_off_pv_a]")
  expect(GemmMsl, "tile_b = ptr_b[tt_off_pv_b]")
  # abstract mmaf: a per-thread FMA loop over the k dimension
  expect(GemmMsl, "float zero = 0.0f")
  expect(GemmMsl, "float tt_mma_sum_acc = 0.0f")
  expect(GemmMsl, "while ((tt_kk < 16)) {")
  expect(GemmMsl, "tt_mma_sum_acc = (tt_mma_sum_acc + (ptr_a[tt_mma_off_a] * ptr_b[tt_mma_off_b]))")
  expect(GemmMsl, "float acc = (tt_mma_sum_acc + zero)")
  # store of the mma result through the view layout
  expect(GemmMsl, "ptr_c[tt_off_pv_c] = acc")

proc checkStable() =
  doAssert GemmMsl == GemmMslAgain,
    "two compiles of the same fixture disagree, the emitted MSL is not stable"
  doAssert GemmMsl.len > 0, "empty MSL"

when isMainModule:
  checkSignature()
  checkBody()
  checkStable()
  echo "[OK] gemm compiles to stable MSL through the bridge and crucible's pipeline"
