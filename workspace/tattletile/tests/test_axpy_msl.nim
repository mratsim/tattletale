# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile IR to MSL through the tattletile bridge and crucible's shipped pipeline
##
## Test that the axpy fixture compiles to the expected Metal Shading
## Language: the kernel signature with buffer slots and coordinate builtins,
## the fused per-thread value loop, and stable output across compiles.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_axpy_msl.nim

import std/strutils

import workspace/tattletile/tattletile

const AxpyMsl = compileTileIrToMsl(
  parseTileIrFile("../fixtures/vec_add_axpy.tileir"), "axpy", threads = 4)

const AxpyMslAgain = compileTileIrToMsl(
  parseTileIrFile("../fixtures/vec_add_axpy.tileir"), "axpy", threads = 4)

proc expect(msl, fragment: string) =
  doAssert fragment in msl, "emitted MSL misses:\n  " & fragment &
    "\nemitted:\n" & msl

proc checkSignature() =
  expect(AxpyMsl, "kernel void axpy(")
  # pointers take explicit buffer slots in launch order, the store target
  # first (the launch binds the output to slot 0), the shape scalars bind
  # by value between them and consume a slot each
  expect(AxpyMsl, "device float* ptr_c [[buffer(0)]]")
  expect(AxpyMsl, "device float* ptr_a [[buffer(2)]]")
  expect(AxpyMsl, "device float* ptr_b [[buffer(4)]]")
  # scalar shape arguments bind by value next to their slot
  expect(AxpyMsl, "constant int& shape_a")
  expect(AxpyMsl, "constant int& shape_c")
  # coordinate builtins are emitted as uint3 attribute parameters
  expect(AxpyMsl, "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]")
  expect(AxpyMsl, "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]")

proc checkBody() =
  # block-level index math through the coordinate builtins, cast to int
  expect(AxpyMsl, "thread_position_in_threadgroup.x")
  expect(AxpyMsl, "threadgroup_position_in_grid.x")
  # fused value loop over the thread-value slots, 4 threads x 2 values,
  # while-form loop idiom of the CIRv2 lowering
  expect(AxpyMsl, "int tt_v;")
  expect(AxpyMsl, "tt_v = 0;")
  expect(AxpyMsl, "while ((tt_v < 2)) {")
  expect(AxpyMsl, "tt_v = (tt_v + 1);")
  # loads through the view layout, the tile dataflow in program order
  expect(AxpyMsl, "tile_a = ptr_a[tt_off_pv_a]")
  expect(AxpyMsl, "tile_b = ptr_b[tt_off_pv_b]")
  expect(AxpyMsl, "scaled = (alpha * tile_a)")
  expect(AxpyMsl, "sum = (scaled + tile_b)")
  expect(AxpyMsl, "ptr_c[tt_off_pv_c] = sum")
  # splat constant, f32 literal
  expect(AxpyMsl, "alpha = 3.0f")

proc checkStable() =
  doAssert AxpyMsl == AxpyMslAgain,
    "two compiles of the same fixture disagree, the emitted MSL is not stable"
  doAssert AxpyMsl.len > 0, "empty MSL"

when isMainModule:
  checkSignature()
  checkBody()
  checkStable()
  echo "[OK] axpy compiles to stable MSL through the bridge and crucible's pipeline"
