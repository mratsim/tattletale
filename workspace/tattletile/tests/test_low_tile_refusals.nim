# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CIRv2 bridge: loud refusals and one offset local per view
##
## Out-of-subset kernels refuse through SubsetError at the bridge.
## An in-place kernel (load + store through one view) compiles with exactly one declaration per offset local:
##
## - mixed tile shapes in one kernel refuse
## - one view addressed with two block index lists refuses
## - duplicate SSA result names refuse on the atom path
## - an atom store view of a different partition tile refuses, as does a store target bound nowhere
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_low_tile_refusals.nim

import std/strutils

import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/tattletile/tattletile

proc refuses(body: proc, construct, detail: string, kernel = "probe") =
  ## Runs `body` and requires a SubsetError whose message names `construct`,
  ## `detail` and the kernel.
  try:
    body()
    doAssert false, "expected a refusal naming '" & construct & "'"
  except SubsetError as e:
    doAssert construct in e.msg,
      "refusal does not name '" & construct & "': " & e.msg
    doAssert detail in e.msg,
      "refusal does not name '" & detail & "': " & e.msg
    doAssert "kernel '" & kernel & "'" in e.msg,
      "refusal does not name the kernel: " & e.msg

template bridgeModule(head, body: string): string =
  ## One-entry Tile IR module around a hand-written op body.
  "cuda_tile.module @k { entry @probe(" & head & ") {" & body & " } }"

const AxpyHead = "%ptr_x: tile<ptr<f32>>, %shape_x: tile<i32>"

const MixedShapes = bridgeModule(AxpyHead,
  " %st = constant <i32: 1> : tile<i32>" &
  " %tok = make_token : token" &
  " %tv_a = make_tensor_view %ptr_x, shape = [%shape_x, %shape_x], strides = [%st, %st] : tile<i32> -> tensor_view<?x?xf32, strides=[?,?]>" &
  " %tv_b = make_tensor_view %ptr_x, shape = [%shape_x], strides = [%st] : tile<i32> -> tensor_view<?xf32, strides=[?]>" &
  " %pv_a = make_partition_view %tv_a : partition_view<tile=(2x4), tensor_view<?x?xf32, strides=[?,?]>>" &
  " %pv_b = make_partition_view %tv_b : partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>" &
  " %bid_x, %bid_y, %bid_z = get_tile_block_id : tile<i32>" &
  " %tile_a, %tok_a = load_view_tko weak %pv_a[%bid_x, %bid_y] token = %tok : partition_view<tile=(2x4), tensor_view<?x?xf32, strides=[?,?]>>, tile<i32> -> tile<2x4xf32>, token" &
  " %tile_b, %tok_b = load_view_tko weak %pv_b[%bid_x] token = %tok_a : partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>, tile<i32> -> tile<8xf32>, token" &
  " %tok_c = store_view_tko weak %tile_b, %pv_b[%bid_x] token = %tok_b : tile<8xf32>, partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>, tile<i32> -> token" &
  " yield")
  ## Two tile shapes of one size in one kernel share one coordinate decomposition,
  ## 8 elements each with one thread-value partition.

const InPlace = bridgeModule(AxpyHead,
  " %st = constant <i32: 1> : tile<i32>" &
  " %tok = make_token : token" &
  " %tv_x = make_tensor_view %ptr_x, shape = [%shape_x], strides = [%st] : tile<i32> -> tensor_view<?xf32, strides=[?]>" &
  " %pv_x = make_partition_view %tv_x : partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>" &
  " %bid_x, %bid_y, %bid_z = get_tile_block_id : tile<i32>" &
  " %tile_x, %tok_a = load_view_tko weak %pv_x[%bid_x] token = %tok : partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>, tile<i32> -> tile<8xf32>, token" &
  " %alpha = constant <f32: 3.0> : tile<8xf32>" &
  " %scaled = mulf %alpha, %tile_x : tile<8xf32>" &
  " %tok_out = store_view_tko weak %scaled, %pv_x[%bid_x] token = %tok_a : tile<8xf32>, partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>, tile<i32> -> token" &
  " yield")
  ## Load and store through one view: the natural in-place kernel.

const GemmHead = "%ptr_x: tile<ptr<f32>>"

const GemmBody =
  " %tv_a = make_tensor_view %ptr_x, shape = [16, 16], strides = [16, 1] : tensor_view<16x16xf32, strides=[16,1]>" &
  " %tv_b = make_tensor_view %ptr_x, shape = [16, 16], strides = [16, 1] : tensor_view<16x16xf32, strides=[16,1]>" &
  " %tv_c = make_tensor_view %ptr_x, shape = [16, 16], strides = [16, 1] : tensor_view<16x16xf32, strides=[16,1]>" &
  " %pv_a = make_partition_view %tv_a : partition_view<tile=(16x16), tensor_view<16x16xf32, strides=[16,1]>>" &
  " %pv_b = make_partition_view %tv_b : partition_view<tile=(16x16), tensor_view<16x16xf32, strides=[16,1]>>" &
  " %pv_c = make_partition_view %tv_c : partition_view<tile=(16x16), tensor_view<16x16xf32, strides=[16,1]>>" &
  " %bid_x, %bid_y, %bid_z = get_tile_block_id : tile<i32>" &
  " %tile_a, %tok_a = load_view_tko weak %pv_a[%bid_x, %bid_y] token = %tok : partition_view<tile=(16x16), tensor_view<16x16xf32, strides=[16,1]>>, tile<i32> -> tile<16x16xf32>, token" &
  " %tile_b, %tok_b = load_view_tko weak %pv_b[%bid_x, %bid_y] token = %tok_a : partition_view<tile=(16x16), tensor_view<16x16xf32, strides=[16,1]>>, tile<i32> -> tile<16x16xf32>, token"

const AtomDupNames = bridgeModule(GemmHead,
  " %tok = make_token : token" & GemmBody &
  " %zero = constant <f32: 0.0> : tile<16x16xf32>" &
  " %zero = constant <f32: 1.0> : tile<16x16xf32>" &
  " %acc = mmaf %tile_a, %tile_b, %zero : tile<16x16xf32>, tile<16x16xf32>, tile<16x16xf32>" &
  " %tok_out = store_view_tko weak %acc, %pv_c[%bid_x, %bid_y] token = %tok_b : tile<16x16xf32>, partition_view<tile=(16x16), tensor_view<16x16xf32, strides=[16,1]>>, tile<i32> -> token" &
  " yield")
  ## Two constants named %zero: the atom accumulator init must not silently
  ## become the last splat's value.

const AtomStoreShape = bridgeModule(GemmHead,
  " %tok = make_token : token" &
  " %tv_a = make_tensor_view %ptr_x, shape = [16, 16], strides = [16, 1] : tensor_view<16x16xf32, strides=[16,1]>" &
  " %tv_b = make_tensor_view %ptr_x, shape = [16, 16], strides = [16, 1] : tensor_view<16x16xf32, strides=[16,1]>" &
  " %tv_c = make_tensor_view %ptr_x, shape = [8, 32], strides = [32, 1] : tensor_view<8x32xf32, strides=[32,1]>" &
  " %pv_a = make_partition_view %tv_a : partition_view<tile=(16x16), tensor_view<16x16xf32, strides=[16,1]>>" &
  " %pv_b = make_partition_view %tv_b : partition_view<tile=(16x16), tensor_view<16x16xf32, strides=[16,1]>>" &
  " %pv_c = make_partition_view %tv_c : partition_view<tile=(8x32), tensor_view<8x32xf32, strides=[32,1]>>" &
  " %bid_x, %bid_y, %bid_z = get_tile_block_id : tile<i32>" &
  " %tile_a, %tok_a = load_view_tko weak %pv_a[%bid_x, %bid_y] token = %tok : partition_view<tile=(16x16), tensor_view<16x16xf32, strides=[16,1]>>, tile<i32> -> tile<16x16xf32>, token" &
  " %tile_b, %tok_b = load_view_tko weak %pv_b[%bid_x, %bid_y] token = %tok_a : partition_view<tile=(16x16), tensor_view<16x16xf32, strides=[16,1]>>, tile<i32> -> tile<16x16xf32>, token" &
  " %zero = constant <f32: 0.0> : tile<16x16xf32>" &
  " %acc = mmaf %tile_a, %tile_b, %zero : tile<16x16xf32>, tile<16x16xf32>, tile<16x16xf32>" &
  " %tok_out = store_view_tko weak %acc, %pv_c[%bid_x, %bid_y] token = %tok_b : tile<16x16xf32>, partition_view<tile=(8x32), tensor_view<8x32xf32, strides=[32,1]>>, tile<i32> -> token" &
  " yield")
  ## Same-size store view tile of a different shape: the fragment addressing
  ## derives from the mmaf tile and reads the strides of the store view.

const UnboundStoreTarget = bridgeModule(AxpyHead,
  " %st = constant <i32: 1> : tile<i32>" &
  " %tok = make_token : token" &
  " %tv_x = make_tensor_view %ptr_x, shape = [%ghost], strides = [%st] : tile<i32> -> tensor_view<?xf32, strides=[?]>" &
  " %pv_x = make_partition_view %tv_x : partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>" &
  " %bid_x, %bid_y, %bid_z = get_tile_block_id : tile<i32>" &
  " %tile_x, %tok_a = load_view_tko weak %pv_x[%bid_x] token = %tok : partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>, tile<i32> -> tile<8xf32>, token" &
  " %tok_out = store_view_tko weak %tile_x, %pv_x[%bid_x] token = %tok_a : tile<8xf32>, partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>, tile<i32> -> token" &
  " yield")
  ## Store view shape entry bound nowhere in the kernel.

proc checkBridgeRefusals() =
  refuses(proc() = discard parseTileIr(MixedShapes).getEntry("probe")
      .runTilePipeline(4).lowerTileToCirv2(false),
    "thread-value partitions", "disagree on the tile shape")
  refuses(proc() = discard parseTileIr(AtomDupNames).getEntry("probe")
      .runTilePipeline(32).lowerTileToCirv2(true),
    "constant %zero", "redefines a kernel-level name")
  refuses(proc() = discard parseTileIr(AtomStoreShape).getEntry("probe")
      .runTilePipeline(32).lowerTileToCirv2(true),
    "store_view_tko", "partition tile")
  refuses(proc() = discard parseTileIr(UnboundStoreTarget).getEntry("probe")
      .runTilePipeline(4).lowerTileToCirv2(false),
    "store_view_tko", "is not bound in the kernel")

static:
  # compile-time entry: the same refusals raise, catchable in the VM
  try:
    discard compileTileIrToMsl(parseTileIr(MixedShapes), "probe", 4)
    doAssert false, "expected the mixed-shape kernel to refuse at compile time"
  except SubsetError as e:
    doAssert "thread-value partitions" in e.msg, e.msg
  try:
    discard compileTileIrToMsl(parseTileIr(AtomDupNames), "probe", 32)
    doAssert false, "expected the duplicate-name kernel to refuse at compile time"
  except SubsetError as e:
    doAssert "redefines a kernel-level name" in e.msg, e.msg

  # in-place kernel: exactly one declaration per offset local
  const InPlaceMsl = compileTileIrToMsl(parseTileIr(InPlace), "probe", 4)
  doAssert InPlaceMsl.count("int tt_block_off_pv_x = ") == 1,
    "one block offset local for the view, got " &
    $InPlaceMsl.count("int tt_block_off_pv_x = ")
  doAssert InPlaceMsl.count("int tt_off_pv_x = ") == 1,
    "one per-thread offset local for the view, got " &
    $InPlaceMsl.count("int tt_off_pv_x = ")

when isMainModule:
  checkBridgeRefusals()
  echo "[OK] CIRv2 bridge: out-of-subset kernels refuse loudly, the in-place kernel dedupes its offset locals"
