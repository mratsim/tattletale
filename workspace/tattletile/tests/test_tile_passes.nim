# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile passes: view resolution, thread-value layout inference, lowering
##
## Test that the axpy fixture resolves to the python reference
## implementation layouts and program ops, thread t owning tile coordinates
## {t, t + 4}, that the gemm fixture resolves its 2-d views and mmaf op,
## and that out-of-subset shapes refuse loudly.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_tile_passes.nim

import std/os
import std/strutils
import std/tables

import workspace/tattletile/tattletile

const FixturesDir = currentSourcePath().parentDir().parentDir() / "fixtures"

proc axpyEntry(): Entry =
  parseTileIr(readFile(FixturesDir / "vec_add_axpy.tileir")).getEntry("axpy")

proc checkResolveViews() =
  let entry = axpyEntry()
  let views = entry.resolveViews()
  doAssert views.len == 3, "three partition views, got " & $views.len
  let a = views["pv_a"]
  doAssert a.base == "ptr_a", "pv_a base"
  doAssert a.shapeEntries == @[ssaEntry("shape_a")], "pv_a shape entries"
  doAssert a.strideEntries == @[ssaEntry("st")], "pv_a stride entries"
  doAssert a.tileShape == @[8], "pv_a tile shape"
  doAssert views["pv_c"].base == "ptr_c", "pv_c base"

proc checkInferLayouts() =
  let entry = axpyEntry()
  let layouts = entry.inferThreadValueLayouts(threads = 4)
  # every tile<8xf32> value: thread t owns {t, t + 4}, the coalesced partition
  for name in ["tile_a", "tile_b", "alpha", "scaled", "sum"]:
    doAssert name in layouts, name & " has a layout assignment"
    let info = layouts[name]
    doAssert info.tileShape == @[8], name & " tile shape"
    doAssert info.elem == eF32, name & " element"
    doAssert info.threads == 4, name & " thread count"
    doAssert info.tv.threads == 4 and info.tv.values == 2, name & " tv shape"
  # ownership map through the tv call, the partition-identity check
  var owned: seq[seq[int]] = @[]
  for t in 0 ..< 4:
    var row: seq[int] = @[]
    for v in 0 ..< 2:
      row.add layouts["tile_a"].tv.tvCall(t, v)
    owned.add(row)
  doAssert owned == @[@[0, 4], @[1, 5], @[2, 6], @[3, 7]],
    "thread t owns {t, t + 4}, got " & $owned
  # 0-d i32 constant: replicated, no partition
  doAssert "st" notin layouts, "0-d scalars carry no partition"

proc checkLowerToProgram() =
  let entry = axpyEntry()
  let prog = entry.runTilePipeline(threads = 4)
  doAssert prog.name == "axpy", "program name"
  doAssert prog.args.len == 6, "six program args"
  var kinds: seq[ProgramOpKind] = @[]
  for op in prog.ops:
    kinds.add op.kind
  doAssert kinds == @[pgConstScalar, pgGetBlockId, pgLoadTile, pgLoadTile,
                      pgConstTile, pgBinary, pgBinary, pgStoreTile, pgYield],
    "program op order, got " & $kinds
  let mul = prog.ops[5]
  doAssert mul.binKind == bkMulf and mul.binLhs == "alpha" and mul.binRhs == "tile_a",
    "mulf alpha, tile_a"
  let add = prog.ops[6]
  doAssert add.binKind == bkAddf and add.binLhs == "scaled" and add.binRhs == "tile_b",
    "addf scaled, tile_b"
  let store = prog.ops[7]
  doAssert store.storeValue == "sum" and store.storeView == "pv_c", "store of sum to pv_c"
  doAssert store.storeIndices == @["bid_x"], "store index"

proc gemmEntry(): Entry =
  parseTileIr(readFile(FixturesDir / "gemm.tileir")).getEntry("gemm")

proc checkGemmViews() =
  let entry = gemmEntry()
  let views = entry.resolveViews()
  doAssert views.len == 3, "three partition views, got " & $views.len
  let a = views["pv_a"]
  doAssert a.base == "ptr_a", "pv_a base"
  doAssert a.shapeEntries == @[staticEntry(16), staticEntry(16)], "pv_a shape entries"
  doAssert a.strideEntries == @[staticEntry(16), staticEntry(1)], "pv_a stride entries"
  doAssert a.tileShape == @[16, 16], "pv_a tile shape"
  doAssert views["pv_b"].base == "ptr_b", "pv_b base"
  doAssert views["pv_c"].base == "ptr_c", "pv_c base"

proc checkGemmLayouts() =
  let entry = gemmEntry()
  let layouts = entry.inferThreadValueLayouts(threads = 4)
  # every tile<16x16xf32> value: thread t owns {t, t + 4, ..., t + 252}
  for name in ["tile_a", "tile_b", "zero", "acc"]:
    doAssert name in layouts, name & " has a layout assignment"
    let info = layouts[name]
    doAssert info.tileShape == @[16, 16], name & " tile shape"
    doAssert info.elem == eF32, name & " element"
    doAssert info.tv.threads == 4 and info.tv.values == 64, name & " tv shape"
  var owned: seq[seq[int]] = @[]
  for t in 0 ..< 4:
    var row: seq[int] = @[]
    for v in 0 ..< 3:
      row.add layouts["acc"].tv.tvCall(t, v)
    owned.add(row)
  doAssert owned == @[@[0, 4, 8], @[1, 5, 9], @[2, 6, 10], @[3, 7, 11]],
    "thread t owns {t, t + 4, ...}, got " & $owned
  doAssert layouts["acc"].tv.tvCall(3, 63) == 255, "the last owned coordinate"

proc checkGemmLowerToProgram() =
  let entry = gemmEntry()
  let prog = entry.runTilePipeline(threads = 4)
  doAssert prog.name == "gemm", "program name"
  doAssert prog.args.len == 3, "three program args, the static views need no shape args"
  var kinds: seq[ProgramOpKind] = @[]
  for op in prog.ops:
    kinds.add op.kind
  doAssert kinds == @[pgGetBlockId, pgLoadTile, pgLoadTile, pgConstTile,
                      pgMma, pgStoreTile, pgYield],
    "program op order, got " & $kinds
  let mma = prog.ops[4]
  doAssert mma.mmaA == "tile_a" and mma.mmaB == "tile_b" and mma.mmaAcc == "zero",
    "mmaf tile_a, tile_b, zero"
  doAssert mma.mmaInfo.tileShape == @[16, 16], "mmaf result tile shape"
  let store = prog.ops[5]
  doAssert store.storeValue == "acc" and store.storeView == "pv_c", "store of acc to pv_c"
  doAssert store.storeIndices == @["bid_x", "bid_y"], "2-d store index"

proc refuses(body: proc, construct: string, kernel = "axpy") =
  ## Runs `body` and requires a SubsetError whose message names `construct`
  ## and the kernel.
  try:
    body()
    doAssert false, "expected a refusal naming '" & construct & "'"
  except SubsetError as e:
    doAssert construct in e.msg,
      "refusal does not name '" & construct & "': " & e.msg
    doAssert "kernel '" & kernel & "'" in e.msg,
      "refusal does not name the kernel: " & e.msg

proc textWithFragment(fragment: string): string =
  ## Axpy module text with one op line replaced, keeping the surrounding
  ## entry structure for pass-level refusal tests.
  let text = readFile(FixturesDir / "vec_add_axpy.tileir")
  text.replace("%sum = addf %scaled, %tile_b : tile<8xf32>", fragment)

proc checkRefusals() =
  let shard = textWithFragment("%bad = mmaf %scaled, %tile_b, %sum : tile<8xf32>, tile<8xf32>, tile<8xf32>")
  refuses(proc() = discard parseTileIr(shard).getEntry("axpy").runTilePipeline(threads = 4),
          "mmaf")

  block: # thread count that does not divide the tile
    let entry = axpyEntry()
    refuses(proc() = discard entry.runTilePipeline(threads = 3), "thread-value partition")

  block: # a load whose token chain is broken
    let text = readFile(FixturesDir / "vec_add_axpy.tileir")
      .replace("token = %tok : partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>, tile<i32> -> tile<8xf32>, token",
               "token = %tok_b : partition_view<tile=(8), tensor_view<?xf32, strides=[?]>>, tile<i32> -> tile<8xf32>, token")
    refuses(proc() = discard parseTileIr(text).getEntry("axpy").runTilePipeline(threads = 4),
            "load_view_tko")

  block: # a dynamically shaped tile element op, the layout pass refuses it
    let text = readFile(FixturesDir / "vec_add_axpy.tileir")
      .replace("%sum = addf %scaled, %tile_b : tile<8xf32>",
               "%sum = addf %scaled, %tile_b : tile<?xf32>")
    refuses(proc() = discard parseTileIr(text).getEntry("axpy").runTilePipeline(threads = 4),
            "addf")

proc checkNewOpRefusals() =
  ## Compiler-emitted op kinds parse (test_tile_ir_captured), the tile-passes lowering subset ends at yield.
  ## Every refusal names construct and kernel.
  template entryModule(body: string): string =
    "cuda_tile.module @k { entry @mykernel() {" & body & " } }"
  block: # assume
    let m = parseTileIr(entryModule(
      " %c = constant <f32: 1.0> : tile<8xf32>" &
      " %a = assume bounded<0, ?>, %c : tile<8xf32> yield"))
    refuses(proc() = discard m.getEntry("mykernel").runTilePipeline(threads = 4),
            "assume", "mykernel")
  block: # reshape
    let m = parseTileIr(entryModule(
      " %c = constant <f32: 1.0> : tile<8xf32>" &
      " %r = reshape %c : tile<8xf32> -> tile<2x4xf32> yield"))
    refuses(proc() = discard m.getEntry("mykernel").runTilePipeline(threads = 4),
            "reshape", "mykernel")
  block: # broadcast
    let m = parseTileIr(entryModule(
      " %c = constant <f32: 1.0> : tile<8xf32>" &
      " %b = broadcast %c : tile<8xf32> -> tile<2x4xf32> yield"))
    refuses(proc() = discard m.getEntry("mykernel").runTilePipeline(threads = 4),
            "broadcast", "mykernel")
  block: # for region
    let m = parseTileIr(entryModule(
      " %z = constant <i32: 0> : tile<i32>" &
      " %o = constant <i32: 8> : tile<i32>" &
      " %r = for %i in (%z to %o, step %z) : tile<i32> { continue }" &
      " yield"))
    refuses(proc() = discard m.getEntry("mykernel").runTilePipeline(threads = 4),
            "for", "mykernel")
  block: # return terminator with no operand
    let m = parseTileIr(entryModule(" %c = constant <f32: 1.0> : tile<8xf32> return"))
    refuses(proc() = discard m.getEntry("mykernel").runTilePipeline(threads = 4),
            "return", "mykernel")

proc checkGemmRefusals() =
  block: # an mmaf whose accumulator operand has a different tile shape
    let text = readFile(FixturesDir / "gemm.tileir")
      .replace("%acc = mmaf %tile_a, %tile_b, %zero : tile<16x16xf32>, tile<16x16xf32>, tile<16x16xf32>",
               "%acc = mmaf %tile_a, %tile_b, %zero : tile<8x8xf32>, tile<16x16xf32>, tile<16x16xf32>")
    refuses(proc() = discard parseTileIr(text).getEntry("gemm").runTilePipeline(threads = 4),
            "mmaf", "gemm")
  block: # an mmaf operand that is not a tile value
    let text = readFile(FixturesDir / "gemm.tileir")
      .replace("%acc = mmaf %tile_a, %tile_b, %zero",
               "%acc = mmaf %tok, %tile_b, %zero")
    refuses(proc() = discard parseTileIr(text).getEntry("gemm").runTilePipeline(threads = 4),
            "mmaf", "gemm")
  block: # a 2-d view indexed with one block coordinate
    let text = readFile(FixturesDir / "gemm.tileir")
      .replace("%pv_c[%bid_x, %bid_y]", "%pv_c[%bid_x]")
    refuses(proc() = discard parseTileIr(text).getEntry("gemm").runTilePipeline(threads = 4),
            "store_view_tko", "gemm")

when isMainModule:
  checkResolveViews()
  checkInferLayouts()
  checkLowerToProgram()
  checkGemmViews()
  checkGemmLayouts()
  checkGemmLowerToProgram()
  checkNewOpRefusals()
  checkRefusals()
  checkGemmRefusals()
  echo "[OK] tile passes: axpy and gemm resolve to the reference layouts, ownership and program ops"
