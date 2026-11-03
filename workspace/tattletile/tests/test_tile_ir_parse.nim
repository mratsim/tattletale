# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile IR parser: fixture acceptance and print round-trip fixed point
##
## Test that every fixture parses, that printing reproduces the parsed module
## exactly (parse o print o parse equals the first parse), and that zero-padded
## integer tokens parse by value.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_tile_ir_parse.nim

import std/os
import std/strutils

import workspace/tattletile/tattletile

const FixturesDir = currentSourcePath().parentDir().parentDir() / "fixtures"

proc checkRoundTrip(name: string) =
  let text = readFile(FixturesDir / name)
  let m1 = parseTileIr(text)
  let m2 = parseTileIr(printTileIr(m1))
  let m3 = parseTileIr(printTileIr(m2))
  doAssert m1 == m2, name & ": print o parse is not the identity\n" & printTileIr(m1)
  doAssert m2 == m3, name & ": not a fixed point"

proc checkAxpyShape(m: TileIrModule) =
  doAssert m.name == "kernels", "module name"
  doAssert m.entries.len == 1, "one entry"
  let e = m.entries[0]
  doAssert e.name == "axpy", "entry name"
  doAssert e.args.len == 6, "six arguments"
  doAssert e.args[0].name == "ptr_a", "first arg name"
  doAssert e.args[0].typ == ptrTile(eF32), "first arg type"
  doAssert e.argType("shape_a") == scalarTile(eI32), "shape arg type"
  doAssert e.ops.len == 16, "op count"
  doAssert e.ops[0].kind == okConstant, "op 0 kind"
  doAssert e.ops[0].constValue.scalar == intScalar(1), "op 0 value"
  doAssert e.ops[0].resultTypes[0] == scalarTile(eI32), "op 0 type"
  doAssert e.ops[2].kind == okMakeTensorView, "op 2 kind"
  doAssert e.ops[2].dynShape == @[ssaEntry("shape_a")], "op 2 dynamic shape"
  doAssert e.ops[2].dynStrides == @[ssaEntry("st")], "op 2 dynamic strides"
  doAssert e.ops[2].resultTypes[0].kind == tkTensorView, "op 2 result type kind"
  doAssert e.ops[2].resultTypes[0].viewShape == @[DynamicDim], "op 2 dynamic view dim"
  doAssert e.ops[6].kind == okMakePartitionView, "op 6 kind"
  doAssert e.ops[6].resultTypes[0].pvTile == @[8], "op 6 tile shape"
  doAssert e.ops[8].kind == okGetTileBlockId, "op 8 kind"
  doAssert e.ops[8].results == @["bid_x", "bid_y", "bid_z"], "op 8 results"
  doAssert e.ops[9].kind == okLoadViewTko, "op 9 kind"
  doAssert e.ops[9].token == "tok", "op 9 token input"
  doAssert e.ops[11].kind == okConstant, "op 11 kind"
  doAssert e.ops[11].constValue.scalar == floatScalar(3.0), "op 11 value"
  doAssert e.ops[12].kind == okMulf, "op 12 kind"
  doAssert e.ops[13].kind == okAddf, "op 13 kind"
  doAssert e.ops[14].kind == okStoreViewTko, "op 14 kind"
  doAssert e.ops[15].kind == okYield, "op 15 kind"
  doAssert e.ops[2].readNames() == @["ptr_a", "shape_a", "st"], "op 2 reads"
  doAssert e.ops[9].readNames() == @["pv_a", "bid_x", "tok"], "op 9 reads"

proc checkGemmShape(m: TileIrModule) =
  doAssert m.entries.len == 1, "one entry"
  let e = m.entries[0]
  doAssert e.name == "gemm", "entry name"
  doAssert e.args.len == 3, "three arguments"
  doAssert e.ops.len == 14, "op count"
  doAssert e.ops[0].kind == okMakeToken, "op 0 kind"
  doAssert e.ops[1].kind == okMakeTensorView, "op 1 kind"
  doAssert e.ops[1].dynShape == @[staticEntry(16), staticEntry(16)], "op 1 static shape"
  doAssert e.ops[1].dynStrides == @[staticEntry(16), staticEntry(1)], "op 1 static strides"
  doAssert e.ops[1].resultTypes[0].viewShape == @[16, 16], "op 1 view shape"
  doAssert "tile<i32> -> " notin opToText(e.ops[1]), "static view prints no operand-type prefix"
  doAssert e.ops[7].kind == okGetTileBlockId, "op 7 kind"
  doAssert e.ops[8].kind == okLoadViewTko, "op 8 kind"
  doAssert e.ops[8].indices == @["bid_x", "bid_y"], "op 8 indices"
  doAssert e.ops[11].kind == okMmaf, "op 11 kind"
  doAssert e.ops[11].resultTypes.len == 3, "op 11 three result types"
  doAssert e.ops[12].kind == okStoreViewTko, "op 12 kind"
  doAssert e.ops[13].kind == okYield, "op 13 kind"

proc checkZeroPaddedIntegers() =
  ## Zero-padded integer tokens parse by value: the digit prescreen counts
  ## significant digits only and the i32 value check decides.
  let m = parseTileIr(
    "cuda_tile.module @k { entry @mykernel(%p: tile<ptr<f32>>) {" &
    " %a = constant <f32: 1.0> : tile<00000008xf32>" &
    " %b = constant <f32: 1.0> : tile<00000000000000000001xf32> } }")
  let e = m.entries[0]
  doAssert e.ops[0].resultTypes[0].tileShape == @[8], "00000008 parses as 8"
  doAssert e.ops[1].resultTypes[0].tileShape == @[1], "a 20-digit zero-padded 1 parses as 1"

checkRoundTrip("vec_add_axpy.tileir")
checkRoundTrip("gemm.tileir")
checkAxpyShape(parseTileIr(readFile(FixturesDir / "vec_add_axpy.tileir")))
checkGemmShape(parseTileIr(readFile(FixturesDir / "gemm.tileir")))
checkZeroPaddedIntegers()
echo "test_tile_ir_parse: PASS"
