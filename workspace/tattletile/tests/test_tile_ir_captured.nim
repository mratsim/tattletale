# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile IR parser: compiler-emitted fixtures
##
## Test that the cutile-rs-captured fixtures parse and print to a round-trip fixed
## point (parse o print o parse), the real-IR-only constructs included.
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_tile_ir_captured.nim

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

proc checkVecAdd(m: TileIrModule) =
  doAssert m.name == "my_module", "module name"
  doAssert m.entries.len == 1, "one entry"
  let e = m.entries[0]
  doAssert e.name == "saxpy_entry", "entry name"
  doAssert e.diName == "saxpy", "di_name entry attribute"
  doAssert e.args.len == 15, "fifteen arguments"
  doAssert e.args[0].typ == ptrTile(eF32), "first arg type"
  doAssert e.args[9].typ == scalarTile(eF32), "scalar arg type"
  doAssert e.ops[^1].kind == okReturn, "bare return terminator"
  # assume chain: bounded<0, ?> then div_by<4> on the same value
  doAssert e.ops[2].kind == okAssume, "op 2 kind"
  doAssert e.ops[2].assumeKind == akBounded, "op 2 predicate"
  doAssert e.ops[2].assumeLo == 0, "op 2 lower bound"
  doAssert e.ops[2].assumeHi == DynamicDim, "op 2 open upper bound"
  doAssert e.ops[3].kind == okAssume, "op 3 kind"
  doAssert e.ops[3].assumeKind == akDivBy, "op 3 predicate"
  doAssert e.ops[3].assumeDiv == 4, "op 3 divisor"
  doAssert e.ops[3].resultTypes[0] == scalarTile(eI32), "op 3 type"
  # partition view with padding_value = zero
  doAssert e.ops[49].kind == okMakePartitionView, "op 49 kind"
  doAssert e.ops[49].resultTypes[0].pvPadding, "op 49 padding_value = zero"
  doAssert e.ops[61].kind == okMakePartitionView, "op 61 kind"
  doAssert not e.ops[61].resultTypes[0].pvPadding, "op 61 unpadded view"
  # scalar enters the tile through reshape + broadcast, not a splat constant
  doAssert e.ops[27].kind == okReshape, "op 27 kind"
  doAssert e.ops[27].operandTypes[0] == scalarTile(eF32), "op 27 input type"
  doAssert e.ops[27].resultTypes[0].tileShape == @[1, 1], "op 27 output shape"
  doAssert e.ops[32].kind == okBroadcast, "op 32 kind"
  doAssert e.ops[32].resultTypes[0].tileShape == @[4, 8], "op 32 output shape"
  # elementwise math and the tko pair stay as in the hand fixtures
  doAssert e.ops[63].kind == okMulf, "op 63 kind"
  doAssert e.ops[64].kind == okAddf, "op 64 kind"
  doAssert e.ops[76].kind == okStoreViewTko, "op 76 kind"
  doAssert e.ops[76].indices == @["93", "94"], "op 76 indices"

proc checkGemm(m: TileIrModule) =
  doAssert m.name == "gemm_module", "module name"
  let e = m.entries[0]
  doAssert e.name == "gemm_entry", "entry name"
  doAssert e.diName == "gemm", "di_name entry attribute"
  doAssert e.args.len == 19, "nineteen arguments"
  doAssert e.ops[^1].kind == okReturn, "bare return terminator"
  # K/BK loop as a for region with one carried accumulator
  let forOp = e.ops[58]
  doAssert forOp.kind == okFor, "op 58 kind"
  doAssert forOp.loopVar == "83", "loop variable"
  doAssert forOp.loopVarType == scalarTile(eI32), "loop variable type"
  doAssert forOp.operands == @["79", "81", "82"], "loop bounds and step"
  doAssert forOp.iterNames == @["84"], "iter name"
  doAssert forOp.iterInits == @["77"], "iter init"
  doAssert forOp.resultTypes[0].tileShape == @[16, 16], "carried type"
  doAssert forOp.body.len == 9, "body op count"
  doAssert forOp.body[^1].kind == okContinue, "region closes with continue"
  doAssert forOp.body[^1].operands == @["94"], "continue carries the accumulator"
  # mmaf inside the region has the real A/B/acc shapes
  doAssert forOp.body[7].kind == okMmaf, "body mmaf kind"
  doAssert forOp.body[7].resultTypes[2].tileShape == @[16, 16], "acc shape"
  # mixed dynamic/static view dims with concrete strides
  doAssert e.ops[15].kind == okMakeTensorView, "op 15 kind"
  doAssert e.ops[15].resultTypes[0].viewShape == @[DynamicDim, 64], "op 15 view shape"
  doAssert e.ops[15].resultTypes[0].viewStrides == @[64, 1], "op 15 strides"

checkRoundTrip("captured_vec_add_axpy.tileir")
checkRoundTrip("captured_gemm.tileir")
checkVecAdd(parseTileIr(readFile(FixturesDir / "captured_vec_add_axpy.tileir")))
checkGemm(parseTileIr(readFile(FixturesDir / "captured_gemm.tileir")))
echo "test_tile_ir_captured: PASS"
