# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/math
import ./tiles
import ./tile_foreach
import workspace/crucible

# ############################################################
#
#             Element-wise unary functions
#
# ############################################################

# Destination buffer
# ------------------

func neg*(dst: var Tile, a: Tile) {.inline.} =
  forEach var d in dst, x in a:
    d = -x

func abs*(dst: var Tile, a: Tile) {.inline.} =
  forEach var d in dst, x in a:
    d = abs(x)

func sqrt*(dst: var Tile, a: Tile) {.inline.} =
  forEach var d in dst, x in a:
    d = sqrt(x)

func rsqrt*(dst: var Tile, a: Tile) {.inline.} =
  forEach var d in dst, x in a:
    d = rsqrt(x)

func exp2*(dst: var Tile, a: Tile) {.inline.} =
  forEach var d in dst, x in a:
    d = exp2(x)

func log2*(dst: var Tile, a: Tile) {.inline.} =
  forEach var d in dst, x in a:
    d = log2(x)

func exp*(dst: var Tile, a: Tile) {.inline.} =
  forEach var d in dst, x in a:
    d = exp(x)

func log*(dst: var Tile, a: Tile) {.inline.} =
  ## natural logarithm, math.log takes an explicit base
  forEach var d in dst, x in a:
    d = ln(x)

# Out-of-place
# ------------

func `-`*(a: Tile): Tile {.inline.} =
  result.neg(a)

func abs*(a: Tile): Tile {.inline.} =
  result.abs(a)

func sqrt*(a: Tile): Tile {.inline.} =
  result.sqrt(a)

func rsqrt*(a: Tile): Tile {.inline.} =
  result.rsqrt(a)

func exp2*(a: Tile): Tile {.inline.} =
  result.exp2(a)

func log2*(a: Tile): Tile {.inline.} =
  result.log2(a)

func exp*(a: Tile): Tile {.inline.} =
  result.exp(a)

func log*(a: Tile): Tile {.inline.} =
  result.log(a)

# In place
# --------

func neg*(a: var Tile) {.inline.} =
  forEach var d in a:
    d = -d

func abs*(a: var Tile) {.inline.} =
  forEach var d in a:
    d = abs(d)

func sqrt*(a: var Tile) {.inline.} =
  forEach var d in a:
    d = sqrt(d)

func rsqrt*(a: var Tile) {.inline.} =
  forEach var d in a:
    d = rsqrt(d)

func exp2*(a: var Tile) {.inline.} =
  forEach var d in a:
    d = exp2(d)

func log2*(a: var Tile) {.inline.} =
  forEach var d in a:
    d = log2(d)

func exp*(a: var Tile) {.inline.} =
  forEach var d in a:
    d = exp(d)

func log*(a: var Tile) {.inline.} =
  forEach var d in a:
    d = ln(d)

template downcast*[TIn: SomeFloat, TOut: float16 or bfloat16 or fp8e4m3 or fp8e5m2, Sh, St](
    src: Tile[TIn, Sh, St],
    _: typedesc[TOut]): auto =
  var outTile: tile(TOut, src.getLayout())
  forEach var d in outTile, x in src:
    when TOut is float16:
      d = toFp16(x)
    elif TOut is bfloat16:
      d = toBf16(x)
    elif TOut is fp8e4m3:
      d = toFp8e4m3(x)
    else:
      d = toFp8e5m2(x)
  outTile

template downcast*[T: SomeFloat, Sh, St](src: Tile[T, Sh, St], _: typedesc[T]): Tile[T, Sh, St] =
  src

template upcast*[TIn: float16 or bfloat16 or fp8e4m3 or fp8e5m2, Sh, St](
    src: Tile[TIn, Sh, St],
    _: typedesc[float32]): auto =
  var outTile: tile(float32, src.getLayout())
  forEach var d in outTile, x in src:
    when TIn is float16:
      d = toFp32(x)
    elif TIn is bfloat16:
      d = toBf32(x)
    elif TIn is fp8e4m3:
      d = toFp32(x)
    else:
      d = toFp32(x)
  outTile

