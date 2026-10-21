# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/math
import ./tiles
import ./tile_foreach

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
