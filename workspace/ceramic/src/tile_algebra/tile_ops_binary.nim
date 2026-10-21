# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ./tiles
import ./tile_foreach

# ############################################################
#
#            Element-wise binary functions
#
# ############################################################

# Destination buffer
# ------------------

func add*(dst: var Tile, a, b: Tile) {.inline.} =
  forEach var d in dst, x in a, y in b:
    d = x + y

func sub*(dst: var Tile, a, b: Tile) {.inline.} =
  forEach var d in dst, x in a, y in b:
    d = x - y

func mul*(dst: var Tile, a, b: Tile) {.inline.} =
  forEach var d in dst, x in a, y in b:
    d = x * y

# In-place accumulation
# ---------------------

func `+=`*(a: var Tile, b: Tile) {.inline.} =
  forEach var d in a, x in b:
    d += x

func `-=`*(a: var Tile, b: Tile) {.inline.} =
  forEach var d in a, x in b:
    d -= x

func `*=`*(a: var Tile, b: Tile) {.inline.} =
  forEach var d in a, x in b:
    d *= x

func `/=`*(a: var Tile, b: Tile) {.inline.} =
  forEach var d in a, x in b:
    d /= x

func `*=`*(a: var Tile, s: a.T) {.inline.} =
  forEach var d in a:
    d = d * s

func `/=`*(a: var Tile, s: a.T) {.inline.} =
  forEach var d in a:
    d = d / s

# Out-of-place
# ------------

func `+`*(a, b: Tile): Tile {.inline.} =
  result.add(a, b)

func `-`*(a, b: Tile): Tile {.inline.} =
  result.sub(a, b)

func `*.`*(a, b: Tile): Tile {.inline.} =
  ## Element-wise multiplication. Hadamard product.
  result.mul(a, b)

# Broadcasts
# ----------
# https://github.com/mratsim/Arraymancer/issues/39
# https://github.com/gchanan/pytorch/wiki/Broadcasting-Notes
# https://github.com/mratsim/Arraymancer/pull/410

func `*`*(a: Tile, s: a.T): Tile {.inline.} =
  forEach var r in result, x in a:
    r = x * s

template `*`*[T, Sh, St](s: T, a: Tile[T, Sh, St]): Tile[T, Sh, St] =
  a * s

func `/`*(a: Tile, s: a.T): Tile {.inline.} =
  forEach var r in result, x in a:
    r = x / s

func `+.`*(a: Tile, s: a.T): Tile {.inline.} =
  forEach var r in result, x in a:
    r = x + s

func `-.`*(a: Tile, s: a.T): Tile {.inline.} =
  forEach var r in result, x in a:
    r = x - s

func `+.`*[T, Sh, St](s: T, a: Tile[T, Sh, St]): Tile[T, Sh, St] {.inline.} =
  forEach var r in result, x in a:
    r = s + x

func `-.`*[T, Sh, St](s: T, a: Tile[T, Sh, St]): Tile[T, Sh, St] {.inline.} =
  forEach var r in result, x in a:
    r = s - x
