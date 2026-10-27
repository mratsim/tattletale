# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[math, macros]
import workspace/ceramic/src/int_tuples/int_tuples_compiletime
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

func divide*(dst: var Tile, a, b: Tile) {.inline.} =
  forEach var d in dst, x in a, y in b:
    d = x / y

func max*(dst: var Tile, a, b: Tile) {.inline.} =
  forEach var d in dst, x in a, y in b:
    d = max(x, y)

func min*(dst: var Tile, a, b: Tile) {.inline.} =
  forEach var d in dst, x in a, y in b:
    d = min(x, y)

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

# Broadcast compound assignment: the right operand broadcasts over size-1 modes
# -----------------------------------------------------------------------------

func `+.=`*[TA, ShA, StA, TB, ShB, StB](a: var Tile[TA, ShA, StA], b: Tile[TB, ShB, StB]) {.inline.} =
  forEachBroadcast var d in a, x in b:
    d += x

func `-.=`*[TA, ShA, StA, TB, ShB, StB](a: var Tile[TA, ShA, StA], b: Tile[TB, ShB, StB]) {.inline.} =
  forEachBroadcast var d in a, x in b:
    d -= x

func `*.=`*[TA, ShA, StA, TB, ShB, StB](a: var Tile[TA, ShA, StA], b: Tile[TB, ShB, StB]) {.inline.} =
  forEachBroadcast var d in a, x in b:
    d = d * x

func `/.=`*[TA, ShA, StA, TB, ShB, StB](a: var Tile[TA, ShA, StA], b: Tile[TB, ShB, StB]) {.inline.} =
  forEachBroadcast var d in a, x in b:
    d = d / x

func `+=`*(a: var Tile, s: a.T) {.inline.} =
  forEach var d in a:
    d += s

func `-=`*(a: var Tile, s: a.T) {.inline.} =
  forEach var d in a:
    d -= s

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

func max*(a, b: Tile): Tile {.inline.} =
  result.max(a, b)

func min*(a, b: Tile): Tile {.inline.} =
  result.min(a, b)

# Broadcasts
# ----------
# https://github.com/mratsim/Arraymancer/issues/39
# https://github.com/gchanan/pytorch/wiki/Broadcasting-Notes
# https://github.com/mratsim/Arraymancer/pull/410

macro broadcastShape(shA, shB: typed): untyped =
  proc unwrapTypeDesc(n: NimNode): NimNode =
    if n.kind == nnkBracketExpr and n[0].eqIdent("typeDesc"):
      n[1]
    else:
      n

  proc resolveTy(n: NimNode): NimNode =
    ## Expands typeDesc wrappers and named type aliases
    result = unwrapTypeDesc(n)
    while result.kind == nnkSym:
      let impl = result.getImpl
      if impl.kind == nnkTypeDef:
        result = unwrapTypeDesc(impl[2])
      else:
        break

  proc unwrapField(n: NimNode): NimNode =
    if n.kind == nnkIdentDefs:
      n[^2]
    else:
      n

  proc isTupleTy(n: NimNode): bool =
    n.kind in {nnkTupleTy, nnkTupleConstr, nnkPar} and
      (n.kind != nnkPar or n.len > 1)

  proc walk(a, b: NimNode): NimNode =
    let a = resolveTy(unwrapField(a))
    let b = resolveTy(unwrapField(b))
    let aStatic = a.getStaticInt()
    let bStatic = b.getStaticInt()
    if aStatic != DynamicSentinel and bStatic != DynamicSentinel:
      return newLit(max(aStatic, bStatic))
    doAssert aStatic == DynamicSentinel and bStatic == DynamicSentinel,
      "broadcastShape: leaf against tuple shape"
    doAssert isTupleTy(a) and isTupleTy(b) and a.len == b.len,
      "broadcastShape: rank mismatch"
    result = nnkPar.newTree()
    for i in 0 ..< a.len:
      result.add walk(a[i], b[i])

  result = walk(shA.getTypeInst(), shB.getTypeInst())

func `*.`*[TA, ShA, StA, TB, ShB, StB](a: Tile[TA, ShA, StA], b: Tile[TB, ShB, StB]): auto {.inline.} =
  ## Element-wise multiply, broadcasting over size-1 modes.
  var outTile: tile(TA, make_layout(broadcastShape(ShA, ShB)))
  forEachBroadcast var r in outTile, x in a, y in b:
    r = x * y
  outTile

func `+.`*[TA, ShA, StA, TB, ShB, StB](a: Tile[TA, ShA, StA], b: Tile[TB, ShB, StB]): auto {.inline.} =
  var outTile: tile(TA, make_layout(broadcastShape(ShA, ShB)))
  forEachBroadcast var r in outTile, x in a, y in b:
    r = x + y
  outTile

func `-.`*[TA, ShA, StA, TB, ShB, StB](a: Tile[TA, ShA, StA], b: Tile[TB, ShB, StB]): auto {.inline.} =
  var outTile: tile(TA, make_layout(broadcastShape(ShA, ShB)))
  forEachBroadcast var r in outTile, x in a, y in b:
    r = x - y
  outTile

func `/.`*[TA, ShA, StA, TB, ShB, StB](a: Tile[TA, ShA, StA], b: Tile[TB, ShB, StB]): auto {.inline.} =
  var outTile: tile(TA, make_layout(broadcastShape(ShA, ShB)))
  forEachBroadcast var r in outTile, x in a, y in b:
    r = x / y
  outTile

# Scalar forms
# ------------

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
