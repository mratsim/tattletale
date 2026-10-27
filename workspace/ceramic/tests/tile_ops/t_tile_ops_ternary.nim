# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test that the ternary tile ops mulsub and where compute element-wise
## over host register tiles in dst and value form, beside the muladd family.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests workspace/ceramic/tests/tile_ops/t_tile_ops_ternary.nim

import workspace/ceramic/src/tile_algebra

proc mulsubCase =
  ## mulsub in dst form and in value form
  var a: tile(float32, (4, 4))
  var b: tile(float32, (4, 4))
  var c: tile(float32, (4, 4))
  for i in 0 ..< 16:
    a.data[i] = float32(i)
    b.data[i] = float32(i) + 1.0'f32
    c.data[i] = float32(i) + 2.0'f32
  var d: tile(float32, (4, 4))
  d.mulsub(a, b, c)
  for i in 0 ..< 16:
    doAssert d.data[i] == float32(i) * (float32(i) + 1.0'f32) -
      (float32(i) + 2.0'f32), "mulsub dst at slot " & $i
  let e = mulsub(a, b, c)
  for i in 0 ..< 16:
    doAssert e.data[i] == float32(i) * (float32(i) + 1.0'f32) -
      (float32(i) + 2.0'f32), "mulsub value at slot " & $i

proc whereCase =
  ## where in dst and value form, with a tile and with a scalar fill.
  ## The cond is a boolean tile over the same shape as the operands.
  var cond: tile(bool, (4, 4))
  var a: tile(float32, (4, 4))
  for i in 0 ..< 16:
    cond.data[i] = (i mod 3) == 0
    a.data[i] = float32(i)

  block:
    var keep: tile(float32, (4, 4))
    for i in 0 ..< 16:
      keep.data[i] = a.data[i]
    keep.where(cond, a, -1.0'f32)
    for i in 0 ..< 16:
      let expected = if (i mod 3) == 0: float32(i) else: -1.0'f32
      doAssert keep.data[i] == expected, "where dst scalar at slot " & $i

  block:
    let sel = where(cond, a, -1.0'f32)
    for i in 0 ..< 16:
      let expected = if (i mod 3) == 0: float32(i) else: -1.0'f32
      doAssert sel.data[i] == expected, "where value scalar at slot " & $i

  block:
    # both fills scalar
    var fill: tile(float32, (4, 4))
    fill.where(cond, 1.5'f32, -1.5'f32)
    for i in 0 ..< 16:
      let expected = if (i mod 3) == 0: 1.5'f32 else: -1.5'f32
      doAssert fill.data[i] == expected, "where dst two scalars at slot " & $i
    let picked = where(cond, 1.5'f32, -1.5'f32)
    for i in 0 ..< 16:
      let expected = if (i mod 3) == 0: 1.5'f32 else: -1.5'f32
      doAssert picked.data[i] == expected, "where value two scalars at slot " & $i

  block:
    var b: tile(float32, (4, 4))
    for i in 0 ..< 16:
      b.data[i] = float32(i) + 10.0'f32
    var d: tile(float32, (4, 4))
    d.where(cond, a, b)
    for i in 0 ..< 16:
      let expected = if (i mod 3) == 0: float32(i) else: float32(i) + 10.0'f32
      doAssert d.data[i] == expected, "where dst tile at slot " & $i

proc main =
  mulsubCase()
  whereCase()
  echo "t_tile_ops_ternary: all cases passed"

when isMainModule:
  main()
