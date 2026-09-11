# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import defunctional

proc collectScalars() =
  let xs = collect(iota(1, 11).filter(gt(4)))
  doAssert xs == @[5, 6, 7, 8, 9, 10], $xs
  doAssert collect(iota(1, 1).filter(gt(0))).len == 0  # empty range (exclusive stop)

type Longish* = object
  n: int
func longish*(n: int): Longish {.inline.} =
  Longish(n: n)
func matches*(p: Longish; v: openArray[char]): bool {.inline.} =
  v.len > p.n

type Twice* = object
func apply*(t: Twice; x: int): int {.inline.} =
  x * 2

type Offset* = object
  by: int
func offset*(by: int): Offset {.inline.} =
  Offset(by: by)
func apply*(t: Offset; x: int): int {.inline.} =
  x + t.by

proc mapChainDemo() =
  let ys = collect(iota(1, 6).map(Twice()))
  doAssert ys == @[2, 4, 6, 8, 10], $ys
  # chained maps + a filter stage in the middle:
  let zs = collect(iota(1, 11).map(Twice()).filter(gt(3)).map(offset(100)))
  doAssert zs == @[104, 106, 108, 110, 112, 114, 116, 118, 120], $zs
  doAssert sum(iota(1, 6).map(Twice())) == 30

proc collectChars() =
  doAssert collect("abc") == "abc"

proc collectViews() =
  let ls = collect(lines("alpha beta\ngamma delta\n\nomega").filter(longish(5)))
  doAssert ls == @["alpha beta", "gamma delta"], $ls

proc reDriveIsSafe() =
  let p = iota(0, 100).filter(gt(94))
  let first = collect(p)
  let second = collect(p)
  doAssert first == second and first.len == 5

proc main =
  collectScalars()
  mapChainDemo()
  collectChars()
  collectViews()
  reDriveIsSafe()
  echo "OK: t_collections"

main()
