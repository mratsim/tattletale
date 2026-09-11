# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import defunctional

proc sumOfChain() =
  doAssert sum(iota(1, 101).filter(gt(50))) == 3775
  doAssert sum(iota(1, 11)) == 55

proc countOfChain() =
  doAssert count(iota(1, 101).filter(gt(50))) == 50
  doAssert count(iota(1, 1)) == 0  # exclusive stop: [1, 1) is empty

proc extremaOfChain() =
  doAssert max(iota(3, 10)) == 9
  doAssert min(iota(3, 10)) == 3

proc reDriveIsSafe() =
  let p = iota(1, 101).filter(gt(50))
  doAssert sum(p) == 3775
  doAssert sum(p) == 3775

proc strideDemo() =
  doAssert collect(iota(0, 10, 3)) == @[0, 3, 6, 9]
  doAssert sum(iota(0, 10, 2)) == 20

proc main =
  strideDemo()
  sumOfChain()
  countOfChain()
  extremaOfChain()
  reDriveIsSafe()
  echo "OK: t_numeric"

main()
