## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Unsanctioned helpers, pending rationalization.
##
## Helpers landing here bypass the sanctioned op surface, the funnel pass
## later promotes, keeps, or kills each one.
##
## - nothing outside layout_algebra imports this file
## - sanctioned modules re-export what survives rationalization
import std/macros

proc shapeRank*(shTyp: NimNode): int {.compileTime.} =
  ## Rank of a layout given its shape type node, tuple constr = element count, scalar = 1.
  if shTyp.kind == nnkTupleConstr:
    shTyp.len
  else:
    1

proc dimCount*(ty: NimNode): int {.compileTime.} =
  ## Top-level dimension count of a shape or layout type node.
  if ty.kind in {nnkTupleConstr, nnkTupleTy}:
    ty.len
  else:
    1
