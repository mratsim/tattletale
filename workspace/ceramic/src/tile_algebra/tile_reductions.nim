# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile-level reductions.
##
## `sum(src)`: the full tile, every lane holds the total.

import workspace/crucible
import ./tiles

proc sum*[T, Sh, St](src: Tile[T, Sh, St]): T {.inline.} =
  ## Full-tile sum, every lane holds the total.
  let owned = src.size() div getLaneWidth()
  var acc = zero(T)
  for i in 0 ..< owned:
    acc += src.data[i]
  reduceAdd(acc)
