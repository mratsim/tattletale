# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Data structures
runnableExamples:
  import std/math
  import ./src/wavl_tree

  var links: seq[WavlLink] = @[]
  var root: int32 = WavlNil
  wavlInit(links, root)

  # Insert three nodes
  links.setLen(3)
  wavlInsert(links, root, 0, proc(a, b: int32): int = cmp(a, b))
  wavlInsert(links, root, 1, proc(a, b: int32): int = cmp(a, b))
  wavlInsert(links, root, 2, proc(a, b: int32): int = cmp(a, b))

  # Find
  doAssert wavlFind(links, root, proc(idx: int32): int = cmp(1, idx)) == 1
  doAssert wavlFind(links, root, proc(idx: int32): int = cmp(42, idx)) == WavlNil

  # Min/Max
  doAssert wavlMin(links, root) == 0
  doAssert wavlMax(links, root) == 2

  # Delete
  wavlDelete(links, root, 1)
  doAssert wavlFind(links, root, proc(idx: int32): int = cmp(1, idx)) == WavlNil
  doAssert wavlFind(links, root, proc(idx: int32): int = cmp(0, idx)) == 0
  doAssert wavlFind(links, root, proc(idx: int32): int = cmp(2, idx)) == 2

  # removeChild dance
  var data = @[10, 20, 30, 40]
  links.setLen(4)
  root = WavlNil
  wavlInsert(links, root, 0, proc(a, b: int32): int = cmp(data[a], data[b]))
  wavlInsert(links, root, 1, proc(a, b: int32): int = cmp(data[a], data[b]))
  wavlInsert(links, root, 2, proc(a, b: int32): int = cmp(data[a], data[b]))
  wavlInsert(links, root, 3, proc(a, b: int32): int = cmp(data[a], data[b]))

  let lastIdx = int32(data.len - 1)
  wavlDelete(links, root, 1)
  data.del(1)
  links.del(1)
  fixLinksAfterDataDeletion(links, root, lastIdx, 1)

import ./src/wavl_tree

export wavl_tree
