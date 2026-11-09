# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Stateful-machinery test helpers, libtorch-free.
##
## The trie suites graft synthetic page tables, the pool suites stage device state on top.

import
  std/[math, random],
  workspace/tattleformers/src/stateful/kvcache

func makePages*(nTokens: int): seq[int] =
  ## Create sequential page indices `[1, 2, 3, ...]` for `nTokens`.
  let nPages = ceilDiv(nTokens, TokensPerPage)
  result = newSeq[int](nPages)
  for i in 0..<nPages:
    result[i] = i + 1

proc makeTokens*(n: int): seq[uint32] =
  ## Sequential tokens `[0, 1, 2, ..., n-1]` for exact-match LPM measurement.
  result = newSeq[uint32](n)
  for i in 0..<n:
    result[i] = uint32(i)

proc randomTokens*(n: int): seq[uint32] =
  ## Random tokens for stress-testing.
  result = newSeq[uint32](n)
  for i in 0..<n:
    result[i] = uint32(rand(1000))
