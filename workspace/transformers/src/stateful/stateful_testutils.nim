## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared test/benchmark utilities for the stateful KV cache module.
##
## Centralizes helpers that were previously duplicated across
## test_kvcache, test_fork_stability, test_radix_invariants,
## bench_kvcache, bench_kvcache_extreme, and bench_kvcache_multiuser.

import
  std/math,
  std/random,
  ./kvcache  # ceilDiv, TokensPerPage

export ceilDiv, TokensPerPage  # re-export so consumers don't need both imports

type PagePayload* = int  # Page type alias for trie-only tests (no GPU pool needed)

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
