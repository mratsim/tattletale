## Tattletale — PagedRadixTrie benchmarks
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Benchmarks for PagedRadixTrie operations:
##   LPM latency vs prefix length
##   Eviction latency vs tree size
##   graftPages (fork) latency
##
## Usage:
##   nim cpp --outdir:build/wip -d:release bench/bench_kvcache.nim
##   ./build/wip/bench_kvcache

import
  std/monotimes,
  std/times,
  std/strformat,
  std/strutils,
  std/random,
  std/math,
  ../src/stateful/kvcache {.all.}

randomize(42)

# ── Timing infrastructure ────────────────────────────────────────────────

template measure*(iters: int; body: untyped): int64 =
  ## Measure execution time of `body` over `iters` iterations.
  ## Returns nanoseconds per iteration.
  let startTime = getMonotime()
  for _ in 0 ..< iters:
    body
  let elapsed = (getMonotime() - startTime).inNanoseconds
  elapsed div iters

proc reportLine*(name: string; ns: int64; ops: float) =
  echo &"{name:<40} {ns:>8} ns/op  {ops:>8.1f} ops/s"

# ── Helpers ──────────────────────────────────────────────────────────────

type BenchPayload = int

proc makePages(nTokens: int): seq[BenchPayload] =
  result = newSeq[BenchPayload](ceilDiv(nTokens, 256))
  for i in 0..<result.len: result[i] = i + 1

proc randomTokens(n: int): seq[uint32] =
  result = newSeq[uint32](n)
  for i in 0..<n:
    result[i] = uint32(rand(1000))

proc makeTokens(prefixLen: int): seq[uint32] =
  ## Sequential tokens for exact-match LPM measurement.
  result = newSeq[uint32](prefixLen)
  for i in 0..<prefixLen:
    result[i] = uint32(i)

proc buildFlatTree(cache: var KVCache[uint32, BenchPayload]; nLeaves: int) =
  ## Flat tree: root with `nLeaves` direct children, each with 256 tokens.
  for i in 0..<nLeaves:
    var tokens = newSeq[uint32](256)
    for j in 0..<256:
      tokens[j] = uint32(i * 256 + j)
    discard cache.lpm(tokens)
    cache.graftPages(tokens, makePages(256))

proc buildDeepChain(cache: var KVCache[uint32, BenchPayload]; depth: int) =
  ## Chain of `depth` nodes, each with 256-token pages.
  var tokens = newSeq[uint32](depth * 256)
  for i in 0..<depth * 256:
    tokens[i] = uint32(i)
  discard cache.lpm(tokens)
  cache.graftPages(tokens, makePages(depth * 256))

# ── Benchmarks ──────────────────────────────────────────────────────────

proc benchLPM() =
  echo "\n── LPM Latency vs Prefix Length (flat tree, 100 leaves) ──"

  for (name, prefixLen, iters) in [("LPM 10 tokens", 10, 100_000),
                                   ("LPM 100 tokens", 100, 50_000),
                                   ("LPM 256 tokens (1 page)", 256, 20_000)]:
    var cache = KVCache[uint32, BenchPayload].new()
    cache.buildFlatTree(100)

    # Use tokens that exactly match leaf 0
    let matchingTokens = makeTokens(prefixLen)
    let pages = makePages(prefixLen)

    let ns = measure(iters):
      discard cache.lpm(matchingTokens)
      # graftPages: full match → timestamps update, no tree mutation
      cache.graftPages(matchingTokens, pages)

    reportLine(name & " x" & $iters, ns, 1e9 / float64(ns))

  echo "\n── LPM Latency vs Tree Size (100 tokens, flat) ──"
  for (name, nLeaves, iters) in [("100 leaves", 100, 50_000),
                                  ("1000 leaves", 1000, 20_000),
                                  ("10000 leaves", 10000, 2_000)]:
    var cache = KVCache[uint32, BenchPayload].new()
    cache.buildFlatTree(nLeaves)

    let matchingTokens = makeTokens(100)
    let pages = makePages(100)

    let ns = measure(iters):
      discard cache.lpm(matchingTokens)
      cache.graftPages(matchingTokens, pages)

    reportLine(name & " x" & $iters, ns, 1e9 / float64(ns))

proc benchEviction() =
  echo "\n── Eviction Latency vs Tree Size ──"

  for (name, nLeaves, iters) in [("evict 100 leaves", 100, 200),
                                  ("evict 1000 leaves", 1000, 100),
                                  ("evict 10000 leaves", 10000, 20)]:
    # Build tree once, drain via evicts, time each evict individually.
    # Use `small iters` x `drain` approach: rebuild after draining tree.
    let drain = nLeaves div 2  # evict half the tree per rebuild
    var nsAcc: int64
    for s in 0..<iters:
      var cache = KVCache[uint32, BenchPayload].new()
      cache.buildFlatTree(nLeaves)
      var t0 = getMonotime()
      for i in 0..<drain:
        cache.evict()
      nsAcc += (getMonotime() - t0).inNanoseconds
    let ns = nsAcc div (iters * drain)

    reportLine(name & " x" & $iters & "x" & $drain, ns, 1e9 / float64(ns))

proc benchGraftPages() =
  echo "\n── graftPages Latency (first-population & fork scenarios) ──"

  for (name, nTokens, iters) in [("graftPages (fresh root, 256 tok)", 256, 100_000),
                                  ("graftPages (fresh root, 512 tok)", 512, 50_000)]:
    var tokens = makeTokens(nTokens)

    let ns = measure(iters):
      var cache = KVCache[uint32, BenchPayload].new()
      discard cache.lpm(tokens)
      cache.graftPages(tokens, makePages(nTokens))

    reportLine(name & " x" & $iters, ns, 1e9 / float64(ns))

  # Rebuild + fork measurement
  for (name, iters) in [("graftPages (fork, 256 tok leaf → sibling)", 100_000)]:
    var tokensA = makeTokens(256)
    var tokensB = makeTokens(256)
    # tokensB has same prefix length as tokensA (both sequential 0..255)
    # Since they're identical, there's no fork. Need differing tokens.
    for i in 0..<256:
      tokensB[i] = uint32(256 + i)

    var cache = KVCache[uint32, BenchPayload].new()
    discard cache.lpm(tokensA)
    cache.graftPages(tokensA, makePages(256))

    let ns = measure(iters):
      discard cache.lpm(tokensB)
      # Creates a sibling (fork) since tokensB differs from existing content
      cache.graftPages(tokensB, makePages(256))

    reportLine(name & " x" & $iters, ns, 1e9 / float64(ns))

proc benchChainLPM() =
  echo "\n── Chain LPM (deep tree) ──"

  for (name, depth, iters) in [("chain depth=10", 10, 50_000),
                                ("chain depth=50", 50, 10_000),
                                ("chain depth=100", 100, 5_000)]:
    var cache = KVCache[uint32, BenchPayload].new()
    cache.buildDeepChain(depth)

    let matchingTokens = makeTokens(depth * 256)
    let pages = makePages(depth * 256)

    let ns = measure(iters):
      discard cache.lpm(matchingTokens)
      cache.graftPages(matchingTokens, pages)

    reportLine(name & " x" & $iters, ns, 1e9 / float64(ns))

# ── Main ────────────────────────────────────────────────────────────────

when isMainModule:
  echo "PagedRadixTrie Benchmarks"
  echo "========================"
  echo "Date: ", now()
  echo ""

  benchLPM()
  benchEviction()
  benchGraftPages()
  benchChainLPM()

  echo "\nDone."
