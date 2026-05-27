## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Multi-user PagedRadixTrie benchmark
##
## Simulates 100 users: 80@8K, 18@64K, 2@800K with 4K shared system prompt.
## Measures tree shape, fork cost, and per-token continuation cost.

import std/monotimes, std/times, std/strformat, std/strutils, std/random,
  std/math, std/importutils, ../src/stateful/kvcache {.all.},
  ../src/stateful/stateful_testutils

privateAccess(PagedRadixNode)
privateAccess(KVCache)

randomize(42)

const
  ShortUsers   = 80
  MediumUsers  = 18
  LongUsers    = 2
  ShortLen     = 8192
  MediumLen    = 64 * 1024
  LongLen      = 800 * 1024
  SystemPrefix = 4096

proc makeTokens(grp, uid, n: int): seq[uint32] =
  result = newSeq[uint32](n)
  for i in 0..<min(n, SystemPrefix):
    result[i] = uint32(i)
  if n <= SystemPrefix: return
  for i in 0..<min(n-SystemPrefix, 512):
    result[SystemPrefix+i] = (grp*1000 + i).uint32
  let uOff = SystemPrefix + 512
  for i in uOff..<n:
    let x = uid*10000 + (i-uOff) mod 997
    result[i] = x.uint32


proc addUser(cache: var KVCache[uint32, int]; tokens: seq[uint32]) =
  ## LPM then graftPages — ensures tree structure is created.
  discard cache.lpm(tokens)
  cache.graftPages(tokens, makePages(tokens.len))

proc populate(): KVCache[uint32, int] =
  result = KVCache[uint32, int].new()
  for u in 0..<ShortUsers:
    addUser(result, makeTokens(0, u, ShortLen))
  for u in 0..<MediumUsers:
    addUser(result, makeTokens(1, u, MediumLen))
  for u in 0..<LongUsers:
    addUser(result, makeTokens(2, u, LongLen))

template measure(iters: int; body: untyped): int64 =
  let t0 = getMonotime()
  for _ in 0 ..< iters: body
  (getMonotime() - t0).inNanoseconds div iters

type TreeStats = object
  nodes, leaves, maxDepth, totalDepth: int

proc stats(n: PagedRadixNode[uint32, int]; d: int; s: var TreeStats) =
  inc s.nodes
  if n.isLeaf: inc s.leaves; s.totalDepth += d; s.maxDepth = max(s.maxDepth, d)
  for c in n.children: stats(c, d+1, s)

proc benchPopulation() =
  echo "\n=== Tree Population ==="
  echo "Users: ", ShortUsers, "@", ShortLen div 1024, "K, ",
       MediumUsers, "@", MediumLen div 1024, "K, ",
       LongUsers, "@", LongLen div 1024, "K"
  echo "System prompt: ", SystemPrefix, " tokens shared"

  let t0 = getMonotime()
  var cache = populate()
  let buildMs = (getMonotime() - t0).inNanoseconds.float / 1e6

  var s: TreeStats; stats(cache.root, 0, s)
  echo "  Build time: ", buildMs.formatFloat(ffDecimal, 1), " ms"
  echo "  Nodes: ", s.nodes, " | Leaves: ", s.leaves
  echo "  Max depth: ", s.maxDepth
  echo "  Avg leaf depth: ", (s.totalDepth.float / s.leaves.float).formatFloat(ffDecimal, 1)
  echo "  Root subtree_sum_leaves: ", cache.root.subtree_sum_leaves
  echo "  Root children: ", cache.root.children.len

proc benchForkDepth() =
  echo "\n=== Fork at various prefix depths ==="
  let tests = [
    ("depth=256   (early sys prompt)",   256,  500_000),
    ("depth=1024  (mid sys prompt)",     1024, 500_000),
    ("depth=4096  (end of sys prompt)",  4096, 100_000),
    ("depth=8192  (user conversation)",  8192, 50_000)]

  for (label, depth, iters) in tests:
    var cache = populate()
    let newUser = makeTokens(3, 99, 16384)
    let ns = measure(iters):
      discard cache.lpm(newUser)
    echo "  ", label, "  ", ns, " ns  (", iters, " iters)"

proc benchContinuation() =
  echo "\n=== Continuation per-token cost ==="
  let tests = [
    ("extend 8K -> 8K+1   (short)",  ShortLen,  100_000),
    ("extend 64K -> 64K+1 (medium)", MediumLen, 10_000),
    ("extend 800K -> 800K+1 (long)", LongLen,   500)]

  for (label, seqLen, iters) in tests:
    var cache = populate()
    let existing = makeTokens(0, 0, seqLen)
    var oneMore = makeTokens(0, 0, seqLen)
    oneMore.add(uint32(seqLen + 100))

    var lpmNs, graftNs: int64
    for i in 0..<iters:
      let t0 = getMonotime()
      let r = cache.lpm(oneMore)
      lpmNs += (getMonotime() - t0).inNanoseconds

      let t1 = getMonotime()
      cache.graftPages(oneMore, makePages(oneMore.len))
      graftNs += (getMonotime() - t1).inNanoseconds

    let avgLpm = lpmNs div iters
    let avgGraft = graftNs div iters
    let total = avgLpm + avgGraft
    echo "  ", label
    echo "    LPM: ", avgLpm, " ns | Graft: ", avgGraft, " ns | Total: ", total, " ns | ", total div seqLen, " ns/tok"

proc benchLPM() =
  echo "\n=== LPM throughput ==="
  let tests = [
    ("LPM 8K (80-user tree)",   ShortLen,  100_000),
    ("LPM 64K (98-user tree)",  MediumLen, 10_000),
    ("LPM 800K (100-user tree)", LongLen,  500)]

  for (label, seqLen, iters) in tests:
    var cache = populate()
    let tokens = makeTokens(0, 0, seqLen)
    let ns = measure(iters):
      discard cache.lpm(tokens)
    echo "  ", label, "  ", ns, " ns  (", iters, " iters, ", (1e9/float64(ns)).formatFloat(ffDecimal,0), " ops/s)"

when isMainModule:
  echo "PagedRadixTrie — Multi-User Benchmarks"
  echo "Date: ", now()
  benchPopulation()
  benchForkDepth()
  benchContinuation()
  benchLPM()
  echo "\nDone."
