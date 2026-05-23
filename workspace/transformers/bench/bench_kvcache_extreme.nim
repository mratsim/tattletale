## Extreme stress tests for PagedRadixTrie KV Cache
##
## Pushes the implementation to its limits:
##   1. Wide tree (100K children at root)
##   2. Deep chain + deep fork (10K depth)
##   3. 1M tokens in a single leaf (page append)
##   4. Eviction drain on wide tree
##   5. Fork-after-page at extreme depth
##   6. Sub-page branching with massive sibling
##
## Usage: nim cpp -d:release --outdir:build/wip bench/bench_kvcache_extreme.nim
##        ./build/wip/bench_kvcache_extreme

import
  std/monotimes,
  std/times,
  std/strformat,
  std/strutils,
  std/math,
  std/importutils,
  ../src/stateful/kvcache {.all.}

privateAccess(PagedRadixNode)
privateAccess(KVCache)

# ── Timing ──────────────────────────────────────────────────────────────
template measure(iters: int; body: untyped): int64 =
  let t0 = getMonotime()
  for _ in 0 ..< iters: body
  (getMonotime() - t0).inNanoseconds div iters

proc report(name: string; ns: int64; ops: float; note = "") =
  let n = if note.len > 0: "  # " & note else: ""
  echo &"  {name:<48} {ns:>8} ns/op  {ops:>8.1f} ops/s{n}"

# ── Helpers ─────────────────────────────────────────────────────────────
type Payload = int

proc makePages(n: int): seq[Payload] =
  result = newSeq[Payload](ceilDiv(n, 256))
  for i in 0..<result.len: result[i] = i + 1

proc makeTokens(n: int): seq[uint32] =
  result = newSeq[uint32](n)
  for i in 0..<n: result[i] = uint32(i)

proc makeTokens(start, n: int): seq[uint32] =
  result = newSeq[uint32](n)
  for i in 0..<n: result[i] = uint32(start + i)

# ════════════════════════════════════════════════════════════════════════
# 1. Wide tree — massive fan-out at root
# ════════════════════════════════════════════════════════════════════════
proc benchWideTree() =
  echo "\n" & repeat("=", 60)
  echo "1. WIDE TREE: massive fan-out at root"
  echo repeat("=", 60)
  echo "  Build a flat tree with N children, measure LPM hit + miss."
  echo ""

  for (label, nLeaves) in [("10K leaves", 10_000),
                            ("100K leaves", 100_000)]:
    var cache = KVCache[uint32, Payload].new()

    let t0 = getMonotime()
    for i in 0..<nLeaves:
      var tokens = newSeq[uint32](256)
      for j in 0..<256:
        # Ensure unique first token so LargeMatch finds the right child
        tokens[j] = uint32(i * 256 + j)
      discard cache.lpm(tokens)
      cache.graftPages(tokens, makePages(256))
    let buildNs = (getMonotime() - t0).inNanoseconds div nLeaves

    echo &"  {label} built: {(getMonotime() - t0).inNanoseconds.float/1e6:.1f} ms"
    echo &"    Root children: {cache.root.children.len}, Leaves: {cache.root.subtree_sum_leaves}"

    # LPM hit: pick a specific leaf's first 16 tokens
    let hitTokens = makeTokens(0, 16)
    let hitNs = measure(50_000): discard cache.lpm(hitTokens)
    report("  LPM hit (16 tok)", hitNs, 1e9/float64(hitNs))

    # LPM miss: tokens that don't match any child
    let missTokens = makeTokens(16_000_000, 16)
    let missNs = measure(10_000): discard cache.lpm(missTokens)
    report("  LPM miss (16 tok)", missNs, 1e9/float64(missNs), "all children skipped")

    # LPM long hit: match 512 tokens across 2 pages
    let longTokens = makeTokens(0, 512)
    let longNs = measure(10_000): discard cache.lpm(longTokens)
    report("  LPM 512 tok hit", longNs, 1e9/float64(longNs), "2 pages match")

    # graftPages: insert last child (worst-case LargeMatch iteration)
    let gTokens = makeTokens(nLeaves * 256, 256)
    let gPages = makePages(256)
    let gNs = measure(10_000):
      discard cache.lpm(gTokens)
      cache.graftPages(gTokens, gPages)
    report("  graftPages last child", gNs, 1e9/float64(gNs), "worst-case LargeMatch")

    # graftPages: full match (update timestamps)
    let fmNs = measure(100_000):
      discard cache.lpm(hitTokens)
      cache.graftPages(hitTokens, makePages(16))
    report("  graftPages full match", fmNs, 1e9/float64(fmNs))

    echo ""

# ════════════════════════════════════════════════════════════════════════
# 2. Deep chain + deep fork
# ════════════════════════════════════════════════════════════════════════
proc benchDeepChain() =
  echo repeat("=", 60)
  echo "2. DEEP CHAIN: 5K depth + fork at 2.5K"
  echo repeat("=", 60)

  for (label, depth, forkPt) in [("5K depth, fork at 2.5K", 5_000, 2_500),
                                  ("1K depth, fork at 500",  1_000,   500)]:
    var cache = KVCache[uint32, Payload].new()
    let totalLen = depth * 256
    let full = makeTokens(totalLen)

    let t0 = getMonotime()
    discard cache.lpm(full)
    cache.graftPages(full, makePages(totalLen))
    let buildNs = (getMonotime() - t0).inNanoseconds
    echo &"\n  {label} built in {buildNs.float/1e6:.1f} ms"
    echo &"    Depth: {depth}, leaf tokens: {cache.root.tokens.len}"

    # LPM on full sequence
    let lpmNs = measure(1_000): discard cache.lpm(full)
    report("  LPM full (all pages)", lpmNs, 1e9/float64(lpmNs))

    # Fork at forkPt * 256 tokens — create sibling with diverging content
    let forkPoint = forkPt * 256
    var diverge = makeTokens(forkPoint)
    diverge.add([999_999_999'u32, 999_999_998'u32])  # diverge
    let divergePages = makePages(diverge.len)

    let t1 = getMonotime()
    discard cache.lpm(diverge)
    cache.graftPages(diverge, divergePages)
    let forkNs = (getMonotime() - t1).inNanoseconds
    report("  fork at depth (graftPages)", forkNs, 1e9/float64(forkNs))

    # LPM on the new branch
    let branchNs = measure(1_000): discard cache.lpm(diverge)
    report("  LPM new branch", branchNs, 1e9/float64(branchNs))

    # walkUp cost: graftPages on existing content (full match)
    let fullMatchNs = measure(500):
      discard cache.lpm(diverge)
      cache.graftPages(diverge, divergePages)
    report("  graftPages full match", fullMatchNs, 1e9/float64(fullMatchNs))

    # LPM on original branch (should still work)
    let origLpmNs = measure(1_000): discard cache.lpm(full)
    report("  LPM original branch", origLpmNs, 1e9/float64(origLpmNs))

    echo &"    Tree: nodes={depth+1+1} depth={depth+1}"

# ════════════════════════════════════════════════════════════════════════
# 3. Massive leaf — 10K pages in a single conversation
# ════════════════════════════════════════════════════════════════════════
proc benchMassiveLeaf() =
  echo "\n" & repeat("=", 60)
  echo "3. MASSIVE LEAF: 10K pages (2.56M tokens) in one node"
  echo repeat("=", 60)

  let nPages = 10_000
  let nTokens = nPages * 256
  var cache = KVCache[uint32, Payload].new()

  # Build leaf in chunks to measure linear growth
  echo "\n  Building leaf incrementally:"
  for chunk in [1, 10, 100, 1000, nPages]:
    var batch = makeTokens(chunk * 256)
    let t0 = getMonotime()
    discard cache.lpm(batch)
    cache.graftPages(batch, makePages(chunk * 256))
    let ns = (getMonotime() - t0).inNanoseconds
    echo &"    +{chunk:>5} pages → {cache.root.tokens.len div 256:>5} pages  {ns:>6} ns total"

  echo &"\n  Final: {cache.root.tokens.len} tokens in root node"
  echo &"    Root subtree_sum_pages: {cache.root.subtree_sum_pages}"

  # LPM on full leaf
  let full = makeTokens(nTokens)
  let lpmNs = measure(500): discard cache.lpm(full)
  report("  LPM full (10K pages)", lpmNs, 1e9/float64(lpmNs))

  # LPM on partial (1 page)
  let partial = makeTokens(256)
  let partialMinNs = measure(500): discard cache.lpm(partial)
  report("  LPM partial (1 page)", partialMinNs, 1e9/float64(partialMinNs))

  # Append 1 more page
  let extTokens = makeTokens(nTokens, 256)
  let extNs = measure(500):
    discard cache.lpm(extTokens)
    cache.graftPages(extTokens, makePages(256))
  report("  graftPages +1 page", extNs, 1e9/float64(extNs))

# ════════════════════════════════════════════════════════════════════════
# 4. Eviction drain: measure evict cost as tree shrinks
# ════════════════════════════════════════════════════════════════════════
proc benchEvictionDrain() =
  echo "\n" & repeat("=", 60)
  echo "4. EVICTION DRAIN: evict all leaves from wide tree"
  echo repeat("=", 60)

  for (label, nLeaves, rebuilds) in [("10K leaves", 10_000, 10),
                                      ("100K leaves", 100_000, 2)]:
    var totalEvicts: int64
    for r in 0..<rebuilds:
      var cache = KVCache[uint32, Payload].new()
      for i in 0..<nLeaves:
        var tokens = newSeq[uint32](256)
        for j in 0..<256: tokens[j] = uint32(i * 256 + j)
        discard cache.lpm(tokens)
        cache.graftPages(tokens, makePages(256))

      let t0 = getMonotime()
      var evicted = 0
      while cache.root.subtree_sum_leaves > 0:
        cache.evict()
        inc evicted
      let elapsed = (getMonotime() - t0).inNanoseconds
      totalEvicts += elapsed

    let avgNs = totalEvicts div (nLeaves * rebuilds)
    let totalMs = totalEvicts.float / 1e6
    let nTotalEv = nLeaves * rebuilds
    let rate = nTotalEv.float / (totalMs / 1000)
    echo &"\n  {label}: drained {nTotalEv} leaves in {totalMs:.1f} ms"
    echo &"    Avg evict: {avgNs} ns  ({rate:.0f} evictions/s)"
    report("    Evict single", avgNs, 1e9/float64(avgNs))

# ════════════════════════════════════════════════════════════════════════
# 5. Fork-after-page at extreme depth
# ════════════════════════════════════════════════════════════════════════
proc benchDeepForkSplit() =
  echo "\n" & repeat("=", 60)
  echo "5. DEEP FORK-SPLIT: split node at max depth (within page)"
  echo repeat("=", 60)

  # Build a chain where the leaf has 300 tokens (1 full page + 44 tokens)
  # Then fork with completely different content — triggers partial-match
  # sibling creation at maximum tree depth.
  let depth = 1_000
  let chainLen = depth * 256
  var cache = KVCache[uint32, Payload].new()

  let full = makeTokens(chainLen)
  discard cache.lpm(full)
  cache.graftPages(full, makePages(chainLen))
  echo &"\n  Chain depth {depth} built: root has {cache.root.totalTokenCount} tokens"

  # Fork at 300 tokens past the last page boundary
  # This creates a sibling at the leaf level via COW
  var diverge = makeTokens(chainLen + 300)
  for i in chainLen ..< diverge.len:
    diverge[i] = uint32(999_999_999 - (i - chainLen))
  let divergePages = makePages(diverge.len)

  let t0 = getMonotime()
  discard cache.lpm(diverge)
  cache.graftPages(diverge, divergePages)
  let forkNs = (getMonotime() - t0).inNanoseconds
  echo &"  Fork at depth {depth}+300: {forkNs} ns (includes LPM + graftPages)"
  report("  fork (+300 tok at depth)", forkNs, 1e9/float64(forkNs))

  # Verify both branches accessible
  let hit1 = measure(1_000): discard cache.lpm(full)
  report("  LPM original branch", hit1, 1e9/float64(hit1))
  let hit2 = measure(1_000): discard cache.lpm(diverge)
  report("  LPM forked branch", hit2, 1e9/float64(hit2))

# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
when isMainModule:
  echo "PagedRadixTrie — Extreme Stress Benchmarks"
  echo "Date: ", now()

  # Build tree (warmup)
  var warmup = KVCache[uint32, Payload].new()
  discard warmup.lpm(@[1'u32, 2, 3])
  warmup.graftPages(@[1'u32, 2, 3], [1])

  benchWideTree()
  benchDeepChain()
  benchMassiveLeaf()
  benchEvictionDrain()
  benchDeepForkSplit()

  echo "\nDone."
