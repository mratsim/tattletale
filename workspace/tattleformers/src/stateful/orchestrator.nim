# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/ceramic/src/tensors,
  workspace/tensorcery/src/tensor_datatypes,
  ./kvcache,
  ./page_pool,
  ./inference_context,
  ../ops_Ni1o/pool_write

# ═══════════════════════════════════════════════════════════════════════
#  Orchestrator, paged KV cache lifecycle manager
# ═══════════════════════════════════════════════════════════════════════
#
# ## Lifecycle (single sequence, MVP)
#
# ```
#         Orchestrator.init       startSequence       appendToken × N      endSequence
#              │                       │                   │                   │
#              ▼                       ▼                   ▼                   ▼
#     ┌──────────────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────────────┐
#     │ Create PagePool  │   │ 1. clearState    │   │ Append   │   │ 1. Collect all   │
#     │ Create KVCache   │──▶│ 2. LPM (trie)    │──▶│ token to │──▶│    tokens+pages  │
#     │ Create InfCtx    │   │ 3. COW partial   │   │ tracking │   │ 2. graftPages    │
#     │                  │   │ 4. Borrow pages  │   │ Borrow   │   │    (sink into    │
#     │                  │   │ 5. Set pos_ids   │   │ page if  │   │     trie)        │
#     │                  │   │                  │   │ boundary │   │ 3. clearState    │
#     └──────────────────┘   └──────────────────┘   └──────────┘   └──────────────────┘
# ```
#
# ## Field lifecycles
#
# Two critical fields in `InferenceContext` manage write tracking.
# They serve DIFFERENT purposes and must not be confused:
#
# ### `cached_tokens`: attention write-skip offset
#
# Purpose: tell the attention layer how many prefix positions the LPM match already covers,
#          so those covered positions stay untouched as immutable pages.
#
# ```
#   Phases:        startSequence                       appendToken × N
#                  │                                   │
#                  ▼                                   ▼
#     ┌─────────────────────┐           ┌────────────────────┐
#     │ LPM → cached_tokens │           │  cached_tokens     │
#     │ = matched count     │──────────▶│  unchanged         │
#     │                     │           │  (never updated    │
#     │ Attention read:     │           │   after LPM)       │
#     │ writeStart =        │           │                    │
#     │   max(0, cached     │           │ Attention reads    │
#     │    - offset)        │           │ same cached_tokens │
#     └─────────────────────┘           └────────────────────┘
# ```
#
# Lifecycle:
#   1. `startSequence`: set to `matched.totalTokenMatched` (0 for empty trie).
#   2. **Never updated again** during the sequence (stable across decode steps).
#   3. `clearState`: reset to 0.
#
# Attention's `writeStart` formula:
#
#   writeStart = max(0, cached_tokens - offset)
#
# - First prefill: cached_tokens=0, offset=0 → writeStart=0, ALL positions written.
# - COW continuation: cached_tokens=300, offset=0 → writeStart=300, cached prefix skipped.
# - Decode: cached_tokens=0, offset=28 → writeStart=0, writes 1 new token.
#
# ### `kv_position`: page allocation write cursor
#
# Purpose: track the total tokens written to the KV cache so far.
#          Used ONLY by `appendToken` for page-boundary detection.
#
# ```
#            startSequence         generate() after          appendToken × N
#                                    prefill forward
#                 │                       │                       │
#                 ▼                       ▼                       ▼
#     ┌─────────────────────┐   ┌─────────────────────┐   ┌────────────────────┐
#     │ kv_position = 0    │   │ setKvPosition(      │   │ Check: kv_position  │
#     │ (no tokens written  │──▶│   ids.len)          │──▶│  > 0 and mod 256    │
#     │  yet, LPM matched   │   │                     │   │  == 0 → borrow page │
#     │  tokens are in trie,│   │ kv_position now     │   │ kv_position += 1    │
#     │  not in local pages)│   │ reflects total      │   │                     │
#     │                     │   │ written tokens      │   │ Positions 0..N-1    │
#     └─────────────────────┘   └─────────────────────┘   │ are in local pages  │
#                                                         └────────────────────┘
# ```
#
# Lifecycle:
#   1. `startSequence`: set to 0 (no local tokens written yet).
#   2. `setKvPosition(n)` (called by generate() AFTER prefill forward): set to ids.len.
#   3. `appendToken`: checked for page boundary, then incremented by 1.
#   4. `clearState`: reset to 0.
#
# ### Why two fields? (BUG-B-001 history)
#
# Originally `kv_position` served BOTH purposes.
# `startSequence` set it to `input_ids.len`. The attention layer
# computed writeStart = seq_len for the first prefill, skipping ALL
# KV writes. The KV cache stayed empty, and decode steps read garbage.
#
# Separating `cached_tokens` (write-skip, stable after LPM) from `kv_position`
# (write-cursor, updated after forward pass) eliminates this coupling.
#
# ## Sequence lifecycle
#
# 1. **init**: creates the page pool (eager allocation), the PagedRadixTrie
#    (logical_map), and an InferenceContext. Pool size comes from the model's
#    `max_position_embeddings` via `computeNumPages`.
#
# 2. **startSequence**: resets the InferenceContext, finds the shared
#    prefix via LPM. Handles Copy-on-Write for the partially-matched
#    page (match ends mid-page). Borrows fresh pages for unmatched
#    prompt tokens. Sets prefill position_ids and `cached_tokens`.
#
# 3. **appendToken (× N)**: appends one token to the tracking sequence,
#    borrows a new page when the write cursor crosses a page boundary,
#    updates position_ids for the single-token decode forward pass,
#    and checks `kv_position` for page boundaries (not `cached_tokens`).
#
# 4. **endSequence**: calls `graftPages` with ALL sequence tokens and pages.
#    Trie internals handle matching, forking, appending.
#    LPM locks are released as a side effect. The InferenceContext resets.
#
# ## Memory model
#
# ```
# OrchestratorObj (ref object)
#   ├── page_pool: PagePool (owns both KV slabs, k_buffer and v_buffer)
#   │     (destructed LAST: field order ensures trie freed before pool)
#   ├── logical_map: KVCache      (owns PagedRadixTrie with Page refs)
#   │     (destructed FIRST: Pages → views release pool buffer refs)
#   │     └── PagedRadixNode
#   │           └── pages: seq[Page]
#   │                 └── PageObj
#   │                       ├── k_view: Tensor   (view into pool.k_buffer)
#   │                       └── v_view: Tensor   (view into pool.v_buffer)
#   └── active_context: InferenceContext
#         └── pages: seq[Page]    (pages for current sequence)
# ```
#
# **Destruction order** (reverse declaration order):
#
#   1. `logical_map` → trie freed → Pages freed → view tensors destroyed first
#   2. `page_pool` → k_buffer/v_buffer freed last (after the views drop)
#
# No custom `=destroy` operators: ORC's default field-by-field destruction
# handles the Nim Tensor → device buffer chain.
#


# ═══════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════

func computeNumPages*(maxContextTokens, concurrentRequests: int): int =
  ## Compute the number of KV cache pages needed for a pool.
  ## Each page stores TokensPerPage (256) KV entries.
  ##
  ## Args:
  ##
  ## - maxContextTokens: Maximum context length (model's max_position_embeddings)
  ## - concurrentRequests: Concurrent sequences (1 for single-sequence MVP)
  ##
  ## Returns:
  ##   Number of pages with headroom for COW partial pages + lazy decode.
  let pagesPerRequest = kvcache.ceilDiv(maxContextTokens, TokensPerPage)
  let basePages = pagesPerRequest * concurrentRequests
  result = basePages + concurrentRequests  # headroom

func computePageSizeBytes*(num_layers, kv_heads, head_dim: int): int64 =
  ## Memory size of a single page in bytes (K + V), bfloat16 elements.
  ##
  ## Args:
  ##
  ## - num_layers: Number of transformer layers
  ## - kv_heads: Number of KV heads (GQA)
  ## - head_dim: Dimension per head
  let elPerBuf = num_layers.int64 * TokensPerPage.int64 * kv_heads.int64 * head_dim.int64
  result = elPerBuf * 2 * 2  # 2 bytes per bfloat16 element, K+V


# ═══════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════

type
  Orchestrator* = ref object
    ## High-level orchestration for inference with paged KV cache.
    ##
    ## Owns:
    ## - `page_pool`: KV buffer allocator (k_buffer, v_buffer)
    ## - `logical_map`: PagedRadixTrie mapping token sequences → Page refs
    ## - `active_context`: Mutable state for the current sequence
    ##
    ## **Lifecycle**: `init` → `startSequence` → `appendToken`* → `endSequence`
    ##
    ## **Destruction order** (reverse field order):
    ##
    ## - `logical_map` freed before `page_pool`: the trie's Page refs view
    ##   the pool's k_buffer/v_buffer
    ## - trie-first destruction releases every buffer reference up front,
    ##   and the pool's buffers free cleanly
    ##
    ## No custom `=destroy` operator: ORC default field destruction handles
    ## the Tensor → device buffer chain correctly without extra nil-ing.
    page_pool: PagePool # first field so it's destroyed last so pages in flight in `logical_map` don't access an invalid pointer.
    active_context: InferenceContext[uint32, Page]
    num_layers: int
    logical_map: KVCache[uint32, Page]

# ═══════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════

proc init*(_: type Orchestrator, engine: AnyEngine,
            num_layers, batch_size, kv_heads, max_seq, head_dim: int,
            num_pages: int): Orchestrator =
  ## Create an Orchestrator with eager KV page pool allocation.
  ##
  ## Args:
  ##
  ## - engine: allocates the K and V slabs
  ## - num_layers, batch_size: transformer layer count, batch dimension
  ## - kv_heads, max_seq, head_dim: GQA width, sequence limit, head width
  ## - num_pages: total pages to pre-allocate in the pool
  ##
  ## Pool is created eagerly (not lazily) so OOM is detected at init time.
  let pool = PagePool.init(engine, num_pages, num_layers, kv_heads, head_dim)
  result = Orchestrator(
    logical_map: KVCache[uint32, Page].new(),
    page_pool: pool,
    active_context: InferenceContext[uint32, Page].init(
      num_layers, batch_size, kv_heads, max_seq, head_dim),
    num_layers: num_layers
  )
  result.active_context.pool = pool

proc getInferenceContextMut*(orc: var Orchestrator): var InferenceContext[uint32, Page] {.inline.} =
  ## Get the active inference context.
  orc.active_context

proc lpm*(orc: var Orchestrator, tokens: openArray[uint32]): LongestPrefixMatch[Page] =
  ## Longest Prefix Match over the orchestrator's trie, read-only.
  ##
  ## Precondition: the caller MUST end the sequence with `endSequence`
  ## (graftPages) to release the lock this acquires.
  orc.logical_map.lpm(tokens)

proc setKvPosition*(orc: var Orchestrator, pos: int) {.inline.} =
  ## Set the write cursor position (called after prefill forward completes).
  orc.active_context.kv_position = pos

# ═══════════════════════════════════════════════════════════════════════
# Pool management
# ═══════════════════════════════════════════════════════════════════════

proc ensurePoolCapacity(orc: var Orchestrator, needed: int) =
  ## Ensure the pool has at least `needed` free pages by evicting
  ## trie leaves when necessary.
  ##
  ## Raises:
  ##
  ## - `ValueError` when no eviction candidates exist, every page
  ##   locked on an active decode path
  # TODO (serving API): add max eviction loop count + request budget
  #   When we have a serving API, the orchestrator will run in an event
  #   loop with rate-limiting and per-request fairness budgets.
  while orc.page_pool.pagesAvailable() < needed:
    let freed = orc.logical_map.evict()
    if freed == 0:
      raise newException(ValueError,
        "[ttt] KVCache OOM: no evictable pages (all active/locked)")
    # evict() clears the leaf's seq[Page], Page refs hit 0,
    # =destroy fires via GC, indices return to pool


# ═══════════════════════════════════════════════════════════════════════
# COW (Copy-on-Write) helper
# ═══════════════════════════════════════════════════════════════════════

proc cowPartialPage(orc: var Orchestrator, dst, src: Page, partialTokens: int) =
  ## Copy partial KV content from the cached page (`src`) into a newly
  ## borrowed COW page (`dst`). Used when LPM matches a page partially:
  ## the trie keeps the matched prefix, and the private page receives
  ## the suffix copy.
  ##
  ## Copies `partialTokens` positions across all `numLayers` layers:
  ##
  ## - per layer, one pool-write op reads a rank-4 view of the src page
  ##   slab and writes the dst pool slot
  ## - all layer nodes are enqueued and evaluated together, one eval per COW
  ## - the src views alias the pool's own buffers, so no host round trip
  ##   and no temporary allocation happen
  let engine = orc.page_pool.kBuffer().engine
  let dstSlot = int(dst.pageIndex())
  var nodes: seq[TensorNode]
  let
    kImpl = src.k_view.node.getImpl()
    vImpl = src.v_view.node.getImpl()
    kShape = kImpl.shape()
    vShape = vImpl.shape()
    kvHeads = kShape[2]
    headDim = kShape[3]
  for layer in 0 ..< orc.num_layers:
    let kSrc = src.k_view.view(
      @[1, partialTokens, kvHeads, headDim],
      @[kImpl.strides()[1], kImpl.strides()[1], kImpl.strides()[2], kImpl.strides()[3]],
      kImpl.offset() + layer * kImpl.strides()[0])
    let vSrc = src.v_view.view(
      @[1, partialTokens, kvHeads, headDim],
      @[vImpl.strides()[1], vImpl.strides()[1], vImpl.strides()[2], vImpl.strides()[3]],
      vImpl.offset() + layer * vImpl.strides()[0])
    nodes.add(tn_pool_write(engine,
      orc.page_pool.kBuffer().node, orc.page_pool.vBuffer().node,
      kSrc.node, vSrc.node, layer, 0, partialTokens, 0, @[dstSlot]))
  engine.eval(nodes.toSpan())


# ═══════════════════════════════════════════════════════════════════════
# Sequence lifecycle
# ═══════════════════════════════════════════════════════════════════════

proc startSequence*(orc: var Orchestrator, input_ids: seq[uint32]) =
  ## Start a new sequence.
  ##
  ## 1. Resets the InferenceContext (clears pages, tokens, position).
  ## 2. Runs LPM on the trie to find a shared prefix with cached sequences,
  ##    then handles Copy-on-Write when the match ends mid-page.
  ## 3. Borrows fresh pages for unmatched prompt tokens.
  ## 4. Sets position_ids for the prefill forward pass.
  ##
  ## Args:
  ##   input_ids: Input token IDs (prompt)

  if input_ids.len == 0:
    raise newException(ValueError,
      "[ttt] Empty prompt")

  # Reset context for fresh sequence
  orc.active_context.clearState()
  var ctx = orc.active_context

  # ── 1. LPM, find shared prefix ──
  ctx.input_tokens = input_ids
  let matched = orc.logical_map.lpm(input_ids)
  ctx.cached_tokens = matched.totalTokenMatched  # for attention writeStart

  # ── 2. COW partial page handling ──
  let partialTokens = ctx.cached_tokens mod TokensPerPage
  var cowPage: Page = nil
  var cowPageUsed = false
  if partialTokens > 0 and matched.pages.len > 0:
    # Last matched page is partial, borrow a new page, copy partial content
    let cachedPage = matched.pages[^1]
    orc.ensurePoolCapacity(1)
    cowPage = orc.page_pool.borrow()
    orc.cowPartialPage(cowPage, cachedPage, partialTokens)
    cowPageUsed = true

  # ── 3. Add fully-matched pages from trie (in order) ──
  let fullMatchCount = matched.pages.len - (if cowPageUsed: 1 else: 0)
  for i in 0 ..< fullMatchCount:
    ctx.pages.add(matched.pages[i])

  # ── 3b. Append COW page after fully-matched pages ──
  if cowPageUsed:
    ctx.pages.add(cowPage)

  # ── 4. Borrow new pages for unmatched prompt portion ──
  let promptPages = kvcache.ceilDiv(input_ids.len, TokensPerPage)
  doAssert promptPages >= ctx.pages.len,
    "[ttt] Invariant: prompt should need >= pages than already matched by trie"
  let newPagesNeeded = promptPages - ctx.pages.len
  if newPagesNeeded > 0:
    orc.ensurePoolCapacity(newPagesNeeded)
    for _ in 0 ..< newPagesNeeded:
      ctx.pages.add(orc.page_pool.borrow())

  # ── 5. Set position_ids for prefill ──
  ctx.setPositionIdsArange(input_ids.len, offset = 0)

proc appendToken*(orc: var Orchestrator, position: int, token_id: uint32) =
  ## Prepare for a single decode step (one token).
  ##
  ## 1. Appends the token to input_tokens tracking.
  ## 2. Lazily borrows a new pool page on a page-boundary crossing.
  ## 3. Updates position_ids for the single-token forward pass.
  ##
  ## NOTE: kv_position is NOT incremented here. The caller (generate())
  ## increments it AFTER the forward call.
  ##
  ## During the attention forward, `ctx.kv_position` then equals
  ## `position_ids.min()`, and the attention layer takes its write
  ## offset from `kv_position` without a synchronous device readback
  ## on every forward pass.
  ##
  ## Lifecycle for one decode step:
  ##
  ## ```
  ##   appendToken(position=300) -> forward() -> generate: setKvPosition(+1)
  ##   position_ids = [300], kv_position unchanged at 300
  ##   attn reads kv_position (=300) == position_ids.min(), writes offset 300
  ##   after forward: kv_pos advanced to 301
  ## ```
  ##
  ## Args:
  ##
  ## - position: current position (cumulative sequence length - 1)
  ## - token_id: the token being decoded
  ##
  ## Note: KV pages are NOT reset, they accumulate across decode steps.
  var ctx = orc.active_context

  # 1. Append token to input_tokens tracking
  ctx.input_tokens.add(token_id)

  # 2. Lazily allocate pages on page-boundary crossing
  #    (kv_position still reflects total written tokens before this decode)
  if ctx.kv_position > 0 and (ctx.kv_position mod TokensPerPage) == 0:
    # About to write into a new page, borrow one
    orc.ensurePoolCapacity(1)
    ctx.pages.add(orc.page_pool.borrow())

  # 3. Update position_ids for single token: [position]
  #    kv_position is NOT incremented here, generate() does it after forward
  #    so that attention's `let offset = ctx.kv_position` gives the correct
  #    write position (equal to position_ids.min()) without GPU→CPU sync.
  ctx.setPositionIds(@[position])

proc endSequence*(orc: var Orchestrator) =
  ## End the current sequence and commit all data to the trie.
  ##
  ## 1. Collects ALL tokens and ALL page references accumulated during
  ##    the sequence (prefill + decode steps).
  ## 2. `graftPages` sinks the pages into the trie, handling matching, forking, and appending internally.
  ## 3. Locks acquired by LPM are released as a side effect.
  ## 4. Resets the InferenceContext for the next sequence.
  ##
  ## After this call:
  ##   - Page refs are held only by the trie (GC manages lifetime).
  ##   - Pool indices will be recycled when the trie evicts those pages.
  ##   - A new `startSequence` can begin immediately.

  var ctx = orc.active_context

  # Collect ALL tokens and ALL pages
  let fullTokens = ctx.input_tokens
  var fullPages = newSeq[Page](ctx.pages.len)
  for i, p in ctx.pages:
    fullPages[i] = p

  # graftPages sinks Page refs into the trie, releases LPM locks
  orc.logical_map.graftPages(fullTokens, fullPages)

  # Reset context (Page refs drop to just trie's refs, GC manages lifetime)
  ctx.clearState()
