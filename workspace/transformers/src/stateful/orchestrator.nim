# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  ./kvcache,
  ./page_pool,
  ./inference_context


# ═════════════════════════════════════════════════════════════════════════════
#
#  Orchestrator — paged KV cache lifecycle manager
#
# ═════════════════════════════════════════════════════════════════════════════
#
# ## Lifecycle (single sequence, MVP)
#
# ```
#         Orchestrator.init       startSequence       decodeStep × N      endSequence
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
# ### `cached_tokens` — Attention write-skip offset
#
# Purpose: Tell the attention layer how many prefix positions are ALREADY in the
#          trie from LPM, so it can skip writing them (protecting immutable pages).
#
# ```
#            startSequence                     decodeStep × N
#                 │                                │
#                 ▼                                ▼
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
#   writeStart = max(0, cached_tokens - offset)
#   - First prefill: cached_tokens=0, offset=0 → writeStart=0, ALL positions written.
#   - COW continuation: cached_tokens=300, offset=0 → writeStart=300, cached prefix skipped.
#   - Decode: cached_tokens=0, offset=28 → writeStart=0, writes 1 new token.
#
# ### `kv_position` — Page allocation write cursor
#
# Purpose: Track total tokens written to the KV cache so far in this sequence.
#          Used ONLY by `decodeStep` for page-boundary detection.
#
# ```
#            startSequence         generate() after          decodeStep × N
#                                    prefill forward
#                 │                       │                       │
#                 ▼                       ▼                       ▼
#     ┌─────────────────────┐   ┌─────────────────────┐   ┌────────────────────┐
#     │ kv_position = 0    │   │ setKvPosition(      │   │ Check: kv_position  │
#     │ (no tokens written  │──▶│   ids.len)          │──▶│  > 0 and mod 256    │
#     │  yet, LPM matched   │   │                     │   │  == 0 → borrow page │
#     │  tokens are in trie,│   │ kv_position now     │   │ kv_position += 1    │
#     │  not in local pages)│   │ reflects total      │   │                     │
#     └─────────────────────┘   │ written tokens      │   │ Positions 0..N-1    │
#                              └─────────────────────┘   │ are in local pages  │
#                                                         └────────────────────┘
# ```
#
# Lifecycle:
#   1. `startSequence`: set to 0 (no local tokens written yet).
#   2. `setKvPosition(n)` (called by generate() AFTER prefill forward): set to ids.len.
#   3. `decodeStep`: checked for page boundary, then incremented by 1.
#   4. `clearState`: reset to 0.
#
# ### Why two fields? (BUG-B-001 history)
#
# Originally `kv_position` was used for BOTH purposes. Setting it to `input_ids.len`
# in `startSequence` caused the attention layer's `writeStart = max(0, kv_position - offset)`
# to compute writeStart = seq_len for the first prefill — skipping ALL KV writes.
# The KV cache was empty, and every decode step read garbage.
#
# Separating `cached_tokens` (write-skip, stable after LPM) from `kv_position`
# (write-cursor, updated after forward pass) eliminates this coupling.
#
# ## Sequence lifecycle
#
# 1. **init** — Creates the GPU page pool (eager allocation), the PagedRadixTrie
#    (logical_map), and an InferenceContext.  Pool size is computed from the
#    model's `max_position_embeddings` via `computeNumPages`.
#
# 2. **startSequence** — Resets the InferenceContext, runs Longest Prefix Match
#    on the trie to find shared prefix with existing sequences.  Handles
#    Copy-on-Write for the partially-matched page (if the match ends mid-page).
#    Borrows fresh pages from the pool for any unmatched prompt tokens.  Sets
#    position_ids for the prefill forward pass.  Sets `cached_tokens` from LPM.
#
# 3. **decodeStep (× N)** — Appends one token to the tracking sequence.  If
#    the write cursor crosses a page boundary, borrows a new page.  Updates
#    position_ids for the single-token decode forward pass.  Checks `kv_position`
#    for page boundaries (not `cached_tokens`).
#
# 4. **endSequence** — Collects ALL tokens and ALL pages accumulated during the
#    sequence, then sinks them into the trie via `graftPages`.  The trie
#    handles matching, forking, and appending internally.  Locks acquired by
#    LPM are released as a side effect.  The InferenceContext is reset for the
#    next sequence.
#
# ## Memory model
#
# ```
# OrchestratorObj
#   ├── page_pool: PagePool       — owns GPU k_buffer / v_buffer
#   │     (destructed LAST — field order ensures trie freed before pool)
#   ├── logical_map: KVCache      — owns PagedRadixTrie with Page refs
#   │     (destructed FIRST — Pages → views release pool buffer refs)
#   │     └── PagedRadixNode
#   │           └── pages: seq[Page]
#   │                 └── PageObj
#   │                       ├── k_view: Tensor   (view into pool.k_buffer)
#   │                       ├── v_view: Tensor   (view into pool.v_buffer)
#   │                       └── pool: PagePool   (cursor, not counted)
#   └── active_context: InferenceContext
#         └── pages: seq[Page]    — pages for current sequence
# ```
#
# **Destruction order** (reverse declaration order):
#   1. `logical_map` → trie freed → Pages freed → k_view/v_view TorchTensors
#      destroyed → pool Storage refcount decremented
#   2. `page_pool` → k_buffer/v_buffer TorchTensors destroyed → Storage freed
#      (if refcount reached 0 in step 1)
#
# No custom `=destroy` hooks — ORC's default field-by-field destruction
# correctly handles the Nim Tensor ref → C++ TorchTensor → intrusive_ptr →
# Storage → CUDA allocator chain.
#


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

func computeNumPages*(maxContextTokens: int; concurrentRequests: int): int =
  ## Compute the number of KV cache pages needed for a pool.
  ## Each page stores TokensPerPage (256) KV entries.
  ##
  ## Args:
  ##   maxContextTokens: Maximum context length (model's max_position_embeddings)
  ##   concurrentRequests: Concurrent sequences (1 for single-sequence MVP)
  ##
  ## Returns:
  ##   Number of pages with headroom for COW partial pages + lazy decode.
  let pagesPerRequest = ceilDiv(maxContextTokens, TokensPerPage)
  let basePages = pagesPerRequest * concurrentRequests
  result = basePages + concurrentRequests  # headroom

func computePageSizeBytes*(num_layers, kv_heads, head_dim: int; dtype: ScalarKind): int64 =
  ## Memory size of a single page in bytes (K + V).
  ##
  ## Args:
  ##   num_layers: Number of transformer layers
  ##   kv_heads: Number of KV heads (GQA)
  ##   head_dim: Dimension per head
  ##   dtype: Element data type (e.g. kBFloat16)
  ##
  ## Returns:
  ##   Total bytes for one page's K and V across all layers.
  let elementSize = case dtype
    of kFloat32: 4
    of kFloat16, kBFloat16: 2
    of kFloat64: 8
    of kInt8, kUint8, kQint8, kQuint8: 1
    of kInt16, kUint16: 2
    of kInt32, kUint32: 4
    of kInt64, kUint64: 8
    else: 2
  let elPerBuf = num_layers.int64 * TokensPerPage.int64 * kv_heads.int64 * head_dim.int64
  result = elPerBuf * elementSize.int64 * 2  # K+V


# ═══════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════

type
  Orchestrator* = ref object
    ## High-level orchestration for inference with paged KV cache.
    ##
    ## Owns:
    ##   - `page_pool`: GPU KV buffer allocator (k_buffer, v_buffer)
    ##   - `logical_map`: PagedRadixTrie mapping token sequences → Page refs
    ##   - `active_context`: Mutable state for the current sequence
    ##
    ## **Lifecycle**: `init` → `startSequence` → `decodeStep`* → `endSequence`
    ##
    ## **Destruction order** (reverse field order):
    ##   `logical_map` before `page_pool` — the trie's Page refs hold views
    ##   into the pool's k_buffer/v_buffer.  When the trie is freed first,
    ##   those views are destroyed, releasing their Storage references, so
    ##   the pool's buffers can be freed cleanly.
    ##
    ## No custom `=destroy` hook — ORC default field destruction handles
    ## the Nim Tensor → C++ TorchTensor → intrusive_ptr → CUDA allocator
    ## chain correctly without extra nil-ing.
    page_pool: PagePool
    active_context: InferenceContext
    num_layers: int
    device: DeviceKind
    position_ids_buf: Tensor  # pre-allocated 1-element tensor for decodeStep
    logical_map: KVCache[uint32, Page]

# ═══════════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════════

proc init*(_: type Orchestrator;
            num_layers, batch_size, kv_heads, max_seq, head_dim: int;
            num_pages: int;
            dtype: ScalarKind; device: DeviceKind): Orchestrator =
  ## Create an Orchestrator with eager GPU page pool allocation.
  ##
  ## Args:
  ##   num_layers: number of transformer layers
  ##   batch_size: batch dimension (metadata for InferenceContext)
  ##   kv_heads: number of KV heads (GQA)
  ##   max_seq: maximum sequence length
  ##   head_dim: dimension per head
  ##   num_pages: total pages to pre-allocate in the GPU pool
  ##   dtype: element data type (kBFloat16, etc.)
  ##   device: target device (kCUDA, etc.)
  ##
  ## Pool is created eagerly (not lazily) so OOM is detected at init time.
  result = Orchestrator(
    logical_map: KVCache[uint32, Page].new(),
    page_pool: PagePool.init(num_pages, num_layers, kv_heads, head_dim, dtype, device),
    active_context: InferenceContext.init(
      num_layers, batch_size, kv_heads, max_seq, head_dim),
    num_layers: num_layers,
    device: device,
    position_ids_buf: F.zeros(1, F.tensorOptions(F.kInt64, device))
  )

proc getInferenceContextMut*(orc: var Orchestrator): var InferenceContext {.inline.} =
  ## Get the active inference context.
  orc.active_context

proc setKvPosition*(orc: var Orchestrator, pos: int) {.inline.} =
  ## Set the write cursor position (called after prefill forward completes).
  orc.active_context.kv_position = pos

# Pool management
# ═══════════════════════════════════════════════════════════════════════════

proc ensurePoolCapacity(orc: var Orchestrator, needed: int) =
  ## Ensure the pool has at least `needed` free pages, evicting from the
  ## trie if necessary.
  ##
  ## Raises `ValueError` if no eviction candidates are available (all pages
  ## are locked on active decode paths).
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


# ═══════════════════════════════════════════════════════════════════════════
# COW (Copy-on-Write) helper
# ═══════════════════════════════════════════════════════════════════════════

proc cowPartialPage(dst, src: Page; partialTokens, numLayers: int) =
  ## Copy partial KV content from the cached page (`src`) into a newly
  ## borrowed COW page (`dst`).  Used when LPM matches a page partially
  ## — the matched prefix is kept in the trie, the suffix is copied to
  ## a private page for this sequence.
  ##
  ## Copies `partialTokens` positions across all `numLayers` layers.
  ## The tensor slices are temporaries destroyed on return; they do not
  ## leak references to the pool's buffers.
  dst.k_view[0 ..< numLayers, 0 ..< partialTokens].copyFrom(
    src.k_view[0 ..< numLayers, 0 ..< partialTokens])
  dst.v_view[0 ..< numLayers, 0 ..< partialTokens].copyFrom(
    src.v_view[0 ..< numLayers, 0 ..< partialTokens])


# ═══════════════════════════════════════════════════════════════════════════
# Sequence lifecycle
# ═══════════════════════════════════════════════════════════════════════════

proc startSequence*(
    orc: var Orchestrator,
    input_ids: seq[uint32]) =
  ## Start a new sequence.
  ##
  ## 1. Resets the InferenceContext (clears pages, tokens, position).
  ## 2. Runs LPM on the trie to find shared prefix with cached sequences.
  ## 3. Handles Copy-on-Write if the match ends mid-page.
  ## 4. Borrows fresh pages from the pool for unmatched prompt tokens.
  ## 5. Sets position_ids for the prefill forward pass.
  ##
  ## Args:
  ##   input_ids: Input token IDs (prompt)

  if input_ids.len == 0:
    raise newException(ValueError,
      "[ttt] Empty prompt")

  # Reset context for fresh sequence
  orc.active_context.clearState()
  var ctx = orc.active_context

  # ── 1. LPM — find shared prefix ──
  ctx.input_tokens = input_ids
  let matched = orc.logical_map.lpm(input_ids)
  ctx.cached_tokens = matched.totalTokenMatched  # for attention writeStart

  # ── 2. COW partial page handling ──
  let partialTokens = ctx.cached_tokens mod TokensPerPage
  var cowPage: Page = nil
  var cowPageUsed = false
  if partialTokens > 0 and matched.pages.len > 0:
    # Last matched page is partial — borrow a new page, copy partial content
    let cachedPage = matched.pages[^1]
    orc.ensurePoolCapacity(1)
    cowPage = orc.page_pool.borrow()
    cowPartialPage(cowPage, cachedPage, partialTokens, orc.num_layers)
    cowPageUsed = true

  # ── 3. Add fully-matched pages from trie (in order) ──
  let fullMatchCount = matched.pages.len - (if cowPageUsed: 1 else: 0)
  for i in 0 ..< fullMatchCount:
    ctx.pages.add(matched.pages[i])

  # ── 3b. Append COW page after fully-matched pages ──
  if cowPageUsed:
    ctx.pages.add(cowPage)

  # ── 4. Borrow new pages for unmatched prompt portion ──
  let promptPages = ceilDiv(input_ids.len, TokensPerPage)
  doAssert promptPages >= ctx.pages.len,
    "[ttt] Invariant: prompt should need >= pages than already matched by trie"
  let newPagesNeeded = promptPages - ctx.pages.len
  if newPagesNeeded > 0:
    orc.ensurePoolCapacity(newPagesNeeded)
    for _ in 0 ..< newPagesNeeded:
      ctx.pages.add(orc.page_pool.borrow())

  # ── 5. Set position_ids for prefill ──
  ctx.setPositionIdsArange(input_ids.len, offset = 0, device = orc.device)

proc decodeStep*(orc: var Orchestrator, position: int, token_id: uint32,
                 device: DeviceKind | Device = kCPU) =
  ## Prepare for a single decode step (one token).
  ##
  ## 1. Appends the token to input_tokens tracking.
  ## 2. If the write cursor crosses a page boundary, lazily borrows a new
  ##    page from the pool.
  ## 3. Updates position_ids for the single-token forward pass.
  ##
  ## NOTE: kv_position is NOT incremented here — the caller (generate())
  ## increments it AFTER the forward call.  This ensures that during the
  ## attention forward, `ctx.kv_position` equals `position_ids.min()`,
  ## so the attention layer can use `kv_position` as the write offset
  ## instead of reading position_ids.min().item(int) (which forces a
  ## GPU→CPU synchronous read on every forward pass).
  ##
  ## Lifecycle for one decode step:
  ##   decodeStep(position=300) -> forward() -> generate: setKvPosition(+1)
  ##     |                            |
  ##     | position_ids = [300]       | attn reads kv_position (=300)
  ##     | kv_position unchanged      | which equals position_ids.min()
  ##     |                            | writes at offset 300
  ##     V                            V
  ##   kv_pos=300 matches pos.min()   kv_pos advanced to 301 after forward
  ## Args:
  ##   position: Current position (cumulative sequence length - 1)
  ##   token_id: The token being decoded
  ##   device: Device for position_ids tensor
  ##
  ## Note: KV pages are NOT reset — they accumulate across decode steps.
  var ctx = orc.active_context

  # 1. Append token to input_tokens tracking
  ctx.input_tokens.add(token_id)

  # 2. Lazily allocate pages on page-boundary crossing
  #    (kv_position still reflects total written tokens before this decode)
  if ctx.kv_position > 0 and (ctx.kv_position mod TokensPerPage) == 0:
    # About to write into a new page — borrow one
    orc.ensurePoolCapacity(1)
    ctx.pages.add(orc.page_pool.borrow())

  # 3. Update position_ids for single token: [position] on device
  #    kv_position is NOT incremented here — generate() does it after forward
  #    so that attention's `let offset = ctx.kv_position` gives the correct
  #    write position (equal to position_ids.min()) without GPU→CPU sync.
  orc.position_ids_buf[0] = position.int64
  ctx.position_ids = orc.position_ids_buf
proc endSequence*(orc: var Orchestrator) =
  ## End the current sequence and commit all data to the trie.
  ##
  ## 1. Collects ALL tokens and ALL page references accumulated during
  ##    the sequence (prefill + decode steps).
  ## 2. Calls `graftPages` which sinks the pages into the trie.  The trie
  ##    handles matching, forking, and appending internally.
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

  # graftPages sinks Page refs into the trie — releases LPM locks
  orc.logical_map.graftPages(fullTokens, fullPages)

  # Reset context (Page refs drop to just trie's refs, GC manages lifetime)
  ctx.clearState()
