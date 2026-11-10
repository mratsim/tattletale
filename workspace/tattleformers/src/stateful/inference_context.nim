# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## InferenceContext: the per-forward-pass KV state container over the page pool and trie.
##
## setRopeForPositions (the rope-slice flow) consumes the rotary
## embedding layer and comes with the transformer layers port.

import
  ./kvcache,
  ./page_pool

type InferenceContext*[T, P] = ref object
  ## State container for a SINGLE forward pass.
  ## Created by orchestrator per request, passed through layers.
  ##
  ## LIFECYCLE:
  ##   - Created at start of sequence (prefill)
  ##   - Reused across decode steps (pages accumulate)
  ##   - position_ids updated each forward pass
  ##   - Discarded when sequence completes

  pages*: seq[Page]           # KV pages: matched from trie + newly borrowed
  pool*: PagePool             # Orchestrator-owned page pool, set at init.
                              # Attention layers read the pool slabs through
                              # this field. Stable per orchestrator.
  kv_position*: int            # Write cursor (next token position to write into), for page allocation
  cached_tokens*: int          # Tokens already in trie from LPM (for attention writeStart)
  input_tokens*: seq[T]        # Token sequence tracking (for graftPages at sequence end)

  position_ids*: seq[int]      # [0,1,2] for prefill, [3] for decode, [6,3,11] for ragged
  ## Debug metadata, describes the context configuration
  num_layers*: int
  batch_size*: int
  kv_heads*: int
  max_seq*: int
  head_dim*: int

proc init*[T, P](
    _: type InferenceContext[T, P],
    num_layers: int,
    batch_size: int,
    kv_heads: int,
    max_seq: int,
    head_dim: int): InferenceContext[T, P] =
  ## Initialize InferenceContext with empty KV page tracking.
  ## KV buffer dimensions are on PagePool (owned by orchestrator).
  ##
  ## Args:
  ##
  ## - num_layers: number of transformer layers
  ## - batch_size: number of sequences in the batch (metadata)
  ## - kv_heads: number of KV heads (GQA, metadata)
  ##
  ## - max_seq: maximum sequence length (metadata)
  ## - head_dim: dimension per head (metadata)

  InferenceContext[T, P](
    num_layers: num_layers,
    batch_size: batch_size,
    kv_heads: kv_heads,
    max_seq: max_seq,
    head_dim: head_dim
  )

proc clearState*[T, P](ctx: InferenceContext[T, P]) =
  ## Clear KV state for reuse in a new sequence.
  ##
  ## Postcondition: pages, kv_position, cached_tokens, input_tokens, position_ids reset to their empty state.
  ## Metadata fields (num_layers, head_dim, etc.) stay for reuse.
  ## Page references drop (GC may recycle to pool).
  ctx.pages = default(seq[Page])
  ctx.kv_position = 0
  ctx.cached_tokens = 0
  ctx.input_tokens = default(seq[T])
  ctx.position_ids = default(seq[int])

proc pageSlots*[T, P](ctx: InferenceContext[T, P]): seq[int] =
  ## Returns the page table consumed by the pool write and jagged
  ## attention ops: the pool slot of each sequence page, in page order.
  ## Position p lives in page p div TokensPerPage and its slot is
  ## pageSlots[p div TokensPerPage].
  result = newSeq[int](ctx.pages.len)
  for i in 0 ..< ctx.pages.len:
    result[i] = int(ctx.pages[i].pageIndex())

proc setPositionIds*[T, P](ctx: InferenceContext[T, P], position_ids: seq[int]) =
  ## Set position_ids for current forward pass.
  ##
  ## Args:
  ##   position_ids: One flat position per token
  ##
  ## Note: Called every forward pass (prefill + each decode step)
  ctx.position_ids = position_ids

proc setPositionIdsArange*[T, P](ctx: InferenceContext[T, P], seq_len: int, offset: int = 0) =
  ## Set position_ids to arange(offset, offset+seq_len).
  ##
  ## Args:
  ##
  ## - seq_len: sequence length
  ## - offset: starting offset, 0 for prefill
  ctx.position_ids = newSeq[int](seq_len)
  for i in 0 ..< seq_len:
    ctx.position_ids[i] = offset + i
