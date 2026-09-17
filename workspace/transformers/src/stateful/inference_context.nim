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
  ../layers/rope

type InferenceContext* = ref object
  ## State container for a SINGLE forward pass.
  ## Created by orchestrator per request, passed through layers.
  ##
  ## LIFECYCLE:
  ##   - Created at start of sequence (prefill)
  ##   - Reused across decode steps (pages accumulate)
  ##   - position_ids updated each forward pass
  ##   - Discarded when sequence completes

  pages*: seq[Page]           # KV pages: matched from trie + newly borrowed
  kv_position*: int            # Write cursor (next token position to write into), for page allocation
  cached_tokens*: int          # Tokens already in trie from LPM (for attention writeStart)
  input_tokens*: seq[uint32]   # Token sequence tracking (for graftPages at sequence end)

  position_ids*: Tensor       # [0,1,2] for prefill, [3] for decode, [6,3,11] for ragged
  ## Debug metadata — describes the context configuration
  num_layers*: int
  batch_size*: int
  kv_heads*: int
  max_seq*: int
  head_dim*: int

  ## RoPE cos/sin for current forward pass.
  ## Sliced from the model's precomputed cache using position_ids.
  cos*: Tensor   ## (seq_len, head_dim) — valid after setRopeForPositions()
  sin*: Tensor   ## (seq_len, head_dim) — valid after setRopeForPositions()
  ## Gather buffers for K/V page gathering into contiguous tensors for SDPA.
  ## Pre-allocated at max_seq size to avoid per-forward-pass GPU allocations.
  ## Owned here (not on the layer) to keep attention stateless.
  k_gather_buf*: Tensor  ## (1, max_seq, kv_heads, head_dim) — allocated lazily on first forward
  v_gather_buf*: Tensor  ## (1, max_seq, kv_heads, head_dim) — allocated lazily on first forward
  ## GDN (Gated DeltaNet) per-layer recurrent state. Unlike the paged KV
  ## cache, GDN layers carry their whole context in these two tensors:
  ## the causal-conv history and the f32 SSM state. Allocated lazily on
  ## first use by the GDN layer forward, indexed by layer index.
  gdnConvState*: seq[Tensor]  ## Per GDN layer: [conv_dim, 3] BF16 causal-conv history
  gdnSsmState*: seq[Tensor]   ## Per GDN layer: [num_v_heads, Dk, Dv] F32 SSM state
  ## KDA per-layer recurrent state, same lifecycle as the GDN slots:
  ## causal-conv history is per branch, q/k/v each keeping k-1 tokens
  ## (the KDA checkpoints carry one conv weight per branch) and the f32
  ## SSM state is the per-channel-decay delta-rule core.
  kdaConvState*: seq[array[3, Tensor]]
    ## Per KDA layer: branch q/k/v histories, each [branch_dim, k-1] BF16
  kdaSsmState*: seq[Tensor]   ## Per KDA layer: [num_heads, Dk, Dv] F32 SSM state

proc init*(
    _: type InferenceContext,
    num_layers: int,
    batch_size: int,
    kv_heads: int,
    max_seq: int,
    head_dim: int): InferenceContext =
  ## Initialize InferenceContext with empty KV page tracking.
  ## KV buffer dimensions are on PagePool (owned by orchestrator).
  ##
  ## Args:
  ##   num_layers: Number of transformer layers
  ##   batch_size: Number of sequences in batch (metadata)
  ##   kv_heads: Number of KV heads (GQA) (metadata)
  ##   max_seq: Maximum sequence length (metadata)
  ##   head_dim: Dimension per head (metadata)

  InferenceContext(
    num_layers: num_layers,
    batch_size: batch_size,
    kv_heads: kv_heads,
    max_seq: max_seq,
    head_dim: head_dim
  )


proc ensureGdnStates*(
    ctx: var InferenceContext,
    layer_idx, convDim, numVHeads, keyDim, valueDim: int,
    device: DeviceKind) =
  ## Allocate the GDN conv and SSM states for `layer_idx` (zeros).
  ##
  ## The GDN layer forward calls this on every pass. The seqs are sized to
  ## the context's layer count on first use, then only nil slots are filled;
  ## the seq payloads never reallocate mid-forward. clearState drops the
  ## seqs so a new sequence starts from zero state.
  ##
  ## TODO: chunked GDN prefill and long-context state management (rolling
  ## the conv/SSM window past the kernel size) are future work. The current
  ## state is the full per-sequence cache.
  if ctx.gdnConvState.len <= layer_idx:
    ctx.gdnConvState.setLen(max(ctx.num_layers, layer_idx + 1))
    ctx.gdnSsmState.setLen(max(ctx.num_layers, layer_idx + 1))
  if ctx.gdnConvState[layer_idx].isNil:
    ctx.gdnConvState[layer_idx] = F.zeros(
      convDim, 3, F.tensorOptions(F.kBFloat16, device))
    ctx.gdnSsmState[layer_idx] = F.zeros(
      numVHeads, keyDim, valueDim, F.tensorOptions(F.kFloat32, device))

proc ensureKdaStates*(
    ctx: var InferenceContext,
    layer_idx, queryDim, keyDim, valueDim, numHeads, keyHeadDim, valueHeadDim,
    convKernel: int,
    device: DeviceKind) =
  ## Allocate layer layer_idx's KDA states zero-filled: three bf16 conv
  ## tail buffers for the q/k/v branches (tail = convKernel - 1) and one
  ## f32 SSM state numHeads x keyHeadDim x valueHeadDim. The state seqs
  ## grow to the layer count once, only nil slots allocate; clearState
  ## drops both seqs so the next sequence restarts from zero.
  if ctx.kdaConvState.len <= layer_idx:
    ctx.kdaConvState.setLen(max(ctx.num_layers, layer_idx + 1))
    ctx.kdaSsmState.setLen(max(ctx.num_layers, layer_idx + 1))
  if ctx.kdaConvState[layer_idx][0].isNil:
    let tail = convKernel - 1
    ctx.kdaConvState[layer_idx][0] = F.zeros(
      queryDim, tail, F.tensorOptions(F.kBFloat16, device))
    ctx.kdaConvState[layer_idx][1] = F.zeros(
      keyDim, tail, F.tensorOptions(F.kBFloat16, device))
    ctx.kdaConvState[layer_idx][2] = F.zeros(
      valueDim, tail, F.tensorOptions(F.kBFloat16, device))
    ctx.kdaSsmState[layer_idx] = F.zeros(
      numHeads, keyHeadDim, valueHeadDim, F.tensorOptions(F.kFloat32, device))

proc setRopeForPositions*(ctx: var InferenceContext, rotary: RotaryPositionEmbedding) =
  ## Populate ctx.cos and ctx.sin from the model's RoPE cache.
  ##
  ## Internally calls `rotary.ropeByPositions(ctx.position_ids)` and stores
  ## the result in `ctx.cos` / `ctx.sin` for downstream attention layers.
  ##
  ## Called once per forward pass. The model is responsible for calling this
  ## so the orchestrator stays ignorant of rope variants.
  (ctx.cos, ctx.sin) = rotary.ropeByPositions(ctx.position_ids)

proc clearState*(ctx: var InferenceContext) =
  ## Clear KV state for reuse in a new sequence.
  ## Drops page references (GC may recycle to pool).
  ## Keeps metadata fields (num_layers, head_dim, etc.) for reuse.
  ##
  ## NOTE: cos/sin (RoPE) are NOT cleared — they are stable per model
  ## and are overwritten by the next `setRopeForPositions` call.
  ## Stale cos/sin cannot leak between sequences because
  ## `setRopeForPositions` is always called before any forward pass.
  ctx.pages = default(seq[Page])
  ctx.kv_position = 0
  ctx.cached_tokens = 0
  ctx.input_tokens = default(seq[uint32])
  ctx.position_ids = nil
  ctx.k_gather_buf = nil
  ctx.v_gather_buf = nil
  ctx.gdnConvState = default(seq[Tensor])
  ctx.gdnSsmState = default(seq[Tensor])
  ctx.kdaConvState = default(seq[array[3, Tensor]])
  ctx.kdaSsmState = default(seq[Tensor])

proc setPositionIds*(ctx: var InferenceContext, position_ids: Tensor) =
  ## Set position_ids for current forward pass.
  ##
  ## Args:
  ##   position_ids: Tensor of shape (batch, seq_len) or (seq_len,)
  ##
  ## Note: Called every forward pass (prefill + each decode step)
  ctx.position_ids = position_ids

proc setPositionIdsArange*(ctx: var InferenceContext, seq_len: int, offset: int = 0, device: DeviceKind = kCPU) =
  ## Set position_ids to arange(offset, offset+seq_len), kInt64.
  ##
  ## Args:
  ##   seq_len: Sequence length
  ##   offset: Starting offset (default 0 for prefill)
  ##   device: Device for tensor
  let opts = F.tensorOptions(F.kInt64, device)
  ctx.position_ids = F.arange(offset, offset + seq_len, opts)
