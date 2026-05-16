# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  ./kvcache,
  ./inference_context

type Orchestrator* = object
  ## High-level orchestration for inference.
  ##
  ## **input_ids vs position_ids**:
  ##
  ##   The orchestrator owns both, but they serve different roles:
  ##
  ##   - `input_ids`: Token IDs passed to `model.forward()`. *What* to compute.
  ##     Example: `[9707, 11, 1246]` = "Hello, how". Set by the caller before
  ##     invoking the model.
  ##
  ##   - `position_ids`: Stored in `InferenceContext.position_ids`. *Where* each
  ##     token sits in the absolute sequence. Set by the orchestrator based on
  ##     scheduler state (prefill offset, decode step, etc.).
  ##
  ##   For single-sequence sync inference:
  ##     prefill:  position_ids = [0, 1, 2, ..., seq_len-1]
  ##     decode:   position_ids = [current_position]
  ##
  ##   They diverge for continuous batching, prefix caching, and speculative decoding.
  ##
  ## MVP: Single active sequence.
  ## Future: Multiple contexts for continuous batching.

  active_context*: InferenceContext  # Current sequence context
  is_active*: bool                   # True if sequence in progress
  num_layers*: int

proc init*(_: type Orchestrator, num_layers: int): Orchestrator =
  ## Initialize orchestrator.
  ##
  ## Args:
  ##   num_layers: Number of transformer layers
  Orchestrator(
    active_context: InferenceContext.init(num_layers, 1, 1, 1, 1, F.kFloat32, F.kCPU),
    is_active: false,
    num_layers: num_layers
  )

proc startSequence*(orch: var Orchestrator, batch_size, kv_heads, max_seq, head_dim: int,
                    dtype: ScalarKind, device: DeviceKind, seq_len: int) =
  ## Start new sequence (prefill phase).
  ##
  ## Creates fresh InferenceContext with preallocated KV caches.
  ##
  ## Args:
  ##   batch_size: Batch size (1 for single sequence)
  ##   kv_heads: Number of KV heads
  ##   max_seq: Maximum sequence length
  ##   head_dim: Dimension per head
  ##   dtype: Data type
  ##   device: Device
  ##   seq_len: Initial sequence length (prompt length)

  # Create fresh context with allocated caches
  orch.active_context = InferenceContext.init(
    orch.num_layers, batch_size, kv_heads, max_seq, head_dim, dtype, device
  )

  # Set position_ids for prefill: [0, 1, 2, ..., seq_len-1]
  orch.active_context.setPositionIdsArange(seq_len, offset=0, device=device)

  orch.is_active = true

proc decodeStep*(orch: var Orchestrator, position: int, device: DeviceKind | Device = kCPU) =
  ## Prepare for decode step.
  ##
  ## Args:
  ##   position: Current position (cumulative sequence length)
  ##   device: Device for position_ids tensor
  ##
  ## Note: KV caches NOT reset — they accumulate across decode steps

  # Update position_ids for single token: [position]
  orch.active_context.position_ids = [position].toTensor().to(device)

proc endSequence*(orch: var Orchestrator) =
  ## End current sequence.
  ##
  ## Resets context for next sequence.
  orch.active_context.reset()
  orch.is_active = false
