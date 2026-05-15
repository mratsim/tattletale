# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Nim ↔ Python wrapper for transformers model inference.
## Follows the HuggingFace Transformers pattern:
##   - `model(input_ids)` — single-pass, no cache, positions auto-derived
##   - `model(input_ids, cache=cache)` — with caller-managed cache (future)

import
  nimpy,
  workspace/libtorch,
  workspace/libtorch/src/tensors_py,
  workspace/transformers/src/models,
  workspace/transformers/src/stateful/inference_context

type
  ModelRef* = ref object of PyNimObjectExperimental
    model: Model

proc init_model*(path: string): ModelRef {.exportpy.} =
  ## Load a model from the given path.
  ## Follows HF pattern: ``model = pytttransformers.init_model(path)``
  result = ModelRef()
  result.model = loadModel(path, kCPU)

proc forward*(self: ModelRef, inputIds: PyObject): PyObject {.exportpy.} =
  ## Run a forward pass on the model.
  ##
  ## Follows the HF Transformers pattern:
  ##   - ``logits = model.forward(input_ids)`` — single-pass prefix, no cache
  ##   - Positions are auto-derived as ``arange(seq_len)``
  ##   - KV cache is empty (prefix pass)
  ##
  ## Args:
  ##   input_ids: torch.Tensor of shape ``(batch, seq_len)``, dtype ``int64``.
  ##
  ## Returns:
  ##   torch.Tensor of shape ``(batch, seq_len, vocab_size)`` — raw logits.

  # Python → Nim: extract input tensor
  let input = tensorFromPyObject(inputIds)

  # Derive dimensions from input + model config
  let batch = input.shape[0]
  let seqLen = input.shape[1]
  let cfg = self.model.getConfig()

  # Create InferenceContext with preallocated KV caches
  var ctx = InferenceContext.init(
    cfg.num_hidden_layers,
    batch,
    cfg.num_key_value_heads,
    cfg.max_position_embeddings,
    cfg.head_dim,
    kBFloat16,
    kCPU
  )

  # Set position_ids for prefill: [0, 1, 2, ..., seq_len-1]
  ctx.setPositionIdsArange(seqLen, offset = 0, device = kCPU)

  # Model forward populates ctx.cos/ctx.sin via setRopeForPositions internally
  let logits = self.model.forward(ctx, input)

  # Nim → Python: return logits tensor
  tensorToPyObject(logits)

setModuleDocString(
  "Nim transformers model inference — compatible with HuggingFace Transformers API"
)
setDocStringForType(
  ModelRef,
  "Opaque wrapper around a loaded Nim transformers model. " &
  "Use ``model.forward(input_ids)`` to get logits."
)
