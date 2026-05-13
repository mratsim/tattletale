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
  workspace/transformers/src/layers,
  workspace/transformers/src/models

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

  # Auto-derive positions: (1, seq_len) tensor of 0..seq_len-1
  # Matches HF: ``position_ids = arange(seq_len) + past_seen_tokens``
  # For prefix pass: past_seen_tokens = 0
  let seqLen = input.shape[1]
  var posSeq: seq[int64] = newSeq[int64](seqLen)
  for i in 0..<seqLen: posSeq[i] = i.int64
  let positions = posSeq.toTensor().unsqueeze(0)

  # Empty cache for prefix pass (cache is per-layer inside RopeGQAttention)
  var cache = KVCache.init()

  # Run forward
  let logits = self.model.forward(input, positions, cache)

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
