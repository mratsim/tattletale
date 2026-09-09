# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Inference context with its page pool for the q_bf16 stateful suites.
## The pool ref is returned with the context so the borrowed pages stay
## alive for the test duration.

import
  std/math,
  workspace/libtorch as F,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool

proc newKVContext*(
    numLayers, kvHeads, headDim: int,
    maxSeq = 512, device = F.kCPU
  ): (InferenceContext, PagePool) =
  ## Fresh InferenceContext with a page pool sized for `maxSeq` tokens.
  var ctx = InferenceContext.init(
    num_layers = numLayers, batch_size = 1,
    kv_heads = kvHeads, max_seq = maxSeq, head_dim = headDim)
  let pool = PagePool.init(
    64, num_layers = numLayers, kv_heads = kvHeads, head_dim = headDim,
    dtype = F.kBFloat16, device = device)
  let numPages = ceilDiv(maxSeq, TokensPerPage)
  for i in 0 ..< numPages:
    ctx.pages.add(pool.borrow())
  (ctx, pool)
