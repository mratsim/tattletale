# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared helpers for the transformer test suites that drive stateful layers:
## one home for the inference context with its page pool, the tensor
## difference helper, and the safetensor opener.

import
  std/math,
  std/memfiles,
  std/os,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool

proc newKVContext*(
    numLayers, kvHeads, headDim: int,
    maxSeq = 512
  ): (InferenceContext, PagePool) =
  ## Fresh InferenceContext with a page pool sized for `maxSeq` tokens.
  ## The pool ref is returned with the context so the borrowed pages stay
  ## alive for the test duration.
  var ctx = InferenceContext.init(
    num_layers = numLayers, batch_size = 1,
    kv_heads = kvHeads, max_seq = maxSeq, head_dim = headDim)
  let pool = PagePool.init(
    64, num_layers = numLayers, kv_heads = kvHeads, head_dim = headDim,
    dtype = F.kBFloat16, device = F.kCPU)
  let numPages = ceilDiv(maxSeq, TokensPerPage)
  for i in 0 ..< numPages:
    ctx.pages.add(pool.borrow())
  (ctx, pool)

proc maxAbsDiff*(a, b: Tensor): float64 =
  ## Maximum absolute elementwise difference of two tensors, compared in f32.
  (a.to(F.kFloat32) - b.to(F.kFloat32)).abs().max().item(float64)

proc openSafetensor*(path: string): (MemFile, Safetensor) =
  ## Open one safetensors file. The memfile must stay open: the Safetensor
  ## holds zero-copy views into the file.
  let memFile = memFiles.open(path, mode = fmRead)
  result = (memFile, safetensors.load(memFile))

proc openSafetensor*(dir: string, name: string): (MemFile, Safetensor) =
  ## Open the safetensors file `dir / name`. The memfile must stay open
  ## while the Safetensor lives: it holds zero-copy views into the file.
  openSafetensor(dir / name)
