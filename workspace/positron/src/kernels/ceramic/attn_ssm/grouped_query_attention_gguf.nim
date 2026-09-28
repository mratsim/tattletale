## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.


# ############################################################
#
#     GGUF attention launchers (grouped_query_attention_gguf)
#
# ############################################################

## GGUF attention launcher set, one Metal block `ggufAttnMsl` hosting
## these launchers, all at D = 128:
## - the quantized-linear launchers, one per GGufScheme
## - the fused qk-norm+rope launcher
## - the paged-attention launcher
## plus the device kernels they call. Each launcher's first parameter
## is the output buffer. engine.run binds the separate outBuf
## argument to it and the args tuple to the rest.
##
## Orchestration across kernels (projection → qk-norm+rope → cache
## write → paged attention → o_proj) is the caller's job.
## Requires H % 8 == 0, Nkv % 8 == 0 and the paged attention's
## compiled-in page_size 16, a static binding of the launcher.

import workspace/crucible
import ../linear_quant/linear_gguf
export linear_gguf.GGufScheme
import ./grouped_query_attention_qk_norm_rope
import ./grouped_query_attention_paged

const ggufAttnMsl* = metal:
  proc ggufLinearQ8(
      Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      w: ptr UncheckedArray[uint8], M, K, N, rowBytes: int32) {.global.} =
    gguf_linear_fwd(Out, x, w, M, K, N, rowBytes, gsQ8_0)

  proc ggufLinearQ4K(
      Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      w: ptr UncheckedArray[uint8], M, K, N, rowBytes: int32) {.global.} =
    gguf_linear_fwd(Out, x, w, M, K, N, rowBytes, gsQ4_K)

  proc ggufLinearIQ4XS(
      Out: ptr UncheckedArray[float16], x: ptr UncheckedArray[float16],
      w: ptr UncheckedArray[uint8], M, K, N, rowBytes: int32) {.global.} =
    gguf_linear_fwd(Out, x, w, M, K, N, rowBytes, gsIQ4_XS)

  proc ggufQkNormRopeD128(
      Out: ptr UncheckedArray[float16], X, G: ptr UncheckedArray[float16],
      Cos, Sin: ptr UncheckedArray[float32],
      xTokenStride, cosTokenStride, headBlocks, xColBase: int32,
      eps: float32) {.global.} =
    qk_norm_rope_fwd(Out, X, G, Cos, Sin, xTokenStride, cosTokenStride,
                     headBlocks, xColBase, eps)

  proc ggufPagedD128(
      o, q: ptr UncheckedArray[float16],
      k_cache, v_cache: ptr UncheckedArray[float16],
      block_table, cache_seqlens, cu_seqlens_q: ptr UncheckedArray[int32],
      num_seqs, H, Nkv, max_pages, page_size: int32) {.global.} =
    paged_attn_fwd(o, q, k_cache, v_cache, block_table, cache_seqlens,
                   cu_seqlens_q, num_seqs, H, Nkv, max_pages,
                   num_layers = 1, layer = 0, page_size = 16, D = 128)
