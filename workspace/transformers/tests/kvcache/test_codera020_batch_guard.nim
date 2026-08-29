## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Anti-regression test for CODERA-020:
## Paged KV attention must reject batch_size > 1.

import
  std/options,
  std/importutils,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/transformers/src/stateful/kvcache {.all.},
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/layers/attn_ssm/grouped_query_attention {.all.},
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/layers/linear {.all.},
  workspace/transformers/src/layers/norm {.all.}

privateAccess(RopeGQAttention)
privateAccess(GroupedQueryAttention)

proc main() =

  runCppTest "forward with batch=2 raises ValueError":
    proc(): bool =
      # Create dummy weights for projections
      let headDim = 64
      let hiddenSize = 256
      let numQoHeads = 4
      let numKvHeads = 2
      let weightOpts = F.tensorOptions(F.kFloat32, F.kCPU)

      let qWeight = F.randn([hiddenSize, numQoHeads * headDim], weightOpts)
      let kWeight = F.randn([hiddenSize, numKvHeads * headDim], weightOpts)
      let vWeight = F.randn([hiddenSize, numKvHeads * headDim], weightOpts)
      let oWeight = F.randn([numQoHeads * headDim, hiddenSize], weightOpts)
      let qNormWeight = F.randn([headDim], weightOpts)
      let kNormWeight = F.randn([headDim], weightOpts)

      let qProj = Linear.init(qWeight)
      let kProj = Linear.init(kWeight)
      let vProj = Linear.init(vWeight)
      let oProj = Linear.init(oWeight)
      let qNorm = RmsNorm.init(qNormWeight)
      let kNorm = RmsNorm.init(kNormWeight)

      # RoPE
      let rotary = RotaryPositionEmbeddingRef.new(
        headDim, max_seq_len = 4096, rope_theta = 10000.0,
        dtype = F.kFloat32, device = F.kCPU)

      # InferenceContext
      var ctx = InferenceContext.init(
        num_layers = 1, batch_size = 1,
        kv_heads = numKvHeads, max_seq = 4096, head_dim = headDim)

      # PagePool + borrow pages
      let pool = PagePool.init(
        64, num_layers = 1,
        kv_heads = numKvHeads, head_dim = headDim,
        dtype = F.kFloat32, device = F.kCPU)
      for i in 0 ..< ceilDiv(4096, TokensPerPage):
        ctx.pages.add(pool.borrow())

      # Create attention layer
      var attn = RopeGQAttention.init(
        0, "test_layer",
        qProj, kProj, vProj, oProj,
        numQoHeads, numKvHeads, headDim, rotary,
        q_norm = some(qNorm), k_norm = some(kNorm))

      # Set up cos/sin
      ctx.position_ids = F.arange(0, 10).unsqueeze(0)
      ctx.cos = F.zeros([1, 10, headDim])
      ctx.sin = F.zeros([1, 10, headDim])

      # Input with batch=2
      let x = F.randn([2, 10, hiddenSize], weightOpts)

      try:
        discard attn.forward(ctx, x)
        echo "❌ Expected ValueError but no exception was raised"
        return false
      except ValueError:
        return true
      except CatchableError as e:
        echo "❌ Expected ValueError but got: ", e.msg
        return false

when isMainModule:
  main()
