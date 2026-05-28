## Anti-regression test for CODERA-020:
## Paged KV attention must reject batch_size > 1.

import
  std/unittest,
  std/options,
  std/importutils,
  workspace/libtorch as F,
  ../../src/stateful/kvcache {.all.},
  ../../src/stateful/page_pool,
  ../../src/stateful/inference_context,
  ../../src/layers/attn {.all.},
  ../../src/layers/rope {.all.},
  ../../src/layers/linear {.all.},
  ../../src/layers/norm {.all.}

privateAccess(RopeGQAttention)
privateAccess(GroupedQueryAttention)

proc testBatchGuard*(): bool =
  ## CODERABBIT-020: forward with batch != 1 must raise ValueError.

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
    qNorm, kNorm,
    numQoHeads, numKvHeads, headDim, rotary)

  # Set up cos/sin
  ctx.position_ids = F.arange(0, 10).unsqueeze(0)
  ctx.cos = F.zeros([1, 10, headDim])
  ctx.sin = F.zeros([1, 10, headDim])

  # Input with batch=2
  let x = F.randn([2, 10, hiddenSize], weightOpts)

  expect(ValueError):
    discard attn.forward(ctx, x)

  result = true

proc runTests*() =
  suite "CODERA-020 (batch guard)":
    test "forward with batch=2 raises ValueError":
      check testBatchGuard()

when isMainModule:
  runTests()
