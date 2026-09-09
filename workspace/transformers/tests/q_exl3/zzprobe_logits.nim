import
  std/os, std/options, std/importutils, std/strformat,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.}

{.experimental: "callOperator".}
privateAccess(Qwen3Model)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])

const
  Dir03 = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-03-full-forward-to-logits" / "Qwen3-0.6B-EXL3-5bpw"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"

proc main() =
  let model = loadQwen3ModelRaw(ModelPath, kCPU)
  var ctx = InferenceContext.init(
    num_layers = model.config.num_hidden_layers,
    batch_size = 1, kv_heads = model.config.num_key_value_heads,
    max_seq = 4096, head_dim = model.config.head_dim)
  let pool = PagePool.init(64, num_layers = model.config.num_hidden_layers,
    kv_heads = model.config.num_key_value_heads,
    head_dim = model.config.head_dim,
    dtype = F.kFloat16, device = F.kCPU)
  for i in 0 ..< ceilDiv(4096, TokensPerPage):
    ctx.pages.add(pool.borrow())
  let inputIds = @[9707.int64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0)
  var hidden = model.embedTokens(inputIds)
  var residual: Option[Tensor] = none(Tensor)
  for layerIdx in 0 ..< model.layers.len:
    var st = Safetensor.open(Dir03 / &"layer-{layerIdx:02d}.safetensor")
    let fixtureOutput = st.getTensorOwned("layer_output", kCPU)
    let layer = model.layers[layerIdx]
    ctx.kv_position = 0
    ctx.position_ids = arange(hidden.size(1)).unsqueeze(0).to(kInt64)
    ctx.setRopeForPositions(model.rotary)
    let (output, newResidual) = layer(ctx, hidden, residual)
    let nimSum = output + newResidual
    let x = nimSum.contiguous().to(kCPU).to(kFloat32)
    let y = fixtureOutput.contiguous().to(kCPU).to(kFloat32)
    let rx = cast[ptr UncheckedArray[float32]](x.data_ptr(float32))
    let ry = cast[ptr UncheckedArray[float32]](y.data_ptr(float32))
    var maxU = 0'i64
    var over = 0
    let n = x.numel()
    for i in 0 ..< n:
      let ux = cast[uint32](rx[i]).uint64
      let uy = cast[uint32](ry[i]).uint64
      if ux == uy: continue
      let ox = if (ux and 0x80000000'u64) != 0'u64: (not ux) and 0xFFFFFFFF'u64 else: ux or 0x80000000'u64
      let oy = if (uy and 0x80000000'u64) != 0'u64: (not uy) and 0xFFFFFFFF'u64 else: uy or 0x80000000'u64
      var d = ox.int64 - oy.int64
      if d < 0: d = -d
      d = d div 8192
      if d > maxU: maxU = d
      if d > 4: inc over
    echo "layer ", layerIdx, ": maxU ", maxU, " over4 ", over, "/", n
    hidden = output
    residual = some(newResidual)
  let finalNorm = model.norm(hidden + residual.get(hidden))
  let finalLogits = model.lmHead(finalNorm)
  let flat = finalLogits.contiguous().to(kCPU).to(kFloat16).contiguous()
  let viewed = flat.view(kInt16).contiguous()
  var f = open("/tmp/nim_logits_f16.bin", fmWrite)
  var buf = newSeq[int16](viewed.numel())
  copyMem(buf[0].addr, viewed.data_ptr(int16), viewed.numel() * 2)
  discard f.writeBuffer(buf[0].addr, viewed.numel() * 2)
  f.close()
  echo "logits written ", flat.shape

main()
