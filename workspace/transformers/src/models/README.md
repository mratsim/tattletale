# Model families

Every model module under `src/models/` follows the same load convention,
three procs per family:

- `load*Config`, the typed config record parsed from `config.json`
- `load*ModelRaw`, weights plus tokenizer
- `load*Model`, the iface-erased wrapper registered in `ModelRegistry`
  under the checkpoint's `architectures[0]` value

Weight tensors never carry per-family loader wrappers. Every family builds
through the shared `.load` methods on the layer types:

- `Embedding.load`, `RmsNorm.load`, `Linear.load`
- `GatedDenseFFN.load`, `BlockSparseFFN.load`, `LMHead.load`
- the attention loaders in `src/deserialization.nim`

These serve zero-copy mmap views from the safetensors collection, or owned
copies per the device argument.

| Module | Registry key (`architectures[0]`) |
|---|---|
| `qwen3.nim` | `Qwen3ForCausalLM` |
| `qwen35.nim` | `Qwen3_5ForConditionalGeneration` |
| `qwen35_moe.nim` | `Qwen3_5MoeForConditionalGeneration` |
| `glm47_flash.nim` | `Glm4MoeLiteForCausalLM` |
| `moonlight.nim` | `DeepseekV3ForCausalLM` |
| `ling3.nim` | `BailingMoeV3ForCausalLM` |
| `kimi_linear.nim` | `KimiLinearForCausalLM` |
