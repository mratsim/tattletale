# Windowed attention: one spelling, per-layer-kind data

## One windowed spelling

One windowed attention spelling serves every sliding-window family, `RopeGQAttention[QKNorm]`.

Per-layer-kind data parameterizes it, the same typed implementation serves the whole family set.

| parameter       | field                 | source of the datum                          |
| --------------- | --------------------- | -------------------------------------------- |
| visibility band | `window: int`         | checkpoint `sliding_window`, per layer kind  |
| rope theta      | `rotary` table ref    | checkpoint rope theta, per layer kind        |
| attention scale | `softmaxScale`        | checkpoint `query_pre_attn_scalar`, or head  |
| qk norm         | `QKNorm` static param | checkpoint norm class (`RmsNormOne`, `void`) |

`FullVisibilityWindow` (`int.high`) removes the band, every key at or before the query stays visible.

Full-attention families instantiate the spelling at `FullVisibilityWindow`, the `window` parameter default.

Their compiled forward path is the plain causal path, the sentinel never binds a mask.

## Mask as data

The visibility band builds a mask tensor at forward time, never a special-cased kernel.

`windowedCausalMask` emits a `(1, 1, qLen, kvLen)` bool mask, `true` keeps a key visible, `false` masks it.

The mask follows the reference sliding-window causal rule `kv_idx <= q_idx` plus `kv_idx > q_idx - window`.

| forward shape         | dispatch                                           |
| --------------------- | -------------------------------------------------- |
| prefill, band unbound | `is_causal = true`, no mask tensor                 |
| decode, band unbound  | no mask, the whole cached history is visible       |
| any shape, band bound | `windowedCausalMask`, absolute-position arithmetic |

Absolute-position arithmetic keeps the band row correct at any cache offset.

A decode query at position `p` sees exactly the keys `p - window + 1 .. p`.

The production page-pool KV cache stores every token.

The band mask hides what a bounded sliding cache would have evicted, the softmax reads the same key set.

## Dual theta as data

Rope rows are request state inside `InferenceContext` (`ctx.cos`, `ctx.sin`), tables live per model.

A dual-theta model builds one `RotaryPositionEmbedding` per layer kind, gemma-3 carries 1e4 sliding and 1e6 full.

The model calls `ctx.setRopeForPositions` with the layer's table before each layer forward.

The attention layer stays theta-agnostic and consumes whatever rows the model wired into the context.

## Family coverage

| family                                | windowed datum               | fit as data                                                |
| ------------------------------------- | ---------------------------- | ---------------------------------------------------------- |
| Mistral                               | all-sliding 32x window 4096  | window 4096 on every layer, one theta                      |
| North                                 | f-first 4:1 pattern          | per-layer kind sequence, window and theta per kind         |
| Laguna                                | yarn + partial rotary 0.5    | partial width sits on the rotary table build               |
| gemma-4                               | dual theta + partial 0.25    | two theta tables, rotary width per kind                    |
| gemma-4 KV tying (`attention_k_eq_v`) | separate K and V cache paths | the pages write and gather apart                             |
| gemma-4-E2B PLE                       | model-level construct        | no attention axis                                          |
| gpt-oss attention sinks               | not a datum of this spelling | needs a new attention-call axis                            |

### Yarn (Laguna)

Yarn rescales the inverse frequencies at rotary table derivation.

The windowed spelling consumes `ctx.cos` and `ctx.sin` rows and never derives them.

Yarn stays a `RotaryPositionEmbedding` construction concern, orthogonal to the spelling.

The Laguna port verifies yarn with a rope-parity replay against the reference rows.

### KV tying (gemma-4)

The reference derives `V = v_norm(k_proj(h))` and `K = rope(k_norm(k_proj(h)))`.

The cache stores K and V separately, no tied-cache tensor exists in the reference.

The page cache already holds independent `k_view` and `v_view` pages.

The spelling writes and gathers K and V separately.

### Sinks (gpt-oss)

A per-head sink logit joins the softmax denominator without attending.

No window, theta or mask datum expresses it, support needs a new attention-call axis.

The gpt-oss port stays unbuilt, its fixture families stay recorded without a consuming suite.

A gpt-oss spelling forks from this one if the sink axis joins the attention call.

## Seating

`alkSlidingAttention` joins `AttentionLayerKind` with the `sliding_attention` config mapping.

A gemma-style block instantiates `SandwichDecoderLayer[RopeGQAttention[RmsNormOne], GatedDenseFFN, RmsNormOne]`.

It converts to `AnyDecoderLayer` like any KDA or MLA layer.

`SandwichDecoderLayer` carries the four-norm placement of the gemma lineage.

Both post-norms normalize their sublayer output alone, the stream joins after the norm.

It returns the same long-residual pair as `DecoderLayer`, contribution plus residual equals the block output.
