# Transformers

The inference engine for Tattletale: model loading, tokenization, transformer
layers, KV cache, and sampling, all in Nim on a libtorch tensor backend.

## What it is

`workspace/transformers` is the highest-level project in the Tattletale
monorepo. It ties the lower-level libraries together into a runnable inference
pipeline: safetensors model I/O, toktoktok tokenization, libtorch tensor ops,
and positron kernels. Its public entry point is [`transformers.nim`](transformers.nim),
which imports and re-exports [`src/models.nim`](src/models.nim).

`src/models.nim` defines the two top-level operations:
- `loadModel(modelPath, device)` — reads `config.json`, dispatches on
  `architectures[0]` through a compile-time `ModelRegistry`, and returns a
  `Model` implementing the `forward`/`getConfig`/`getTokenizer`/`getDeviceKind`
  interface (`src/models/all_interfaces.nim`).
- `generate(model, prompt, temp, maxTokens, maxContextLen)` — tokenizes,
  runs a prefill forward over the full prompt, then a decode loop one token at
  a time, sampling with Gumbel-max (`src/samplers.nim`).

## Headline capabilities

### 1. EXL3 quantization

EXL3 is the highest-quality quantization scheme currently supported. It uses
random Hadamard rotations for input/output incoherence processing plus trellis
and lattice codebooks. Implementation lives in
[`src/quantizations/`](src/quantizations/) (`exl3.nim`, `exl3_codecs.nim`,
`datatypes.nim`, `unquantized_codecs.nim`). The 128-block Fast Walsh-Hadamard
Transform is implemented in positron at
`workspace/positron/src/kernels/portable/hadamard_transforms.nim`.

Test coverage is in [`tests/q_exl3/`](tests/q_exl3/) (see its
[`README.md`](tests/q_exl3/README.md)). Because of floating-point associativity
and warp-shuffle reductions, these fixtures cannot match on CPU — EXL3 tests
must run on a CUDA backend. Fixture-generation conventions are documented in
[`tests/testgen/FIXTURE_GENERATION.md`](tests/testgen/FIXTURE_GENERATION.md).

### 2. KV cache (PagedRadixTrie)

The KV cache in [`src/stateful/kvcache.nim`](src/stateful/kvcache.nim) is an
`IntrusiveAttention`: a PagedRadixTrie — a compressed Radix/Patricia trie over
token sequences operating at 256-token page granularity — where split/graft
operations happen at page boundaries. It deliberately avoids hashmaps (no
collision resolution, no early exit), avoids amortized/global heap eviction
(no tombstones, no refcounting churn), and indexes child pages with an
intrusive WAVL tree used as a longest-prefix-match (LPM) index rather than an
exact-key BST. This gives guaranteed worst-case latency with no rebuilding,
rehashing, or tombstones, and roughly O(log n) LPM plus memory bandwidth for
the prefix comparison.

Measured in the file's design notes for a 100K-child flat tree: LPM hit 53 ns,
LPM miss 32 ns (down from 690 µs before the WAVL index), graftPages on the last
child 203 ns, and a 100K-leaf tree build of 47 ms. The comparison-reported
costs are `~50 ns + O(memory bandwidth)` for prefix matching.

The design is partially formally verified in Lean 4: the specification is at
[`src/stateful/kvcache.lean`](src/stateful/kvcache.lean) with the WAVL-tree
formalization in `formalities/kvcache.lean` and `formalities/wavl_tree.lean`.

## Honest WIP status

Radix attention is **not** complete. The trie stores and matches KV pages, but
the attention layer does not yet consume the matched pages via a CSR
(compressed sparse row) style attention kernel. Today the layer copies page KV
into contiguous per-layer attention views — see the `copyFrom` path in
[`src/layers/attn.nim`](src/layers/attn.nim) (around lines 254–257). Implementing
CSR-style attention over the matched pages is pending; the radix trie itself
should not be presented as finished end-to-end attention.

## Source layout

```
src/
  models.nim            # loadModel + generate entry points
  models/               # Model iface, ModelRegistry, qwen3/qwen3_5/lfm2 implementations
  layers.nim            # Layer union type + device/dtype conversion
  layers/               # attn, embedding, linear, lmhead, mlp, norm, rope, short_conv, transformer
  quantizations/        # exl3, exl3_codecs, datatypes, unquantized_codecs
  samplers.nim          # Gumbel-max sampling
  deserialization.nim   # safetensors / weight loading
  stateful/             # kvcache, orchestrator, page_pool, inference_context
transformers.nim        # public entry: imports + exports src/models
```

## Build / run

Transformers is a Nim package (`transformers.nimble`, version 0.1.0, dual
MIT/Apache 2.0 license). Tasks are defined at the monorepo root in
`config.nims`; from the `tattletale` root:

```bash
nim install_deps
nim test_transformers        # run the transformers test suite
```

Single-file tests (per `AGENTS.md`):

```bash
nim cpp -r --verbosity:0 --hints:off --warnings:off \
  --outdir:build/tests/test_name --nimcache:nimcache/tests/test_name \
  workspace/transformers/tests/test_sampler.nim
```

EXL3 tests must run on CUDA; inject the CUDA runtime at link time via
`LD_PRELOAD` of `libtorch_cuda.so` (see `AGENTS.md`). Example:

```bash
# Compile one EXL3 test at a time (no -r), then run the binary under CUDA preload.
SITE_PKGS="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
TORCH_LIB="$SITE_PKGS/torch/lib"
CUDA_LIB="$(dirname "$(find "$SITE_PKGS/nvidia" -maxdepth 1 -type d -name 'cu*' | head -n1)")/lib"

nim cpp --hints:off --warnings:off \
  --outdir:build/wip --nimcache:nimcache/wip \
  workspace/transformers/tests/q_exl3/test_exl3_hadamard.nim

LD_PRELOAD="$TORCH_LIB/libtorch_cuda.so" \
LD_LIBRARY_PATH="$(pwd)/.venv/lib:$TORCH_LIB:$CUDA_LIB" \
build/wip/test_exl3_hadamard
```

## Related docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — position in the stack, data flow, extension points
- [`docs/DESIGN.md`](docs/DESIGN.md) — design principles (stateless forward)
- `AGENTS.md` — monorepo build/test/lint conventions
- Upstream libraries this project composes:
  - `workspace/libtorch` — tensor layer
  - `workspace/toktoktok` — tokenizer
  - `workspace/safetensors` — model I/O
  - `workspace/positron` — kernels (e.g. Hadamard transforms)
