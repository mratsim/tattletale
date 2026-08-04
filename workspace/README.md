# Tattletale Workspace

This directory contains the sub-projects that make up Tattletale. Each
`workspace/<name>/` directory has a `<name>.nim` entry file that re-exports its
`src/` modules.

See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for how these projects connect, from
a high-level inference goal down to emitted GPU code. Agent guidelines live in
[`AGENTS.md`](../AGENTS.md), [`crucible/AGENTS.md`](crucible/AGENTS.md), and
[`ceramic/AGENTS.md`](ceramic/AGENTS.md).

## Importing the packages

Each project exposes a `<name>.nim` shim at the workspace root that imports
the project's public entry module and re-exports its API:

```nim
# workspace/crucible.nim
import workspace/crucible/crucible
export crucible
```

Import the shim to pull in a project's core modules under one name, instead
of reaching into individual `src/` paths:

```nim
import workspace/ceramic        # Layout, Int[N], layout algebra, tensors, fill/copy/gemm kernels
import workspace/crucible       # GPU code generator (CUDA / OpenCL / Vulkan / WebGPU)
import workspace/transformers   # inference engine (models, generate, KV cache)
import workspace/libtorch       # tensor layer
import workspace/toktoktok      # BPE tokenizer
import workspace/safetensors    # model I/O
import workspace/data_structures # WAVL tree / longest-prefix-match
import workspace/positron       # portable kernels
```

The entry `src/<name>/<name>.nim` is the project's public API; the workspace
shim re-exports it verbatim. See `ARCHITECTURE.md` for how the projects
connect.

## Showcase

These are the differentiating projects, each with a dedicated README and
architecture document.

- **Crucible** — GPU kernel code generator. Nim macros consume ceramic layout
  algebra and positron kernel specs and emit native CUDA, OpenCL, Vulkan and
  WebGPU code.
  - [`crucible/README.md`](crucible/README.md),
    [`crucible/ARCHITECTURE.md`](crucible/ARCHITECTURE.md)
  - [`crucible/crucible.nim`](crucible/crucible.nim),
    [`crucible/src/codegen/gpu_compiler.nim`](crucible/src/codegen/gpu_compiler.nim)

- **Ceramic** — layout algebra library (an Nvidia CuTe Layout Algebra implementation and Nvidia Cutlass port). Provides
  `Layout[Shape, Stride]`, `Int[N]`, and layout transformations (`coalesce`,
  `complement`, `compose`, `logical_divide`), plus CPU/GPU fill, copy and GEMM
  kernels.
  - [`ceramic/README.md`](ceramic/README.md),
    [`ceramic/ARCHITECTURE.md`](ceramic/ARCHITECTURE.md)
  - [`ceramic/ceramic.nim`](ceramic/ceramic.nim),
    [`ceramic/src/layout_algebra.nim`](ceramic/src/layout_algebra.nim),
    [`ceramic/src/kernel_gemm_gpu.nim`](ceramic/src/kernel_gemm_gpu.nim)

## Core

These are the working layers of the inference engine. Most have no dedicated
README; the source files are the reference.

- **Transformers** — the inference engine. Stateless, pure `forward()` passes
  (see [`transformers/docs/DESIGN.md`](transformers/docs/DESIGN.md)) with
  stateful pieces (KV cache, page pool, orchestrator) isolated under
  [`transformers/src/stateful/`](transformers/src/stateful/).
  - [`transformers/transformers.nim`](transformers/transformers.nim),
    [`transformers/src/stateful/kvcache.nim`](transformers/src/stateful/kvcache.nim),
    [`transformers/src/stateful/kvcache.lean`](transformers/src/stateful/kvcache.lean)
  - [`transformers/README.md`](transformers/README.md),
    [`transformers/ARCHITECTURE.md`](transformers/ARCHITECTURE.md)

- **LibTorch** — the tensor layer wrapping the C++ libtorch runtime. The
  README notes this dependency is planned for removal in favor of a Nim-native
  tensor library.
  - [`libtorch/libtorch.nim`](libtorch/libtorch.nim),
    [`libtorch/src/tensors.nim`](libtorch/src/tensors.nim),
    [`libtorch/src/tensors_nn.nim`](libtorch/src/tensors_nn.nim)

- **Positron** — portable kernel specs plus a prebuilt CUDA static library
  (Hadamard FWHT-128 + RMSNorm) built from
  [`positron/make_libpositron_cuda.cu`](positron/make_libpositron_cuda.cu) and
  wired through [`positron/libpositron_cuda.nim`](positron/libpositron_cuda.nim).
  - [`positron/README.md`](positron/README.md),
    [`positron/positron.nim`](positron/positron.nim),
    [`positron/src/kernels/portable/`](positron/src/kernels/portable/)

- **Toktoktok** — BPE tokenizer. Text-to-token and token-to-text, with a
  tiktoken-compatible serialization and regex-based tokenizer rules.
  - [`toktoktok/toktoktok.nim`](toktoktok/toktoktok.nim),
    [`toktoktok/src/bpe_codec.nim`](toktoktok/src/bpe_codec.nim),
    [`toktoktok/src/serialization.nim`](toktoktok/src/serialization.nim)

- **Safetensors** — model I/O. Loads safetensors files with a bridge into the
  libtorch tensor layer.
  - [`safetensors/safetensors.nim`](safetensors/safetensors.nim),
    [`safetensors/src/safetensors.nim`](safetensors/src/safetensors.nim),
    [`safetensors/src/safetensors_libtorch.nim`](safetensors/src/safetensors_libtorch.nim)

- **Data structures** — an intrusive WAVL (Weak AVL) tree with
  longest-prefix-match (radix) support, backing the KV cache.
  - [`data_structures/data_structures.nim`](data_structures/data_structures.nim),
    [`data_structures/src/wavl_tree.nim`](data_structures/src/wavl_tree.nim),
    [`data_structures/src/wavl_tree.lean`](data_structures/src/wavl_tree.lean)

## Utility

Single-purpose helpers, kept minimal.

- **Bencher** — benchmarking and reporting helpers
  ([`bencher/bencher.nim`](bencher/bencher.nim),
  [`bencher/src/benchmarking.nim`](bencher/src/benchmarking.nim)).
- **Cpuplatforms** — x86 CPU detection and SIMD helpers
  ([`cpuplatforms/loadtime_functions.nim`](cpuplatforms/loadtime_functions.nim),
  [`cpuplatforms/x86/cpudetect_x86.nim`](cpuplatforms/x86/cpudetect_x86.nim)).
- **Pcre2** — vendored PCRE2 bindings
  ([`pcre2/pcre2.nim`](pcre2/pcre2.nim), vendored under
  [`pcre2/vendor/pcre2/`](pcre2/vendor/pcre2)).

## Formal verification

A differentiator: correctness-critical, stateful data structures are
formalized in Lean4, and the specs are symlinked from [`formalities/`](../formalities/)
(see [`formalities/README.md`](../formalities/README.md)).

- [`data_structures/src/wavl_tree.lean`](data_structures/src/wavl_tree.lean) —
  intrusive WAVL tree.
- [`transformers/src/stateful/kvcache.lean`](transformers/src/stateful/kvcache.lean) —
  PagedRadixTrie KV cache.
