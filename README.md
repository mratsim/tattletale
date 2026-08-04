# Tattletale

A high-performance inference engine project.

See motivation and MVP goals at https://github.com/mratsim/tattletale/issues/1

TL;DR of goals (README-driven development):
- High-performance: concurrent queries, 1M+ context per queries, highly-tuned kernels, fused kernels, SOTA CPU threadpool and kernels
- Embeddable: Single-binary, callable from C, C++, Rust, Python, ...
- Multi-hardware: Currently Cuda, OpenCL, Vulkan, WebGPU. Future HIP and Metal and why not DX12
- Multi-modality: Audio and Image input AND generation
- Maintainable and easy to extend
- Cryptography-inspired engineering practices: Lean4 formalization of complex state management
- First class support of EXL3 quantization, the SOTA quantization in 2026 with random Welsh-Hadamard transforms, trellis and lattice codebooks.

Killer apps:
- Being an engine embeddable in a game that can handle parallel NPC dialogs and actions
- Text + Audio + Video assistant on desktop and phone with concurrent agents handling

## Highlights

_At the moment the project is still in its infancy, we present key differentiators that will hopefully snowball into an unique product in the landscape._

- Embeddable, single dependency on drivers + libTorch C++.
  LibTorch dependency will be removed in the future.

- Bidirectional Python, C, C++ integration:
  - Nim can call Python and Python can call Nim\
    https://github.com/mratsim/tattletale/blob/9975f37/workspace/libtorch/tests/python_integration/test_tensor_bridge.nim#L25-L85
  - Nim can call C/C++ and C/C++ can call Nim (by virtue of compiling to C/C++ as an intermediate language.

- Nim -> Cuda, OpenCL, Vulkan, WebGPU compiler implemented in Nim macros.\
  Build time or runtime portable code generation on any accelerator:\
  https://github.com/mratsim/tattletale/tree/dbb44dd/workspace/positron/src/codegen

- IntrusiveAttention, a PagedRadixTrie implemented on top of intrusive WAVL-tree for guaranteed worst-case latency.\
  No rebuilding, rehashing or tombstones like with hashmaps\
  ~50ns+O(memory bandwidth) for prefix matching whatever the fan-out or the depth\
  allowing handling 100K+ cached requests with a single machine (for example for a router)\
  Partial formal verification in Lean4.\
  - https://github.com/mratsim/tattletale/blob/dbb44dd/workspace/transformers/src/stateful/kvcache.nim
  - https://github.com/mratsim/tattletale/blob/dbb44dd/workspace/transformers/src/stateful/kvcache.lean

- EXL3 quant support, currently the highest quality quantization scheme
  using random Hadamard rotations, trellis and lattice codebooks.

## Future highlights

- [WIP] Porting CuteDSL/Cutlass/TileLang to Nim and enabling them across hardware vendors
- No more libTorch dependencies, let's write my yet-another-tensor library in Nim
  - Previous large: https://github.com/mratsim/Arraymancer
  - Previous mini: https://github.com/mratsim/nim-julia-challenge/blob/master/src/tensor.nim
  - Previous compiler-based: https://github.com/mratsim/laser/blob/master/laser/lux_compiler/lux_dsl.nim

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layer stack, data flow, repo layout, extension points.
- [`workspace/README.md`](workspace/README.md) — per-project catalog of the sub-projects.
