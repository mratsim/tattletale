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

- Promising performance with very high-level approach (50LOC for matmul) based on tile algebra implemented on top of CuTe Layout Algebra:
  
  | M = N = K | Tattletale/Ceramic 32×32×16 fp16 | Apple MLX steel (Tensor core API) 64×64×16 | Apple MLX NAX (Neural Accelerator API) 64×64×256 |
  |----------:|-----------------------:|-------------------:|------------------:|
  |       256 |                 1.21  |              1.09  |             1.00  |
  |       512 |                 5.32  |              4.36  |             5.17  |
  |      1024 |                11.09  |             11.15  |            11.67  |
  |      2048 |                12.21  |             13.00  |            12.48  |
  |      4096 |                12.85  |             13.29  |            12.83  |
  
  *Apple M4 Max, Metal, fp16 inputs / fp32 accumulate, TFlops = 2·M·N·K / time.*
  
  [[Crucible/Ceramic] Fix lowering of struct containing pointers - Fix GEMM bench: 99% 🔥 vs Apple MLX and Apple Neural Accelerator (NAX) - #140](https://github.com/mratsim/tattletale/pull/140)

- Embeddable, single dependency on drivers + libTorch C++.
  LibTorch dependency will be removed in the future.

- Bidirectional Python, C, C++ integration:
  - Nim can call Python and Python can call Nim\
    https://github.com/mratsim/tattletale/blob/9975f37/workspace/libtorch/tests/python_integration/test_tensor_bridge.nim#L25-L85
  - Nim can call C/C++ and C/C++ can call Nim (by virtue of compiling to C/C++ as an intermediate language.

- Nim -> Cuda, OpenCL, Metal, Vulkan, WebGPU compiler implemented in Nim macros.

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

- No more libTorch dependencies, let's write my yet-another-tensor library in Nim
  - [[Design] Tensorcery: Migrating away from libtorch](https://github.com/mratsim/tattletale/issues/111)
  - [[Design] Ceramic tile algebra & Tensorcery tensor library #127
](https://github.com/mratsim/tattletale/issues/127)

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layer stack, data flow, repo layout, extension points.
- [`workspace/README.md`](workspace/README.md) — per-project catalog of the sub-projects.
