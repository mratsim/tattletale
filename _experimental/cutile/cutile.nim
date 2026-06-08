# cutile
# Nim port of cutile-rs (NVIDIA cuTile Rust library)
#
# Tile-based GPU kernel compilation and execution for NVIDIA GPUs.
# Compiles Nim-expressed tile kernels to CUDA Tile IR bytecode,
# then to cubin via tileiras, and launches via CUDA Driver API.
#
# Architecture (layered, each layer independently testable):
#
#   Layer 0: cuda_driver.nim  — CUDA Driver API wrappers (cuInit, cuModuleLoad, etc.)
#   Layer 1: bytecode.nim     — TileIR bytecode writer (Module, Ops, serialization)
#   Layer 2: compiler.nim     — tileiras invocation (.bc → .cubin)
#   Layer 3: runtime.nim      — Load + launch (.cubin → CUmodule → cuLaunchKernel)
#   Layer 4: dsl.nim          — Nim DSL (tileKernel macro, tile ops)
#
# Pipeline:
#   Nim DSL → TileIR Bytecode (.bc) → tileiras → Cubin (.cubin)
#     → cuModuleLoad → cuModuleGetFunction → cuLaunchKernel → GPU
