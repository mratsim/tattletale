# libtorch

Nim tensor API over libtorch (C++). Wraps low-level `TorchTensor` and ancillary types (e.g. `IntArrayRef`) into a high-level `Tensor` and Nim API.

## What it provides

Entry point: [`libtorch.nim`](libtorch.nim) imports and re-exports [`src/tensors.nim`](src/tensors.nim) and [`src/tensors_nn.nim`](src/tensors_nn.nim).

- **Tensor API** — `src/tensors.nim` (core tensor type, ops) and `src/tensors_nn.nim` (neural-network layer helpers).
- **Python bridge** — `src/tensors_py.nim`; bidirectional Nim<->Python integration, e.g. `tests/python_integration/test_tensor_bridge.nim`.
- **Raw bindings** — `src/raw/` and `src/raw_libtorch.nim`. See [`src/raw/abi/torch_tensors.md`](src/raw/abi/torch_tensors.md) (raw `torch::` bindings guide: naming conventions, `std::optional`, `TensorOptions`, typedesc/static handling) and [`src/README.md`](src/README.md) (design notes on the double refcount / indirection).
- **Vendor tooling** — `vendor/` for fetching/installing libtorch (see `libtorch_installer.nim`).
- Three distinct notions of equality are documented in `libtorch.nim`: referential, value (`equal`), and elementwise (`eq`).

## Dependencies and direction

This is the single dependency on libTorch C++ in the project; the root README states it will be removed in the future (`../../README.md`). `libtorch.nimble` declares `backend = "cpp"` and deps `zip` and `nimpy`.

## Tests

- `tests/tensors/`, `tests/raw_torch_tensors/`, and `tests/python_integration/`.

## Related

- Consumed by the model I/O bridge: [`../safetensors/README.md`](../safetensors/README.md)
- Root project: [`../../README.md`](../../README.md)
