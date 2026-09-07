# safetensors

Nim reader for the [safetensors](https://huggingface.co/docs/safetensors) model format, with a bridge to libtorch tensors.

## What it provides

Entry point: [`safetensors.nim`](safetensors.nim) re-exports [`src/safetensors.nim`](src/safetensors.nim) and [`src/safetensors_libtorch.nim`](src/safetensors_libtorch.nim).

- **Format reader** (`src/safetensors.nim`)
  - `load*(path: string): Safetensor` — memory-maps the file, parses the header (JSON via `jsony`) and validates offsets (contiguity, no overlap, no reads past EOF) with `validate_offsets`. Raises `ValueError` naming the path for a missing or malformed file.
  - `close*(st: var Safetensor)` — releases the mapping `load` opened.
  - `Safetensor` / `TensorInfo` types, `ST_dtype` enum covering BOOL through U64 including MX formats (`F4`, `F6_E2M3`, `F6_E3M2`, `F8_E5M2`, `F8_E4M3`, `F8_E8M0`).
  - `getMmapView` — zero-copy `MemSlice` into the memory-mapped file, enabling direct-to-GPU loading without materializing tensors in RAM.
  - `MAX_HEADER_SIZE` guard (100 MB) against oversized-header attack vectors.
  - Error model: exceptions (see header notes in `src/safetensors.nim`).
- **libtorch bridge** (`src/safetensors_libtorch.nim`)
  - `toTorchType` — maps `ST_dtype` to libtorch `ScalarKind`.
  - `getTensorView` — zero-copy `Tensor` over the mmap (`from_blob`).
  - `getTensorOwned` — owned copy onto a target device, safe after the reader goes out of scope.

## Lifetime / memory safety

`open(path)` acquires the memory mapping and the returned `Safetensor` owns it for its whole lifetime; the value's `=destroy` hook releases the mapping when the value goes out of scope. `MemSlice` and `getTensorView` views are valid while the value is in scope, dangling after its destructor ran; `getTensorOwned` copies remain valid afterwards. The borrow by views is a documented contract, not a compiler-enforced one (MemFile predates `lent` and view openarrays).

## Tests

- `tests/` — view/owned loading and `from_blob` aliasing tests against `tests/fixtures` and `tests/testgen`.

## Status

Loading and libtorch bridging are implemented. Writing/saving is not yet implemented (the file is read-only); the individual-tensor API is labeled WIP in the source.

## Related

- Uses the tensor layer: [`../libtorch/README.md`](../libtorch/README.md)
- Root project: [`../../README.md`](../../README.md)
- Package manifest: `safetensors.nimble` (license: MIT or Apache 2.0, deps: `jsony`, `stew`, `nim >= 2.2.0`)
