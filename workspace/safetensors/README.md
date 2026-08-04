# safetensors

Nim reader for the [safetensors](https://huggingface.co/docs/safetensors) model format, with a bridge to libtorch tensors.

## What it provides

Entry point: [`safetensors.nim`](safetensors.nim) re-exports [`src/safetensors.nim`](src/safetensors.nim) and [`src/safetensors_libtorch.nim`](src/safetensors_libtorch.nim).

- **Format reader** (`src/safetensors.nim`)
  - `load*(memFile: MemFile): Safetensor` — parses the header (JSON via `jsony`) and validates offsets (contiguity, no overlap, no reads past EOF) with `validate_offsets`.
  - `Safetensor` / `TensorInfo` types, `ST_dtype` enum covering BOOL through U64 including MX formats (`F4`, `F6_E2M3`, `F6_E3M2`, `F8_E5M2`, `F8_E4M3`, `F8_E8M0`).
  - `getMmapView` — zero-copy `MemSlice` into the memory-mapped file, enabling direct-to-GPU loading without materializing tensors in RAM.
  - `MAX_HEADER_SIZE` guard (100 MB) against oversized-header attack vectors.
  - Error model: exceptions (see header notes in `src/safetensors.nim`).
- **libtorch bridge** (`src/safetensors_libtorch.nim`)
  - `toTorchType` — maps `ST_dtype` to libtorch `ScalarKind`.
  - `getTensorView` — zero-copy `Tensor` over the mmap (`from_blob`).
  - `getTensorOwned` — owned copy onto a target device (safe after the `MemFile` is closed).

## Lifetime / memory safety

All views are derived from the caller-provided `MemFile`; they MUST NOT outlive the underlying memory mapping. This is documented in the source but not yet compiler-enforced (see the nimony borrow-check references in `src/safetensors.nim`).

## Tests

- `tests/` — view/owned loading and `from_blob` aliasing tests against `tests/fixtures` and `tests/testgen`.

## Status

Loading and libtorch bridging are implemented. Writing/saving is not yet implemented (the file is read-only); the individual-tensor API is labeled WIP in the source.

## Related

- Uses the tensor layer: [`../libtorch/README.md`](../libtorch/README.md)
- Root project: [`../../README.md`](../../README.md)
- Package manifest: `safetensors.nimble` (license: MIT or Apache 2.0, deps: `jsony`, `stew`, `nim >= 2.2.0`)
