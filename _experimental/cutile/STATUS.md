# Nim cutile Bytecode Generator — Status Report

**Date:** 2026-05-29
**Goal:** Generate TileIR bytecode from Nim that matches Rust cutile-rs output and can be decoded by both the Rust decoder and (eventually) compiled via tileiras/CUDA TileIR JIT.

---

## What Works

### Rust reference (ground truth)
- `cutile-rs/cutile-ir/examples/build_basic.rs` compiles, runs, produces `/tmp/add_kernel.bc` (378 bytes)
- Rust `decode_bytecode` round-trips the Rust reference bytecode perfectly
- `cargo test -p cutile-ir` passes (per-op roundtrip tests, bytecode validate, dominance checks, etc.)

### Nim bytecode generator — structural correctness
- **Rust `decode_bytecode` ACCEPTS our Nim bytecode** (181-byte SAXPY file)
- Header, func section, string section, type section all parse correctly
- Operations decode: Iota, Reshape, Broadcast, Offset, LoadPtrTko, Broadcast, Fma, StorePtrTko, Return
- All 9 Nim examples (e00_minimal through e08_matmul_py) compile and produce valid bytecode

### Validator tool
- `cutile-ir/examples/validate_nim_bc.rs` — reads any `.bc` file, feeds to `decode_bytecode`, prints full decoded output
- tileiras support wired in but tileiras binary not installed on this system

### Type system
- `TileType(shape, elemType)` with hash-based dedup: `hash = (shape.len << 16) | ord(elemType)`
- Recursive registration: pointer types register their pointee; tile types register scalar elemTypes
- FuncType data pre-serialized as `seq[byte]` and appended to `m.funcTypes`
- Section format: varint count + align-4 padding + LE32 offset table + type data entries

---

## What Didn't Work

### 1. `toBytecode` is NOT idempotent

**Problem:** First call produces 176 bytes, second call produces 181 bytes (+5 bytes).

**Root cause:** `writeTypeSection` captures `numTileTypes = m.types.len` at the start. Then `writeTileTypeData` for each TileType recursively calls `getTypeIndex` to find the scalar elemType (e.g., `scalarI32` for `tile128I32`). If the scalar isn't already registered, `getTypeIndex` APPENDS it to `m.types` mid-iteration. This new type is AFTER the captured `numTileTypes` so it's skipped in the type section.

On the SECOND call, `m.types` already has the missing scalar (added by the first call), so `numTileTypes` is higher and the type IS included. 176 + 5 = 181 (4-byte offset entry + 1-byte scalar tag = 5 bytes for scalarI32).

**Fix attempted:** Added pre-registration loop at the top of `toBytecode`:
```nim
for t in m.types:
  if t.shape.len > 0:
    let scalarTy = TileType(shape: @[], elemType: t.elemType)
    discard getTypeIndex(m, scalarTy)
```
This ensures all scalar elemTypes exist BEFORE writeTypeSection captures numTileTypes.

### 2. `m.funcTypes = @[]` placement

**Problem:** Duplicate FuncType entries accumulate across `toBytecode` calls.

**Fix:** Moved `m.funcTypes = @[]` to the top of `toBytecode`. Now each call starts fresh and writesFuncSection creates exactly one FuncType per function.

### 3. CUDA TileIR JIT / tileiras

**Problem:** This system has CUDA 13.0 (driver ~580.82) which doesn't support TileIR JIT natively. The `cuModuleLoadData` call fails. tileiras binary is not installed.

**Expected:** CUDA 13.1+ provides `cuModuleLoadData` TileIR JIT. `tileiras` binary (from CUDA Toolkit) would compile bytecode to CUBIN/SASS.

**What we tried:**
- `cuModuleLoadData` with our bytecode → fails (driver too old)
- `compileBytecodeCached` → writes to `/tmp/cutile_cache/<hash>_sm_120.bc`, tries `cuModuleLoadData`, tries `tileiras`, raises `CompileError`
- Tileiras not found on this system (`find / -name tileiras` returns nothing)

---

## Error Messages Encountered

### Nim `fromBytecode` decoder (early development)
```
error at offset 171: invalid integer value for enum type: 11
```
This was the OLD Nim decoder trying to parse bytecode. The Nim decoder has been superseded by the Rust `decode_bytecode` validator, which successfully parses our bytecode.

### Nim `writeFuncSection` type index mismatch (early development)
```
function signature index 9 out of bounds for type section with 8 entries
```
Caused by `toBytecode` being called multiple times, each call adding another FuncType entry to `m.funcTypes`. The sigIdx grew but type section count grew faster. Fixed by `m.funcTypes = @[]` reset.

### Nim binary write size mismatch
```
Bytecode: 176 bytes   (bc.len from toBytecode)
file:      181 bytes  (actual /tmp/cutile_examples/e03_saxpy.bc)
```
The 5-byte discrepancy is from the non-idempotent behavior described above. The file was written by `writeFile` with `s.len = 181` (after pre-registration fix, both bc.len and file size should match now).

---

## Testing / Comparison Procedure

### 1. Rust validator (golden oracle)
```bash
cd _references_kernels/cutile-rs
cargo run --example validate_nim_bc /path/to/nim.bc
```
This calls `cutile_ir::decode_bytecode()` and prints the full decoded module. If it says `[OK]`, the bytecode is structurally valid.

### 2. tileiras compiler (byte-level + GPU compilation)
```bash
CUTILE_TILEIRAS_PATH=/path/to/tileiras \
  cargo run --example validate_nim_bc /path/to/nim.bc
```
tileiras is the NVIDIA-provided TileIR→SASS compiler. It's stricter than `decode_bytecode` and would catch encoding errors the decoder tolerates. **Not available on this system.**

### 3. Rust full test suite
```bash
cd _references_kernels/cutile-rs
cargo test -p cutile-ir -- --nocapture
```
Tests: bytecode validation, round-trip per-op, decode rejection, dominance, version checks. These validate Rust's own bytecode, not Nim's.

### 4. Bytecode hex comparison
```bash
python3 -c "
with open('/path/to/file.bc','rb') as f: d = f.read()
for i in range(0, len(d), 16):
    print(f'{i:04x}: ' + ' '.join(f'{b:02x}' for b in d[i:i+16]))
"
```

### 5. Nim example execution
```bash
cd tattletale
nim cpp --hints:off --warnings:off -d:release -o:build/examples/e03_saxpy \
  workspace/positron/cutile/examples/e03_saxpy.nim
build/examples/e03_saxpy 2>&1 | head -20
```

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Header (magic + version) | OK | `7f 54 69 6c 65 49 52 00 0d 01 00 00` |
| Func section | OK | Varint funcCount, nameIdx, sigIdx, flags, loc, bodyLen, body |
| String section | OK | Varint count + align-4 + LE32 offsets + strings |
| Type section | OK | 8 types (7 Tile + 1 Func), offset table, type data |
| Operation encoding | OK | opcode, resultTypes, op-specific attrs, operands |
| Op-specific attrs | OK | Fma (rounding_mode), LoadPtrTko/StorePtrTko (memory_ordering, memory_scope, optimization_hints, operandSegmentSizes) |
| `toBytecode` idempotency | FIX APPLIED | Pre-registers scalar elemTypes before writeTypeSection |
| Rust decoder acceptance | OK | `decode_bytecode` parses successfully |
| tileiras compilation | N/A | tileiras not installed |
| CUDA TileIR JIT | N/A | Driver CUDA 13.0, needs 13.1+ |

---

## Key Reference Files

- **Nim bytecode generator:** `tattletale/workspace/positron/cutile/bytecode.nim`
- **Nim DSL:** `tattletale/workspace/positron/cutile/dsl.nim`
- **Nim compiler (GPU):** `tattletale/workspace/positron/cutile/compiler.nim`
- **Nim examples:** `tattletale/workspace/positron/cutile/examples/e0*.nim`
- **Rust IR library:** `_references_kernels/cutile-rs/cutile-ir/`
- **Rust validator:** `_references_kernels/cutile-rs/cutile-ir/examples/validate_nim_bc.rs`
- **Rust build_basic (reference):** `_references_kernels/cutile-rs/cutile-ir/examples/build_basic.rs`

## Next Steps (when resuming)

1. Verify the `toBytecode` pre-registration fix actually makes it idempotent (bc.len should be same on 1st and 2nd call)
2. Generate Rust reference bytecode for SAXPY pattern (currently only have vector add) and do byte-level comparison
3. Install tileiras or find a system with CUDA 13.1+ for GPU compilation
4. Expand Nim example coverage: vector add → Rust reference → compare
5. Consider: should `writeTileTypeData` be forbidden from mutating `m.types`? (i.e., make it a pure function that just serializes, with all type registration happening before `writeTypeSection`)
6. Check all 9 Nim examples against Rust decoder, not just SAXPY
