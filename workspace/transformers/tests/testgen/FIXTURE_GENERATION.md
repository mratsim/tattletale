# Fixture Generation Conventions

This document records the conventions and invariants that fixture generators
must follow to produce deterministic, production-faithful
test data for the Nim inference pipelines.

## Fixture payload tiering rules

Quantization fixture families mirror the bf16 sidecar and summary-surface
comparison contract, except the hash-checked 00 families:

- the codec payload contract verifies bit for bit
- the verification takes the raw inputs plus the recorded kernel hash

The exl3 families carry the bf16 bounds with the ulp unit taken in fp16,
because EXL3 dequantizes to fp16. A value landing exactly on the bound passes,
every bound comparison inclusive.

A fixture family ships sidecar plus slices, never raw output
tensors except the value-bearing slices:

- **Decision payloads** (greedy chains, final logits) are recorded as compact
  JSON decisions:
  - greedy chains carry the chosen token, top-32 ids and f32 logits, argmax
    margin and the explicit `tail_probability` key per step (`ttt-tf-001-greedy-steps-h2`)
  - final logits carry the argmax decision frame
    (`ttt-tf-005-argmax-decisions`), one record per position holding
    the argmax id, the top-32 set with hex f32 logits, the margin,
    the tail probability
  - the frame adds the per-frame `ulp_datatype` key, no serialized instrument constant
  - the [1,6,248320] raw-logits class shrinks from megabytes to tens of KB,
    consumers read argmax and the top-32 set only
- **Boundary slices** that feed the instrument-compared boundaries stay
  raw (layer inputs/outputs, GDN conv/state trajectories), the value-bearing
  slices of the family.
- **Fingerprint sidecars**, `<fixture>.stats.json.zst` frames carrying order
  statistics plus histograms so removed raw tensors keep a distribution
  check. The payload byte format comes from harness/gen_stats.nim
- **GDN family frozen truth**, the GDN conv/state fixtures regenerate
  byte-identically against the frozen recordings. A regeneration that differs
  must stop the change until the cause is found (a stale assumption, a code defect, or the wrong model),
  run the force-fallback verification first, then
  take the finding to review before accepting a new canonical truth
  for the family. Never silently re-record GDN.
- **Re-record records** are kept outside the repo, holding the fixture file
  checksums before and after plus the re-run byte-comparison results.
  The record contract is the "verified re-record record" definition in ../README.md.
  Old raw payloads leave the tree only after the family successor passes.

Any regeneration that changes the recorded fixture files names it
in the commit body (values unchanged, container new by design), a silent
byte change is a defect.

### Recording environment

Fixture generation runs in the uv-synced environment:

    uv sync --group test-vectors

The group locks torch==2.14.0 and transformers==5.16.1
(the recorded versions). Re-check `torch.__version__` against the version row
of the fixture metadata before generating.

The installed transformers (5.16.1) is the source of truth:

- no vendored checkout is consulted
- `_references_*` never appears in comments or fixture generation
- hub kernel packages are absent, so kernel dispatch falls back
  to the pure-torch bodies, which is the reference behavior

### Recording box

`recorded_from` names the recording box plus device ("m4max-cpu", "rtxpro6000-cuda").

The default recording box stays m4max-cpu, a recording on another box sets
`TTT_RECORD_FROM` in the environment.

The value lands in the env frame of the greedy
fixtures:

- the env frame additionally carries an explicit `device` row

Recording runs torch-side on the recording box. Replay picks the device
through `select_device`, GPU over CPU, Metal on m4max, CUDA on rtxpro6000.
cpu replay must not be automatic, a missing device kernel fails loudly
and stays a finding.

- PYTORCH_ENABLE_MPS_FALLBACK is banned everywhere, it re-enables
  the automatic CPU fallback PR #104 removed, the device policy
  linter (`tests/linters/lint_device_policy.py`) counts it
- a `= kCPU` default device parameter is banned in transformers src,
  callers pass the device explicitly

### Exl3 families

- Tier 00 (codec, hadamard), packed trellis inputs plus metadata frames,
  the production-CUDA-kernel weight hash recorded inside the frame,
  bit-exactness the contract, no stats sidecars.
- Tier 01 layer internals, fingerprint stats sidecars per payload,
  quantile-only entries on the linear outputs, histogram entries
  covering attention and block outputs, the margin-critical class.
- Tier 01 block-02 trace, quantile-only stats entries per stage tensor,
  chained stages accumulate drift linearly under the stage-indexed bound,
  the bit-exact input-layernorm reference resetting the count.

The suites compare the per-op ulp rows on the recording device,
the chain checkpoint band elsewhere, elementwise plus stats.

- Tier 03 full forward, per-layer stats entries beside the raw
  boundary payloads, the final logits argmax decision frame
  (`ttt-tf-005-argmax-decisions`) and the frozen quantile entry.
- Tier 04 greedy, `ttt-tf-001-greedy-steps-h2` step records,
  the suites replay with teacher-forced tie recovery.

- Record-time sidecars come from the generators. The raw logits payload is
  the one frozen source. With it retired, the committed decision
  plus stats frames stay frozen data.
- Tier-1 product-property tests live in tests/layer_invariance/:
  they are fixture-free by construction, no fixture material involved.

## Guiding principle

**The production EXL3 kernel (`exllamav3_ext`) is the ground truth.**

Nim reimplementations (e.g. `hadamard_rotate_128`, `linear.forward`) are tested
AGAINST fixtures generated by the C++ kernel. If a reimplementation disagrees
with the kernel, the reimplementation must be fixed, never the fixture or the test tolerance.

---

## 1.  Weight layout

### Convention

| Function / Kernel | Weight shape | Format |
|---|---|---|
| `ext.reconstruct` (C++) | `[in_features, out_features]` | row-major, **non-transposed** |
| `ext.hgemm` (C++) | `[in_features, out_features]` | expects row-major GEMM |
| `torch.nn.functional.linear` | `[out_features, in_features]` | transposed for `input @ Wᵀ` |
| `linear_forward_orig_exl3` (Python) | `[in_features, out_features]` | passes straight to `ext.hgemm` |
| `Linear.forward` (Nim, qEXL3) | `[out_features, in_features]` | transposed for `F.linear` |
| `Linear.load` (Nim, deserialization) | Depends on `quant_format` | qBF16 → transposed; qExl3 → see below |

### Rule

Fixtures must use the **kernel-native** layout (`[in_features, out_features]`, non-transposed).

The fixture generator calls `linear_forward_orig_exl3`, which takes non-transposed
weight and feeds it directly to `ext.hgemm`.

The Nim test loads weights through `Linear.load`, stored in the layout `Linear.forward` expects (transposed for `F.linear`).

If the Nim `Linear.forward` is ever changed to call `ext.hgemm` directly,
the load function must be updated to store non-transposed weights.

**Do NOT transpose in the fixture generator.** The generator calls
the production kernel directly, and no `F.linear` wrapper would require
a transposed weight.

---

## 2.  Cos / sin shape

### Convention

| Context | Shape | Notes |
|---|---|---|
| `precompute_freqs_cis_reimpl_exl3` output | `[max_seq_len, head_dim]` | 2D, fp16 |
| Fixture `cos` / `sin` keys | `[batch, seq, head_dim]` | 3D, sliced and expanded from the 2D table |
| Nim `ctx.cos` after `setRopeForPositions` | `[seq, head_dim]` | 2D, sliced from `rotary.cos_cache` |
| Nim `applyRope` input | 2D `(seq, head_dim)` | unsqueezed internally to `(1, 1, seq, head_dim)` |

### Rule

The fixture saves a **3D** cos/sin tensor `[batch, seq, head_dim]`, sliced
from the 2D table at the same `position_ids` for all batch items. The Nim
test normalises the fixture entry back to 2D via:

```nim
let hfCos2d = if hfCos.dim == 3: hfCos[b] else: hfCos
```

Both sides produce the same values (verified by `assertAllClose`, `rtol=1e-5`, before the full-attention comparison).

---

## 3.  Precision of the Hadamard transform

### Convention

The production kernel `ext.had_r_128(a, b, pre_scale, post_scale, norm)`
performs all arithmetic in **fp32**:

```
output = FWHT(input ⊙ pre_scale) ⊙ post_scale × norm / √128
```

The Nim reimplementation `hadamard_rotate_128` must also do its FWHT
butterflies and scale/norm multiplications in fp32.

Converting intermediate values to fp16 between stages costs precision,
producing results that differ from the kernel by ~0.0003 to 0.0005 absolute,
beyond the recorded `Tol = 1e-4` band.

### Rule

Every stage of `hadamard_rotate_128` (scale, FWHT butterfly, norm multiply)
MUST happen in fp32. The final result is converted to fp16 only at the very
end:

```nim
var blk_f32 = blk.to(kFloat32)
# (pre-scale, FWHT, post-scale, norm) — all fp32
F.copyFrom(blk, blk_f32)  # convert to original dtype at end
```

This matches `fwht_128`, which also operates in-place on a fp32 tensor.

---

## 4.  Scale order in the Hadamard transform

### Convention

The CUDA kernel `ext.had_r_128` takes two scale arguments around the FWHT:
- **`pre_scale` (3rd arg)**, applied element-wise **before** the FWHT,
  used for `suh` (input incoherence scale).
- **`post_scale` (4th arg)**, applied element-wise **after** the FWHT,
  used for `svh` (output incoherence scale).

### Rule

```python
# Input Hadamard (suh is pre-scale):
ext.had_r_128(x, out, suh, None, 1.0)

# Output Hadamard (svh is post-scale):
ext.had_r_128(y, out, None, svh, 1.0)
```

In `hadamard_rotate_128` this is controlled by the `pre_scale` boolean parameter:

```nim
# Input: pre_scale=true
let xh = hadamard_rotate_128(xf16, self.suh, INV_SQRT_128, pre_scale=true)

# Output: pre_scale=false (then apply svh manually)
let yh = hadamard_rotate_128(result, nil, INV_SQRT_128, pre_scale=false)
result = yh * self.svh
```

The ENTIRE `hadamard_rotate_128` call (including scale, not just the FWHT)
happens in fp32 before the final fp16 conversion.

---

## 5.  Normalisation (1/√128)

`ext.had_r_128` always divides the result by √128 internally, before
applying the `norm` multiplier.
```
output = FWHT(input) / √128 × norm
```

So to undo the internal division, the `norm` parameter passed by the caller is
interpreted as an **extra** multiplier on top of `1/√128`, so in practice:
- `norm = 1.0` → output = `FWHT(input) / √128 × 1.0` = `FWHT(input) / √128`.
- `norm = 1/√128` → output = `FWHT(input) / √128 × 1/√128` = `FWHT(input) / 128`.

The Nim reimpl does NOT have this baked-in `1/√128`, so it must apply it
explicitly via `INV_SQRT_128` (= `0.088388347648`):

```nim
# Input:  FWHT(x) / √128 × 1.0     = FWHT(x) / √128
xh = hadamard_rotate_128(x, suh, INV_SQRT_128, pre_scale=true)

# Output: FWHT(y) / √128 × 1.0     = FWHT(y) / √128
yh = hadamard_rotate_128(y, nil, INV_SQRT_128, pre_scale=false)
result = yh * svh
```

---

## 6.  Attention / GQA

The attention fixture uses `repeat_interleave` to expand the K/V heads
before calling `F.scaled_dot_product_attention` (standard SDPA):

- the C++ `enable_gqa=True` path inside the Nim `GroupedQueryAttention`
  produces slightly different fp16 results (differences of ~2 ULPs per element)
- the deterministic choice avoids cascading fp16 differences, the Nim
  attention test follows the Python generator convention, pre-expanding
  the K/V heads and using standard SDPA without `enable_gqa`

---

## 7.  Random seed determinism

Every generator calls `torch.manual_seed` with a file-level seed constant.
CUDA determinism flags are set:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This guarantees identical fixture values across separate Python invocations
on the same GPU architecture.

---

## 8.  Exponential layer sampling for codec fixtures

The codec fixtures (`exl3/`) store the **full decoded FP16 weight** plus
the quantized trellis for each linear projection.

Storage for all 28 layers × 7 projections (196 fixtures) runs ~1.1 GB, so
the fixtures are generated only for an **exponential subset** of layers:

| Layer | Rationale |
|---|---|
| 0 | First layer — simplest residual state |
| 1 | Second layer — immediately adjacent to first |
| 2 | Exponential step (×2) |
| 4 | Exponential step |
| 8 | Exponential step |
| 16 | Exponential step |
| 27 | Last layer (LAYER_COUNT - 1) — accumulates most residual |

This yields **49 fixtures** (7 layers × 7 projections) instead of 196, a ~75%
storage reduction while maintaining coverage of early, middle, and late layers.

The generator `gen_exl3_qwen3_00_codec.py` defaults to this mode, pass
`--all-layers` to cover all 28 layers (for a full verification run, not tracked in git),
or `--layer N` for a single layer.

---

## 9.  Generator file naming

Every fixture generator is named `gen_<quant>_<id>_<slug>_<model>.py`:

| Part | Rule | Examples |
|---|---|---|
| `<quant>` | omitted for the unquantized/bf16 path, a marker for a quantized one | `gen_exl3_*` carries `exl3`, bf16 files carry none |
| `<id>` | the consuming suite's slot, spelled exactly as that suite spells it | `02_first_8_layers_plus_final`, `03_full_forward_to_logits`, `04_greedy_text_generation`, `01_layer_internals`, `01_block_02_trace`, `00_codec`, `00_hadamard` |
| `<slug>` | the fixture concern (the fixture family directory name) | `codec`, `first-8-layers-plus-final`, `full-forward-to-logits`, `greedy-text-generation`, `layer-internals` |
| `<model>` | the checkpoint name | `Qwen3-0.6B`, `Qwen3.5-0.8B`, `Qwen3.6-35B-A3B` |

- the `<id>` names the fixture family the consuming suite names, so suite,
  generator and fixture directory carry the same name
- the model name lives in the filename, the concern name is the consuming
  suite spelling, never a private name of a port's own invention

A non-generator never wears the `gen_` prefix. The shared EXL3 helpers live
outside testgen/ in `tests/quant_utils/exl3_utils.py` (kernel-reconstruction and forward logic).

- every generator with no Qwen3 precedent carries a justification header
  at the top of its docstring, what it is, which suite consumes it, and why
  no Qwen3 analog exists
- the name itself is the justification when a Qwen3 file of the same concern exists

## 10.  JSON fixture payloads ship compressed

JSON fixture payloads ship as single zstd frames (`.json.zst`) so pretty-printed
numerics stay out of text diffs and history bloat.

Container contract:

- producers write level 19 frames with the content size and a checksum
  recorded in the frame header, the same shape from both producer sides
  (python `compression.zstd` and the Nim binding)
- suites inflate them in memory through the `zstdReadFixture` reader
  (`harness/recording.nim`, vendored zstd binding) and parse through
  jsony against a declared schema type
- the reader asserts the recorded content size instead of guessing buffers,
  a corrupt or content-size-unknown frame raises an error
- retired containers keep their re-record records outside the repo,
  the retired `.json.zip` paths are removable by a git history rewrite
  that drops the retired blobs

Every fixture data file is blob material, review the generators plus
the harness, never the recorded payloads. The recorded json sidecars ship
as `.json.zst` frames the same way, the reader is `readJsonFixture` (`harness/recording.nim`).

- the system dynlib is the default binding path (`TTT_USE_SYSTEM_ZSTD=true`)
- the vendored static build is the opt-in (`-d:TTT_USE_SYSTEM_ZSTD=false`)
  and requires a materialized `workspace/zstd/vendor/zstd` submodule
- on macOS the binary self-locates the dylib through a baked rpath, no
  environment variables involved
- the round-trip test (`workspace/zstd/tests/t_zstd_roundtrip.nim`) documents both invocations
- generators emit the frames directly

## 11.  Size budgets for fixture files

New fixture files respect a size budget so the tree weight stays flat:

- hard cap 256 kiB per new fixture file, a larger fixture needs a reviewed
  exception before it is added
- soft target 64 kiB for committed text payloads (decisions, sidecars), advisory
- per-model fixture directory total 1.5 MiB over the first two path
  components under `tests/fixtures/`, a directory that new files push past
  the total is a defect, directories already over the budget are recorded
  baseline exceptions until the payload shrinks

## 12.  Re-recording the EXL3 fixtures on the CUDA box

The production EXL3 kernels are the ground truth, a re-record runs on the CUDA
box with `exllamav3_ext` importable. One env preamble from the repo root:

    export CUDA_HOME="$(pwd)/.venv/lib/python3.14/site-packages/nvidia/cu13"  # the CUDA toolkit root
    export PATH="$(pwd)/.venv/bin:$PATH"  # the venv tools first on the path

- the venv ships `exllamav3_ext`, the generators locate it through the normal import
- the preamble above is the only environment setup
- the backend dispatch is one shape across the rung 01/03/04 generators
- `exllamav3_cuda` when the `exllamav3_ext` module imports and a CUDA device exists
- the pure-torch `pytorch` fallback otherwise
- run every generator from `workspace/transformers/tests` via `uv run python testgen/<script>`

Per script:

| script | invocation | emits | provenance |
|---|---|---|---|
| `gen_exl3_qwen3_00_codec.py` | `--device cuda:0 --backend both --check` | `fixtures/exl3-00-codec/Qwen3-0.6B-EXL3-5bpw/` payloads plus metadata frames | metadata `backend: "both"`, the decoder max diff printed |
| `gen_exl3_qwen3_01_layer_internals.py` | plain | `fixtures/exl3-01-layer-internals/Qwen3-0.6B-EXL3-5bpw-layer-0/` payloads, `.metadata.json.zst`, 004 stats frames | metadata `backend: "exllamav3_cuda"` |
| `gen_exl3_qwen3_03_full_forward_to_logits.py` | plain | per-layer `layer-NN.safetensor.stats.json.zst` (004, fp16 grid) plus `final_logits.decisions.json.zst` (005, `ulp_datatype` "fp16") | `Backend: exllamav3_cuda` banner, the pre-write guard compares against the committed frame |
| `gen_exl3_qwen3_04_greedy_text_generation.py` | plain | per-prompt `<prompt>.json.zst` (001 greedy steps) plus `<prompt>.decisions.json.zst` (005, `ulp_datatype` "fp16") | `Backend: exllamav3_cuda` banner, env frame `recorded_from` |

Post-run checks:

- `uv run python linters/lint_fixtures.py .` stays clean
- the consuming suites rerun green:
  - `q_exl3/t_exl3_qwen3_00_codec`, `t_exl3_qwen3_00_hadamard`, `t_exl3_qwen3_01_layer_internals`
  - `t_exl3_qwen3_03_full_forward_to_logits` plus `t_exl3_qwen3_04_greedy_text_generation` rerun green
- a pre-write guard rejection means the machine does not reproduce the committed recording
- the guards fire on the 00 codec --check diff and the 03 argmax/margin comparison
- stop and take the rejection to review, never force a re-record over it

The 00 codec is the payload family, its fixtures record the pure-torch
decoder by design:
- the recorded default is `--backend pytorch`
- the verification runs `--backend both --check` on the CUDA box
