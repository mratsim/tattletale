# Fixture Generation Conventions

This document records the conventions and invariants that fixture generators must
follow to produce deterministic, production-faithful test data for the Nim
inference pipelines.

## Fixture payload tiering rules

Quantization fixture families mirror the bf16 contract (stats + descriptor
sidecars, summary-surface comparisons) except the hash-anchored 00 families,
whose bit-exactness contract requires the raw inputs plus the recorded kernel
hash. The exl3 families carry the bf16 bounds with the ulp unit taken in
fp16, because EXL3 dequantizes to fp16; every bound comparison is inclusive,
a value landing exactly on the bound passes.


A fixture family ships sidecar plus slices plus PROVENANCE.md, never raw output
tensors except the value-bearing slices:

- **Decision payloads** (greedy chains, final logits) are recorded as compact
  JSON decisions: greedy chains carry the chosen token, top-32 ids and f32
  logits, argmax margin and the explicit `tail_probability` key per step
  (ttt-tf-001-greedy-steps-h2). Final logits carry the decision projection
  ttt-tf-002-logit-decisions-probe-h2, per
  position the argmax id, the top-2 competing pair with f32 logits, the argmax
  margin and the softmax probability beyond the top-2 pair, plus the strided
  512-word bit-exact probe row of every deciding row. The [1,6,248320]
  raw-logits class shrinks from megabytes to tens of KB, consumers read argmax
  and top-2 only.
- **Boundary slices** that feed bit-exact or band-compared boundaries stay raw
  (layer inputs/outputs, GDN conv/state trajectories), they are the
  value-bearing slices of the family.
- **Fingerprint sidecars** (`<fixture>.stats.json.zst` frames, payload
  byte format of harness/gen_stats.nim) carry order statistics and
  histograms so removed raw tensors keep a distribution check.
- **PROVENANCE.md** per fixture family, generated at record time with
  `fixture_stats.write_provenance`:
  date, dtype, generator, model, platform, python/torch/transformers
  versions, recorded_from, seed, kept light enough to
  recreate the recording run.
  No device row and no num_threads row:
  the recording device derives from the recorded_from row, its last
  dash-separated component (harness/device.nim `recordedDevice`).
  No git metadata: a re-run depends only
  on the recorded versions and the generator beside the fixtures, and
  hashes dangle under squash merges and rewrites.
- **GDN family frozen truth**: the GDN conv/state fixtures regenerate
  byte-identically against the frozen recordings (the re-record records are
  archived with the work reports). A regeneration that differs must stop the
  change until the cause is found (a stale assumption, a code defect, or the
  wrong model): run the force-fallback verification first, then escalate
  before accepting a new canonical truth for the family. Never silently
  re-record GDN.
- **Re-record records** (fixture file checksums before and after, plus the
  re-run results) are archived with the work reports.
  Old raw payloads leave the tree only after the family successor passes.

### Recording environment

Fixture generation runs in the uv-synced environment:

    uv sync --group test-vectors

CAUTION: the group pins torch==2.11.0, so a sync DOWNGRADES torch from the
recorded 2.14.0. Restore the recorded version after every sync, then re-check
that `torch.__version__` matches the version recorded in the fixture metadata
before generating:

    uv pip install torch==2.14.0

The installed transformers (5.16.1) is the source of truth:
no vendored checkout is consulted, `_references_*` never appears in comments or
fixture generation, and hub kernel packages are absent so kernel dispatch falls
back to the pure-torch bodies, which is the reference behavior.

### Recording box

`recorded_from` names the recording box plus device ("m4max-cpu",
"rtxpro6000-cuda"). The default recording box stays m4max-cpu; a recording on
another box sets `TTT_RECORD_FROM` in the environment, the value lands in the
PROVENANCE.md rows and in the env frame of the greedy fixtures. Every EXL3
fixture family writes its PROVENANCE.md at record time through
`fixture_exl3_common.write_family_provenance`.

### Fixed quantile method and bucket spec

- Quantiles: exact order statistic of the ascending sort, index
  `floor(p * (n - 1))`, no interpolation, computed in f64 indices over values
  promoted from the native dtype to f32 for storage. Fixed
  probabilities: 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, plus
  min and max, stored as hex f32 bit patterns.
- Histograms: binade-log soft buckets over the bf16 bit pattern. Bucket =
  (sign, binade -64..63, mantissa top 6 bits), 14-bit keys in sparse sorted
  storage, plus dedicated zero (65534) and subnormal (65535) bins. Soft
  bucketing: each element contributes (2 - low) toward its bucket and low
  toward the neighbor bucket, low being the dropped 8th mantissa bit, integer
  math, bit-exact cross-implementation. Bucket width is 4 ulp. NaN and
  +Inf raise an error, -Inf is whitelisted only through allow_minus_inf and stays
  unbucketed (a quantile-only path).
- The cross-implementation contract is carried by the committed stats corpus:
  tests/harness/stats-corpus, verified inside the harness selftest, python and Nim
  fingerprints bit-exact on order statistics and histograms, the histogram
  total exact as the summation statistic.

### Fixture classes and the family ladder

- The fixture tree carries exactly two quant classes:
  the bf16-* families and the exl3-* families, both under tests/fixtures/.
- The bf16 and exl3 family prefixes climb one ladder:
  the number is the rung.
- Rung 00:
  codec and hadamard primitives, no layer context.
- Rung 01:
  one decoder layer, its internal operations.
- Rung 02:
  several decoder layers in sequence.
- Rung 03:
  the whole forward pass to logits.
- Rung 04:
  autoregressive text generation.
- Each rung contains the previous rung plus more:
  a missing rung is simply not yet recorded.
- Harness self-test material lives in tests/harness/, never under fixtures/:
  the stats corpus sits at tests/harness/stats-corpus, it is the
  instrument-check input of the harness machinery.

### The exl3 families

- Rung 00 (codec, hadamard): packed trellis inputs plus metadata frames with
  the production-CUDA-kernel weight hash, bit-exactness the contract, no
  stats sidecars.
- Rung 01 layer internals: fingerprint stats sidecars per payload,
  quantile-only entries on the linear outputs and histogram entries on the
  attention and block outputs, the margin-critical class. The suites compare
  the per-op ulp rows on the recording device and the chain checkpoint band
  elsewhere, elementwise plus stats.
- Rung 01 block-02 trace: quantile-only stats entries per stage tensor, the
  chained stages accumulating drift linearly under the stage-indexed bound,
  the bit-exact input-layernorm anchor resetting the count.
- Rung 03 full forward: per-layer quantile-only stats entries beside the
  raw boundary payloads, the final logits decision projection
  (ttt-tf-002-logit-decisions-probe-h2) plus the frozen quantile entry of the
  retired raw tensor. The 1.7 MB raw logits tensor stays out of the tree.
- Rung 04 greedy: ttt-tf-001-greedy-steps-h2 step records (top-32 support, argmax margin,
  tail probability); the suites replay with teacher-forced tie recovery.
- Record-time sidecars come from the generators; the mac battery between
  record waves regenerates the identical frames from the committed payload
  bytes through testgen/fixture_sidecar_bootstrap.py. The raw logits payload
  is the one frozen source: with it retired, the committed decision and
  stats frames stay frozen data.
- Tier-1 synthetic product-property tests live in tests/synthetic/:
  they are fixture-free by construction, no fixture material involved.

## Guiding principle

**The production EXL3 kernel (`exllamav3_ext`) is the ground truth.**

Nim reimplementations (e.g. `hadamard_rotate_128`, `linear.forward`) are tested
AGAINST fixtures generated by the C++ kernel.  If a reimplementation disagrees
with the kernel, the reimplementation must be fixed — never adjust the fixture
or the test tolerance.

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

Fixtures must use the **kernel-native** layout (`[in_features, out_features]`,
non-transposed).  The fixture generator calls `linear_forward_orig_exl3` which
takes non-transposed weight and feeds it directly to `ext.hgemm`.

The Nim test loads weights via `Linear.load`, which already stores them in the
layout expected by `Linear.forward` (transposed for `F.linear`).  If the Nim
`Linear.forward` is ever changed to call `ext.hgemm` directly, the load function
must be updated to store non-transposed weights.

**Do NOT transpose in the fixture generator.**  The generator calls the
production kernel directly — there is no `F.linear` wrapper that would require
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

The fixture saves a **3D** cos/sin tensor `[batch, seq, head_dim]` obtained by
slicing the 2D table at the same `position_ids` for all batch items.  The Nim
test normalises the fixture entry back to 2D via:

```nim
let hfCos2d = if hfCos.dim == 3: hfCos[b] else: hfCos
```

Both sides produce the same values (verified by `assertAllClose` with
`rtol=1e-5` before the full-attention comparison).

---

## 3.  Precision of the Hadamard transform

### Convention

The production kernel `ext.had_r_128(a, b, pre_scale, post_scale, norm)`
performs all arithmetic in **fp32**:

```
output = FWHT(input ⊙ pre_scale) ⊙ post_scale × norm / √128
```

The Nim reimplementation `hadamard_rotate_128` must also work in fp32 for the
FWHT butterflies and scale/norm multiplications.  Converting intermediate
values to fp16 between stages loses precision and produces results that differ
from the kernel by ~0.0003–0.0005 absolute, which exceeds `Tol = 1e-4`.

### Rule

Every stage of `hadamard_rotate_128` (scale, FWHT butterfly, norm multiply)
MUST happen in fp32.  The final result is converted to fp16 only at the very
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

The CUDA kernel `ext.had_r_128` has two scale arguments:
- **`pre_scale` (3rd arg)**: applied element-wise **before** the FWHT.
  Used for `suh` (input incoherence scale).
- **`post_scale` (4th arg)**: applied element-wise **after** the FWHT.
  Used for `svh` (output incoherence scale).

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

`ext.had_r_128` always divides the result by √128 internally before applying
the `norm` multiplier:
```
output = FWHT(input) / √128 × norm
```

So to undo the internal division, the `norm` parameter passed by the caller is
interpreted as an **extra** multiplier on top of `1/√128`.  In practice:
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

The attention fixture uses `repeat_interleave` to expand K/V heads before
calling `F.scaled_dot_product_attention` (standard SDPA).  This differs from
the C++ `enable_gqa=True` path in the Nim `GroupedQueryAttention`, which
produces slightly different fp16 results (differences of ~2 ULPs per element).

To keep the test deterministic and avoid cascading fp16 differences, the
Nim attention test should match the Python generator's convention: pre-expand
K/V heads and use standard SDPA without `enable_gqa`.

---

## 7.  Random seed determinism

Every generator calls `torch.manual_seed` with a file-level seed constant.
CUDA determinism flags are set:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This guarantees identical fixture values across separate Python invocations on
the same GPU architecture.

---

## 8.  Exponential layer sampling for codec fixtures

The codec fixtures (`exl3/`) store the **full decoded FP16 weight** plus the
quantized trellis for each linear projection.  To keep storage tractable (~1.1 GB
for all 28 layers × 7 projections = 196 fixtures), fixtures are generated only
for an **exponential subset** of layers:

| Layer | Rationale |
|---|---|
| 0 | First layer — simplest residual state |
| 1 | Second layer — immediately adjacent to first |
| 2 | Exponential step (×2) |
| 4 | Exponential step |
| 8 | Exponential step |
| 16 | Exponential step |
| 27 | Last layer (LAYER_COUNT - 1) — accumulates most residual |

This yields 7 layers × 7 projections = **49 fixtures** instead of 196, a ~75%
storage reduction while maintaining coverage of early, middle, and late layers.

The generator `gen_exl3_codec_fixtures.py` defaults to this mode.  Pass
`--all-layers` to generate for all 28 layers (e.g. for a full verification run,
not tracked in git), or `--layer N` for a single layer.

---

## 9.  Generator file naming

Every fixture generator is named `gen_<quant>_<id>_<slug>_<model>.py`:

| Part | Rule | Examples |
|---|---|---|
| `<quant>` | omitted for the unquantized/bf16 path, a marker for a quantized one | `gen_exl3_*` carries `exl3`, bf16 files carry none |
| `<id>` | the consuming suite's slot, spelled exactly as that suite spells it | `02_first_8_layers_plus_final`, `03_full_forward_to_logits`, `04_greedy_text_generation`, `01_layer_internals`, `01_layer_internals_attn`, `01_layer_internals_gdn`, `01_layer_internals_moe`, `01_block_02_trace`, `00_codec`, `00_hadamard` |
| `<slug>` | the fixture concern (the fixture family directory name) | `first-8-layers-plus-final`, `full-forward-to-logits`, `greedy-text-generation` |
| `<model>` | the checkpoint name | `Qwen3-0.6B`, `Qwen3.5-0.8B`, `Qwen3.6-35B-A3B` |

The `<id>` is the fixture family the consuming suite names, so suite,
generator and fixture directory carry the same name. The model name lives in the
filename, and the concern name is the one the consuming suite spells, never
a private name of a port's own invention.

A non-generator never wears the `gen_` prefix. Shared helpers take
`fixture_*`: `fixture_exl3_common.py` (kernel-reconstruction and forward logic).

Every generator with no Qwen3 precedent carries a justification header at
the top of its docstring: what it is, which suite consumes it, and why no
Qwen3 analog exists. The name itself is the justification when a Qwen3
file of the same concern exists.

## 10.  JSON fixture payloads ship compressed

JSON fixture payloads ship as single zstd frames (`.json.zst`) so
pretty-printed numerics stay out of text diffs and history bloat.

Container contract:

- producers write level 19 frames with the content size and a checksum
  recorded in the frame header, the same shape from both producer sides
  (python `compression.zstd` and the Nim binding)
- suites inflate them in memory through the `zstdReadFixture` reader
  (`harness/recording.nim`, vendored zstd binding) and parse through
  jsony against a declared schema type
- the reader asserts the recorded content size instead of guessing
  buffers, a corrupt or content-size-unknown frame raises an error
- retired containers keep their re-record records outside the repo,
  the retired `.json.zip` paths are removable by a git history rewrite
  that drops the retired blobs

Every fixture data file is blob material: review the generators and
the harness, never the recorded payloads. The recorded
json sidecars ship as `.json.zst` frames the same way, the reader is
`readJsonFixture` (`harness/recording.nim`).

The system dynlib is the default binding path
(`TTT_USE_SYSTEM_ZSTD=true`), the vendored static build is the
opt-in (`-d:TTT_USE_SYSTEM_ZSTD=false`) and requires a materialized
`workspace/zstd/vendor/zstd` submodule. On macOS the binary
self-locates the dylib through a baked rpath, no environment
variables involved. The round-trip test
(`workspace/zstd/tests/t_zstd_roundtrip.nim`) documents both
invocations. Generators emit the frames directly.

## 11.  Size budgets and the ratchet check

New fixture files respect a size budget so the tree weight stays flat:

- Hard cap 256 kiB per new fixture file. A larger fixture needs an
  explicit operator decision before it is added.
- Soft target 64 kiB for committed text payloads (decisions,
  sidecars). The target is advisory; the check reports it as a note.
- Per-model fixture directory total 1.5 MiB. A directory (first two path
  components under `tests/fixtures/`) that the new files push past the
  total is a defect; directories already over the budget at the check's
  base are recorded baseline exceptions, not failures.

`testgen/check_fixture_filesize.py` is the check. Call it with:
`python3 check_fixture_filesize.py <repo> <base> <head-or-''> [paths...]`. It prints one line per finding plus a summary block and always
exits 0; automation greps the `summary: defect kinds: {...}` line for a
non-empty dict. Run it once with `<base>` set to `-` to list the current
over-budget files and directories as baseline exceptions (sweep mode).
Baseline exceptions are removed when their payload is removed (zip,
projection, or deletion), never by editing the budget constants.
