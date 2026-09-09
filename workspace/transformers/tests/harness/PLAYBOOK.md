# PLAYBOOK: transformer test suites

How to write and run a suite against the harness. SPEC.md states the
check semantics.

## New-suite checklist

1. File at `workspace/transformers/tests/<area>/t_<name>.nim`, license
   header, compile command in the module doc comment.
2. Imports: `workspace/libtorch as F`, `workspace/libtorch_testutils`,
   `workspace/transformers/tests/harness/harness`, the module under test.
3. One `runCppTest` per unit under test. Body: build inputs (seeded
   `Torch.manual_seed` for property mode, `Safetensor.open` for wiring
   mode, the comparison against recorded fixtures), call the unit, assert.

```nim
import
  std/os,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/transformers/src/layers/norm,
  workspace/transformers/tests/harness/harness

proc main() =
  runCppTest "RmsNormOne vs fixture":
    proc(): bool =
      var st = Safetensor.open(FixtureDir / "norm.safetensor")
      let x = st.getTensorOwned("input")
      let w = st.getTensorOwned("weight")
      let expected = st.getTensorOwned("output")
      let norm = RmsNormOne.init(w, eps = 1e-6)
      let got = norm.forward(x)
      # deterministic op: bit-exact against the recording
      assertAllClose(got, expected, rtol = 0.0, abstol = 0.0)
      # analytic property on the same tensors
      checkRmsNormScalePreservation(x, w, got, biasOne = true)
      true

when isMainModule:
  main()
```

4. Per unit: one invariant call (property) plus one tolerance call
   (comparison). Deterministic ops compare bit-exact against recordings.
   Same-device computed outputs compare under a budget row, never
   bit-exact.
5. For op outputs feeding a fingerprint family: compute the stats file with
   harness/gen_stats.nim (see below), then check the output with
   `assertMatchRate` plus `assertStats` under the op's budget row.

## Fingerprint stats files

Bootstrap (once per fixture family, committed):

    nim cpp -r --hints:off --warnings:off --outdir:build/tools \
      --nimcache:nimcache/tools \
      workspace/transformers/tests/harness/gen_stats.nim

Recorded fixtures and their stats tensors are listed in gen_stats.nim.
Histograms are stored only on the margin-critical tensor per family, add a new
tensor as `(name, false)` unless it feeds small-margin steps.
Regenerate only after a sanctioned re-record; the stats file is committed
data and its diff must be reviewed like a fixture change.

## Running

- One suite: `nim test_tf_<suite>` (per-suite tasks in
  config.nims), one family: `nim test_tf_family name=...`.
  The TTT_TEST_ON environment value flips the device
  (harness/device.nim): `TTT_TEST_ON=cpu nim test_tf_family
  name=chain` runs the chain suites on the reference device.
- Full set: `nim test_transformers` (config.nims scans tests/,
  tests/q_bf16/, tests/q_exl3/ for `test_*` / `t_*` files,
  non-recursive: a new test subdir needs a `getTestCommands` line in
  config.nims). Final verification only, the per-suite tasks cover
  everyday runs.
- Single file:
  `nim cpp -r --hints:off --warnings:off --outdir:build/tests/<name>
  --nimcache:nimcache/tests/<name> <path>`
- Harness selftest: part of the suite through
  tests/harness/t_harness_selftest.nim.

## Extending the fault corpus

Add the fault kind to `FaultKind`, implement it in `applyFault`, add a
rejection case in `runSelftest`, and add a detection-floor row to the
selftest module doc. A fault the current checks cannot detect (see
tail probability vs row sums) needs a new check, not a bigger corpus.

## Recordings

Sidecars load with `zstdReadFixture(zipPath)` (one JSON entry, jsony-parsed
by the suite). After a re-record pass, regenerate PROVENANCE.md:

```nim
writeProvenance("PROVENANCE.md", @[
  ("python", "3.14.7"), ("torch", "2.14.0"),
  ("transformers", "5.16.1"), ("recorded_from", "m4max-cpu")])
doAssert verifyProvenance("PROVENANCE.md")
```
