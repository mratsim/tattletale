# Writing Docs — Examples

Banned vocabulary with replacements, and before/after pairs from actual slop
seen in this repo.

## Banned vocabulary (hard blocklist)

Campaign and harness words that must NEVER appear in any committed doc
comment, module header, or prose. A reviewer greps for these, so even a
domain-legitimate use should be rephrased to the replacement below.

| Banned | Replacement |
|---|---|
| `pin` / `pinned` | "verified against", "checked by" (never "pins behavior"). CUDA page-locked memory: write "page-locked", never "pinned" |
| `gate` / `gates` / `gating` (for tests) | "test", "check", "assert" |
| `draw` / `draws` (test/harness verb) | "read", "take", "use" |
| `bite` | "chunk", "step", "case" |
| `mission` (campaign phase label) | the module's real name or path ("ex02a", never "Mission 02") |
| `fingerprint` | "identifier", "signature" |
| `digest` (test-run summary sense) | "summary", "report" (the cryptographic sense stays: "Returns the SHA-256 digest" is canonical) |
| `sentinel` | "marker", "guard value" |
| `mutation` (test-theater sense) | "change", "variation" |
| `RED` / `GREEN` (test-status theater) | never. State the invariant or behavior in present tense |
| `oracle` | "reference implementation" |
| `probe` | "test". Test files are `test_*.nim`, never "probes" |
| `harness` | "test suite", "runner" |
| `smoke` (as in "smoke test") | "sanity", "quick", "basic" |
| `deviation class` | describe the actual difference (ulps, bytes, layout) |
| `load-bearing` | "essential", "critical", "necessary" ("the load-bearing guard" → "the essential guard") |
| `seam` | "boundary", "interface", "edge" ("the seam between tiles" → "the boundary between tiles") |

## Before → after

From actual slop seen in this repo:

| Sloppy (banned) | House style |
|---|---|
| "pinned by the probe's host mirror" | "verified against the host reference" |
| "probe gates cb2 with that deviation class" | "cb2 matches the CUDA-faithful rounding to within a few fp16 ulps" |
| "Missions 02/03 import this module" | "The ex02a microkernel examples import this module" |
| "the harness draws a sample from the fixture" | "the test reads a sample from the fixture" |
| "gates this check with RED until the fix" | "checks that the packed layout round-trips" |
| "the oracle emits the reference layout" | "the reference implementation emits the expected layout" |
| "smoke test covers the load path" | "sanity test covers the load path" |
| "a sentinel row marks the tile end" | "a marker row ends the tile" |
